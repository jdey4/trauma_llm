#!/usr/bin/env python3
"""
Compare clinician and model free-text responses using Gemini embeddings.

What this script does
---------------------
1. Loads clinician responses from the survey Excel file.
2. Loads model responses from output_free_text.csv.
3. Normalizes case IDs so clinicians and models align correctly.
4. Embeds both clinicians and model responses using Gemini embeddings.
5. Runs a joint MDS sweep over dimensions using cosine distances.
6. Saves stress-vs-components plot and chooses an optimal dimension by elbow.
7. Saves pairplots of the shared MDS embedding, color-coded by Clinician/Leah/Gemini/GPT.
8. Repeatedly samples 5 cases and computes MMD(model, experts) minus expert split-half MMD.
9. Saves a strip plot of deviation-from-experts for each model.

Requirements
------------
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl google-genai python-dotenv
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances, pairwise_distances
from dotenv import load_dotenv

try:
    from google import genai
except ImportError as e:
    raise ImportError(
        "Missing google-genai. Install with: pip install google-genai"
    ) from e


load_dotenv()

RESPONSE_PREFIX = "How would you personally respond to this case?"
YEARS_COL = "How many years of experience do you have in your field?"
SPECIALTY_COL = "What is your medical specialty?"
NAME_COLS = ["First name", "Last name"]


# -----------------------------
# Utilities
# -----------------------------
def normalize_case_num(x) -> int | None:
    """
    Robustly convert case labels like:
      3
      '3'
      'case_3'
      'Case 3'
      'prompt_03'
    into integer case IDs.
    """
    if pd.isna(x):
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float) and x.is_integer():
        return int(x)

    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))
    return None


def clean_response_text(df: pd.DataFrame, text_col: str = "response_text") -> pd.DataFrame:
    df = df.copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].notna()]
    df = df[df[text_col].ne("")]
    df = df[df[text_col].str.lower().ne("nan")]
    return df.reset_index(drop=True)


# -----------------------------
# Data loading
# -----------------------------
def load_clinician_responses(
    excel_path: str | Path,
    sheet_name: str = "Experts",
    response_prefix: str = RESPONSE_PREFIX,
) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    response_cols = [c for c in df.columns if c.startswith(response_prefix)]
    if not response_cols:
        raise ValueError(
            f'No response columns found starting with "{response_prefix}" in sheet "{sheet_name}".'
        )

    out_frames = []
    for i, col in enumerate(response_cols, start=1):
        tmp = pd.DataFrame({
            "response_id": [f"clinician_case{i}_{j}" for j in range(len(df))],
            "source": "Clinician",
            "model_name": "Clinician",
            "case_num": i,
            "response_text": df[col],
            "clinician_id": np.arange(len(df)),
            "clinician_name": (
                df.get(NAME_COLS[0], pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
                + " "
                + df.get(NAME_COLS[1], pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
            ).str.strip(),
            "years_experience": pd.to_numeric(df.get(YEARS_COL, np.nan), errors="coerce"),
            "specialty": df.get(SPECIALTY_COL, pd.Series([np.nan] * len(df))),
        })
        out_frames.append(tmp)

    long_df = pd.concat(out_frames, ignore_index=True)
    long_df = clean_response_text(long_df, "response_text")
    long_df["case_num"] = long_df["case_num"].apply(normalize_case_num)
    long_df = long_df[long_df["case_num"].notna()].reset_index(drop=True)
    long_df["case_num"] = long_df["case_num"].astype(int)
    return long_df


def load_model_responses(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    rename_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "response":
            rename_map[c] = "response_text"
        elif cl == "model":
            rename_map[c] = "model_name"
        elif cl == "case":
            rename_map[c] = "case_num"
        elif cl == "rep":
            rename_map[c] = "rep"

    df = df.rename(columns=rename_map)

    required = ["response_text", "model_name", "case_num"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in model CSV: {missing}")

    if "rep" not in df.columns:
        df["rep"] = 0

    df["model_name"] = df["model_name"].astype(str).str.strip()
    df["case_num"] = df["case_num"].apply(normalize_case_num)

    df = clean_response_text(df, "response_text")
    df = df[df["case_num"].notna()].reset_index(drop=True)
    df["case_num"] = df["case_num"].astype(int)

    df["source"] = "Model"
    df["response_id"] = [
        f"{m}_case{c}_rep{r}_{i}"
        for i, (m, c, r) in enumerate(zip(df["model_name"], df["case_num"], df["rep"]))
    ]

    keep_cols = ["response_id", "source", "model_name", "case_num", "response_text", "rep"]
    return df[keep_cols].reset_index(drop=True)


# -----------------------------
# Gemini embeddings
# -----------------------------
def make_genai_client(api_key: str | None = None):
    api_key = (
        api_key
        or os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key:
        raise EnvironmentError(
            "No Gemini API key found. Put GOOGLE_GENERATIVE_AI_API_KEY or GOOGLE_API_KEY in your .env file."
        )
    return genai.Client(api_key=api_key)


def get_gemini_embedding(
    client,
    text: str,
    model: str = "gemini-embedding-001",
    max_retries: int = 5,
    initial_sleep: float = 2.0,
) -> np.ndarray:
    sleep_s = initial_sleep
    last_err = None

    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=model,
                contents=text,
            )

            if hasattr(result, "embeddings") and len(result.embeddings) > 0:
                return np.array(result.embeddings[0].values, dtype=np.float32)

            raise RuntimeError("Embedding response did not contain embeddings.")

        except Exception as e:
            last_err = e
            if attempt == max_retries - 1:
                break
            print(f"Embedding failed on attempt {attempt + 1}/{max_retries}. Retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)
            sleep_s *= 2.0

    raise RuntimeError(f"Failed to embed text after {max_retries} attempts: {last_err}")


def embed_dataframe(
    df: pd.DataFrame,
    text_col: str = "response_text",
    embedding_model: str = "gemini-embedding-001",
    cache_path: str | Path | None = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    if cache_path is not None and Path(cache_path).exists():
        cached = pd.read_json(cache_path)
        cached["embedding"] = cached["embedding"].apply(np.array)
        X = np.vstack(cached["embedding"].values)
        return cached, X

    client = make_genai_client()
    embeddings = []

    for idx, text in enumerate(df[text_col].tolist(), start=1):
        emb = get_gemini_embedding(client=client, text=text, model=embedding_model)
        embeddings.append(emb)
        if idx % 25 == 0 or idx == len(df):
            print(f"Embedded {idx}/{len(df)} responses")

    out_df = df.copy()
    out_df["embedding"] = [x.tolist() for x in embeddings]
    X = np.vstack(embeddings)

    if cache_path is not None:
        out_df.to_json(cache_path, orient="records", indent=2)

    return out_df, X


# -----------------------------
# Joint MDS sweep
# -----------------------------
def fit_mds_range(
    X: np.ndarray,
    min_components: int = 1,
    max_components: int = 50,
    random_state: int = 42,
    distance_metric: str = "cosine",
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    """
    Run joint MDS across a range of dimensions on the combined clinician+model embeddings.
    """
    max_allowed = min(X.shape[0] - 1, X.shape[1])
    max_components = min(max_components, max_allowed)

    if max_components < min_components:
        raise ValueError("max_components is smaller than min_components after adjustment.")

    print(f"Requested joint MDS sweep up to {max_components} dimensions (max allowed = {max_allowed})")

    if distance_metric == "cosine":
        mds_input = cosine_distances(X)
        dissimilarity_mode = "precomputed"
    elif distance_metric == "euclidean":
        mds_input = X
        dissimilarity_mode = "euclidean"
    else:
        raise ValueError("distance_metric must be 'cosine' or 'euclidean'.")

    stress_rows = []
    projections: Dict[int, np.ndarray] = {}

    for k in range(min_components, max_components + 1):
        mds = MDS(
            n_components=k,
            metric=True,
            n_init=4,
            max_iter=300,
            eps=1e-6,
            dissimilarity=dissimilarity_mode,
            normalized_stress="auto",
            random_state=random_state,
        )
        coords = mds.fit_transform(mds_input)
        stress_rows.append({"n_components": k, "stress": float(mds.stress_)})
        projections[k] = coords
        print(f"Finished joint MDS for k={k}, stress={mds.stress_:.4f}")

    return pd.DataFrame(stress_rows), projections


def choose_elbow_dimension(stress_df: pd.DataFrame) -> int:
    pts = stress_df[["n_components", "stress"]].to_numpy(dtype=float)

    if len(pts) <= 2:
        return int(pts[min(1, len(pts) - 1), 0])

    p1 = pts[0]
    p2 = pts[-1]
    line_vec = p2 - p1
    line_norm = np.linalg.norm(line_vec)

    if line_norm == 0:
        return int(pts[0, 0])

    distances = []
    for p in pts:
        dist = np.abs(np.cross(line_vec, p - p1)) / line_norm
        distances.append(dist)

    elbow_idx = int(np.argmax(distances))
    return int(pts[elbow_idx, 0])


def save_stress_plot(
    stress_df: pd.DataFrame,
    optimal_k: int,
    outpath: str | Path,
) -> None:
    sns.set_context("talk")
    plt.figure(figsize=(9, 5.5))
    ax = sns.lineplot(data=stress_df, x="n_components", y="stress", marker="o")

    ax.axvline(optimal_k, linestyle="--", linewidth=1.2)
    ax.scatter(
        [optimal_k],
        [float(stress_df.loc[stress_df["n_components"] == optimal_k, "stress"].iloc[0])],
        s=110,
        zorder=5,
    )

    ax.set_title("Joint MDS stress vs. number of components")
    ax.set_xlabel("Number of MDS components")
    ax.set_ylabel("Stress")

    max_k = int(stress_df["n_components"].max())
    if max_k <= 20:
        ticks = list(range(1, max_k + 1, 2))
    elif max_k <= 50:
        ticks = list(range(1, max_k + 1, 5))
    else:
        ticks = list(range(1, max_k + 1, 10))
    if optimal_k not in ticks:
        ticks.append(optimal_k)
        ticks = sorted(set(ticks))
    ax.set_xticks(ticks)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def save_pairplot(
    projected_df: pd.DataFrame,
    dims_to_plot: List[str],
    outpath: str | Path,
) -> None:
    sns.set_context("talk")

    order = ["Clinician"] + [m for m in sorted(projected_df["model_name"].unique()) if m != "Clinician"]
    palette = dict(zip(order, sns.color_palette("Set2", n_colors=len(order))))

    cols = dims_to_plot + ["model_name"]

    g = sns.pairplot(
        projected_df[cols],
        vars=dims_to_plot,
        hue="model_name",
        hue_order=order,
        palette=palette,
        diag_kind="hist",
        corner=True,
        plot_kws={"alpha": 0.65, "s": 28},
    )
    g.figure.suptitle("Joint MDS pairplot of clinician and model Gemini embeddings", y=1.02)

    if g._legend is not None:
        g._legend.set_title("Group")
        for text in g._legend.texts:
            text.set_fontsize(9)

    g.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(g.figure)


# -----------------------------
# MMD
# -----------------------------
def median_heuristic_sigma(X: np.ndarray) -> float:
    if len(X) < 2:
        return 1.0
    dists = pairwise_distances(X, metric="euclidean")
    tri = dists[np.triu_indices_from(dists, k=1)]
    tri = tri[tri > 0]
    if len(tri) == 0:
        return 1.0
    return float(np.median(tri))


def rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    gamma = 1.0 / (2.0 * sigma * sigma + 1e-12)
    d2 = pairwise_distances(X, Y, metric="sqeuclidean")
    return np.exp(-gamma * d2)


def mmd2_unbiased(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    n = X.shape[0]
    m = Y.shape[0]

    if n < 2 or m < 2:
        return np.nan

    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)

    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    term_x = Kxx.sum() / (n * (n - 1))
    term_y = Kyy.sum() / (m * (m - 1))
    term_xy = 2.0 * Kxy.mean()

    return float(term_x + term_y - term_xy)


def casewise_mmd_against_experts(
    combined_df: pd.DataFrame,
    model_name: str,
    case_num: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Returns:
        model_vs_expert_mmd, expert_split_half_mmd
    """
    experts = combined_df[
        (combined_df["model_name"] == "Clinician") &
        (combined_df["case_num"] == case_num)
    ]
    model = combined_df[
        (combined_df["model_name"] == model_name) &
        (combined_df["case_num"] == case_num)
    ]

    if len(experts) < 4 or len(model) < 2:
        return np.nan, np.nan

    X_exp = np.vstack(experts["embedding"].apply(np.array).values)
    X_mod = np.vstack(model["embedding"].apply(np.array).values)

    pooled = np.vstack([X_exp, X_mod])
    sigma = median_heuristic_sigma(pooled)

    model_vs_exp = mmd2_unbiased(X_mod, X_exp, sigma)

    idx = rng.permutation(len(X_exp))
    half = len(idx) // 2
    if half < 2 or len(idx) - half < 2:
        return model_vs_exp, np.nan

    A = X_exp[idx[:half]]
    B = X_exp[idx[half:]]
    exp_vs_exp = mmd2_unbiased(A, B, sigma)

    return model_vs_exp, exp_vs_exp


def subsampled_mmd_experiment(
    combined_df: pd.DataFrame,
    n_cases_sample: int = 5,
    n_trials: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    all_models = sorted([m for m in combined_df["model_name"].unique() if m != "Clinician"])

    # Only keep cases that exist for clinicians and for every model
    expert_cases = set(combined_df.loc[combined_df["model_name"] == "Clinician", "case_num"].unique())
    common_cases = expert_cases.copy()
    for m in all_models:
        model_cases = set(combined_df.loc[combined_df["model_name"] == m, "case_num"].unique())
        common_cases = common_cases.intersection(model_cases)

    common_cases = sorted(common_cases)

    if len(common_cases) < n_cases_sample:
        raise ValueError(
            f"Not enough common cases across clinicians and all models. "
            f"Found {len(common_cases)} common cases, need at least {n_cases_sample}."
        )

    rows = []

    for trial in range(n_trials):
        sampled_cases = rng.choice(common_cases, size=n_cases_sample, replace=False)

        for model_name in all_models:
            model_scores = []
            baseline_scores = []

            for case_num in sampled_cases:
                mmd_model, mmd_baseline = casewise_mmd_against_experts(
                    combined_df=combined_df,
                    model_name=model_name,
                    case_num=int(case_num),
                    rng=rng,
                )

                if np.isfinite(mmd_model):
                    model_scores.append(mmd_model)
                if np.isfinite(mmd_baseline):
                    baseline_scores.append(mmd_baseline)

            if len(model_scores) == 0:
                continue

            model_avg = float(np.mean(model_scores))
            baseline_avg = float(np.mean(baseline_scores)) if len(baseline_scores) > 0 else np.nan
            deviation = model_avg - baseline_avg if np.isfinite(baseline_avg) else np.nan

            rows.append({
                "trial": trial,
                "model_name": model_name,
                "mmd_model_vs_expert": model_avg,
                "mmd_expert_split_baseline": baseline_avg,
                "deviation_from_experts": deviation,
            })

    return pd.DataFrame(rows)


def save_mmd_stripplot(df_scores: pd.DataFrame, outpath: str | Path) -> None:
    sns.set_context("talk")
    plt.figure(figsize=(7.5, 6))

    order = sorted(df_scores["model_name"].unique())

    ax = sns.stripplot(
        data=df_scores,
        x="model_name",
        y="deviation_from_experts",
        order=order,
        jitter=0.18,
        alpha=0.85,
        size=5,
    )

    means = (
        df_scores.groupby("model_name", as_index=False)["deviation_from_experts"]
        .mean()
        .rename(columns={"deviation_from_experts": "mean_deviation"})
    )

    for _, row in means.iterrows():
        xpos = order.index(row["model_name"])
        ax.scatter(
            xpos,
            row["mean_deviation"],
            marker="D",
            s=120,
            zorder=5,
            color="red",
        )

    ax.axhline(0, linestyle="--", linewidth=1.5, color="black")
    ax.set_xlabel("")
    ax.set_ylabel("MMD(model, experts) - MMD(expert split, expert split)")
    ax.set_title("Model deviation from expert embedding distributions")
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", type=str, default="trauma_data/clinical survey responses_ 3-19-26.xlsx")
    parser.add_argument("--sheet", type=str, default="Experts")
    parser.add_argument("--model_csv", type=str, default="output_free_text.csv")
    parser.add_argument("--outdir", type=str, default="compare_outputs")
    parser.add_argument("--embedding_model", type=str, default="gemini-embedding-001")
    parser.add_argument("--distance_metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--max_components", type=int, default=50)
    parser.add_argument("--pairplot_max_dims", type=int, default=6)
    parser.add_argument("--n_cases_sample", type=int, default=5)
    parser.add_argument("--n_trials", type=int, default=200)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading clinician responses...")
    clinician_df = load_clinician_responses(args.excel, args.sheet)

    print("Loading model responses...")
    model_df = load_model_responses(args.model_csv)

    keep_cols = ["response_id", "source", "model_name", "case_num", "response_text"]
    combined_df = pd.concat(
        [
            clinician_df[keep_cols],
            model_df[keep_cols],
        ],
        ignore_index=True,
    )

    combined_df.to_csv(outdir / "combined_responses_long.csv", index=False)

    print("Embedding all responses with Gemini...")
    combined_df, X = embed_dataframe(
        combined_df,
        text_col="response_text",
        embedding_model=args.embedding_model,
        cache_path=outdir / "combined_embeddings.json",
    )

    print("Running joint MDS sweep...")
    stress_df, projections = fit_mds_range(
        X,
        min_components=1,
        max_components=args.max_components,
        random_state=args.random_state,
        distance_metric=args.distance_metric,
    )
    stress_df.to_csv(outdir / "joint_mds_stress_by_components.csv", index=False)

    optimal_k = choose_elbow_dimension(stress_df)
    print(f"Chosen optimal dimensionality by elbow heuristic: k={optimal_k}")

    save_stress_plot(
        stress_df=stress_df,
        optimal_k=optimal_k,
        outpath=outdir / "joint_mds_stress_vs_components.png",
    )

    print("Saving joint MDS pairplot...")
    X_opt = projections[optimal_k]
    plot_k = max(2, min(optimal_k, args.pairplot_max_dims))

    projected_df = combined_df.copy()
    for i in range(plot_k):
        projected_df[f"MDS{i+1}"] = X_opt[:, i]

    projected_df.to_json(outdir / "combined_with_mds.json", orient="records", indent=2)

    mds_cols = [f"MDS{i+1}" for i in range(plot_k)]
    save_pairplot(
        projected_df=projected_df,
        dims_to_plot=mds_cols,
        outpath=outdir / f"joint_pairplot_mds_k{optimal_k}_showing_{plot_k}dims.png",
    )

    print("Running subsampled MMD experiment...")
    mmd_scores = subsampled_mmd_experiment(
        combined_df=combined_df,
        n_cases_sample=args.n_cases_sample,
        n_trials=args.n_trials,
        random_state=args.random_state,
    )
    mmd_scores.to_csv(outdir / "mmd_subsample_scores.csv", index=False)

    save_mmd_stripplot(
        mmd_scores,
        outpath=outdir / "mmd_stripplot.png",
    )

    print("\nDone.")
    print(f"Saved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()