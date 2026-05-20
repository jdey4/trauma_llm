#!/usr/bin/env python3

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
    raise ImportError("Missing google-genai. Install with: pip install google-genai") from e


load_dotenv()

RESPONSE_PREFIX = "How would you personally respond to this case?"
YEARS_COL = "How many years of experience do you have in your field?"
SPECIALTY_COL = "What is your medical specialty?"
NAME_COLS = ["First name", "Last name"]


def resolve_existing_path(path_like: str | Path, filename: str | None = None) -> Path:
    p = Path(path_like)
    if p.exists():
        return p

    search_name = filename or p.name
    candidates = [
        Path.cwd() / search_name,
        Path.cwd() / "trauma_data" / search_name,
        Path.cwd() / "data" / search_name,
        Path.cwd() / "output" / search_name,
        Path.cwd() / "outputs" / search_name,
    ]

    for c in candidates:
        if c.exists():
            print(f"Found file at: {c}")
            return c

    matches = list(Path.cwd().rglob(search_name))
    if matches:
        print(f"Found file at: {matches[0]}")
        return matches[0]

    raise FileNotFoundError(f"Could not find {search_name}")


def normalize_case_num(x) -> int | None:
    if pd.isna(x):
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float) and x.is_integer():
        return int(x)
    m = re.search(r"(\d+)", str(x).strip())
    return int(m.group(1)) if m else None


def clean_response_text(df: pd.DataFrame, text_col: str = "response_text") -> pd.DataFrame:
    df = df.copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].notna()]
    df = df[df[text_col].ne("")]
    df = df[df[text_col].str.lower().ne("nan")]
    return df.reset_index(drop=True)


def load_clinician_responses(
    excel_path: str | Path,
    sheet_name: str = "Experts",
) -> pd.DataFrame:
    excel_path = resolve_existing_path(
        excel_path,
        filename="clinical survey responses_ 3-19-26.xlsx",
    )

    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    response_cols = [
        c for c in df.columns
        if str(c).startswith(RESPONSE_PREFIX)
    ]

    print(f"Found {len(response_cols)} expert response columns.")

    out_frames = []

    for i, col in enumerate(response_cols, start=1):
        tmp = pd.DataFrame({
            "response_id": [f"clinician_case{i}_expert{j}" for j in range(len(df))],
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
    long_df["case_num"] = long_df["case_num"].astype(int)

    print("\nExpert rows by case:")
    print(long_df.groupby("case_num").size())

    return long_df


def load_model_responses(csv_path: str | Path) -> pd.DataFrame:
    csv_path = resolve_existing_path(csv_path, filename="output_free_text.csv")
    df = pd.read_csv(csv_path)

    rename_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "response":
            rename_map[c] = "response_text"
        elif cl == "model":
            rename_map[c] = "model_name"
        elif cl in ["case", "case_num"]:
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

    valid_cases = df["case_num"].dropna().astype(int)
    if valid_cases.min() == 0:
        print("Detected 0-based model cases. Converting 0–16 to 1–17.")
        df["case_num"] = df["case_num"] + 1

    df = clean_response_text(df, "response_text")
    df = df[df["case_num"].notna()].reset_index(drop=True)
    df["case_num"] = df["case_num"].astype(int)

    df["source"] = "Model"
    df["response_id"] = [
        f"{m}_case{c}_rep{r}_{i}"
        for i, (m, c, r) in enumerate(zip(df["model_name"], df["case_num"], df["rep"]))
    ]

    # IMPORTANT:
    # Do NOT keep any existing embedding column from the model CSV.
    # It may be 3072-dim and incompatible with Gemini embeddings.
    out = df[["response_id", "source", "model_name", "case_num", "response_text", "rep"]].copy()

    print("\nModel rows by model/case:")
    print(out.groupby(["model_name", "case_num"]).size())

    return out.reset_index(drop=True)


def filter_cases_by_length_agreement(
    combined_df: pd.DataFrame,
    tolerance: int = 100,
    outdir: str | Path | None = None,
) -> pd.DataFrame:
    df = combined_df.copy()
    df["resp_len"] = df["response_text"].astype(str).str.strip().str.len()

    expert_mean = (
        df[df["model_name"] == "Clinician"]
        .groupby("case_num")["resp_len"]
        .mean()
        .rename("expert_mean_len")
        .reset_index()
    )

    model_means = (
        df[df["model_name"] != "Clinician"]
        .groupby(["case_num", "model_name"])["resp_len"]
        .mean()
        .rename("model_mean_len")
        .reset_index()
    )

    report = model_means.merge(expert_mean, on="case_num", how="left")
    report["abs_diff_from_expert_mean"] = (
        report["model_mean_len"] - report["expert_mean_len"]
    ).abs()
    report["model_passes"] = report["abs_diff_from_expert_mean"] <= tolerance

    case_pass = (
        report.groupby("case_num")["model_passes"]
        .all()
        .rename("case_passes")
        .reset_index()
    )

    kept_cases = sorted(case_pass.loc[case_pass["case_passes"], "case_num"].tolist())
    dropped_cases = sorted(case_pass.loc[~case_pass["case_passes"], "case_num"].tolist())

    filtered_df = df[df["case_num"].isin(kept_cases)].copy().reset_index(drop=True)

    print("\n================ LENGTH AGREEMENT FILTER ================")
    print(f"Tolerance: ±{tolerance} characters")
    print(f"Kept cases: {kept_cases}")
    print(f"Dropped cases: {dropped_cases}")
    print(f"Rows before filtering: {len(df)}")
    print(f"Rows after filtering:  {len(filtered_df)}")

    print("\nRows by group after filtering:")
    print(filtered_df["model_name"].value_counts())

    print("\nRows by case/group after filtering:")
    print(filtered_df.groupby(["case_num", "model_name"]).size())

    print("=========================================================\n")

    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        report.to_csv(outdir / "case_length_agreement_model_report.csv", index=False)
        case_pass.to_csv(outdir / "case_length_agreement_case_report.csv", index=False)
        filtered_df.to_csv(outdir / "combined_responses_after_length_filter.csv", index=False)

    if len(kept_cases) == 0:
        raise ValueError(
            "No cases survived length-agreement filtering. "
            "Increase --length_tolerance."
        )

    return filtered_df


def make_genai_client(api_key: str | None = None):
    api_key = (
        api_key
        or os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key:
        raise EnvironmentError("No Gemini API key found in .env.")
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
            print(f"Embedding failed attempt {attempt + 1}/{max_retries}; retrying in {sleep_s:.1f}s")
            time.sleep(sleep_s)
            sleep_s *= 2.0

    raise RuntimeError(f"Failed to embed after {max_retries} attempts: {last_err}")


def embed_dataframe(
    df: pd.DataFrame,
    text_col: str = "response_text",
    embedding_model: str = "gemini-embedding-001",
    cache_path: str | Path | None = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Always produces same-dimension Gemini embeddings for every row.
    Ignores any old embedding columns.
    """
    if cache_path is not None and Path(cache_path).exists():
        cached = pd.read_json(cache_path)
        if cached["response_id"].tolist() == df["response_id"].tolist():
            print(f"Using valid embedding cache: {cache_path}")
            X = np.vstack(cached["embedding"].apply(lambda z: np.array(z, dtype=np.float32)).values)
            return cached, X
        else:
            print("Embedding cache exists but does not match current rows. Recomputing.")

    client = make_genai_client()
    embeddings = []

    print(f"Embedding {len(df)} responses with Gemini...")

    for idx, text in enumerate(df[text_col].tolist(), start=1):
        emb = get_gemini_embedding(
            client=client,
            text=text,
            model=embedding_model,
        )
        embeddings.append(emb)

        if idx % 25 == 0 or idx == len(df):
            print(f"Embedded {idx}/{len(df)} responses")

    X = np.vstack(embeddings)

    out_df = df.copy()
    out_df["embedding"] = [x.tolist() for x in X]

    if cache_path is not None:
        out_df.to_json(cache_path, orient="records", indent=2)

    return out_df, X


def fit_mds_range(
    X: np.ndarray,
    min_components: int = 1,
    max_components: int = 50,
    random_state: int = 42,
    distance_metric: str = "cosine",
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    max_allowed = min(X.shape[0] - 1, X.shape[1])
    max_components = min(max_components, max_allowed)

    print(f"Requested MDS sweep up to {max_components} dimensions")

    if distance_metric == "cosine":
        mds_input = cosine_distances(X)
        dissimilarity_mode = "precomputed"
    elif distance_metric == "euclidean":
        mds_input = X
        dissimilarity_mode = "euclidean"
    else:
        raise ValueError("distance_metric must be cosine or euclidean.")

    stress_rows = []
    projections = {}

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
        print(f"Finished MDS for k={k}, stress={mds.stress_:.4f}")

    return pd.DataFrame(stress_rows), projections


def choose_elbow_dimension(stress_df: pd.DataFrame) -> int:
    pts = stress_df[["n_components", "stress"]].to_numpy(dtype=float)

    if len(pts) <= 2:
        return int(pts[-1, 0])

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

    return int(pts[int(np.argmax(distances)), 0])


def save_stress_plot(stress_df: pd.DataFrame, optimal_k: int, outpath: str | Path) -> None:
    sns.set_context("talk")
    plt.figure(figsize=(9, 5.5))

    ax = sns.lineplot(
        data=stress_df,
        x="n_components",
        y="stress",
        marker="o",
    )

    ax.axvline(optimal_k, linestyle="--", linewidth=1.2)
    ax.scatter(
        [optimal_k],
        [float(stress_df.loc[stress_df["n_components"] == optimal_k, "stress"].iloc[0])],
        s=110,
        zorder=5,
    )

    ax.set_title("Length-matched MDS stress vs. number of components")
    ax.set_xlabel("Number of MDS components")
    ax.set_ylabel("Stress")

    max_k = int(stress_df["n_components"].max())
    ticks = list(range(1, max_k + 1, 5))
    if optimal_k not in ticks:
        ticks = sorted(set(ticks + [optimal_k]))
    ax.set_xticks(ticks)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def save_pairplot(projected_df: pd.DataFrame, dims_to_plot: List[str], outpath: str | Path) -> None:
    sns.set_context("talk")

    order = ["Clinician"] + [
        m for m in sorted(projected_df["model_name"].unique())
        if m != "Clinician"
    ]

    palette = dict(zip(order, sns.color_palette("Set2", n_colors=len(order))))

    cols = dims_to_plot + ["model_name"]

    g = sns.pairplot(
        projected_df[cols],
        vars=dims_to_plot,
        hue="model_name",
        hue_order=order,
        palette=palette,
        diag_kind="hist",
        corner=False,
        plot_kws={"alpha": 0.65, "s": 28},
    )

    g.figure.suptitle("Length-matched joint MDS pairplot", y=1.02)

    if g._legend is not None:
        g._legend.set_title("Group")

    g.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(g.figure)


def median_heuristic_sigma(X: np.ndarray) -> float:
    dists = pairwise_distances(X, metric="euclidean")
    tri = dists[np.triu_indices_from(dists, k=1)]
    tri = tri[tri > 0]
    return float(np.median(tri)) if len(tri) else 1.0


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

    return float(
        Kxx.sum() / (n * (n - 1))
        + Kyy.sum() / (m * (m - 1))
        - 2.0 * Kxy.mean()
    )


def casewise_mmd_against_experts(
    combined_df: pd.DataFrame,
    model_name: str,
    case_num: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    experts = combined_df[
        (combined_df["model_name"] == "Clinician")
        & (combined_df["case_num"] == case_num)
    ]

    model = combined_df[
        (combined_df["model_name"] == model_name)
        & (combined_df["case_num"] == case_num)
    ]

    if len(experts) < 4 or len(model) < 2:
        return np.nan, np.nan

    X_exp = np.vstack(experts["embedding"].apply(lambda z: np.array(z, dtype=np.float32)).values)
    X_mod = np.vstack(model["embedding"].apply(lambda z: np.array(z, dtype=np.float32)).values)

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

    all_models = sorted([
        m for m in combined_df["model_name"].unique()
        if m != "Clinician"
    ])

    common_cases = set(
        combined_df.loc[
            combined_df["model_name"] == "Clinician",
            "case_num",
        ].unique()
    )

    for model_name in all_models:
        model_cases = set(
            combined_df.loc[
                combined_df["model_name"] == model_name,
                "case_num",
            ].unique()
        )
        common_cases = common_cases.intersection(model_cases)

    common_cases = sorted(common_cases)

    if len(common_cases) < n_cases_sample:
        raise ValueError(
            f"Only {len(common_cases)} common length-matched cases survived, "
            f"but n_cases_sample={n_cases_sample}. "
            f"Use --n_cases_sample {len(common_cases)} or increase --length_tolerance."
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

            if not model_scores:
                continue

            model_avg = float(np.mean(model_scores))
            baseline_avg = float(np.mean(baseline_scores)) if baseline_scores else np.nan
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
    ax.set_title("Length-matched model deviation from experts")

    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--excel", type=str, default="trauma_data/clinical survey responses_ 3-19-26.xlsx")
    parser.add_argument("--sheet", type=str, default="Experts")
    parser.add_argument("--model_csv", type=str, default="output_free_text.csv")
    parser.add_argument("--outdir", type=str, default="compare_outputs_length_matched")
    parser.add_argument("--embedding_model", type=str, default="gemini-embedding-001")
    parser.add_argument("--distance_metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--max_components", type=int, default=50)
    parser.add_argument("--pairplot_max_dims", type=int, default=6)
    parser.add_argument("--n_cases_sample", type=int, default=5)
    parser.add_argument("--n_trials", type=int, default=200)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--length_tolerance", type=int, default=100)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading clinician responses...")
    clinician_df = load_clinician_responses(args.excel, args.sheet)

    print("Loading model responses...")
    model_df = load_model_responses(args.model_csv)

    print("\nExpert cases:", sorted(clinician_df["case_num"].unique()))
    print("Model cases:", sorted(model_df["case_num"].unique()))

    keep_cols = ["response_id", "source", "model_name", "case_num", "response_text"]

    combined_df = pd.concat(
        [
            clinician_df[keep_cols],
            model_df[keep_cols],
        ],
        ignore_index=True,
    )

    combined_df.to_csv(outdir / "combined_responses_before_length_filter.csv", index=False)

    print("\nApplying case-level length-agreement filter...")
    combined_df = filter_cases_by_length_agreement(
        combined_df=combined_df,
        tolerance=args.length_tolerance,
        outdir=outdir,
    )

    print("Embedding length-matched responses...")
    combined_df, X = embed_dataframe(
        combined_df,
        text_col="response_text",
        embedding_model=args.embedding_model,
        cache_path=outdir / f"gemini_embeddings_length_matched_tol{args.length_tolerance}.json",
    )

    print("Running MDS sweep...")
    stress_df, projections = fit_mds_range(
        X,
        min_components=1,
        max_components=args.max_components,
        random_state=args.random_state,
        distance_metric=args.distance_metric,
    )

    stress_df.to_csv(outdir / "length_matched_mds_stress_by_components.csv", index=False)

    optimal_k = choose_elbow_dimension(stress_df)
    print(f"Chosen optimal dimensionality: k={optimal_k}")

    save_stress_plot(
        stress_df=stress_df,
        optimal_k=optimal_k,
        outpath=outdir / "length_matched_mds_stress_vs_components.png",
    )

    X_opt = projections[optimal_k]
    plot_k = max(2, min(optimal_k, args.pairplot_max_dims))

    projected_df = combined_df.copy()

    for i in range(plot_k):
        projected_df[f"MDS{i+1}"] = X_opt[:, i]

    projected_df.to_json(
        outdir / "length_matched_combined_with_mds.json",
        orient="records",
        indent=2,
    )

    mds_cols = [f"MDS{i+1}" for i in range(plot_k)]

    save_pairplot(
        projected_df=projected_df,
        dims_to_plot=mds_cols,
        outpath=outdir / f"length_matched_pairplot_mds_k{optimal_k}_showing_{plot_k}dims.png",
    )

    print("Running length-matched MMD experiment...")
    mmd_scores = subsampled_mmd_experiment(
        combined_df=combined_df,
        n_cases_sample=args.n_cases_sample,
        n_trials=args.n_trials,
        random_state=args.random_state,
    )

    mmd_scores.to_csv(outdir / "length_matched_mmd_subsample_scores.csv", index=False)

    save_mmd_stripplot(
        mmd_scores,
        outpath=outdir / "length_matched_mmd_stripplot.png",
    )

    print("\nDone.")
    print(f"Saved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()