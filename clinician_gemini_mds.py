#!/usr/bin/env python3
"""
Clinician response embedding analysis with Gemini embeddings + MDS.

What this script does
---------------------
1. Reads clinician free-text responses from the survey Excel file.
2. Extracts all columns whose names begin with:
       "How would you personally respond to this case?"
3. Reshapes them into a long table: one row per (clinician, case).
4. Loads GOOGLE_API_KEY automatically from a .env file.
5. Calls Gemini embeddings ("gemini-embedding-001") on each clinician response.
6. Runs metric MDS for a wide range of embedding dimensions.
7. Plots stress vs. number of MDS components.
8. Selects an "optimal" number of dimensions using a simple elbow heuristic.
9. Creates a seaborn pairplot using the optimal projected dimensions.
10. Color-codes all 17 cases consistently.

Requirements
------------
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl google-genai python-dotenv
"""
#%%
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

try:
    from google import genai
except ImportError as e:
    raise ImportError(
        "Missing google-genai. Install with: pip install google-genai"
    ) from e


load_dotenv()
#%%
RESPONSE_PREFIX = "How would you personally respond to this case?"
YEARS_COL = "How many years of experience do you have in your field?"
SPECIALTY_COL = "What is your medical specialty?"
NAME_COLS = ["First name", "Last name"]

# Force consistent hue handling for all 17 cases
TOTAL_CASES = 17


def load_clinician_responses(
    excel_path: str | Path,
    sheet_name: str = "Experts",
    response_prefix: str = RESPONSE_PREFIX,
) -> pd.DataFrame:
    """
    Read survey responses and reshape into one row per clinician response.

    Returns columns:
        clinician_id, clinician_name, years_experience, specialty,
        case_num, response_col, response_text
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    response_cols = [c for c in df.columns if c.startswith(response_prefix)]
    if not response_cols:
        raise ValueError(
            f'No response columns found starting with "{response_prefix}" in sheet "{sheet_name}".'
        )

    out_frames = []
    for i, col in enumerate(response_cols, start=1):
        tmp = pd.DataFrame({
            "clinician_id": np.arange(len(df)),
            "clinician_name": (
                df.get(NAME_COLS[0], pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
                + " "
                + df.get(NAME_COLS[1], pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
            ).str.strip(),
            "years_experience": pd.to_numeric(df.get(YEARS_COL, np.nan), errors="coerce"),
            "specialty": df.get(SPECIALTY_COL, pd.Series([np.nan] * len(df))),
            "case_num": i,
            "response_col": col,
            "response_text": df[col],
        })
        out_frames.append(tmp)

    long_df = pd.concat(out_frames, ignore_index=True)

    long_df["response_text"] = long_df["response_text"].astype(str).str.strip()
    long_df = long_df[long_df["response_text"].notna()]
    long_df = long_df[long_df["response_text"].ne("")]
    long_df = long_df[long_df["response_text"].str.lower().ne("nan")]
    long_df = long_df.reset_index(drop=True)

    return long_df


def make_genai_client(api_key: str | None = None):
    api_key = api_key or os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not found. Put it in your .env file or export it in the environment."
        )

    return genai.Client(api_key=api_key)


def get_gemini_embedding(
    client,
    text: str,
    model: str = "gemini-embedding-001",
    max_retries: int = 5,
    initial_sleep: float = 2.0,
) -> np.ndarray:
    """
    Embed one text with Gemini embeddings and return a 1D numpy array.
    """
    sleep_s = initial_sleep
    last_err = None

    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=model,
                contents=text,
            )

            if hasattr(result, "embeddings") and len(result.embeddings) > 0:
                emb = np.array(result.embeddings[0].values, dtype=np.float32)
                return emb

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
    model: str = "gemini-embedding-001",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Add Gemini embeddings to a dataframe row-by-row.

    Returns:
        df_with_embedding_col, embedding_matrix
    """
    client = make_genai_client()
    embeddings: List[np.ndarray] = []

    for idx, text in enumerate(df[text_col].tolist(), start=1):
        emb = get_gemini_embedding(client=client, text=text, model=model)
        embeddings.append(emb)
        if idx % 10 == 0 or idx == len(df):
            print(f"Embedded {idx}/{len(df)} responses")

    emb_mat = np.vstack(embeddings)
    out_df = df.copy()
    out_df["embedding"] = [x.tolist() for x in embeddings]

    return out_df, emb_mat


def fit_mds_range(
    X: np.ndarray,
    min_components: int = 1,
    max_components: int = 50,
    random_state: int = 42,
    normalize_before_mds: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Run metric MDS across a range of target dimensions.

    Returns:
        stress_df with columns [n_components, stress]
        projections dict: k -> projected array of shape (n_samples, k)
    """
    if normalize_before_mds:
        X = StandardScaler().fit_transform(X)

    max_allowed = min(X.shape[0] - 1, X.shape[1])
    max_components = min(max_components, max_allowed)

    if max_components < min_components:
        raise ValueError("max_components is smaller than min_components after adjustment.")

    print(f"Requested MDS sweep up to {max_components} dimensions (max allowed = {max_allowed})")

    stress_rows = []
    projections = {}

    for k in range(min_components, max_components + 1):
        mds = MDS(
            n_components=k,
            metric=True,
            n_init=4,
            max_iter=300,
            eps=1e-6,
            dissimilarity="euclidean",
            normalized_stress="auto",
            random_state=random_state,
        )
        X_proj = mds.fit_transform(X)
        stress_rows.append({"n_components": k, "stress": float(mds.stress_)})
        projections[k] = X_proj
        print(f"Finished MDS for k={k}, stress={mds.stress_:.4f}")

    stress_df = pd.DataFrame(stress_rows)
    return stress_df, projections


def choose_elbow_dimension(stress_df: pd.DataFrame) -> int:
    """
    Elbow detection by maximum distance to the straight line joining
    the first and last points of the stress curve.
    """
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
    """
    Plot full stress curve but use sparse x-axis labeling for readability.
    """
    sns.set_context("talk")

    plt.figure(figsize=(9, 5.5))
    ax = sns.lineplot(
        data=stress_df,
        x="n_components",
        y="stress",
        marker="o",
    )

    # Highlight elbow
    ax.axvline(optimal_k, linestyle="--", linewidth=1.2)
    ax.scatter(
        [optimal_k],
        [float(stress_df.loc[stress_df["n_components"] == optimal_k, "stress"].iloc[0])],
        s=110,
        zorder=5,
    )

    ax.set_title("MDS stress vs. number of components")
    ax.set_xlabel("Number of MDS components")
    ax.set_ylabel("Stress")

    # -------- KEY FIX --------
    # Sparse tick labeling (adaptive)
    max_k = stress_df["n_components"].max()

    if max_k <= 20:
        ticks = list(range(1, max_k + 1, 5))
    elif max_k <= 50:
        ticks = list(range(1, max_k + 1, 5))
    else:
        ticks = list(range(1, max_k + 1, 5))  
    
    ax.set_xticks(ticks)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def save_pairplot(
    projected_df: pd.DataFrame,
    dims_to_plot: List[str],
    outpath: str | Path,
    total_cases: int = TOTAL_CASES,
) -> None:
    """
    Create pairplot of projected MDS dimensions with consistent coloring
    across all 17 cases.
    """
    sns.set_context("talk")

    df_plot = projected_df.copy()

    hue_col = None
    palette = None
    hue_order = None

    if "case_num" in df_plot.columns:
        hue_col = "case_num"

        # Force all 17 cases into the categorical order, even if some are missing
        hue_order = list(range(1, total_cases + 1))
        df_plot[hue_col] = pd.Categorical(df_plot[hue_col], categories=hue_order, ordered=True)

        # Enough distinct colors for 17 classes
        palette = sns.color_palette("husl", n_colors=total_cases)

    cols = dims_to_plot.copy()
    if hue_col and hue_col not in cols:
        cols.append(hue_col)

    g = sns.pairplot(
        df_plot[cols],
        vars=dims_to_plot,
        hue=hue_col,
        hue_order=hue_order,
        palette=palette,
        diag_kind="hist",
        corner=True,
        plot_kws={"alpha": 0.75, "s": 32},
    )

    g.figure.suptitle(
        "Pairplot of clinician-response Gemini embeddings after MDS",
        y=1.02
    )

    if g._legend is not None:
        g._legend.set_title("Case")
        for text in g._legend.texts:
            text.set_fontsize(9)

    g.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(g.figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--excel",
        type=str,
        default="trauma_data/clinical survey responses_ 3-19-26.xlsx",
        help="Path to survey Excel file",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default="Experts",
        help="Sheet name to use",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="mds_outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--max_components",
        type=int,
        default=50,
        help="Largest MDS dimension to test for the stress plot",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="gemini-embedding-001",
        help="Gemini embedding model name",
    )
    parser.add_argument(
        "--pairplot_max_dims",
        type=int,
        default=10,
        help="Cap pairplot dims for readability",
    )
    parser.add_argument(
        "--total_cases",
        type=int,
        default=17,
        help="Total number of cases for consistent color coding",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading clinician responses...")
    response_df = load_clinician_responses(
        excel_path=args.excel,
        sheet_name=args.sheet,
    )
    response_df.to_csv(outdir / "clinician_responses_long.csv", index=False)
    print(f"Loaded {len(response_df)} non-empty clinician responses")

    print("Step 2: Embedding responses with Gemini...")
    embedded_df, X = embed_dataframe(
        response_df,
        text_col="response_text",
        model=args.embedding_model,
    )
    embedded_df.to_json(
        outdir / "clinician_responses_with_embeddings.json",
        orient="records",
        indent=2,
    )

    print("Step 3: Running MDS over a range of dimensions...")
    stress_df, projections = fit_mds_range(
        X,
        min_components=1,
        max_components=args.max_components,
        random_state=42,
        normalize_before_mds=True,
    )
    stress_df.to_csv(outdir / "mds_stress_by_components.csv", index=False)

    optimal_k = choose_elbow_dimension(stress_df)
    print(f"Chosen optimal dimensionality by elbow heuristic: k={optimal_k}")

    print("Step 4: Saving stress plot...")
    save_stress_plot(
        stress_df=stress_df,
        optimal_k=optimal_k,
        outpath=outdir / "mds_stress_vs_components.png",
    )

    print("Step 5: Saving pairplot at optimal dimensionality...")
    X_opt = projections[optimal_k]
    plot_k = max(2, min(optimal_k, args.pairplot_max_dims))

    mds_cols = [f"MDS{i+1}" for i in range(plot_k)]
    projected_df = embedded_df.copy()

    for i in range(plot_k):
        projected_df[f"MDS{i+1}"] = X_opt[:, i]

    projected_df.to_csv(outdir / "clinician_responses_projected.csv", index=False)

    save_pairplot(
        projected_df=projected_df,
        dims_to_plot=mds_cols,
        outpath=outdir / f"pairplot_mds_k{optimal_k}_showing_{plot_k}dims.png",
        total_cases=args.total_cases,
    )

    print("\nDone.")
    print(f"Outputs saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()