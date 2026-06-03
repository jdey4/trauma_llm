#!/usr/bin/env python3

from __future__ import annotations

import os
import re
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_distances

try:
    from google import genai
except ImportError as e:
    raise ImportError("Install google-genai: pip install google-genai") from e


load_dotenv()

RESPONSE_PREFIX = "How would you personally respond to this case?"


def resolve_existing_path(path_like, filename=None):
    p = Path(path_like)
    if p.exists():
        return p

    search_name = filename or p.name
    candidates = [
        Path.cwd() / search_name,
        Path.cwd() / "trauma_data" / search_name,
        Path.cwd() / "data" / search_name,
        Path.cwd() / "outputs" / search_name,
        Path.cwd() / "compare_outputs" / search_name,
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


def normalize_case_num(x):
    if pd.isna(x):
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float) and x.is_integer():
        return int(x)

    m = re.search(r"(\d+)", str(x).strip())
    return int(m.group(1)) if m else None


def clean_response_text(df, text_col="response_text"):
    df = df.copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].notna()]
    df = df[df[text_col].ne("")]
    df = df[df[text_col].str.lower().ne("nan")]
    return df.reset_index(drop=True)


def load_expert_responses(excel_path, sheet_name="Experts"):
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

    rows = []

    for case_idx, col in enumerate(response_cols, start=1):
        for expert_idx, response in enumerate(df[col].tolist()):
            rows.append({
                "response_id": f"expert_case{case_idx}_expert{expert_idx}",
                "source": "Expert",
                "case_num": case_idx,
                "response_text": response,
            })

    out = pd.DataFrame(rows)
    out = clean_response_text(out, "response_text")
    out["case_num"] = out["case_num"].astype(int)

    print("\nExpert responses per case:")
    print(out.groupby("case_num").size())

    return out


def load_model_responses(csv_path):
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
        raise ValueError(f"Missing model CSV columns: {missing}")

    if "rep" not in df.columns:
        df["rep"] = np.arange(len(df))

    df["case_num"] = df["case_num"].apply(normalize_case_num)

    valid_cases = df["case_num"].dropna().astype(int)
    if valid_cases.min() == 0:
        print("Detected 0-based model cases. Converting 0–16 to 1–17.")
        df["case_num"] = df["case_num"] + 1

    df = clean_response_text(df, "response_text")
    df["case_num"] = df["case_num"].astype(int)
    df["model_name"] = df["model_name"].astype(str).str.strip()

    df["response_id"] = [
        f"model_{m}_case{c}_rep{r}_{i}"
        for i, (m, c, r) in enumerate(
            zip(df["model_name"], df["case_num"], df["rep"])
        )
    ]

    out = df[["response_id", "model_name", "case_num", "response_text"]].copy()
    out["source"] = "Model"

    print("\nModel responses per case:")
    print(out.groupby("case_num").size())

    return out


def make_genai_client():
    api_key = (
        os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )

    if not api_key:
        raise EnvironmentError(
            "No Gemini API key found. Add GOOGLE_API_KEY or "
            "GOOGLE_GENERATIVE_AI_API_KEY to your .env file."
        )

    return genai.Client(api_key=api_key)


def get_gemini_embedding(client, text, model="gemini-embedding-001"):
    sleep_s = 2.0
    last_err = None

    for attempt in range(5):
        try:
            result = client.models.embed_content(
                model=model,
                contents=text,
            )
            return np.array(result.embeddings[0].values, dtype=np.float32)

        except Exception as e:
            last_err = e
            if attempt == 4:
                break
            print(f"Embedding failed attempt {attempt+1}/5. Retrying in {sleep_s:.1f}s")
            time.sleep(sleep_s)
            sleep_s *= 2

    raise RuntimeError(f"Embedding failed: {last_err}")


def embed_dataframe(df, cache_path, embedding_model="gemini-embedding-001"):
    cache_path = Path(cache_path)

    if cache_path.exists():
        cached = pd.read_json(cache_path)

        if cached["response_id"].tolist() == df["response_id"].tolist():
            print(f"Using embedding cache: {cache_path}")
            X = np.vstack(
                cached["embedding"].apply(lambda x: np.array(x, dtype=np.float32)).values
            )
            return cached, X

        print("Cache exists but row order/content differs. Recomputing embeddings.")

    client = make_genai_client()
    embeddings = []

    print(f"Embedding {len(df)} responses with Gemini...")

    for i, text in enumerate(df["response_text"].tolist(), start=1):
        emb = get_gemini_embedding(client, text, model=embedding_model)
        embeddings.append(emb)

        if i % 25 == 0 or i == len(df):
            print(f"Embedded {i}/{len(df)}")

    out = df.copy()
    out["embedding"] = [e.tolist() for e in embeddings]
    X = np.vstack(embeddings)

    out.to_json(cache_path, orient="records", indent=2)

    return out, X


def average_pairwise_cosine_distance(X):
    if len(X) < 2:
        return np.nan

    D = cosine_distances(X)

    upper = D[np.triu_indices_from(D, k=1)]
    return float(np.mean(upper))


def compute_case_disagreement(df):
    rows = []

    for case_num, sub in df.groupby("case_num"):
        X = np.vstack(
            sub["embedding"].apply(lambda x: np.array(x, dtype=np.float32)).values
        )

        rows.append({
            "case_num": int(case_num),
            "avg_pairwise_cosine_distance": average_pairwise_cosine_distance(X),
            "n_responses": len(sub),
        })

    return pd.DataFrame(rows).sort_values("case_num")


def plot_disagreement_heatmap(model_scores, expert_scores, outpath):
    all_cases = list(range(1, 18))

    model_map = dict(
        zip(model_scores["case_num"], model_scores["avg_pairwise_cosine_distance"])
    )
    expert_map = dict(
        zip(expert_scores["case_num"], expert_scores["avg_pairwise_cosine_distance"])
    )

    heatmap_df = pd.DataFrame(
        [
            [model_map.get(c, np.nan) for c in all_cases],
            [expert_map.get(c, np.nan) for c in all_cases],
        ],
        index=["Model disagreement", "Expert disagreement"],
        columns=[f"Case {c}" for c in all_cases],
    )

    sns.set_context("talk")

    plt.figure(figsize=(18, 4.2))

    ax = sns.heatmap(
        heatmap_df,
        cmap="YlOrRd",
        annot=False,          # no numbers inside blocks
        linewidths=1,
        linecolor="white",
        cbar_kws={
            "label": "Average pairwise \n cosine distance",
            "shrink": 0.9,
            "pad": 0.02,
        },
    )

    ax.set_title("Case-level response embedding disagreement", pad=18)
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Make room for full colorbar label on right
    plt.subplots_adjust(right=0.88, bottom=0.32, left=0.16, top=0.82)

    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    return heatmap_df

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--excel",
        type=str,
        default="trauma_data/clinical survey responses_ 3-19-26.xlsx",
    )
    parser.add_argument("--sheet", type=str, default="Experts")
    parser.add_argument(
        "--model_csv",
        type=str,
        default="output_free_text.csv",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="case_disagreement_outputs",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="gemini-embedding-001",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading expert responses...")
    expert_df = load_expert_responses(args.excel, args.sheet)

    print("\nLoading model responses...")
    model_df = load_model_responses(args.model_csv)

    print("\nEmbedding expert responses...")
    expert_df, _ = embed_dataframe(
        expert_df,
        cache_path=outdir / "expert_gemini_embeddings.json",
        embedding_model=args.embedding_model,
    )

    print("\nEmbedding model responses...")
    model_df, _ = embed_dataframe(
        model_df,
        cache_path=outdir / "model_gemini_embeddings.json",
        embedding_model=args.embedding_model,
    )

    print("\nComputing expert disagreement...")
    expert_scores = compute_case_disagreement(expert_df)
    expert_scores.to_csv(outdir / "expert_case_disagreement.csv", index=False)

    print("\nComputing model disagreement...")
    model_scores = compute_case_disagreement(model_df)
    model_scores.to_csv(outdir / "model_case_disagreement.csv", index=False)

    heatmap_df = plot_disagreement_heatmap(
        model_scores=model_scores,
        expert_scores=expert_scores,
        outpath=outdir / "case_disagreement_heatmap.png",
    )

    heatmap_df.to_csv(outdir / "case_disagreement_heatmap_values.csv")

    print("\nDone.")
    print(f"Saved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()