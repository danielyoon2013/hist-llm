"""
Prepare corpus documents for synthetic data generation.

Replaces the old 3-script sequence (build_index → export → export_additional)
with a single command that produces uniform parquets.

Usage:
    # Full preparation for a period
    python -m src.post_training.prepare --period 1900_1949

    # Only corpus or only additional
    python -m src.post_training.prepare --period 1900_1949 --source corpus
    python -m src.post_training.prepare --period 1900_1949 --source additional

    # Custom quality filter (top 25% instead of top 50%)
    python -m src.post_training.prepare --period 1900_1949 --quality-percentile 75

Output:
    synthetic/input/{collection}.parquet  (columns: doc_name, text)
    One parquet per collection. 10K doc cap per collection.
    Generators sample what they need at runtime via compute_plan().
"""

import os
import re
import argparse
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from src.post_training.config import PERIODS, DATA_ROOT, get_paths


# ---------------------------------------------------------------------------
# Quality model periods (25-year bins used by Ridge classifiers)
# ---------------------------------------------------------------------------

QUALITY_PERIODS = [
    (1678, 1700, "1678_1700"),
    (1701, 1725, "1701_1725"),
    (1726, 1750, "1726_1750"),
    (1751, 1775, "1751_1775"),
    (1776, 1800, "1776_1800"),
    (1801, 1825, "1801_1825"),
    (1826, 1850, "1826_1850"),
    (1851, 1875, "1851_1875"),
    (1876, 1900, "1876_1900"),
    (1901, 1925, "1901_1925"),
    (1926, 1950, "1926_1950"),
    (1951, 1975, "1951_1975"),
    (1976, 2000, "1976_2000"),
    (2001, 2023, "2001_2023"),
]

MODELS_DIR = DATA_ROOT / "processing" / "quality_models"


def year_to_quality_period(year):
    """Map a year to its 25-year quality model period."""
    for start, end, period in QUALITY_PERIODS:
        if start <= year <= end:
            return period
    return None


def load_quality_model(period):
    """Load scaler and Ridge model for a quality period. Returns (scaler, model) or None."""
    import joblib

    scaler_path = MODELS_DIR / f"scaler_{period}.pkl"
    model_path = MODELS_DIR / f"ridge_{period}.pkl"

    if not scaler_path.exists() or not model_path.exists():
        return None

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return scaler, model


# ---------------------------------------------------------------------------
# Corpus preparation (main historical corpus)
# ---------------------------------------------------------------------------

def prepare_corpus(period, quality_percentile=50, max_per_collection=10000, seed=42):
    """Export main corpus collections as parquets with quality filtering.

    Steps:
    1. Build metadata index in-memory (join classified + raw parquets)
    2. Apply quality filter (top N percentile by predicted_quality)
    3. Per collection: cap at max_per_collection docs
    4. Retrieve text from raw parquets, write as {collection}.parquet
    """
    paths = get_paths(period)
    start_year, end_year = paths["start_year"], paths["end_year"]
    input_base = paths["synthetic_dir"] / "input"
    raw_root = paths["raw_data_root"]
    classified_root = paths["classified_root"]

    os.makedirs(input_base, exist_ok=True)

    # Step 1: Build metadata index in-memory
    print("Building metadata index...")
    frames = []
    for year in range(start_year, end_year + 1):
        classified_path = classified_root / f"classified_{year}.parquet"
        if not classified_path.exists():
            continue
        classified = pd.read_parquet(
            classified_path, columns=["identifier", "predicted_quality"]
        )

        year_dir = raw_root / str(year)
        if not year_dir.exists():
            continue
        subset_frames = []
        for f in sorted(glob(str(year_dir / "subset_*.parquet"))):
            df = pd.read_parquet(f, columns=["identifier", "collection"])
            df["subset_file"] = Path(f).name
            subset_frames.append(df)
        if not subset_frames:
            continue

        raw = pd.concat(subset_frames, ignore_index=True)
        merged = classified.merge(raw, on="identifier", how="inner")
        merged["year"] = year
        frames.append(merged)

    if not frames:
        print("No corpus data found for this period.")
        return

    meta = pd.concat(frames, ignore_index=True)
    print(f"  Total documents: {len(meta):,}")

    # Step 2: Quality filter
    threshold = meta["predicted_quality"].quantile(quality_percentile / 100)
    meta = meta[meta["predicted_quality"] >= threshold]
    print(f"  After quality filter (top {100 - quality_percentile}%): {len(meta):,}")

    # Step 3: Per-collection export with cap
    collections = sorted(meta["collection"].unique())
    print(f"\n  Exporting {len(collections)} collections "
          f"(cap {max_per_collection:,} per collection):")

    summary = []
    for collection in collections:
        coll_df = meta[meta["collection"] == collection]
        available = len(coll_df)

        if available > max_per_collection:
            sample = coll_df.sample(n=max_per_collection, random_state=seed)
            action = "sampled"
            n = max_per_collection
        else:
            sample = coll_df
            action = "all"
            n = available

        # Retrieve text from raw parquets
        texts = []
        for (year, subset_file), group in sample.groupby(["year", "subset_file"]):
            parquet_path = raw_root / str(year) / subset_file
            if not parquet_path.exists():
                continue
            needed_ids = set(group["identifier"])
            raw_df = pd.read_parquet(parquet_path, columns=["identifier", "text"])
            matched = raw_df[raw_df["identifier"].isin(needed_ids)]
            for _, row in matched.iterrows():
                if pd.notna(row["text"]) and row["text"].strip():
                    texts.append({"doc_name": row["identifier"], "text": row["text"]})

        if texts:
            safe_name = re.sub(r'[^\w\-]', '_', collection).strip('_')
            safe_name = re.sub(r'_+', '_', safe_name)
            output_path = input_base / f"{safe_name}.parquet"
            pd.DataFrame(texts).to_parquet(output_path, index=False)
            print(f"    {collection}: {len(texts):,} / {available:,} ({action})")
            summary.append((collection, available, len(texts)))
        else:
            summary.append((collection, available, 0))

    total_written = sum(s[2] for s in summary)
    print(f"\n  Corpus total: {total_written:,} documents")
    return summary


# ---------------------------------------------------------------------------
# Additional data preparation (NYT, Economist, FT, Newswire)
# ---------------------------------------------------------------------------

def prepare_additional(period, quality_percentile=50, max_per_collection=10000,
                       min_text_length=200, seed=42):
    """Export additional datasets as parquets with quality filtering.

    Steps per dataset:
    1. Load text via existing loaders
    2. Filter by min text length
    3. If embeddings + quality models available: apply quality filtering
    4. Cap at max_per_collection docs
    5. Write as {dataset}.parquet
    """
    from src.post_training.corpus.export_additional import DATASETS, LOADERS

    paths = get_paths(period)
    start_year, end_year = paths["start_year"], paths["end_year"]
    input_base = paths["synthetic_dir"] / "input"
    embeddings_root = DATA_ROOT / "additional_data" / "embeddings"

    os.makedirs(input_base, exist_ok=True)

    summary = []
    for ds_name, ds_info in DATASETS.items():
        data_dir = paths[ds_info["config_key"]]
        if not data_dir.exists():
            print(f"  {ds_name}: directory not found, skipping")
            summary.append((ds_name, 0, 0))
            continue

        loader = LOADERS[ds_name]
        pool = loader(data_dir, start_year, end_year)

        # Basic text filter
        pool = pool[pool["text"].notna()]
        pool = pool[pool["text"].str.strip().str.len() >= min_text_length]
        raw_count = len(pool)

        if raw_count == 0:
            print(f"  {ds_name}: no documents after filtering")
            summary.append((ds_name, 0, 0))
            continue

        # Quality filtering via embeddings + Ridge models (if available)
        pool = _apply_quality_filter(
            pool, ds_name, embeddings_root, start_year, end_year,
            quality_percentile,
        )
        filtered_count = len(pool)

        # Cap
        if filtered_count > max_per_collection:
            pool = pool.sample(n=max_per_collection, random_state=seed)
            action = "sampled"
        else:
            action = "all"

        # Write parquet
        pool = pool.reset_index(drop=True)
        pool["doc_name"] = [f"doc_{i:05d}" for i in range(len(pool))]
        output_df = pool[["doc_name", "text"]]
        output_path = input_base / f"{ds_name}.parquet"
        output_df.to_parquet(output_path, index=False)

        n_written = len(output_df)
        print(f"  {ds_name}: {n_written:,} / {raw_count:,} ({action})")
        summary.append((ds_name, raw_count, n_written))

    total_written = sum(s[2] for s in summary)
    print(f"\n  Additional total: {total_written:,} documents")
    return summary


def _apply_quality_filter(pool, ds_name, embeddings_root, start_year, end_year,
                          quality_percentile):
    """Apply quality filtering using embeddings + Ridge models if available.

    Falls back to no filtering (text length only) if embeddings or models
    are not available.
    """
    emb_dir = embeddings_root / ds_name
    if not emb_dir.exists():
        print(f"    (no embeddings found for {ds_name}, skipping quality filter)")
        return pool

    # Load embeddings for the year range
    emb_frames = []
    for year in range(start_year, end_year + 1):
        emb_path = emb_dir / f"embeddings_{year}.parquet"
        if emb_path.exists():
            emb_frames.append(pd.read_parquet(emb_path))

    if not emb_frames:
        print(f"    (no embeddings for year range, skipping quality filter)")
        return pool

    emb_df = pd.concat(emb_frames, ignore_index=True)

    # Check if 'embedding' column exists
    if "embedding" not in emb_df.columns:
        print(f"    (no 'embedding' column found, skipping quality filter)")
        return pool

    # Apply quality models per quality-period
    scores = []
    for _, row in emb_df.iterrows():
        year = row.get("year", start_year)
        qp = year_to_quality_period(year)
        if qp is None:
            scores.append(np.nan)
            continue
        # Cache model loading would be nice, but keep it simple for now
        models = load_quality_model(qp)
        if models is None:
            scores.append(np.nan)
            continue
        scaler, model = models
        X = np.array(row["embedding"]).reshape(1, -1)
        X_scaled = scaler.transform(X)
        scores.append(model.predict(X_scaled)[0])

    emb_df["predicted_quality"] = scores

    # Join scores back to text pool
    # Try common identifier columns
    id_col = None
    for candidate in ["source_id", "identifier", "_id", "id", "article_id"]:
        if candidate in pool.columns and candidate in emb_df.columns:
            id_col = candidate
            break

    if id_col is None:
        # Fall back to index-based join if both have same length
        if len(pool) == len(emb_df):
            pool = pool.reset_index(drop=True)
            pool["predicted_quality"] = emb_df["predicted_quality"].values
        else:
            print(f"    (cannot join embeddings to text, skipping quality filter)")
            return pool
    else:
        emb_scores = emb_df[[id_col, "predicted_quality"]].dropna()
        pool = pool.merge(emb_scores, on=id_col, how="left")

    # Filter by percentile (only on rows that have scores)
    has_score = pool["predicted_quality"].notna()
    if has_score.sum() > 0:
        threshold = pool.loc[has_score, "predicted_quality"].quantile(
            quality_percentile / 100
        )
        pool = pool[~has_score | (pool["predicted_quality"] >= threshold)]
        n_filtered = has_score.sum() - (pool["predicted_quality"].notna()).sum()
        print(f"    (quality filter: removed {n_filtered:,} below "
              f"p{quality_percentile} threshold)")

    # Drop the quality column before returning
    pool = pool.drop(columns=["predicted_quality"], errors="ignore")
    return pool


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def prepare(period, source="all", quality_percentile=50, max_per_collection=10000,
            min_text_length=200, seed=42):
    """One-stop preparation: corpus + additional → uniform parquets."""
    paths = get_paths(period)
    input_base = paths["synthetic_dir"] / "input"

    print(f"{'='*60}")
    print(f"Preparing documents for {period}")
    print(f"  Quality filter: top {100 - quality_percentile}%")
    print(f"  Max per collection: {max_per_collection:,}")
    print(f"  Output: {input_base}")
    print(f"{'='*60}")

    if source in ("all", "corpus"):
        print(f"\n--- Main Corpus ---")
        prepare_corpus(
            period,
            quality_percentile=quality_percentile,
            max_per_collection=max_per_collection,
            seed=seed,
        )

    if source in ("all", "additional"):
        print(f"\n--- Additional Datasets ---")
        prepare_additional(
            period,
            quality_percentile=quality_percentile,
            max_per_collection=max_per_collection,
            min_text_length=min_text_length,
            seed=seed,
        )

    # Print summary
    if input_base.exists():
        parquets = sorted(input_base.glob("*.parquet"))
        print(f"\n{'='*60}")
        print(f"Prepared {len(parquets)} collection parquets:")
        total = 0
        for p in parquets:
            n = len(pd.read_parquet(p, columns=["doc_name"]))
            total += n
            print(f"  {p.name}: {n:,} docs")
        print(f"Total: {total:,} documents available for generation")
        print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare corpus documents for synthetic data generation"
    )
    parser.add_argument("--period", type=str, required=True,
                        choices=list(PERIODS.keys()))
    parser.add_argument("--source", type=str, default="all",
                        choices=["all", "corpus", "additional"],
                        help="Which sources to prepare (default: all)")
    parser.add_argument("--quality-percentile", type=int, default=50,
                        help="Quality floor percentile (default: 50 = top 50%%)")
    parser.add_argument("--max-per-collection", type=int, default=10000,
                        help="Max documents per collection (default: 10000)")
    parser.add_argument("--min-text-length", type=int, default=200,
                        help="Minimum text length for additional data (default: 200)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare(
        args.period,
        source=args.source,
        quality_percentile=args.quality_percentile,
        max_per_collection=args.max_per_collection,
        min_text_length=args.min_text_length,
        seed=args.seed,
    )
