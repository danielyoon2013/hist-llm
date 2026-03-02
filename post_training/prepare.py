"""
Prepare corpus documents for synthetic data generation.

Replaces the old 3-script sequence (build_index + export + export_additional)
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
import json
import argparse
from glob import glob
from pathlib import Path

import pandas as pd

from src.post_training.config import PERIODS, DATA_ROOT, get_paths


# ---------------------------------------------------------------------------
# Additional data loaders — each returns DataFrame with [text, source_id]
# ---------------------------------------------------------------------------

ADDITIONAL_DATASETS = {
    "nyt_filtered": {
        "config_key": "nyt_filtered_dir",
        "classified_name": "nyt",
        "description": "NYT filtered abstracts (500+ chars)",
    },
    "economist": {
        "config_key": "economist_dir",
        "classified_name": "economist",
        "description": "The Economist (OCR text)",
    },
    "ft": {
        "config_key": "ft_dir",
        "classified_name": "ft",
        "description": "Financial Times (cleaned text)",
    },
    "newswire": {
        "config_key": "newswire_dir",
        "classified_name": "newswire",
        "description": "US Newswire articles (cleaned)",
    },
}


def _load_nyt_filtered(data_dir, start_year, end_year):
    """Load NYT filtered abstracts from yearly parquet files."""
    frames = []
    for year in range(max(start_year, 1851), min(end_year, 2017) + 1):
        path = data_dir / f"nyt_{year}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["combined_text", "_id"])
        df.rename(columns={"combined_text": "text", "_id": "source_id"}, inplace=True)
        df["year"] = year
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["text", "source_id", "year"])
    return pd.concat(frames, ignore_index=True)


def _load_economist(data_dir, start_year, end_year):
    """Load Economist articles from weekly parquet files."""
    all_files = sorted(data_dir.glob("economist_*.parquet"))
    matching = []
    for f in all_files:
        try:
            year = int(f.stem.split("_")[1].split("-")[0])
        except (IndexError, ValueError):
            continue
        if start_year <= year <= end_year:
            matching.append((f, year))

    if not matching:
        return pd.DataFrame(columns=["text", "source_id", "year"])

    frames = []
    for i, (f, year) in enumerate(matching):
        try:
            df = pd.read_parquet(f, columns=["ocr_text", "article_id"])
            df.rename(columns={"ocr_text": "text", "article_id": "source_id"},
                      inplace=True)
            df["year"] = year
            frames.append(df)
        except Exception as e:
            print(f"  Warning: Failed to read {f.name}: {e}")
        if (i + 1) % 500 == 0:
            print(f"  Read {i + 1}/{len(matching)} Economist files...")

    if not frames:
        return pd.DataFrame(columns=["text", "source_id", "year"])
    return pd.concat(frames, ignore_index=True)


def _load_ft(data_dir, start_year, end_year):
    """Load Financial Times articles from yearly parquet files."""
    frames = []
    for year in range(max(start_year, 1888), min(end_year, 2006) + 1):
        path = data_dir / f"{year}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["text_cleaned", "id"])
        df.rename(columns={"text_cleaned": "text", "id": "source_id"}, inplace=True)
        df["year"] = year
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["text", "source_id", "year"])
    return pd.concat(frames, ignore_index=True)


def _load_newswire(data_dir, start_year, end_year):
    """Load Newswire articles from yearly JSON files."""
    frames = []
    for year in range(max(start_year, 1878), min(end_year, 1977) + 1):
        path = data_dir / f"{year}_data_clean.json"
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  Warning: Skipping {path.name} (corrupted JSON: {e})")
            continue
        records = []
        for idx, item in enumerate(data):
            text = item.get("cleaned_article", "")
            records.append({
                "text": text,
                "source_id": f"newswire_{year}_{idx}",
                "year": year,
            })
        frames.append(pd.DataFrame(records))
    if not frames:
        return pd.DataFrame(columns=["text", "source_id", "year"])
    return pd.concat(frames, ignore_index=True)


_LOADERS = {
    "nyt_filtered": _load_nyt_filtered,
    "economist": _load_economist,
    "ft": _load_ft,
    "newswire": _load_newswire,
}


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

    Uses pre-calculated quality scores from
    D:/hist_LLM/additional_data/classified/{collection}/classified_{year}.parquet
    (generated by src/base_training/quality/classify_additional.py).

    Steps per dataset:
    1. Load text via loaders
    2. Filter by min text length
    3. Join pre-calculated quality scores and apply percentile filter
    4. Cap at max_per_collection docs
    5. Write as {dataset}.parquet
    """
    paths = get_paths(period)
    start_year, end_year = paths["start_year"], paths["end_year"]
    input_base = paths["synthetic_dir"] / "input"
    classified_root = DATA_ROOT / "additional_data" / "classified"

    os.makedirs(input_base, exist_ok=True)

    summary = []
    for ds_name, ds_info in ADDITIONAL_DATASETS.items():
        data_dir = paths[ds_info["config_key"]]
        if not data_dir.exists():
            print(f"  {ds_name}: directory not found, skipping")
            summary.append((ds_name, 0, 0))
            continue

        loader = _LOADERS[ds_name]
        pool = loader(data_dir, start_year, end_year)

        # Basic text filter
        pool = pool[pool["text"].notna()]
        pool = pool[pool["text"].str.strip().str.len() >= min_text_length]
        raw_count = len(pool)

        if raw_count == 0:
            print(f"  {ds_name}: no documents after filtering")
            summary.append((ds_name, 0, 0))
            continue

        # Quality filtering via pre-calculated classified parquets
        classified_name = ds_info["classified_name"]
        pool = _apply_quality_filter_from_classified(
            pool, classified_name, classified_root,
            start_year, end_year, quality_percentile,
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


def _apply_quality_filter_from_classified(pool, classified_name, classified_root,
                                          start_year, end_year, quality_percentile):
    """Apply quality filtering using pre-calculated classified parquets.

    Reads from: classified_root/{classified_name}/classified_{year}.parquet
    These files have columns [identifier, predicted_quality] and are generated
    by src/base_training/quality/classify_additional.py.
    """
    classified_dir = classified_root / classified_name
    if not classified_dir.exists():
        print(f"    (no classified scores for {classified_name}, skipping quality filter)")
        return pool

    # Load quality scores for the year range
    score_frames = []
    for year in range(start_year, end_year + 1):
        path = classified_dir / f"classified_{year}.parquet"
        if path.exists():
            score_frames.append(pd.read_parquet(path))

    if not score_frames:
        print(f"    (no classified scores for year range, skipping quality filter)")
        return pool

    scores_df = pd.concat(score_frames, ignore_index=True)

    # The classified files use 'identifier' which is the original_index from embeddings.
    # Try to join via source_id (which matches the original identifier in the text data).
    id_col = None
    for candidate in ["source_id", "identifier", "_id", "id", "article_id"]:
        if candidate in pool.columns:
            id_col = candidate
            break

    if id_col is None:
        print(f"    (no identifier column in pool, skipping quality filter)")
        return pool

    # Join scores
    scores_df = scores_df.rename(columns={"identifier": id_col})
    before = len(pool)
    pool = pool.merge(
        scores_df[[id_col, "predicted_quality"]].drop_duplicates(subset=[id_col]),
        on=id_col, how="left",
    )

    # Filter by percentile (only on rows that have scores)
    has_score = pool["predicted_quality"].notna()
    n_scored = has_score.sum()

    if n_scored > 0:
        threshold = pool.loc[has_score, "predicted_quality"].quantile(
            quality_percentile / 100
        )
        # Keep: rows without scores (can't filter) OR rows above threshold
        pool = pool[~has_score | (pool["predicted_quality"] >= threshold)]
        n_removed = before - len(pool)
        print(f"    (quality filter: {n_scored:,} scored, removed {n_removed:,} "
              f"below p{quality_percentile})")
    else:
        print(f"    (no scores matched, skipping quality filter)")

    pool = pool.drop(columns=["predicted_quality"], errors="ignore")
    return pool


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def prepare(period, source="all", quality_percentile=50, max_per_collection=10000,
            min_text_length=200, seed=42):
    """One-stop preparation: corpus + additional -> uniform parquets."""
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
