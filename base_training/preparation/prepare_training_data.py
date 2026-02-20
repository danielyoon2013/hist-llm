"""
Prepare training data for nanochat by filtering high-quality documents.

This script:
1. Loads classified documents for each Option A period (English + additional)
2. Filters documents above the quality cutoff threshold
3. Joins with raw text from both English corpus and additional news data
4. Shards data into ~250M character parquet files for nanochat

Sources:
    English:    D:\\hist_LLM\\corpus\\{classified,raw}\\
    Additional: D:\\hist_LLM\\additional_data\\{classified,raw}\\{nyt,economist,ft,newswire}

Optimizations:
- Parallel file reads within each year (16 workers, saturates SSD I/O)
- Sequential year processing (only 1 year in memory at a time)
- Two-pass reads (identifier column first, skip row groups with no matches)
- PyArrow direct (no pandas DataFrame overhead for text loading)

Output structure:
    D:\\hist_LLM\\periods\\
    ├── 1678_1849/
    │   └── base_data/
    │       ├── shard_00000.parquet
    │       └── ...
    └── ...

Usage:
    python prepare_training_data.py                    # Prepare all periods
    python prepare_training_data.py --period 1678_1849 # Prepare single period
    python prepare_training_data.py --dry-run          # Show stats without writing
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import gc

# --- CONFIG ---
# English corpus
CLASSIFIED_DIR = Path(r"D:\hist_LLM\corpus\classified")
RAW_DIR = Path(r"D:\hist_LLM\corpus\raw")

# Additional news data
ADDITIONAL_CLASSIFIED_DIR = Path(r"D:\hist_LLM\additional_data\classified")
ADDITIONAL_RAW_DIR = Path(r"D:\hist_LLM\additional_data\raw")
ADDITIONAL_COLLECTIONS = {
    "nyt": {
        "raw_dir": ADDITIONAL_RAW_DIR / "news_archives" / "NYT_filtered_500char",
        "file_pattern": "nyt_{year}.parquet",
        "text_col": "combined_text",
        "id_col": "_id",
    },
    "economist": {
        "raw_dir": ADDITIONAL_RAW_DIR / "news_archives" / "Economist",
        "file_pattern": "economist_{year}-*.parquet",
        "text_col": "ocr_text",
        "id_col": "article_id",
    },
    "ft": {
        "raw_dir": ADDITIONAL_RAW_DIR / "news_archives" / "FT",
        "file_pattern": "{year}.parquet",
        "text_col": "text_cleaned",
        "id_col": "id",
    },
    "newswire": {
        "raw_dir": ADDITIONAL_RAW_DIR / "newswire",
        "file_pattern": "{year}_data_clean.json",
        "text_col": "cleaned_article",
        "id_col": None,
    },
}

CUTOFF_FILE = Path(r"D:\hist_LLM\processing\quality_graphs\period_summary.csv")
OUTPUT_DIR = Path(r"D:\hist_LLM\periods")
STAGING_DIR = Path(r"D:\hist_LLM\processing\staging")

# Target shard size (~250M characters)
SHARD_SIZE_CHARS = 250_000_000
# Row group size for parquet files (must be >= num_gpus for DDP distribution)
# Matches nanochat's repackage_data_reference.py
ROW_GROUP_SIZE = 1024
# Parallel file reads within each year (saturates NVMe SSD I/O queues)
FILE_READ_WORKERS = 16

# Option A periods
PERIOD_RANGES = [
    (1678, 1849, "1678_1849"),
    (1850, 1899, "1850_1899"),
    (1900, 1949, "1900_1949"),
    (1950, 1999, "1950_1999"),
    (2000, 2009, "2000_2009"),
    (2010, 2023, "2010_2023"),
]


def load_cutoff_scores() -> dict:
    """Load cutoff scores from period_summary.csv."""
    df = pd.read_csv(CUTOFF_FILE)
    return dict(zip(df['period'], df['cutoff_score']))


def _load_classified_year(args):
    """Load high-quality identifiers for a single year. Used by thread pool."""
    year, cutoff, source = args
    if source == "english":
        classified_path = CLASSIFIED_DIR / f"classified_{year}.parquet"
    else:
        classified_path = ADDITIONAL_CLASSIFIED_DIR / source / f"classified_{year}.parquet"
    if not classified_path.exists():
        return source, []
    df = pd.read_parquet(classified_path, columns=['identifier', 'predicted_quality'])
    high_quality = df[df['predicted_quality'] >= cutoff]['identifier'].tolist()
    del df
    return source, high_quality


def get_high_quality_identifiers(start_year: int, end_year: int, cutoff: float, workers: int = 8) -> tuple:
    """Get identifiers of documents above quality cutoff for a period.

    Returns (english_ids: set, additional_ids: dict[collection, set]).
    """
    english_ids = set()
    additional_ids = {c: set() for c in ADDITIONAL_COLLECTIONS}

    tasks = []
    for y in range(start_year, end_year + 1):
        tasks.append((y, cutoff, "english"))
        for collection in ADDITIONAL_COLLECTIONS:
            tasks.append((y, cutoff, collection))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for source, result in tqdm(executor.map(_load_classified_year, tasks),
                                   total=len(tasks), desc="Loading classified", leave=False):
            if source == "english":
                english_ids.update(result)
            else:
                additional_ids[source].update(result)

    return english_ids, additional_ids


def _read_matching_texts(args):
    """Read matching texts from a single parquet file.

    Two-pass at row-group level using pyarrow directly (no pandas):
      Pass 1: Read only identifier column (small/cheap)
      Pass 2: Read text column only for row groups with matches (skip the rest)
    """
    parquet_file, valid_set = args
    try:
        pf = pq.ParquetFile(parquet_file)
        all_texts = []
        for rg_idx in range(pf.metadata.num_row_groups):
            # Pass 1: identifier column only — cheap, small column
            ids = pf.read_row_group(rg_idx, columns=['identifier']).column('identifier').to_pylist()
            mask = [id_val in valid_set for id_val in ids]
            if not any(mask):
                continue  # Skip expensive text column read for this row group
            # Pass 2: text column — only read if row group has matches
            text_col = pf.read_row_group(rg_idx, columns=['text']).column('text').to_pylist()
            all_texts.extend(t for t, m in zip(text_col, mask) if m and t is not None)
        return all_texts
    except Exception as e:
        print(f"  [WARN] Error reading {parquet_file}: {e}", flush=True)
        return []


def load_texts_for_year(year: int, valid_identifiers: set) -> list:
    """Load English text content for documents in valid_identifiers set.

    Uses parallel file reads (FILE_READ_WORKERS threads) to saturate SSD I/O.
    """
    year_dir = RAW_DIR / str(year)
    if not year_dir.exists():
        return []

    parquet_files = sorted(year_dir.glob("*.parquet"))
    if not parquet_files:
        return []

    texts = []
    args_list = [(f, valid_identifiers) for f in parquet_files]
    with ThreadPoolExecutor(max_workers=FILE_READ_WORKERS) as executor:
        for result in executor.map(_read_matching_texts, args_list):
            texts.extend(result)
            del result
    return texts


def load_additional_texts_for_year(year: int, collection: str, valid_ids: set) -> list:
    """Load text from an additional data collection for a given year."""
    if not valid_ids:
        return []

    cfg = ADDITIONAL_COLLECTIONS[collection]
    texts = []

    try:
        if collection == "newswire":
            raw_path = cfg["raw_dir"] / f"{year}_data_clean.json"
            if not raw_path.exists():
                return []
            with open(raw_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for i, doc in enumerate(data):
                idx = f"newswire_{year}_{i}"
                if idx in valid_ids:
                    text = doc.get("cleaned_article", "")
                    if text:
                        texts.append(text)
        elif collection == "economist":
            files = sorted(cfg["raw_dir"].glob(f"economist_{year}-*.parquet"))
            for fp in files:
                df = pd.read_parquet(fp, columns=[cfg["id_col"], cfg["text_col"]])
                for _, row in df.iterrows():
                    if str(row[cfg["id_col"]]) in valid_ids:
                        text = row[cfg["text_col"]]
                        if text and str(text).strip():
                            texts.append(str(text))
                del df
        else:
            pattern = cfg["file_pattern"].format(year=year)
            raw_path = cfg["raw_dir"] / pattern
            if not raw_path.exists():
                return []
            df = pd.read_parquet(raw_path, columns=[cfg["id_col"], cfg["text_col"]])
            for _, row in df.iterrows():
                if str(row[cfg["id_col"]]) in valid_ids:
                    text = row[cfg["text_col"]]
                    if text and str(text).strip():
                        texts.append(str(text))
            del df
    except Exception as e:
        print(f"  [WARN] Error loading {collection}/{year}: {e}", flush=True)

    return texts


def prepare_period(start_year: int, end_year: int, period_name: str,
                   cutoff: float, dry_run: bool = False) -> dict:
    """Prepare training data for a single period (English + additional)."""
    print(f"\n{'='*60}")
    print(f"Preparing {period_name} (cutoff={cutoff:.4f})")
    print(f"{'='*60}")

    # Step 1: Get high-quality document identifiers from both sources
    print("Step 1: Finding high-quality documents...")
    english_ids, additional_ids = get_high_quality_identifiers(start_year, end_year, cutoff)
    total_additional = sum(len(v) for v in additional_ids.values())
    print(f"  English: {len(english_ids):,} docs above cutoff")
    for coll, ids in additional_ids.items():
        if ids:
            print(f"  {coll}: {len(ids):,} docs above cutoff")
    total_above = len(english_ids) + total_additional
    print(f"  Total: {total_above:,} documents above cutoff")

    if dry_run:
        return {
            'period': period_name,
            'docs_above_cutoff': total_above,
            'english_docs': len(english_ids),
            'additional_docs': total_additional,
            'shards': 0,
            'total_chars': 0
        }

    # Step 2: Stream text content to temp file on local disk (avoids Dropbox sync bottleneck)
    print("Step 2: Loading text content...")
    base_data_dir = OUTPUT_DIR / period_name / "base_data"
    base_data_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = STAGING_DIR / period_name
    staging_dir.mkdir(parents=True, exist_ok=True)
    temp_path = staging_dir / "_temp.parquet"
    schema = pa.schema([('text', pa.string())])
    writer = pq.ParquetWriter(temp_path, schema, use_dictionary=False, compression='zstd')
    total_docs = 0
    total_chars = 0

    WRITE_CHUNK_SIZE = 50_000  # Write in chunks to avoid memory spikes during table creation
    # Sequential year processing: only 1 year's texts in memory at a time.
    for year in tqdm(range(start_year, end_year + 1), desc="Loading texts"):
        # English corpus
        texts = load_texts_for_year(year, english_ids)
        # Additional collections
        for collection in ADDITIONAL_COLLECTIONS:
            coll_ids = additional_ids.get(collection, set())
            if coll_ids:
                add_texts = load_additional_texts_for_year(year, collection, coll_ids)
                texts.extend(add_texts)
                del add_texts

        if texts:
            total_docs += len(texts)
            total_chars += sum(len(t) for t in texts)
            for i in range(0, len(texts), WRITE_CHUNK_SIZE):
                chunk = texts[i:i + WRITE_CHUNK_SIZE]
                writer.write_table(pa.Table.from_pydict({"text": chunk}))
        del texts
        gc.collect()
    writer.close()

    print(f"  Loaded {total_docs:,} documents with text")
    print(f"  Total characters: {total_chars:,} ({total_chars/1e9:.2f}B)")

    # Step 3: Shuffle row group order (block-level shuffle)
    print("Step 3: Shuffling documents...")
    np.random.seed(42)
    pf = pq.ParquetFile(temp_path)
    rg_order = np.random.permutation(pf.metadata.num_row_groups)

    # Step 4: Write shards (read row groups in shuffled order, shuffle within each)
    print("Step 4: Writing shards...")
    shard_idx = 0
    current_texts = []
    current_chars = 0

    for rg_i in tqdm(rg_order, desc="Writing shards"):
        rg_texts = pf.read_row_group(int(rg_i)).column('text').to_pylist()
        np.random.shuffle(rg_texts)
        for text in rg_texts:
            if text is None:
                continue
            text_len = len(text)
            if current_chars + text_len > SHARD_SIZE_CHARS and current_texts:
                shard_path = base_data_dir / f"shard_{shard_idx:05d}.parquet"
                pq.write_table(pa.Table.from_pydict({"text": current_texts}),
                               shard_path, row_group_size=ROW_GROUP_SIZE,
                               use_dictionary=False, compression="zstd",
                               compression_level=3, write_statistics=False)
                shard_idx += 1
                current_texts = []
                current_chars = 0
            current_texts.append(text)
            current_chars += text_len
        del rg_texts
        gc.collect()

    if current_texts:
        shard_path = base_data_dir / f"shard_{shard_idx:05d}.parquet"
        pq.write_table(pa.Table.from_pydict({"text": current_texts}),
                       shard_path, row_group_size=ROW_GROUP_SIZE,
                       use_dictionary=False, compression="zstd",
                       compression_level=3, write_statistics=False)
        shard_idx += 1

    del pf  # Close file handle before deleting (Windows file locking)
    temp_path.unlink()
    print(f"  Wrote {shard_idx} shards to {base_data_dir}")

    return {
        'period': period_name,
        'docs_above_cutoff': total_above,
        'english_docs': len(english_ids),
        'additional_docs': total_additional,
        'docs_with_text': total_docs,
        'shards': shard_idx,
        'total_chars': total_chars
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for nanochat")
    parser.add_argument('--period', type=str, help='Single period to process (e.g., 1678_1849)')
    parser.add_argument('--dry-run', action='store_true', help='Show stats without writing files')
    args = parser.parse_args()

    # Load cutoff scores
    cutoff_scores = load_cutoff_scores()
    print("Cutoff scores:")
    for period, cutoff in cutoff_scores.items():
        print(f"  {period}: {cutoff:.4f}")

    # Filter periods if specified
    if args.period:
        periods = [(s, e, n) for s, e, n in PERIOD_RANGES if n == args.period]
        if not periods:
            print(f"Error: Unknown period '{args.period}'")
            return
    else:
        periods = PERIOD_RANGES

    # Process each period
    results = []
    for start, end, period_name in periods:
        cutoff = cutoff_scores.get(period_name)
        if cutoff is None:
            print(f"[SKIP] No cutoff score for {period_name}")
            continue

        result = prepare_period(start, end, period_name, cutoff, dry_run=args.dry_run)
        results.append(result)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

    if not args.dry_run:
        summary_path = OUTPUT_DIR / "preparation_summary.csv"
        # Merge with existing summary to preserve previous runs
        if summary_path.exists():
            existing_df = pd.read_csv(summary_path)
            combined_df = pd.concat([existing_df, summary_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['period'], keep='last')
            combined_df = combined_df.sort_values('period').reset_index(drop=True)
            combined_df.to_csv(summary_path, index=False)
        else:
            summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
