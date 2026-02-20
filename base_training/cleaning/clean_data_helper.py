"""
Cleaning heuristics for historical document text.

Three-stage filter:
  1. Symbol ratio    — rejects garbled OCR (non-word chars > 25% of word count)
  2. Punctuation density — rejects tables/lists (<12% of lines end with sentence punctuation)
  3. Stopwords       — rejects non-English text (fewer than 3 common stopwords)

Usage:
    # Core logic — works on any text Series
    mask = compute_clean_mask(df["text"])
    clean_df = df[mask]

    # File-level wrapper for parallel processing (English corpus)
    result = process_parquet_file(path, masks_dir, text_col="text")
"""

import pandas as pd
import re
import gc
from pathlib import Path


def compute_clean_mask(texts: pd.Series) -> tuple:
    """Apply cleaning heuristics to a Series of text strings.

    Returns:
        (is_clean, stats) where:
            is_clean: pd.Series[bool] — True for rows that pass all filters
            stats: dict with counts of failures per stage
    """
    n = len(texts)

    # Stage 1: Symbol ratio — garbled OCR
    sym_counts = texts.str.count(r'[^\w\s]')
    word_counts = texts.str.split().str.len()
    fail_sym = (sym_counts / (word_counts + 1e-6)) > 0.25

    # Stage 2: Punctuation density — tables/lists/fragments
    def _punct_density(text):
        if not text:
            return 0
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if not lines:
            return 0
        ends = sum(1 for l in lines if l.endswith(('.', '?', '!')))
        return ends / len(lines)

    fail_punct = texts.apply(_punct_density) < 0.12

    # Stage 3: Stopwords — non-English text
    stop_pattern = r'\b(the|and|is|of|to|it|that|in)\b'
    stop_counts = texts.str.count(stop_pattern, flags=re.IGNORECASE)
    fail_stops = stop_counts < 3

    # Combine
    is_clean = ~(fail_sym | fail_punct | fail_stops)

    stats = {
        'total_rows': n,
        'failed_symbols': int(fail_sym.sum()),
        'failed_punct': int(fail_punct.sum()),
        'failed_stopwords': int(fail_stops.sum()),
        'clean_rows': int(is_clean.sum()),
        'reduction_pct': round((1 - is_clean.sum() / n) * 100, 2) if n > 0 else 0,
    }

    return is_clean, stats


def process_parquet_file(file_path, masks_dir, text_col='text'):
    """Clean a single parquet file and save mask. For use with ProcessPoolExecutor.

    Args:
        file_path: Path to parquet file containing text data.
        masks_dir: Directory to save cleaning mask (organized by year or flat).
        text_col: Name of the text column to clean.

    Returns:
        dict with stats, or dict with 'error' key on failure.
    """
    try:
        file_path = Path(file_path)
        df = pd.read_parquet(file_path, columns=[text_col], engine='pyarrow')
        df = df.rename(columns={text_col: '_text'})

        is_clean, stats = compute_clean_mask(df['_text'])
        stats['file'] = file_path.name

        # Save mask: indices of clean rows
        masks_dir = Path(masks_dir)
        masks_dir.mkdir(parents=True, exist_ok=True)
        mask_name = file_path.stem + '_mask.parquet'
        clean_indices = df[is_clean.values].index.to_series()
        pd.DataFrame(clean_indices, columns=['original_index']).to_parquet(
            masks_dir / mask_name, index=False
        )

        del df
        gc.collect()
        return stats

    except Exception as e:
        return {"file": str(file_path), "error": str(e)}
