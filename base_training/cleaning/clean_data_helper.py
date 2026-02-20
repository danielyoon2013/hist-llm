
import pandas as pd
import numpy as np
import re
import gc
from pathlib import Path

def process_single_file(file_path, masks_dir):
    try:
        df_meta = pd.read_parquet(file_path, columns=[], engine='pyarrow')
        if not df_meta.index.is_unique:
            return {"file": str(file_path), "error": "NON-UNIQUE INDEX"}
        del df_meta
        
        # Load only text column to minimize I/O
        df = pd.read_parquet(file_path, columns=['text'], engine='pyarrow')
        original_count = len(df)
        year = Path(file_path).parent.name
        subset_name = Path(file_path).name

        # --- FUNNEL STAGE 1: SYMBOL RATIO ---
        sym_counts = df['text'].str.count(r'[^\w\s]')
        word_counts = df['text'].str.split().str.len()
        df['fail_sym'] = (sym_counts / (word_counts + 1e-6)) > 0.25

        # --- FUNNEL STAGE 2: PUNCTUATION DENSITY ---
        def fast_punct(text):
            if not text: return 0
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            if not lines: return 0
            ends = sum(1 for l in lines if l.endswith(('.', '?', '!')))
            return ends / len(lines)

        df['fail_punct'] = df['text'].apply(fast_punct) < 0.12

        # --- FUNNEL STAGE 3: STOPWORDS ---
        stop_pattern = r'\b(the|and|is|of|to|it|that|in)\b'
        stop_counts = df['text'].str.count(stop_pattern, flags=re.IGNORECASE)
        df['fail_stops'] = stop_counts < 3

        # --- FINAL AGGREGATION ---
        df['is_clean'] = ~(df['fail_punct'] | df['fail_sym'] | df['fail_stops'])

        # --- PREPARE RESULTS DICT ---
        res_dict = {
            'year': year, 
            'file': subset_name, 
            'total_rows': original_count,
            'failed_punct': int(df['fail_punct'].sum()),
            'failed_symbols': int(df['fail_sym'].sum()),
            'failed_stopwords': int(df['fail_stops'].sum()),
            'clean_rows': int(df['is_clean'].sum()),
            'reduction_pct': float((1 - (df['is_clean'].sum() / original_count)) * 100)
        }

        # --- SAVING LOGIC ---
        year_mask_dir = Path(masks_dir) / year
        year_mask_dir.mkdir(parents=True, exist_ok=True)
        
        clean_indices = df[df['is_clean']].index.to_series()
        mask_filename = year_mask_dir / f"{subset_name.replace('.parquet', '_mask.parquet')}"
        pd.DataFrame(clean_indices, columns=['original_index']).to_parquet(mask_filename, index=False)

        # --- CLEANUP ---
        del df
        gc.collect()

        return res_dict
        
    except Exception as e:
        return {"file": str(file_path), "error": str(e)}