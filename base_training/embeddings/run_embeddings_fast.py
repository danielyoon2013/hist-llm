"""
Maximum speed embedding script for BGE-large with truncation.
Optimizations:
1. Half precision (fp16) - 2x memory efficiency, faster on A100
2. Larger batch size (768-1024)
3. Disable tokenizer parallelism warning
4. Optimized encode parameters
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import tarfile
import torch
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
INBOX = Path("/home/ubuntu/embedding/input_data")
OUTBOX = Path("/home/ubuntu/embedding/output_data")
TEMP_EXTRACT = Path("/home/ubuntu/embedding/temp_extract")
for p in [INBOX, OUTBOX, TEMP_EXTRACT]: p.mkdir(exist_ok=True)

# --- MODEL SETUP WITH FP16 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load model in half precision for speed
model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)
model.half()  # Convert to fp16 - faster on A100

instruction = "Represent this historical text for quality classification: "

# A100 40GB can handle large batches with fp16
BATCH_SIZE = 1024  # Increase from 256


def process_year():
    while True:
        archives = sorted(INBOX.glob("year_*.tar"))
        for arch in archives:
            year = arch.stem.split("_")[1]
            print(f"\n{'='*50}")
            print(f"Processing Year {year}")
            print(f"{'='*50}")

            # --- Extract ---
            t0 = time.time()
            try:
                with tarfile.open(arch, "r:") as tar:
                    tar.extractall(path=TEMP_EXTRACT)
            except tarfile.ReadError as e:
                print(f"!!! Error extracting {arch.name}: {e}. Skipping...")
                for junk in TEMP_EXTRACT.glob("*"):
                    if junk.is_file():
                        junk.unlink()
                continue
            print(f"Extraction time: {time.time() - t0:.1f}s")

            # --- Load all parquets ---
            t0 = time.time()
            all_parquets = list(TEMP_EXTRACT.glob("*.parquet"))
            if not all_parquets:
                print(f"No parquet files found. Skipping...")
                arch.unlink()
                continue

            all_dfs = []
            for p_file in all_parquets:
                df = pd.read_parquet(p_file)
                df['subset_source'] = p_file.name
                df['row_idx'] = df.index
                all_dfs.append(df)
                p_file.unlink()

            combined_df = pd.concat(all_dfs, ignore_index=True)
            print(f"Loaded {len(combined_df):,} rows in {time.time() - t0:.1f}s")

            # --- Prepare texts ---
            t0 = time.time()
            # Only take first 2000 chars to speed up tokenization (512 tokens ≈ 2000 chars)
            all_texts = [
                instruction + str(t)[:2000]
                for t in combined_df['text'].tolist()
            ]
            print(f"Text prep: {time.time() - t0:.1f}s")

            # --- Embed with maximum speed ---
            t0 = time.time()
            with torch.inference_mode():  # Faster than no_grad
                all_embeddings = model.encode(
                    all_texts,
                    batch_size=BATCH_SIZE,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                    # normalize_embeddings=False (default) - keeping consistent with previous runs
                )
            embed_time = time.time() - t0
            texts_per_sec = len(all_texts) / embed_time
            print(f"Embedding: {embed_time:.1f}s ({texts_per_sec:,.0f} texts/sec)")

            # --- Save results ---
            t0 = time.time()
            final_df = pd.DataFrame({
                'original_index': combined_df['identifier'],
                'subset_source': combined_df['subset_source'],
                'row_idx': combined_df['row_idx'],
                'embedding': list(all_embeddings)
            })
            out_name = f"embeddings_{year}.parquet"
            final_df.to_parquet(OUTBOX / out_name)
            print(f"Save: {time.time() - t0:.1f}s")

            # Cleanup
            arch.unlink()

            # Wait for local to download (max 2 minutes, then move on)
            print(f"Waiting for download of {out_name}...")
            wait_count = 0
            max_wait = 12  # 12 x 10s = 2 minutes
            while (OUTBOX / out_name).exists() and wait_count < max_wait:
                time.sleep(10)
                wait_count += 1

            if (OUTBOX / out_name).exists():
                print(f"Timeout waiting for download. Moving to next year...")
            else:
                print(f"Year {year} complete!")

        time.sleep(10)


if __name__ == "__main__":
    process_year()
