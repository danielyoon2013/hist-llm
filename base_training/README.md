# Historical Document Quality Classification Pipeline

This pipeline classifies historical documents (1678-2023) by quality on a 1-5 scale using BGE embeddings and Ridge regression.

## Problem

We have ~120 million documents across 346 years. Manually labeling all is impossible, so we:
1. Sample a subset per time period
2. Label samples using OpenAI GPT-4
3. Train a classifier on labeled samples
4. Run inference on the full dataset

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA PREPARATION                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   D:\English\{year}\subset_*.parquet                                        │
│   (~120M documents, 346 years)                                              │
│                                                                             │
│                              │                                              │
│                              ▼                                              │
│                    ┌─────────────────┐                                      │
│                    │  Clean_Data.ipynb│                                      │
│                    │  (Remove noise)  │                                      │
│                    └────────┬────────┘                                      │
│                             │                                               │
│                             ▼                                               │
│              cleaning_masks\{year}\*_mask.parquet                           │
│              (61M clean rows identified)                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SAMPLING & LABELING                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ┌─────────────────┐                                      │
│                    │ Sample_Data.ipynb│                                      │
│                    │ (10k per period) │                                      │
│                    └────────┬────────┘                                      │
│                             │                                               │
│                             ▼                                               │
│                    ┌─────────────────┐                                      │
│                    │ Label_Data.ipynb │                                      │
│                    │ (OpenAI GPT-4)   │                                      │
│                    └────────┬────────┘                                      │
│                             │                                               │
│                             ▼                                               │
│              Labeled samples with quality scores 1-5                        │
│              (~10k samples × 14 periods = 140k labeled)                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EMBEDDING & TRAINING                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   LOCAL (Windows)                      REMOTE (Lambda GPU)                  │
│   ───────────────                      ──────────────────                   │
│                                                                             │
│   Embed_All_Data_Local_Parallel2.ipynb                                      │
│        │                                                                    │
│        │ 1. Slim & archive year ──────► Lambda processes with               │
│        │ 2. Upload via SCP              run_embeddings_fast.py              │
│        │ 3. Wait for results            (BGE-large-en-v1.5, fp16)           │
│        │ 4. Download & verify ◄──────── embeddings_{year}.parquet           │
│        ▼                                                                    │
│   D:\English_Results\embeddings_{year}.parquet                              │
│                                                                             │
│                              │                                              │
│                              ▼                                              │
│                    ┌──────────────────┐                                     │
│                    │train_ridge_models│                                     │
│                    │    .py           │                                     │
│                    └────────┬─────────┘                                     │
│                             │                                               │
│                             ▼                                               │
│              Models\ridge_{period}.pkl  (14 models)                         │
│              Models\scaler_{period}.pkl (14 scalers)                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLASSIFICATION                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ┌────────────────────┐                                   │
│                    │check_and_classify.py│                                   │
│                    └─────────┬──────────┘                                   │
│                              │                                              │
│                              ▼                                              │
│              D:\English_Classified\classified_{year}.parquet                │
│                                                                             │
│              Columns:                                                       │
│              - identifier (document ID)                                     │
│              - predicted_quality (1-5 continuous)                           │
│              - is_clean (True)                                              │
│              - token_count                                                  │
│              - word_count                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FINAL SELECTION                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Sort all documents by predicted_quality (descending)                      │
│   Select top documents until cumulative token_count reaches 35B             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Time Periods

Models are trained separately for 14 periods (25-year windows):

| Period | Years | Period | Years |
|--------|-------|--------|-------|
| 1678_1700 | 1678-1700 | 1851_1875 | 1851-1875 |
| 1701_1725 | 1701-1725 | 1876_1900 | 1876-1900 |
| 1726_1750 | 1726-1750 | 1901_1925 | 1901-1925 |
| 1751_1775 | 1751-1775 | 1926_1950 | 1926-1950 |
| 1776_1800 | 1776-1800 | 1951_1975 | 1951-1975 |
| 1801_1825 | 1801-1825 | 1976_2000 | 1976-2000 |
| 1826_1850 | 1826-1850 | 2001_2023 | 2001-2023 |

## File Reference

### Active Scripts

| File | Purpose | When to Run |
|------|---------|-------------|
| `check_and_classify.py` | Check status & classify new embeddings | After new embeddings are downloaded |
| `download_embeddings_safe.py` | Safe download from Lambda with verification | When downloading results |
| `reprocess_years.py` | Manage corrupted/missing years | When troubleshooting |
| `train_ridge_models.py` | Train Ridge classifiers | One-time (or if retraining) |
| `run_embeddings_fast.py` | Embedding script (deployed on Lambda) | Runs automatically on Lambda |

### Active Notebooks

| File | Purpose |
|------|---------|
| `Embed_All_Data_Local_Parallel2.ipynb` | Main workflow: upload data, wait, download results |
| `Clean_Data.ipynb` | Data cleaning pipeline |
| `Label_Data.ipynb` | OpenAI labeling pipeline |
| `Classify_Data.ipynb` | Classification analysis |

### Legacy/Reference (can be deleted)

| File | Notes |
|------|-------|
| `Embed_All_Data_Local.ipynb` | Superseded by Parallel2 |
| `Embed_All_Data_Local_Parallel.ipynb` | Superseded by Parallel2 |
| `Embed_Data.ipynb` | Early exploration |
| `Sample_Data.ipynb` | One-time sampling |
| `classify_full_data.py` | Superseded by check_and_classify.py |
| `run_embeddings_all.py` | Superseded by run_embeddings_fast.py |
| `benchmark_*.py` | One-time benchmarks |

## Common Commands

```bash
# Check pipeline status
python code/check_and_classify.py --status

# Classify new embeddings (also adds token counts)
python code/check_and_classify.py

# Check reprocessing status
python code/reprocess_years.py

# Safe download from Lambda
python code/download_embeddings_safe.py --year 1800

# Delete corrupted embedding files
python code/reprocess_years.py --delete-corrupted

# Upload year for reprocessing
python code/reprocess_years.py --upload 1800
```

## Directory Structure

```
D:\English\                          # Raw data
    {year}\
        subset_*.parquet             # 100 files per year

D:\English_Results\                  # Embeddings
    embeddings_{year}.parquet

D:\English_Classified\               # Final output
    classified_{year}.parquet

Dropbox\hist_LLM\
    Data\
        Clean_Data\cleaning_masks\   # Cleaning masks
        Classify_Data\Models\        # Ridge models
        Embedding_Data\              # Training embeddings
    code\                            # This folder
```
