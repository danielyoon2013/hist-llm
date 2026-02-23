# Base Training Pipeline

Quality-filtered continued pretraining data from the English historical corpus (1678-2023, ~125M documents).

## Quick Start: Run in Order

| Step | What | How | Output |
|------|------|-----|--------|
| 0 | Clean raw text | `cleaning/Clean_Data.ipynb` | `corpus/cleaning_masks/{year}/` |
| 0.5 | Generate embeddings | `embeddings/run_embeddings_fast.py` (on GPU) | `corpus/embeddings/embeddings_{year}.parquet` |
| 1 | Sample 10K docs/period | `quality/Sample_Data.ipynb` | `processing/sample_data/training_samples_{period}.parquet` |
| 2 | GPT-4o-mini labeling | `quality/Label_Data.ipynb` | `processing/label_data/labeled_data_{period}.parquet` |
| 3 | Join labels + embeddings | `python quality/create_labeled_embeddings.py` | `processing/labeled_embeddings/embeddings_bge_{period}.parquet` |
| 4 | Train Ridge models | `python quality/train_ridge_models.py` | `processing/quality_models/{ridge,scaler}_{period}.pkl` |
| 5 | Classify all docs | `python quality/check_and_classify.py --reclassify` | `corpus/classified/classified_{year}.parquet` |
| 6 | Compute quality cutoffs | `python analysis/compute_quality_cutoffs.py` | `processing/quality_graphs/period_summary.csv` |
| 7 | Shard training data | `python sharding/prepare_base_data.py` | `periods/{period}/base_data[_suffix]/shard_{NNNNN}.parquet` |

All `python` commands are run from the repo root: `python src/base_training/quality/...`

All data paths are on `D:\hist_LLM\` (local SSD).

---

## Pipeline Diagram

```
cleaning/          embeddings/        quality/                      analysis/        sharding/
---------          -----------        --------                      ---------        ---------
Clean_Data.ipynb   run_embeddings  -> Sample_Data.ipynb          -> compute_     -> prepare_
  |                _fast.py            Label_Data.ipynb              quality_       base_
  v                  |                 create_labeled_emb.py         cutoffs.py     data.py
cleaning_masks/      v                 train_ridge_models.py
                  embeddings/          check_and_classify.py
                                         |
                                         v
                                      classified/
```

---

## Step Details

### Step 0: Clean raw text

Apply three heuristic filters to remove garbled OCR, tables/lists, and non-English text. Produces cleaning masks (indices of clean rows) used by Steps 1 and 5.

**Run:** `cleaning/Clean_Data.ipynb`

Three-stage filter (see `clean_data_helper.py`):
1. **Symbol ratio** -- rejects garbled OCR (non-word chars > 25% of words)
2. **Punctuation density** -- rejects tables/lists (<12% of lines end with `.?!`)
3. **Stopwords** -- rejects non-English text (fewer than 3 common stopwords)

**Output:** `D:\hist_LLM\corpus\cleaning_masks\{year}\{subset}_mask.parquet`

### Step 0.5: Generate BGE embeddings

Embed all documents using BGE-large-en-v1.5 (1024-dim). Requires GPU.

**Run:** `embeddings/run_embeddings_fast.py` on a GPU machine, or `embeddings/Embed_All_Data_Local.ipynb` for local processing.

**Output:** `D:\hist_LLM\corpus\embeddings\embeddings_{year}.parquet`

### Step 1: Sample documents for labeling

Sample ~10K clean documents per 25-year period (14 periods). Equal allocation across years within each period. Only samples from documents passing cleaning masks.

**Run:** `quality/Sample_Data.ipynb`

**Output:** `D:\hist_LLM\processing\sample_data\training_samples_{period}.parquet` -- schema: `[text, original_index, year]`

### Step 2: Label samples with GPT-4o-mini

Rate each sampled document on a 1-5 quality scale via OpenAI Batch API (50% cost savings). Scoring based on Information Density and Data Hygiene; final score = min of both.

**Run:** `quality/Label_Data.ipynb`

Three cells to run in sequence:
1. `submit_batches()` -- creates JSONL, uploads, submits batch jobs
2. `check_and_download()` -- poll periodically until all complete, auto-merges results
3. `sanity_check()` -- verify score distributions

**Output:** `D:\hist_LLM\processing\label_data\labeled_data_{period}.parquet` -- adds `score` column (1-5)

### Step 3: Create labeled embeddings

Join GPT quality labels with BGE embeddings to produce Ridge training data.

```bash
python src/base_training/quality/create_labeled_embeddings.py
python src/base_training/quality/create_labeled_embeddings.py --period 1901_1925  # single period
```

**Output:** `D:\hist_LLM\processing\labeled_embeddings\embeddings_bge_{period}.parquet` -- schema: `[original_index, labels, embedding]`

### Step 4: Train Ridge models

Train StandardScaler + RidgeCV pipeline for each 25-year period (14 models total).

```bash
python src/base_training/quality/train_ridge_models.py
```

**Output:** `D:\hist_LLM\processing\quality_models\{ridge,scaler}_{period}.pkl` (28 files + `training_summary.csv`)

### Step 5: Classify all documents

Apply Ridge models to predict quality scores for every clean embedded document. Only classifies rows passing cleaning masks from Step 0.

```bash
python src/base_training/quality/check_and_classify.py                # classify unclassified years
python src/base_training/quality/check_and_classify.py --reclassify   # overwrite all
python src/base_training/quality/check_and_classify.py --status       # check status only
```

**Output:** `D:\hist_LLM\corpus\classified\classified_{year}.parquet` -- schema: `[identifier, predicted_quality, is_clean, token_count, word_count]`

### Step 6: Compute quality cutoffs

Compute quality cutoff scores for each analysis period. For each period, sorts documents by predicted quality and finds the score where cumulative tokens reach the 20B threshold. Also generates cumulative token plots.

```bash
python src/base_training/analysis/compute_quality_cutoffs.py
```

**Output:**
- `D:\hist_LLM\processing\quality_graphs\cumulative_tokens_{period}.png` (6 plots)
- `D:\hist_LLM\processing\quality_graphs\period_summary.csv` (cutoff scores)

### Step 7: Prepare sharded training data

Filter documents above the cutoff score, load raw text, shuffle, and write ~250M character shards for nanochat.

```bash
# Default: use cutoff from period_summary.csv (20B-token threshold)
python src/base_training/sharding/prepare_base_data.py
python src/base_training/sharding/prepare_base_data.py --period 1678_1849  # single period
python src/base_training/sharding/prepare_base_data.py --dry-run           # stats only

# Experimental: manual cutoff or percentile-based filtering
python src/base_training/sharding/prepare_base_data.py --period 1900_1949 --cutoff 0 --output-suffix all       # all clean docs
python src/base_training/sharding/prepare_base_data.py --period 1900_1949 --top-pct 50 --output-suffix top50   # top 50% by quality
```

**Output:** `D:\hist_LLM\periods\{period}\base_data[_suffix]\shard_{NNNNN}.parquet`

---

## Period Naming

- **14 x 25-year periods** (for Ridge models): `1678_1700`, `1701_1725`, ..., `2001_2023`
- **6 analysis periods** (for training shards):

| Period | Era | Years |
|--------|-----|-------|
| `1678_1849` | Early Modern + Industrial Revolution | 1678-1849 |
| `1850_1899` | Victorian Era | 1850-1899 |
| `1900_1949` | World Wars Era | 1900-1949 |
| `1950_1999` | Cold War + Digital Age | 1950-1999 |
| `2000_2009` | Early Internet | 2000-2009 |
| `2010_2023` | Social Media Era | 2010-2023 |

---

## File Reference

```
base_training/
├── cleaning/
│   ├── Clean_Data.ipynb              # Step 0: Heuristic text cleaning
│   └── clean_data_helper.py          # Cleaning utilities (compute_clean_mask)
├── embeddings/
│   ├── run_embeddings_fast.py        # Step 0.5: BGE embedding (GPU)
│   ├── Embed_All_Data_Local.ipynb    # Full corpus embedding (local GPU)
│   ├── Embed_Data.ipynb              # Original embedding notebook
│   └── reprocess_embeddings_years.py # Fix corrupted/missing years
├── quality/
│   ├── Sample_Data.ipynb             # Step 1: Sample docs for labeling
│   ├── Label_Data.ipynb              # Step 2: GPT-4o-mini batch labeling
│   ├── create_labeled_embeddings.py  # Step 3: Join labels + embeddings
│   ├── train_ridge_models.py         # Step 4: Train Ridge regressors
│   ├── check_and_classify.py         # Step 5: Classify all clean docs
│   └── Classify_Data.ipynb           # Interactive classification exploration
├── analysis/
│   ├── compute_quality_cutoffs.py    # Step 6: Quality cutoffs + cumulative token graphs
│   └── Sanity_Check_Data.ipynb       # Data inspection/validation
└── sharding/
    └── prepare_base_data.py          # Step 7: Quality filtering + sharding
```

---

## Data Directory Structure

```
D:\hist_LLM\
├── corpus/                              # English historical corpus (1678-2023)
│   ├── raw/{year}/subset_*.parquet      # Raw text (identifier, text, token_count, word_count)
│   ├── cleaning_masks/{year}/*_mask.parquet  # Clean row indices per subset
│   ├── embeddings/embeddings_{year}.parquet  # BGE embeddings (1024-dim)
│   └── classified/classified_{year}.parquet  # Quality predictions (clean rows only)
│
├── processing/                          # Intermediate pipeline outputs
│   ├── sample_data/training_samples_{period}.parquet
│   ├── label_data/labeled_data_{period}.parquet
│   ├── labeled_embeddings/embeddings_bge_{period}.parquet
│   ├── quality_models/{ridge,scaler}_{period}.pkl
│   └── quality_graphs/{cumulative_tokens_*.png, period_summary.csv}
│
└── periods/{analysis_period}/           # Final sharded training data
    ├── base_data/shard_{NNNNN}.parquet          # Default (20B-token cutoff), ~250M chars each
    ├── base_data_all/shard_{NNNNN}.parquet      # --output-suffix all (no quality filter)
    └── base_data_top50/shard_{NNNNN}.parquet    # --output-suffix top50 (top 50% by quality)
```
