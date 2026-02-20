# hist-LLM

Historical LLM training pipeline. Builds a domain-adapted language model from historical document corpora spanning 1678-2023.

**Two-stage training:**
1. **Base training** (continued pretraining) — Quality-filtered raw text from English historical corpus + curated news archives
2. **Post-training** (instruction tuning) — Synthetic QA/CoT data generated from the same corpora, filtered and split for fine-tuning via [nanochat](https://github.com/your-org/nanochat)

---

## Repository Structure

```
src/
├── base_training/                  # Stage 1: Continued pretraining data pipeline
│   ├── cleaning/                   # Step 0: Heuristic text cleaning
│   │   ├── clean_data_helper.py         # Cleaning utilities (compute_clean_mask)
│   │   └── Clean_Data.ipynb             # Run cleaning on English + additional data
│   ├── embeddings/                 # Step 0.5: BGE embedding generation
│   │   ├── run_embeddings_fast.py       # Fast BGE embedding (English corpus)
│   │   ├── reprocess_embeddings_years.py
│   │   ├── Embed_Data.ipynb             # Original embedding notebook
│   │   ├── Embed_All_Data_Local.ipynb   # Full corpus embedding (local GPU)
│   │   ├── Prepare_Additional_For_Embedding.ipynb   # Prep additional data for embedding
│   │   └── Embed_Additional_Data.ipynb              # Embed additional data with BGE
│   ├── quality/                    # Steps 1–5: Full quality pipeline
│   │   ├── Sample_Data.ipynb            # Step 1: Sample docs for labeling
│   │   ├── Label_Data.ipynb             # Step 2: GPT-4o-mini quality labeling (Batch API)
│   │   ├── create_labeled_embeddings.py # Step 3: Join GPT labels with BGE embeddings
│   │   ├── train_ridge_models.py        # Step 4: Train Ridge regressors per 25-year period
│   │   ├── check_and_classify.py        # Step 5: Apply Ridge to classify all documents
│   │   └── Classify_Data.ipynb          # Interactive classification exploration
│   ├── analysis/                   # Step 6: Cumulative token analysis
│   │   ├── plot_cumulative_tokens.py    # Cumulative token graphs + cutoff scores
│   │   └── Sanity_Check_Data.ipynb      # Data inspection/validation
│   └── sharding/                   # Step 7: Final training data for nanochat
│       └── prepare_training_data.py     # Quality filtering + sharding (English + additional)
│
├── post_training/                  # Stage 2: Instruction tuning data pipeline
│   ├── config.py                   # Central config (periods, paths, API keys)
│   ├── utils.py                    # Shared utilities
│   ├── corpus/                     # Synthetic data generation
│   │   ├── export.py               # Export English corpus documents
│   │   ├── export_additional.py    # Export additional news data
│   │   ├── run_direct.py           # Generate QA pairs via GPT-4o-mini
│   │   ├── convert.py              # Convert individual JSONs to per-collection JSONL
│   │   ├── run_cot.py              # Chain-of-thought generation
│   │   ├── run_curate.py           # Curation pipeline
│   │   └── build_index.py          # Build document index
│   ├── instruct/                   # Instruction tuning pipeline
│   │   ├── download.py             # Download external instruct datasets
│   │   ├── filter.py               # LAB filtering
│   │   ├── score.py                # Quality scoring
│   │   ├── split.py                # Train/test splitting
│   │   └── analyze.py              # Dataset analysis
│   ├── eval/                       # Evaluation
│   │   ├── generate_lab_questions.py
│   │   └── shuffle_lab_answers.py
│   └── identity/
│       └── generate.py             # Identity/system prompt generation
│
└── notebooks/                      # Exploratory notebooks
    ├── analyze_corpus_qa.ipynb
    ├── chat_with_model.ipynb
    ├── plot_eval_results.ipynb
    ├── sharding_adjustment.ipynb
    └── verify_eval_logits.ipynb
```

---

## Data Directory Structure (D: drive)

All data lives on `D:\hist_LLM\` (local SSD, not synced to Dropbox).

```
D:\hist_LLM\
├── corpus/                         # English historical corpus (1678-2023)
│   ├── raw/                        # Raw text by year
│   │   └── {year}/subset_*.parquet     # Columns: identifier, text, token_count, word_count, ...
│   ├── cleaning_masks/             # Heuristic cleaning masks by year
│   │   └── {year}/{subset}_mask.parquet    # Columns: original_index (clean row indices)
│   ├── embeddings/                 # BGE embeddings by year
│   │   └── embeddings_{year}.parquet   # Columns: original_index, embedding (1024-dim)
│   └── classified/                 # Quality predictions by year
│       └── classified_{year}.parquet   # Columns: identifier, predicted_quality, is_clean, ...
│
├── additional_data/                # Curated news collections
│   ├── raw/
│   │   ├── news_archives/
│   │   │   ├── NYT_filtered_500char/   nyt_{year}.parquet          (1851-2016)
│   │   │   ├── Economist/              economist_{year}-*.parquet  (1843-2014)
│   │   │   └── FT/                     {year}.parquet              (1888-2006)
│   │   └── newswire/                   {year}_data_clean.json      (1878-1977)
│   ├── cleaning_masks/             # Heuristic cleaning masks by collection
│   │   └── {collection}/{file}_mask.parquet
│   ├── embeddings/
│   │   └── {collection}/embeddings_{year}.parquet
│   └── classified/
│       └── {collection}/classified_{year}.parquet
│
├── processing/                     # Intermediate pipeline outputs
│   ├── sample_data/                # Sampled docs for labeling
│   │   └── training_samples_{25yr_period}.parquet
│   ├── label_data/                 # GPT quality labels
│   │   └── labeled_data_{25yr_period}.parquet
│   ├── labeled_embeddings/         # Labels joined with embeddings
│   │   └── embeddings_bge_{25yr_period}.parquet
│   ├── quality_models/             # Trained Ridge models (14 periods)
│   │   ├── ridge_{25yr_period}.pkl
│   │   └── scaler_{25yr_period}.pkl
│   ├── quality_graphs/             # Cumulative token plots
│   │   ├── cumulative_tokens_{analysis_period}.png
│   │   └── period_summary.csv      # Cutoff scores per analysis period
│   └── staging/                    # Temp files during shard creation
│
└── periods/                        # Final sharded training data for nanochat
    └── {analysis_period}/
        └── base_data/
            └── shard_{NNNNN}.parquet   # ~250M chars each, zstd compressed
```

**Period naming:**
- **25-year periods** (for Ridge models): `1678_1700`, `1701_1725`, ..., `2001_2023` (14 total)
- **Analysis periods** (for training shards): `1678_1849`, `1850_1899`, `1900_1949`, `1950_1999`, `2000_2009`, `2010_2023` (6 total)

---

## Base Training Pipeline

The pipeline produces quality-filtered, sharded text data from ~146M documents across both the English historical corpus and 4 news collections.

```
cleaning/        embeddings/       quality/                    analysis/       sharding/
─────────        ───────────       ────────                    ─────────       ─────────
Clean_Data  ──►  run_embeddings  ──►  Sample_Data.ipynb   ──►  plot_cumul.  ──►  prepare_
 .ipynb          _fast.py              Label_Data.ipynb         tokens.py        training
                                       create_labeled_emb.py                     _data.py
                                       train_ridge_models.py
                                       check_and_classify.py
```

### Step-by-step instructions

#### Step 0: Clean raw text (heuristic filtering)

Apply three heuristic filters (symbol ratio, punctuation density, stopwords) to remove
garbled OCR, tables/lists, and non-English text. Produces cleaning masks used by
downstream steps.

**Notebook:** `base_training/cleaning/Clean_Data.ipynb`
**Helper:** `base_training/cleaning/clean_data_helper.py`

**Output:**
- English: `D:\hist_LLM\corpus\cleaning_masks\{year}\{subset}_mask.parquet`
- Additional: `D:\hist_LLM\additional_data\cleaning_masks\{collection}\{file}_mask.parquet`

#### Step 1: Sample documents for labeling

Sample ~10K documents per 25-year period from both English corpus and additional news data, proportionally by document count. Samples only from clean documents (uses masks from Step 0).

**Notebook:** `base_training/quality/Sample_Data.ipynb`

**Output:** `D:\hist_LLM\processing\sample_data\training_samples_{period}.parquet`

#### Step 2: Label samples with GPT-4o-mini

Rate each sampled document on a 1-5 quality scale using GPT-4o-mini via the OpenAI Batch API (50% cost savings).

**Notebook:** `base_training/quality/Label_Data.ipynb`

**Output:** `D:\hist_LLM\processing\label_data\labeled_data_{period}.parquet`

#### Step 3: Create labeled embeddings

Join GPT quality labels with BGE embeddings to produce training data for Ridge models.

```bash
python src/base_training/quality/create_labeled_embeddings.py
python src/base_training/quality/create_labeled_embeddings.py --period 1901_1925  # single period
```

**Output:** `D:\hist_LLM\processing\labeled_embeddings\embeddings_bge_{period}.parquet`

#### Step 4: Train Ridge models

Train Ridge regression models (one per 25-year period) to predict quality scores from embeddings.

```bash
python src/base_training/quality/train_ridge_models.py
```

**Output:** `D:\hist_LLM\processing\quality_models\{ridge,scaler}_{period}.pkl` (28 files)

#### Step 5: Classify all documents

Apply Ridge models to predict quality scores for every embedded document.

```bash
# Classify English corpus
python src/base_training/quality/check_and_classify.py
python src/base_training/quality/check_and_classify.py --reclassify  # overwrite existing

# Classify additional news data
python src/base_training/quality/check_and_classify.py --additional
python src/base_training/quality/check_and_classify.py --additional --collection nyt  # single collection

# Check status only
python src/base_training/quality/check_and_classify.py --status
```

**Output:**
- English: `D:\hist_LLM\corpus\classified\classified_{year}.parquet`
- Additional: `D:\hist_LLM\additional_data\classified\{collection}\classified_{year}.parquet`

#### Step 6: Cumulative token analysis

Determine quality cutoff scores by plotting cumulative tokens vs quality for each analysis period. The cutoff is set where cumulative tokens reach the 20B threshold.

```bash
python src/base_training/analysis/plot_cumulative_tokens.py
```

**Output:**
- `D:\hist_LLM\processing\quality_graphs\cumulative_tokens_{period}.png`
- `D:\hist_LLM\processing\quality_graphs\period_summary.csv` (cutoff scores)

#### Step 7: Prepare sharded training data

Filter documents above the cutoff score, load raw text from both English and additional sources, shuffle, and write ~250M character shards.

```bash
python src/base_training/sharding/prepare_training_data.py
python src/base_training/sharding/prepare_training_data.py --period 1678_1849  # single period
python src/base_training/sharding/prepare_training_data.py --dry-run           # stats only
```

**Output:** `D:\hist_LLM\periods\{period}\base_data\shard_{NNNNN}.parquet`

---

## Post-Training Pipeline

Generates synthetic instruction-tuning data from the same historical corpora, then filters and splits for fine-tuning.

```
export_additional.py ──► run_direct.py ──► convert.py ──► filter.py ──► split.py ──► nanochat
(parquet input)          (GPT-4o-mini QA)   (per-collection   (LAB)      (train/test)
                                             JSONL)
```

### Data flow

1. **Export** — Extract documents from classified corpus (`export.py`, `export_additional.py`)
2. **Generate** — Create QA pairs via GPT-4o-mini (`run_direct.py`)
3. **Convert** — Merge individual JSON outputs into per-collection JSONL files (`convert.py`)
4. **Filter** — Apply LAB quality filtering (`filter.py`)
5. **Split** — 95/5 train/test split (`split.py`)

### Output structure

```
{period}/final/
├── filtered/   # Source files: *_filtered.jsonl (external) + hist_*.jsonl (corpus)
├── removed/    # LAB-removed external datasets
├── train/      # 95% train splits (40 files)
└── test/       # 5% test splits (40 files)
```

Post-training config is centralized in `src/post_training/config.py`.

---

## Additional Data Collections

| Collection | Source | Years | Documents | Text Column |
|-----------|--------|-------|-----------|-------------|
| NYT | New York Times | 1851-2016 | ~2.8M | `combined_text` |
| Economist | The Economist | 1843-2014 | ~0.9M | `ocr_text` |
| FT | Financial Times | 1888-2006 | ~14.4M | `text_cleaned` |
| Newswire | Wire services | 1878-1977 | ~2.7M | `cleaned_article` |

**Known issue:** Newswire 1957 JSON is corrupted and is skipped automatically.

---

## Environment

- **Python 3.10+** with PyTorch, transformers, pandas, pyarrow, joblib, tqdm, matplotlib
- **GPU** required for BGE embedding generation
- **OpenAI API key** required for GPT-4o-mini labeling and QA generation (stored in `key.txt`)
- Data on local SSD (`D:\hist_LLM\`) for I/O performance
- Code synced via Dropbox (`C:\Users\danielyoon\Dropbox\hist_LLM\src\`)
