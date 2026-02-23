# hist-LLM

Historical LLM training pipeline. Builds a domain-adapted language model from the English historical corpus spanning 1678-2023 (~125M documents).

**Two-stage training:**
1. **Base training** (continued pretraining) — Quality-filtered raw text from the English historical corpus
2. **Post-training** (instruction tuning) — Synthetic QA/CoT data generated from the same corpus, filtered and split for fine-tuning via nanochat

---

## Repository Structure

```
src/
├── base_training/                  # Stage 1: Continued pretraining data pipeline
│   ├── cleaning/                   # Step 0: Heuristic text cleaning
│   │   ├── clean_data_helper.py         # Cleaning utilities (compute_clean_mask)
│   │   └── Clean_Data.ipynb             # Run cleaning on raw corpus
│   ├── embeddings/                 # Step 0.5: BGE embedding generation
│   │   ├── run_embeddings_fast.py       # Fast BGE embedding (GPU)
│   │   ├── reprocess_embeddings_years.py # Fix corrupted/missing years
│   │   ├── Embed_Data.ipynb             # Original embedding notebook
│   │   └── Embed_All_Data_Local.ipynb   # Full corpus embedding (local GPU)
│   ├── quality/                    # Steps 1–5: Full quality pipeline
│   │   ├── Sample_Data.ipynb            # Step 1: Sample docs for labeling
│   │   ├── Label_Data.ipynb             # Step 2: GPT-4o-mini quality labeling (Batch API)
│   │   ├── create_labeled_embeddings.py # Step 3: Join GPT labels with BGE embeddings
│   │   ├── train_ridge_models.py        # Step 4: Train Ridge regressors per 25-year period
│   │   ├── check_and_classify.py        # Step 5: Classify all clean embedded documents
│   │   └── Classify_Data.ipynb          # Interactive classification exploration
│   ├── analysis/                   # Step 6: Cumulative token analysis
│   │   ├── compute_quality_cutoffs.py   # Quality cutoff scores + cumulative token graphs
│   │   └── Sanity_Check_Data.ipynb      # Data inspection/validation
│   └── sharding/                   # Step 7: Final training data for nanochat
│       └── prepare_base_data.py         # Quality filtering + sharding
│
├── post_training/                  # Stage 2: Instruction tuning data pipeline
│   ├── config.py                   # Central config (periods, paths, API keys)
│   ├── utils.py                    # Shared utilities
│   ├── corpus/                     # Synthetic data generation
│   │   ├── export.py               # Export English corpus documents
│   │   ├── export_additional.py    # Export additional news data
│   │   ├── run_direct.py           # Generate QA pairs via GPT-4o-mini
│   │   ├── run.py                  # Original generation script
│   │   ├── run_cot.py              # Chain-of-thought generation
│   │   ├── run_curate.py           # Curation pipeline
│   │   ├── convert.py              # Convert individual JSONs to per-collection JSONL
│   │   ├── build_index.py          # Build document index
│   │   └── synth_config.yaml       # Synthesis configuration
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
    ├── base_data/shard_{NNNNN}.parquet          # Default (20B-token cutoff)
    ├── base_data_all/shard_{NNNNN}.parquet      # Experimental: no quality filter
    └── base_data_top50/shard_{NNNNN}.parquet    # Experimental: top 50% by quality
```

**Period naming:**
- **25-year periods** (for Ridge models): `1678_1700`, `1701_1725`, ..., `2001_2023` (14 total)
- **Analysis periods** (for training shards): `1678_1849`, `1850_1899`, `1900_1949`, `1950_1999`, `2000_2009`, `2010_2023` (6 total)

---

## Base Training Pipeline

Quality-filtered continued pretraining data from the English historical corpus (1678-2023, ~125M documents). See `base_training/README.md` for full details.

| Step | What | How |
|------|------|-----|
| 0 | Clean raw text | `cleaning/Clean_Data.ipynb` |
| 0.5 | Generate embeddings | `embeddings/run_embeddings_fast.py` (GPU) |
| 1 | Sample 10K docs/period | `quality/Sample_Data.ipynb` |
| 2 | GPT-4o-mini labeling | `quality/Label_Data.ipynb` |
| 3 | Join labels + embeddings | `python quality/create_labeled_embeddings.py` |
| 4 | Train Ridge models | `python quality/train_ridge_models.py` |
| 5 | Classify all docs | `python quality/check_and_classify.py --reclassify` |
| 6 | Compute quality cutoffs | `python analysis/compute_quality_cutoffs.py` |
| 7 | Shard training data | `python sharding/prepare_base_data.py` |

All `python` commands run from repo root: `python src/base_training/quality/...`

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

## Environment

- **Python 3.10+** with PyTorch, transformers, pandas, pyarrow, joblib, tqdm, matplotlib
- **GPU** required for BGE embedding generation
- **OpenAI API key** required for GPT-4o-mini labeling and QA generation (stored in `key.txt`)
- Data on local SSD (`D:\hist_LLM\`) for I/O performance
- Code synced via Dropbox (`C:\Users\danielyoon\Dropbox\hist_LLM\src\`)
