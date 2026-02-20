# Corpus Q&A Generation Pipeline

This pipeline generates synthetic Q&A training data from the historical document corpus using Meta's `synthetic-data-kit`.

## Overview

The pipeline transforms raw documents into instruction-following conversations:

```
Raw Documents (.parquet) → Sampled Docs (.txt) → Q&A Pairs → Curated → Nanochat Format
```

## Directory Structure

```
data/periods_data/{period}/posttraining_data/synthetic/
├── input/
│   ├── USPTO/                    # Documents per collection
│   │   ├── doc_00000.txt
│   │   └── ...
│   ├── Caselaw_Access_Project/
│   └── ...
├── parsed/
│   ├── USPTO/
│   └── ...
├── generated/
│   ├── USPTO/
│   │   ├── doc_00000_qa_pairs.json
│   │   ├── doc_00000_cot_examples.json
│   │   └── ...
│   └── ...
├── curated/
│   ├── USPTO/
│   └── ...
├── output/
│   └── corpus_qa_v2.jsonl        # Final output (all collections combined)
└── document_metadata.parquet     # Metadata index for sampling
```

## Prerequisites

1. **Metadata Index**: Must be built first (one-time per period)
2. **Raw Data**: Documents on D: drive (`D:\English\{year}\*.parquet`)
3. **Quality Scores**: Pre-computed in `D:\English_Classified\`
4. **Dependencies**: `synthetic-data-kit`, `pandas`, `pyarrow`

## Quick Start

### Step 1: Build Metadata Index (One-time)

```bash
python -m src.post_training.corpus.build_index --period 1950_1999
```

### Step 2: Run Pipeline

**Test run** (2 docs per collection):
```bash
python -m src.post_training.corpus.run --period 1950_1999 --max-per-collection 2
```

**Production run with parallelization** (recommended):
```bash
# Fastest: QA only, 16 workers (~9 hours for 27K docs)
python -m src.post_training.corpus.run --period 1950_1999 --workers 16 --skip-cot

# Full: QA + CoT, 16 workers (~18 hours for 27K docs)
python -m src.post_training.corpus.run --period 1950_1999 --workers 16
```

## Scripts

### 1. `build_index.py` - Build Metadata Index

Creates a parquet index joining document identifiers with quality scores and collection metadata.

```bash
python -m src.post_training.corpus.build_index --period 1950_1999
```

**Output**: `synthetic/document_metadata.parquet`

### 2. `export.py` - Export Documents

Exports sampled documents as `.txt` files organized by collection.

```bash
# Test (2 docs per collection)
python -m src.post_training.corpus.export --period 1950_1999 --max-per-collection 2

# Production (10K docs per collection, top 50% quality)
python -m src.post_training.corpus.export --period 1950_1999

# Custom settings
python -m src.post_training.corpus.export --period 1950_1999 \
    --max-per-collection 5000 \
    --quality-percentile 75
```

**Output**: `synthetic/input/{collection}/doc_*.txt`

### 3. `run.py` - Run Full Pipeline

Orchestrates the complete pipeline: export → ingest → QA → CoT → curate → convert

```bash
# Test run (2 docs per collection)
python -m src.post_training.corpus.run --period 1950_1999 --max-per-collection 2 --skip-curate

# Production run (10K docs per collection)
python -m src.post_training.corpus.run --period 1950_1999

# Single collection only
python -m src.post_training.corpus.run --period 1950_1999 --collection USPTO

# Skip stages (for re-running after failure)
python -m src.post_training.corpus.run --period 1950_1999 --skip-export --skip-ingest
```

**Options**:
- `--workers N`: Parallel workers (default: 1, recommend 8-16)
- `--batch-size N`: Docs per batch (default: 100)
- `--max-per-collection N`: Max docs per collection (default: 10000)
- `--quality-percentile N`: Quality floor (default: 50 = top 50%)
- `--collection NAME`: Process single collection
- `--skip-export`: Reuse existing exported docs
- `--skip-cot`: Skip CoT generation (2x faster, QA only)
- `--skip-curate`: Skip curation (default: True for Windows)
- `--num-qa N`: QA pairs per doc chunk (default: 3)
- `--num-cot N`: CoT examples per doc chunk (default: 2)

**Output**: `synthetic/output/corpus_qa_v2.jsonl`

**Progress Monitoring**: Progress is written to `synthetic/progress.json`:
```bash
# Watch progress in real-time
watch -n 5 cat data/periods_data/1950_1999/posttraining_data/synthetic/progress.json
```

### 4. `convert.py` - Convert to Nanochat Format

Converts all collection outputs to a single nanochat JSONL file:

```bash
python -m src.post_training.corpus.convert --period 1950_1999
```

## Typical Workflow

```bash
# 1. Build metadata index (one-time)
python -m src.post_training.corpus.build_index --period 1950_1999

# 2. Test with small sample (2 docs per collection)
python -m src.post_training.corpus.run --period 1950_1999 --max-per-collection 2 --skip-curate

# 3. Verify output
head data/periods_data/1950_1999/posttraining_data/synthetic/output/corpus_qa_v2.jsonl

# 4. Production: run one collection first
python -m src.post_training.corpus.run --period 1950_1999 --collection USPTO

# 5. Run all collections
python -m src.post_training.corpus.run --period 1950_1999
```

## Configuration

Pipeline settings are in `synth_config.yaml`:

- `qa_generation`: Prompts for Q&A pair generation
- `cot_generation`: Prompts for chain-of-thought generation
- `curate`: Quality thresholds for filtering

## Speed Optimization

The pipeline bottleneck is OpenAI API calls. Each document is chunked and each chunk requires API calls.

| Optimization | Speedup | Trade-off |
|--------------|---------|-----------|
| `--workers 16` | 4-8x vs single | No downside (API handles concurrency) |
| `--skip-cot` | 2x | No chain-of-thought examples |
| Larger chunk_size (config) | ~1.5x | Slightly coarser QA coverage |

**Recommended for fastest run:**
```bash
python -m src.post_training.corpus.run --period 1950_1999 --workers 16 --skip-cot
```

**Time estimates for 27K docs:**
| Workers | QA + CoT | QA only |
|---------|----------|---------|
| 1 | ~280 hours | ~140 hours |
| 8 | ~35 hours | ~18 hours |
| 16 | ~18 hours | ~9 hours |

## Cost Estimation

Actual collection sizes (1950_1999 period): **26,767 documents**
- Only USPTO and Caselaw Access Project have >10K docs
- Other 14 collections have fewer docs (sampled all available)

Cost at ~$5 per 1000 docs: **~$135**

## Known Issues

- **Windows curation bug**: The `curate` command fails on Windows due to emoji encoding issues in synthetic-data-kit. Use `--skip-curate` as a workaround.
- **Dropbox locking**: Some folders may be locked by Dropbox sync. Wait for sync to complete before deleting.

## Exploration

Use `explore_collections.ipynb` to explore the corpus:
- View document counts per collection
- See example text from each collection
- Analyze quality score distributions
