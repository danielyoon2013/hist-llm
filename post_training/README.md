# Post-Training Pipeline

Synthetic data generation, quality filtering, and assembly for historical LLM post-training (6 periods, 1678-2023).

## Quick Start: Run in Order

| Step | What | How | Output |
|------|------|-----|--------|
| 0a | Export main corpus | `python -m src.post_training.corpus.export --period {P}` | `synthetic/input/{collection}/*.txt` |
| 0b | Export news archives | `python -m src.post_training.corpus.export_additional --period {P}` | `synthetic/input/{dataset}.parquet` |
| 1 | Generate synthetic data | `python -m src.post_training.generate --period {P} --generators A B C D E F G H` | `synthetic/by_generator/gen_*_{fmt}.jsonl` |
| 2 | Validate + deduplicate | `python -m src.post_training.process --period {P}` | `quality/validated/`, `quality/deduped/` |
| 3 | LAB temporal filter | `python -m src.post_training.instruct.filter --period {P} --submit --input-dir {deduped_dir}` | `final/filtered/` |
| 3b | Check filter status | `python -m src.post_training.instruct.filter --period {P} --check --input-dir {deduped_dir}` | — |
| 3c | Process filter results | `python -m src.post_training.instruct.filter --period {P} --process --input-dir {deduped_dir}` | `final/filtered/` |
| 4 | Assemble train/test | `python -m src.post_training.assemble --period {P}` | `final/train/`, `final/test/` |
| 5 | Train/test split (ext.) | `python -m src.post_training.instruct.split --period {P}` | `final/train/`, `final/test/` |

All `python` commands are run from the repo root: `python -m src.post_training.generate ...`

All data paths are on `D:\hist_LLM\` (local SSD). Period paths defined in `config.py`.

---

## Pipeline Diagram

```
export.py              generate.py          process.py             filter.py            assemble.py
export_additional.py   -----------          ----------             ---------            -----------
--------------------   Generators A-H  -->  Validate + Dedup  -->  LAB Filter    -->   Merge + Split
Export corpus docs       |                    |                    (Batch API)           |
  |                      v                    v                      |                   v
  v                    by_generator/        quality/                 v                 final/train/
synthetic/input/       gen_*_{fmt}.jsonl    validated/             final/              final/test/
{collection}/*.txt     (18 format files)    deduped/               filtered/
{dataset}.parquet                           stats.json
```

---

## The 8 Generators (19 Format Cells)

Each generator produces multiple format variants from a single API call, aligned to external benchmark native formats. This ensures evaluation measures temporal knowledge, not format confusion.

| ID | Name | Input | Supported Formats | Benchmark Alignment | Items/Chunk |
|----|------|-------|-------------------|---------------------|-------------|
| A | Factual QA | corpus chunks | `mc4`, `open` | MMLU, ARC | 3 |
| B | Chain-of-Thought | corpus chunks | `mc4`, `open`, `cot` | ARC, GSM8K | 2 |
| C | Reading Comprehension | corpus passages | `mc4`, `mc4_passage`, `mc2_passage` | HellaSwag, RACE, BoolQ | 3 |
| D | Temporal Reasoning | period metadata | `mc4`, `open` | LAB Eval | 5 |
| E | Quantitative | corpus chunks | `open`, `cot` | GSM8K | 2 |
| F | Sentence Completion | corpus sentences | `mc4`, `mc2` | HellaSwag, WinoGrande | 3 |
| G | Instruction Following | corpus passages | `open`, `mc4_passage` | RACE | 2 |
| H | Anti-Hallucination | period metadata | `mc4`, `open` | LAB Eval | 5 |

### Format Key

| Format | Description | Eval Benchmarks Using It |
|--------|-------------|--------------------------|
| `mc4` | 4-choice MC via `render_mc()` | MMLU, ARC, HellaSwag, LAB Eval |
| `mc2` | 2-choice MC via `render_mc()` | PIQA, WinoGrande |
| `mc4_passage` | Passage prefix + 4-choice MC (RACE-style) | RACE |
| `mc2_passage` | Passage prefix + 2-choice MC (BoolQ-style) | BoolQ |
| `open` | Open-ended question + answer | GSM8K (generative) |
| `cot` | Question + `<think>` reasoning + answer | GSM8K (with reasoning) |

**Corpus-based** (A, B, C, E, F, G): Read text from `synthetic/input/` (parquet or txt), chunk at 6000 chars / 300 overlap, call GPT-4o-mini per chunk.

**Metadata-based** (D, H): Generate from period year range alone (no corpus needed). D creates temporal ordering questions; H creates refusal examples for events after the period's end year.

**MC format** matches nanochat's `render_mc()` exactly (choice text BEFORE letter for better token binding in small models):
```
Multiple Choice question: What caused...?
- The economy grew=A
- War broke out=B
- Population declined=C
- Trade expanded=D

Respond only with the letter of the correct answer.
```

---

## Step Details

### Step 0: Export Corpus Documents

Before generators can run, source documents must be exported into `synthetic/input/`. This is a **one-time setup per period** — once exported, generators can be re-run without re-exporting.

Two scripts handle two types of sources:

**`corpus/export.py`** — Exports from the main historical corpus (parquet shards in `D:\hist_LLM\corpus\raw\`).

- Reads the metadata index (`document_metadata.parquet`) built by `build_index.py`
- Applies a quality floor (top 50% by `predicted_quality` score from the embedding classifier)
- Samples up to 10K documents per collection (deterministic seed=42)
- Retrieves full text from raw parquet shards by identifier
- Writes individual `.txt` files: `synthetic/input/{collection}/doc_00000.txt`
- Covers ~26 collections (Caselaw, English-PD, US-PD-Books, USPTO, Wikisource, etc.)

**`corpus/export_additional.py`** — Exports from 4 curated news archives (`D:\hist_LLM\additional_data\raw\`).

- Loads articles for the period's year range from source-specific formats (parquet/JSON)
- Filters by minimum text length (≥200 chars)
- Samples up to 10K per dataset
- Writes as single parquet files: `synthetic/input/{dataset}.parquet` (columns: `doc_name`, `text`)
- Sources: NYT (filtered abstracts), Economist (OCR text), FT (cleaned), Newswire (cleaned)

**Prerequisite:** `build_index.py` must have been run first to create `document_metadata.parquet`. This joins quality scores from the embedding classifier with collection labels from the raw corpus.

```bash
# Build metadata index (if not already done)
python -m src.post_training.corpus.build_index --period 1900_1949

# Export main corpus (writes ~110K .txt files)
python -m src.post_training.corpus.export --period 1900_1949

# Export news archives (writes 4 .parquet files, ~40K docs)
python -m src.post_training.corpus.export_additional --period 1900_1949

# Custom settings
python -m src.post_training.corpus.export --period 1900_1949 --max-per-collection 5000 --quality-percentile 75
python -m src.post_training.corpus.export_additional --period 1900_1949 --dataset economist ft --max-per-collection 500
```

**Output:** `D:\hist_LLM\periods\{period}\posttraining_data\synthetic\input\` populated with `.txt` directories and `.parquet` files.

Generators read from `synthetic/input/` via `BaseGenerator._load_documents()`, which handles both formats transparently.

### Step 1: Generate Synthetic Data

Run one or more generators for a period. Each generator writes one JSONL file **per supported format** to `synthetic/by_generator/`.

```bash
# Full run (all 8 generators, produces 19 format files)
python -m src.post_training.generate --period 1900_1949 --generators A B C D E F G H

# Test run (3 documents only, < $0.01)
python -m src.post_training.generate --period 1900_1949 --generators A B C D E F G H --max-docs 3

# Specific generators and collections
python -m src.post_training.generate --period 1900_1949 --generators A B --collections economist ft

# Metadata-based only (no corpus needed)
python -m src.post_training.generate --period 1900_1949 --generators D H

# Adjust concurrency
python -m src.post_training.generate --period 1900_1949 --generators A --max-workers 80
```

**Output:** `D:\hist_LLM\periods\{period}\posttraining_data\synthetic\by_generator\gen_{name}_{fmt}.jsonl`

All output is nanochat CustomJSON format: each line is `[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]`

### Step 2: Validate + Deduplicate

Two-step local pipeline (no API calls):

1. **Validate** — format check (alternating user/assistant, >=2 messages) + content check (min 10 chars, max 10K chars; MC single-letter responses exempt)
2. **Dedup** — 3-level deduplication:
   - Level 1: Exact hash (SHA-256 of normalized user message)
   - Level 2: Near-duplicate (MinHash + LSH, threshold 0.8, requires `pip install datasketch`)
   - Level 3: Cross-generator (remove duplicate questions across generators, priority A > B > C > ..., per-format files handled by prefix matching)

```bash
# Full pipeline (validate + dedup)
python -m src.post_training.process --period 1900_1949

# Individual steps
python -m src.post_training.process --period 1900_1949 --step validate
python -m src.post_training.process --period 1900_1949 --step dedup
```

**Output:**
- `D:\hist_LLM\periods\{period}\posttraining_data\quality\validated\gen_*_{fmt}.jsonl`
- `D:\hist_LLM\periods\{period}\posttraining_data\quality\deduped\gen_*_{fmt}.jsonl`
- `D:\hist_LLM\periods\{period}\posttraining_data\quality\stats.json`

### Step 3: LAB Temporal Filter

Filter out conversations requiring knowledge after the period's end year. Uses OpenAI Batch API (3-step async workflow: submit, check, process).

```bash
# Point filter at deduped synthetic data
DEDUP_DIR="D:/hist_LLM/periods/1900_1949/posttraining_data/quality/deduped"

# Submit batch job (~24h turnaround)
python -m src.post_training.instruct.filter --period 1900_1949 --submit --input-dir "$DEDUP_DIR"

# Check status
python -m src.post_training.instruct.filter --period 1900_1949 --check --input-dir "$DEDUP_DIR"

# Download results and filter
python -m src.post_training.instruct.filter --period 1900_1949 --process --input-dir "$DEDUP_DIR"

# Filter external instruct datasets (SmolTalk, MMLU, ARC)
python -m src.post_training.instruct.filter --period 1900_1949 --submit
python -m src.post_training.instruct.filter --period 1900_1949 --process

# Filter corpus Q&A (legacy run_direct.py output)
python -m src.post_training.instruct.filter --period 1900_1949 --submit --corpus
python -m src.post_training.instruct.filter --period 1900_1949 --process --corpus
```

**Output:** `D:\hist_LLM\periods\{period}\posttraining_data\final\filtered\*_filtered.jsonl`

### Step 4: Assemble Train/Test

Merge all generator outputs into a single shuffled dataset, then split 95/5 train/test.

```bash
python -m src.post_training.assemble --period 1900_1949

# Dry run (show plan without writing)
python -m src.post_training.assemble --period 1900_1949 --dry-run

# Override input source
python -m src.post_training.assemble --period 1900_1949 --source deduped   # skip LAB filter
python -m src.post_training.assemble --period 1900_1949 --source validated  # skip dedup too
python -m src.post_training.assemble --period 1900_1949 --source raw        # use raw generator output
```

Auto-detection priority: `final/filtered` > `quality/deduped` > `quality/validated` > `synthetic/by_generator`

**Output:**
- `D:\hist_LLM\periods\{period}\posttraining_data\final\train\hist_synthetic_train.jsonl`
- `D:\hist_LLM\periods\{period}\posttraining_data\final\test\hist_synthetic_test.jsonl`

### Step 5: Split External Datasets

Split LAB-filtered external instruct datasets (SmolTalk, MMLU, etc.) into train/test. This is for the non-synthetic portion of the curriculum.

```bash
python -m src.post_training.instruct.split --period 1900_1949
python -m src.post_training.instruct.split --period 1900_1949 --dry-run
```

**Output:** `final/train/{name}_train.jsonl` and `final/test/{name}_test.jsonl` for each dataset.

---

## API Cost Estimate

All generators use `gpt-4o-mini` ($0.15/1M input, $0.60/1M output).

| Item | Cost per Period |
|------|----------------|
| Generation (8 generators, 19 formats) | ~$4.00-5.50 |
| LAB filtering (Batch API) | ~$2-3 |
| **Total per period** | **~$6-8.50** |
| **All 6 periods** | **~$36-51** |

Multi-format rendering adds zero extra API cost (same API call produces all format variants).

For testing, use `--max-docs 3` to limit to ~6 chunks and ~48 API calls (< $0.01).

---

## 6 Periods

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
post_training/
├── config.py                          # Central config: periods, paths, API settings
├── utils.py                           # Shared: OpenAI client, JSONL I/O, Batch API helpers
├── generate.py                        # CLI: run synthetic data generators
├── process.py                         # CLI: validate + deduplicate pipeline
├── assemble.py                        # CLI: merge generators into train/test
│
├── generators/                        # Synthetic data generators (A-H)
│   ├── __init__.py                    # Registry: {"A": GenAFactual, ...}
│   ├── base.py                        # BaseGenerator, render_mc, make_mc_choices,
│   │                                  #   truncate_passage, chunk_text, call_api
│   ├── prompts.py                     # All 8 prompt templates (with distractor requests)
│   ├── gen_a_factual.py               # A: Factual QA (mc4, open)
│   ├── gen_b_cot.py                   # B: Chain-of-Thought (mc4, open, cot)
│   ├── gen_c_comprehension.py         # C: Reading Comprehension (mc4, mc4_passage, mc2_passage)
│   ├── gen_d_temporal.py              # D: Temporal Reasoning (mc4, open; no corpus)
│   ├── gen_e_quantitative.py          # E: Quantitative / Math (open, cot)
│   ├── gen_f_completion.py            # F: Sentence Completion (mc4, mc2)
│   ├── gen_g_instruct.py              # G: Instruction Following (open, mc4_passage)
│   └── gen_h_antihalluc.py            # H: Anti-Hallucination (mc4, open; no corpus)
│
├── quality/                           # Quality pipeline
│   ├── __init__.py
│   ├── validate.py                    # Format + content validation
│   ├── dedup.py                       # 3-level dedup (exact, MinHash, cross-gen)
│   └── pipeline.py                    # Orchestrator: validate -> dedup
│
├── instruct/                          # External dataset processing
│   ├── filter.py                      # LAB temporal filtering (Batch API)
│   └── split.py                       # Train/test split (95/5)
│
├── corpus/                            # Corpus export + legacy generation
│   ├── build_index.py                 # Build metadata index (quality + collection labels)
│   ├── export.py                      # Export main corpus → synthetic/input/{collection}/*.txt
│   ├── export_additional.py           # Export news archives → synthetic/input/{dataset}.parquet
│   ├── run_direct.py                  # Legacy QA/CoT generation (superseded by generators/)
│   ├── convert.py                     # Format converter
│   └── synth_config.yaml              # Legacy config
│
└── eval/                              # Evaluation
    ├── generate_lab_questions.py       # Generate LAB eval MC questions
    └── shuffle_lab_answers.py          # Randomize answer positions
```

---

## Data Directory Structure

```
D:\hist_LLM\periods\{period}\posttraining_data\
│
├── synthetic/
│   ├── input/                         # Source corpus for generation (Step 0)
│   │   ├── {dataset}.parquet          # News archives: economist, ft, nyt_filtered, newswire
│   │   │                              #   (columns: doc_name, text; from export_additional.py)
│   │   └── {collection}/              # Main corpus collections: Caselaw, English-PD, etc.
│   │       └── doc_00000.txt          #   (individual txt files; from export.py)
│   ├── generated/                     # Legacy run_direct.py output
│   │   └── {collection}/
│   │       └── {collection}_qa_cot.jsonl
│   ├── by_generator/                  # Per-generator, per-format raw output (Step 1)
│   │   ├── gen_a_factual_mc4.jsonl
│   │   ├── gen_a_factual_open.jsonl
│   │   ├── gen_b_cot_mc4.jsonl
│   │   ├── gen_b_cot_open.jsonl
│   │   ├── gen_b_cot_cot.jsonl
│   │   ├── gen_c_comprehension_mc4.jsonl
│   │   ├── gen_c_comprehension_mc4_passage.jsonl
│   │   ├── gen_c_comprehension_mc2_passage.jsonl
│   │   ├── gen_d_temporal_mc4.jsonl
│   │   ├── gen_d_temporal_open.jsonl
│   │   ├── gen_e_quantitative_open.jsonl
│   │   ├── gen_e_quantitative_cot.jsonl
│   │   ├── gen_f_completion_mc4.jsonl
│   │   ├── gen_f_completion_mc2.jsonl
│   │   ├── gen_g_instruct_open.jsonl
│   │   ├── gen_g_instruct_mc4_passage.jsonl
│   │   ├── gen_h_antihalluc_mc4.jsonl
│   │   └── gen_h_antihalluc_open.jsonl
│   └── document_metadata.parquet      # Document provenance index
│
├── quality/                           # Quality pipeline outputs (Step 2)
│   ├── validated/                     # After format + content validation
│   │   └── gen_*_{fmt}.jsonl
│   ├── deduped/                       # After 3-level deduplication
│   │   └── gen_*_{fmt}.jsonl
│   └── stats.json                     # Pipeline statistics
│
├── final/
│   ├── filtered/                      # After LAB temporal filter (Step 3)
│   │   ├── *_filtered.jsonl           # External datasets
│   │   └── hist_*.jsonl               # Historical corpus
│   ├── removed/                       # Removed items for inspection
│   ├── train/                         # Final train splits (Step 4-5)
│   │   ├── hist_synthetic_train.jsonl # Merged synthetic generators
│   │   ├── hist_{collection}_train.jsonl
│   │   ├── smoltalk_train.jsonl
│   │   ├── mmlu_train.jsonl
│   │   └── ...                        # 36 files total for nanochat
│   └── test/                          # Final test splits
│       └── *.jsonl
│
├── eval/
│   └── lab_questions.jsonl            # LAB evaluation MC questions
│
├── LAB_scores/                        # LAB filter scores per dataset
│   └── {dataset}_scores.jsonl
│
└── batch_temp/                        # Batch API temporary files
    ├── *_requests.jsonl
    ├── *_results.jsonl
    └── *_batch_ids.txt
```

---

## nanochat Integration

All output files are in nanochat's **CustomJSON** format (loaded by `nanochat/tasks/customjson.py`):

```jsonl
[{"role":"user","content":"What caused...?"},{"role":"assistant","content":"The main cause..."}]
[{"role":"user","content":"Multiple Choice question: ...\n- Option A=A\n- Option B=B\n..."},{"role":"assistant","content":"B"}]
[{"role":"user","content":"Explain..."},{"role":"assistant","content":"<think>\nStep 1...\n</think>\nThe answer is..."}]
```

The training script `nanochat/runs/speedrun_hist_llm.sh` reads 36 JSONL files from `final/train/` in a `TaskSequence` curriculum (mid-training + SFT phases).

---

## Dependencies

- **Required:** `openai`, `pandas`, `pyarrow` (already installed)
- **Optional:** `datasketch` (for MinHash near-duplicate detection in Step 2)

```bash
pip install datasketch  # ~50KB, pure Python
```
