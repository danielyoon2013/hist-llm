# Post-Training Pipeline

Synthetic data generation, quality filtering, and assembly for historical LLM post-training (6 periods, 1678-2023).

## Quick Start: Run in Order

| Step | What | How | Output |
|------|------|-----|--------|
| 1 | Generate synthetic data | `python -m src.post_training.generate --period {P} --generators A B C D E F G H` | `synthetic/by_generator/gen_*.jsonl` |
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
generate.py          process.py             filter.py            assemble.py
-----------          ----------             ---------            -----------
Generators A-H  -->  Validate + Dedup  -->  LAB Filter    -->   Merge + Split
  |                    |                    (Batch API)           |
  v                    v                      |                   v
by_generator/        quality/                 v                 final/train/
gen_*.jsonl          validated/             final/              final/test/
                     deduped/               filtered/
                     stats.json
```

---

## The 8 Generators

| ID | Name | Input | Format | Items/Chunk |
|----|------|-------|--------|-------------|
| A | Factual QA | corpus chunks | Open-ended Q&A | 3 |
| B | Chain-of-Thought | corpus chunks | `<think>` reasoning + answer | 2 |
| C | Reading Comprehension | corpus passages | MC (A/B/C/D) | 3 |
| D | Temporal Reasoning | period metadata | MC (A/B/C/D) | 5 |
| E | Quantitative | corpus chunks | `<think>` math reasoning | 2 |
| F | Sentence Completion | corpus sentences | MC HellaSwag-style | 3 |
| G | Instruction Following | corpus passages | Instruction + response | 2 |
| H | Anti-Hallucination | period metadata | Refusal for post-period Q | 5 |

**Corpus-based** (A, B, C, E, F, G): Read text from `synthetic/input/` (parquet or txt), chunk at 6000 chars / 300 overlap, call GPT-4o-mini per chunk.

**Metadata-based** (D, H): Generate from period year range alone (no corpus needed). D creates temporal ordering questions; H creates refusal examples for events after the period's end year.

**MC format** (C, D, F): Uses nanochat's `render_mc()` format:
```
Question text?
- choice=A: ...
- choice=B: ...
- choice=C: ...
- choice=D: ...

Respond only with the letter of the correct answer.
```

---

## Step Details

### Step 1: Generate Synthetic Data

Run one or more generators for a period. Each generator writes a separate JSONL file to `synthetic/by_generator/`.

```bash
# Full run (all 8 generators)
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

**Output:** `D:\hist_LLM\periods\{period}\posttraining_data\synthetic\by_generator\gen_*.jsonl`

All output is nanochat CustomJSON format: each line is `[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]`

### Step 2: Validate + Deduplicate

Two-step local pipeline (no API calls):

1. **Validate** — format check (alternating user/assistant, >=2 messages) + content check (min 10 chars, max 10K chars; MC single-letter responses exempt)
2. **Dedup** — 3-level deduplication:
   - Level 1: Exact hash (SHA-256 of normalized user message)
   - Level 2: Near-duplicate (MinHash + LSH, threshold 0.8, requires `pip install datasketch`)
   - Level 3: Cross-generator (remove duplicate questions across generators, priority A > B > C > ...)

```bash
# Full pipeline (validate + dedup)
python -m src.post_training.process --period 1900_1949

# Individual steps
python -m src.post_training.process --period 1900_1949 --step validate
python -m src.post_training.process --period 1900_1949 --step dedup
```

**Output:**
- `D:\hist_LLM\periods\{period}\posttraining_data\quality\validated\gen_*.jsonl`
- `D:\hist_LLM\periods\{period}\posttraining_data\quality\deduped\gen_*.jsonl`
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
| Generation (8 generators) | ~$4.00-5.50 |
| LAB filtering (Batch API) | ~$2-3 |
| **Total per period** | **~$6-8.50** |
| **All 6 periods** | **~$36-51** |

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
│   ├── base.py                        # BaseGenerator: chunking, API, ThreadPool
│   ├── prompts.py                     # All 8 prompt templates
│   ├── gen_a_factual.py               # A: Factual QA
│   ├── gen_b_cot.py                   # B: Chain-of-Thought
│   ├── gen_c_comprehension.py         # C: Reading Comprehension (MC)
│   ├── gen_d_temporal.py              # D: Temporal Reasoning (MC, no corpus)
│   ├── gen_e_quantitative.py          # E: Quantitative / Math
│   ├── gen_f_completion.py            # F: Sentence Completion (MC)
│   ├── gen_g_instruct.py              # G: Instruction Following
│   └── gen_h_antihalluc.py            # H: Anti-Hallucination (no corpus)
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
├── corpus/                            # Legacy corpus Q&A generation
│   ├── run_direct.py                  # Direct QA/CoT generation (generators A+B)
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
│   ├── input/                         # Source corpus for generation
│   │   ├── {collection}.parquet       # Parquet format (economist, ft, etc.)
│   │   └── {collection}/              # Or txt directory format
│   │       └── *.txt
│   ├── generated/                     # Legacy run_direct.py output
│   │   └── {collection}/
│   │       └── {collection}_qa_cot.jsonl
│   ├── by_generator/                  # Per-generator raw output (Step 1)
│   │   ├── gen_a_factual.jsonl
│   │   ├── gen_b_cot.jsonl
│   │   ├── gen_c_comprehension.jsonl
│   │   ├── gen_d_temporal.jsonl
│   │   ├── gen_e_quantitative.jsonl
│   │   ├── gen_f_completion.jsonl
│   │   ├── gen_g_instruct.jsonl
│   │   └── gen_h_antihalluc.jsonl
│   └── document_metadata.parquet      # Document provenance index
│
├── quality/                           # Quality pipeline outputs (Step 2)
│   ├── validated/                     # After format + content validation
│   │   └── gen_*.jsonl
│   ├── deduped/                       # After 3-level deduplication
│   │   └── gen_*.jsonl
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
