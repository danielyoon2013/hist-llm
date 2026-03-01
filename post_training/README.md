# Post-Training Pipeline

Synthetic data generation, quality filtering, and assembly for historical LLM post-training (6 periods, 1678-2023).

## Quick Start: Run in Order

| Step | What | How | Output |
|------|------|-----|--------|
| 0a | Export main corpus | `python -m src.post_training.corpus.export --period {P}` | `synthetic/input/{collection}/*.txt` |
| 0b | Export news archives | `python -m src.post_training.corpus.export_additional --period {P}` | `synthetic/input/{dataset}.parquet` |
| 1a | Submit generation batches | `python -m src.post_training.generate submit --period {P}` | `synthetic/batch_temp/*_batch_id.txt` |
| 1b | Check batch status | `python -m src.post_training.generate check --period {P}` | — |
| 1c | Process batch results | `python -m src.post_training.generate process --period {P}` | `synthetic/by_generator/gen_*_{fmt}.jsonl` |
| 2 | Assemble mid-train/SFT/test | `python -m src.post_training.assemble --period {P}` | `final/train/`, `final/test/` |
| ~~3~~ | ~~Validate + deduplicate~~ | _Deferred — not needed for v1_ | |
| ~~4~~ | ~~LAB temporal filter~~ | _Deferred — not needed for v1_ | |
| ~~5~~ | ~~Train/test split (ext.)~~ | _Deferred — not needed for v1_ | |

### Per-Period Targets

| Output | Size | Description |
|--------|-----:|-------------|
| Mid-train | 1,000,000 | All synthetic examples (pre-quality target) |
| SFT | 10,000 | 1% proportional subsample (from non-H generators) |
| Test | ~50,000 | 5% holdout for training-loss monitoring |

All `python` commands are run from the repo root: `python -m src.post_training.generate ...`

All data paths are on `D:\hist_LLM\` (local SSD). Period paths defined in `config.py`.

---

## Pipeline Diagram

```
Step 0 (one-time)      Step 1 (Batch API)           Step 2
─────────────────      ──────────────────           ──────
export.py              generate.py submit           assemble.py
export_additional.py   generate.py check            merge + subsample
                       generate.py process              |
corpus → input/            |                        final/train/
  ~30 collections      ~192,500 API calls             midtrain.jsonl (1M)
  ~4 news archives     via Batch API (50% off)        sft.jsonl (10K)
                       → by_generator/*.jsonl       final/test/
                                                      test.jsonl (50K)

Steps 3-5 (dedup, LAB filter, external split) deferred for v1.
```

---

## How Allocation Works

The `--target` parameter (default: 1,000,000) drives the entire pipeline. Here's the full flow:

### 1. Split target between corpus and metadata generators

```
Total target: 1,000,000
  ├── Metadata generators (D, H): 2.5% each = 25,000 each = 50,000 total
  └── Corpus generators (A,B,C,E,F,G): remaining 95% = 950,000 total
```

### 2. Distribute corpus share proportionally

Each corpus generator produces `items_per_chunk × num_formats` examples per chunk. This determines their relative weight:

```
Generator   items/chunk × formats = examples/chunk    weight
─────────   ──────────────────────────────────────    ──────
A (Factual)       3    ×    2    =    6                 6/30 = 20%
B (CoT)           2    ×    3    =    6                 6/30 = 20%
C (Comprehend)    3    ×    2    =    6                 6/30 = 20%
E (Quant)         2    ×    2    =    4                 4/30 = 13.3%
F (Completion)    3    ×    2    =    6                 6/30 = 20%
G (Instruct)      2    ×    1    =    2                 2/30 = 6.7%
                                     ──
                              Total: 30 examples/chunk
```

### 3. Compute docs needed

Each document → ~2 chunks → 30 examples/chunk = **60 examples per doc**

```
Docs needed = 950,000 / 60 = 15,833 docs
```

The system randomly samples 15,833 docs from whatever was exported in Step 0. All 6 corpus generators process the **same docs** — each chunk goes through every corpus generator, which is why one doc produces 60 examples total.

### 4. Compute metadata generator API calls

Metadata generators (D, H) don't use corpus docs — they generate from the period's year range alone.

**Gen D (Temporal)**: `25,000 target / 2 formats = 12,500 raw items / 10 items per call = 1,250 API calls`

**Gen H (Historical Facts)**: Generates per-year to avoid duplicates. For 1900-1949 (50 years):
`12,500 raw items / 50 years = 250 items/year / 10 items per call = 25 calls/year × 50 years = 1,250 API calls`

For 1678-1849 (172 years): `73 items/year → 8 calls/year × 172 years = 1,376 API calls`

This ensures D and H produce **exactly 25,000 examples** regardless of how many years the period spans.

### 5. Final allocation table

| Gen | Type | % of 1M | Mid-Train Target | SFT (1%) |
|-----|------|---------|----------------:|--------:|
| A | corpus | 19.0% | 190,000 | 1,949 |
| B | corpus | 19.0% | 190,000 | 1,949 |
| C | corpus | 19.0% | 190,000 | 1,949 |
| D | metadata | 2.5% | 25,000 | 256 |
| E | corpus | 12.7% | 126,666 | 1,299 |
| F | corpus | 19.0% | 190,000 | 1,949 |
| G | corpus | 6.3% | 63,333 | 649 |
| H | metadata | 2.5% | 25,000 | 0 (train-only) |
| **Total** | | **100%** | **1,000,000** | **10,000** |

All of this is computed automatically by `config.py:compute_allocation()`. You only specify `--target`.

---

## The 8 Generators (16 Format Cells)

Each generator produces multiple format variants from a single API call, aligned to external benchmark native formats. This ensures evaluation measures temporal knowledge, not format confusion.

| ID | Name | Input | Formats | Benchmarks | Items/Call |
|----|------|-------|---------|------------|-----------|
| A | Factual QA | corpus | `mc4`, `open` | MMLU, ARC | 3 |
| B | Chain-of-Thought | corpus | `mc4`, `open`, `cot` | ARC, GSM8K | 2 |
| C | Reading Comprehension | corpus | `mc4_passage`, `mc2_passage` | RACE, BoolQ | 3 |
| D | Temporal Reasoning | metadata | `mc4`, `open` | LAB Eval | 10 |
| E | Quantitative | corpus | `open`, `cot` | GSM8K | 2 |
| F | Sentence Completion | corpus | `mc4`, `mc2` | HellaSwag, WinoGrande | 3 |
| G | Instruction Following | corpus | `mc4_passage` | RACE | 2 |
| H | Historical Facts | metadata | `mc4`, `open` | MMLU, LAB Eval | 10 |

**Self-contained questions:** Prompts for non-passage generators (A, B, E, F) explicitly instruct GPT to produce self-contained questions that are answerable without seeing the source text. Passage-based generators (C, G) include the source passage in the training conversation, so their questions may reference it.

### Format Key

| Format | Description | Eval Benchmarks Using It |
|--------|-------------|--------------------------|
| `mc4` | 4-choice MC via `render_mc()` | MMLU, ARC, HellaSwag, LAB Eval |
| `mc2` | 2-choice MC via `render_mc()` | PIQA, WinoGrande |
| `mc4_passage` | Passage prefix + 4-choice MC (RACE-style) | RACE |
| `mc2_passage` | Passage prefix + 2-choice MC (BoolQ-style) | BoolQ |
| `open` | Open-ended question + answer | GSM8K (generative) |
| `cot` | Question + `<think>` reasoning + answer | GSM8K (with reasoning) |

### How Multi-Format Rendering Works

Each generator makes **one API call** that returns all fields needed for every format. The prompt requests the correct answer, distractors, and (for Gen B/E) reasoning steps together. The same raw response is then rendered into multiple formats locally — zero extra API cost.

| Format | Fields consumed from API response |
|--------|----------------------------------|
| `open` | `question` + `answer` |
| `mc4` / `mc2` | `question` + `answer` + `distractors` |
| `cot` | `question` + `reasoning` + `answer` (Gen B/E only) |
| `mc4_passage` / `mc2_passage` | `question` + `answer` + `distractors` + source chunk as passage prefix |

If an item has fewer than 3 distractors (GPT occasionally returns only 1-2), the MC format skips that item (`format_conversation` returns `None`) while the open/cot formats still use it. This is why MC counts can be slightly lower than open counts.

MC answer positioning is balanced via per-format cyclic counters (`self._mc_counters[fmt]`): the 1st item places the correct answer at A, the 2nd at B, the 3rd at C, the 4th at D, then back to A — guaranteeing ~25% uniform distribution across the dataset.

**Corpus-based** (A, B, C, E, F, G): Read text from `synthetic/input/` (parquet or txt), chunk at 6000 chars / 300 overlap, call GPT-4o-mini per chunk.

**Metadata-based** (D, H): Generate from period year range alone (no corpus needed). D creates temporal ordering/reasoning questions in batches; H generates per-year (one API call per year in the period, e.g., 50 calls for 1900-1949) to eliminate duplicate facts across batches.

**Train-only** (H): Historical facts are placed entirely in the training set — no test split. Factual recall is evaluated via external benchmarks (MMLU, LAB Eval), not a held-out test set.

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

Run generators for a period using the OpenAI Batch API (50% cost savings, ~24h turnaround). The `--target` flag controls total output (default: 1M). Allocation across generators is computed automatically.

```bash
# BATCH MODE (production) — 3-step workflow
python -m src.post_training.generate submit  --period 1900_1949   # submit all 8 generator batches
python -m src.post_training.generate check   --period 1900_1949   # check status (repeat until done)
python -m src.post_training.generate process --period 1900_1949   # download results + write output

# Custom target
python -m src.post_training.generate submit --period 1900_1949 --target 500000

# Specific generators only
python -m src.post_training.generate submit --period 1900_1949 --generators A B D H

# SYNC MODE (testing, instant results, uses regular API)
python -m src.post_training.generate --period 1900_1949 --target 180 --sync
```

**Output:** `D:\hist_LLM\periods\{period}\posttraining_data\synthetic\by_generator\gen_{name}_{fmt}.jsonl`

All output is nanochat CustomJSON format: each line is `[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]`

### Step 2: Assemble Mid-Train / SFT / Test

Merge all generator outputs into three files:
- **Mid-train** — all examples (1M target), used for continued pre-training
- **SFT** — 1% proportional subsample (10K default), used for supervised fine-tuning
- **Test** — 5% holdout from non-H generators, for training-loss monitoring

Train-only generators (Gen H: historical facts) go entirely to mid-train — no test split, no SFT. Factual recall is evaluated via external benchmarks (MMLU, LAB Eval).

```bash
# v1: read directly from raw generator output (no dedup/filter)
python -m src.post_training.assemble --period 1900_1949 --source raw

# Custom SFT size
python -m src.post_training.assemble --period 1900_1949 --source raw --sft-size 5000

# Dry run (show plan without writing)
python -m src.post_training.assemble --period 1900_1949 --source raw --dry-run
```

Auto-detection priority: `final/filtered` > `quality/deduped` > `quality/validated` > `synthetic/by_generator`. For v1 we use `--source raw` to read directly from `synthetic/by_generator/`.

**Output:**
- `final/train/hist_synthetic_midtrain.jsonl` — all 1M examples
- `final/train/hist_synthetic_sft.jsonl` — 10K proportional subsample
- `final/test/hist_synthetic_test.jsonl` — 5% holdout

### Deferred Steps (v2)

The following steps are implemented but deferred for v1. They can be added later without changing any generated data:

- **Validate + Deduplicate** (`process.py`): 3-level dedup (exact hash, MinHash, cross-generator). Run before assemble to reduce redundancy.
- **LAB Temporal Filter** (`instruct/filter.py`): Batch API filter to remove questions requiring post-period knowledge. Run after dedup, before assemble.
- **External Dataset Split** (`instruct/split.py`): Split LAB-filtered SmolTalk/MMLU/ARC into train/test for the non-synthetic curriculum.

---

## API Usage and Cost

### Which API is used where?

| Step | API Type | Why |
|------|----------|-----|
| **Step 1: Generate** | **Batch API** (async, 50% discount) | ~192,500 API calls per period — batch saves ~$140/period vs regular API. Submit JSONL, wait ~24h, download results. |

Generation uses `gpt-4o-mini` via the OpenAI Batch API. Multi-format rendering adds zero extra API cost — one API call per chunk returns all fields needed for every format variant.

A sync mode (`--sync`) is available for testing small runs — it uses the regular API with ThreadPoolExecutor (50 workers) for instant results.

### Cost Estimate (v1 — generation only)

GPT-4o-mini Batch API pricing (50% off regular): $0.075/1M input tokens, $0.30/1M output tokens.

| Item | Cost/Period | x 6 Periods |
|------|----------:|----------:|
| Generation (~192,500 API calls via Batch API) | ~$140 | ~$840 |
| **Total** | **~$140** | **~$840** |

Generation breakdown: 15,833 docs × 2 chunks × 6 corpus generators = ~190,000 calls + ~2,500 metadata calls (D+H) = ~192,500 total. At ~$0.00075/call (Batch API average) = ~$140/period.

For testing, use `--target 180 --sync` for ~3 docs worth (< $0.01).

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
├── generate.py                        # CLI: submit/check/process batch generation (+ --sync)
├── process.py                         # CLI: validate + deduplicate pipeline
├── assemble.py                        # CLI: merge generators into train/test
│
├── generators/                        # Synthetic data generators (A-H)
│   ├── __init__.py                    # Registry: {"A": GenAFactual, ...}
│   ├── base.py                        # BaseGenerator: run(), submit_batch_requests(),
│   │                                  #   process_batch_results(), render_mc, call_api
│   ├── prompts.py                     # All 8 prompt templates (with distractor requests)
│   ├── gen_a_factual.py               # A: Factual QA (mc4, open)
│   ├── gen_b_cot.py                   # B: Chain-of-Thought (mc4, open, cot)
│   ├── gen_c_comprehension.py         # C: Reading Comprehension (mc4_passage, mc2_passage)
│   ├── gen_d_temporal.py              # D: Temporal Reasoning (mc4, open; no corpus)
│   ├── gen_e_quantitative.py          # E: Quantitative / Math (open, cot)
│   ├── gen_f_completion.py            # F: Sentence Completion (mc4, mc2)
│   ├── gen_g_instruct.py              # G: Instruction Following (mc4_passage)
│   └── gen_h_histfacts.py             # H: Historical Facts & Dates (mc4, open; no corpus)
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
│   │   ├── gen_c_comprehension_mc4_passage.jsonl
│   │   ├── gen_c_comprehension_mc2_passage.jsonl
│   │   ├── gen_d_temporal_mc4.jsonl
│   │   ├── gen_d_temporal_open.jsonl
│   │   ├── gen_e_quantitative_open.jsonl
│   │   ├── gen_e_quantitative_cot.jsonl
│   │   ├── gen_f_completion_mc4.jsonl
│   │   ├── gen_f_completion_mc2.jsonl
│   │   ├── gen_g_instruct_mc4_passage.jsonl
│   │   ├── gen_h_histfacts_mc4.jsonl
│   │   └── gen_h_histfacts_open.jsonl
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
│   │   ├── hist_synthetic_midtrain.jsonl  # All synthetic (1M target)
│   │   ├── hist_synthetic_sft.jsonl       # 1% proportional subsample (10K)
│   │   ├── hist_{collection}_train.jsonl
│   │   ├── smoltalk_train.jsonl
│   │   ├── mmlu_train.jsonl
│   │   └── ...                        # nanochat training files
│   └── test/                          # Final test splits
│       └── *.jsonl
│
├── eval/
│   └── lab_questions.jsonl            # LAB evaluation MC questions
│
├── LAB_scores/                        # LAB filter scores per dataset
│   └── {dataset}_scores.jsonl
│
└── batch_temp/                        # Batch API temporary files (Step 1 + Step 3)
    ├── gen_*_requests.jsonl           # Batch request files (one per generator)
    ├── gen_*_manifest.jsonl           # custom_id → chunk_text mapping (corpus generators)
    ├── gen_*_batch_id.txt             # OpenAI batch ID for status checks
    ├── gen_*_results.jsonl            # Downloaded batch results
    ├── *_requests.jsonl               # LAB filter batch requests (Step 3)
    ├── *_results.jsonl                # LAB filter batch results
    └── *_batch_ids.txt                # LAB filter batch IDs
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
