# Post-Training Pipeline

Synthetic data generation and assembly for historical LLM post-training (6 periods, 1678-2023).

## Three-Step Pipeline

```bash
# Step 1: Prepare source documents (one-time per period)
python -m src.post_training.prepare --period 1900_1949

# Step 2: Generate synthetic data
python -m src.post_training.generate submit --period 1900_1949     # batch API (50% off)
python -m src.post_training.generate check  --period 1900_1949     # poll status
python -m src.post_training.generate process --period 1900_1949    # download + write

# Step 3: Assemble into train/test splits
python -m src.post_training.assemble --period 1900_1949
```

For testing, use sync mode with a small target:
```bash
python -m src.post_training.generate --period 1900_1949 --target 120 --sync
```

---

## How Allocation Works

One parameter — `--target` (default: 1,000,000) — drives the entire pipeline.

### The Rule: Equal Weight Per Format Slot

Each generator produces 1+ format variants. Every format slot gets an equal share of the target. A generator's allocation is simply `per_slot x number_of_formats`.

```
GENERATOR_SPEC in config.py:

Gen   Name              Formats                       Slots
───   ────              ───────                       ─────
A     Factual QA        mc4, open                     2
B     Chain-of-Thought  mc4, open, cot                3
C     Comprehension     mc4_passage, mc2_passage      2
D     Quantitative      open, cot                     2
E     Completion        mc4, mc2                      2
F     Instruct          mc4_passage                   1
                                              Total: 12 slots
```

At `--target 1,000,000`:

```
per_slot = 1,000,000 / 12 = 83,333

A: 83,333 x 2 = 166,666
B: 83,333 x 3 = 249,999
C: 83,333 x 2 = 166,666
D: 83,333 x 2 = 166,666
E: 83,333 x 2 = 166,666
F: 83,333 x 1 =  83,333
                ─────────
         Total: 999,996  (+4 absorbed into A = 1,000,000)
```

### From Target to Docs Needed

Each document is chunked (~2 chunks per doc at 6000 chars). Each chunk produces 2 items per API call (`ITEMS_PER_CALL = 2`). So:

```
items_per_doc = ITEMS_PER_CALL x CHUNKS_PER_DOC = 2 x 2 = 4

docs_needed (per generator) = ceil(per_slot / items_per_doc)
                             = ceil(83,333 / 4) = 20,834
```

Each generator independently samples its own docs from `synthetic/input/`. All of this is computed by `compute_plan()` — you only specify `--target`.

### Per-Period Output Targets

| Output | Size | Description |
|--------|-----:|-------------|
| Mid-train | 1,000,000 | All synthetic examples |
| SFT | 10,000 | 1% proportional subsample |
| Test | ~50,000 | 5% holdout for training-loss monitoring |

---

## The 6 Generators (12 Format Slots)

All generators are corpus-based — they read documents from `synthetic/input/` and produce training conversations grounded in historical text.

| ID | Name | Formats | Benchmark Alignment | Items/Call |
|----|------|---------|---------------------|-----------|
| A | Factual QA | `mc4`, `open` | MMLU, ARC | 2 |
| B | Chain-of-Thought | `mc4`, `open`, `cot` | ARC, GSM8K | 2 |
| C | Reading Comprehension | `mc4_passage`, `mc2_passage` | RACE, BoolQ | 2 |
| D | Quantitative | `open`, `cot` | GSM8K | 2 |
| E | Sentence Completion | `mc4`, `mc2` | HellaSwag, WinoGrande | 2 |
| F | Instruction Following | `mc4_passage` | RACE | 2 |

### Format Key

| Format | Description | Eval Benchmarks |
|--------|-------------|-----------------|
| `mc4` | 4-choice MC | MMLU, ARC, HellaSwag |
| `mc2` | 2-choice MC | PIQA, WinoGrande |
| `mc4_passage` | Passage + 4-choice MC (RACE-style) | RACE |
| `mc2_passage` | Passage + 2-choice MC (BoolQ-style) | BoolQ |
| `open` | Open-ended Q&A | GSM8K (generative) |
| `cot` | `<think>` reasoning + answer | GSM8K (with reasoning) |

### Multi-Format Rendering

Each generator makes **one API call** per chunk that returns all fields needed for every format. The prompt requests the correct answer, distractors, and reasoning steps together. The same raw response is rendered into multiple formats locally — zero extra API cost.

**Self-contained questions:** Prompts for non-passage generators (A, B, D, E) instruct GPT to produce questions answerable without the source text. Passage-based generators (C, F) include the source passage in the conversation.

**MC answer positioning:** Per-format cyclic counters place the correct answer at A, B, C, D in rotation — guaranteeing ~25% uniform distribution.

---

## Step 1: Prepare Source Documents

`prepare.py` exports documents from two sources into uniform parquets at `synthetic/input/{collection}.parquet` (columns: `doc_name`, `text`).

### Main Corpus (~26 collections)
- Joins quality scores (`corpus/classified/classified_{year}.parquet`) with collection labels from raw parquets
- Applies quality percentile filter (default: top 50%)
- Caps at 10K docs per collection

### Additional Data (4 news archives)
- NYT, Economist, FT, Newswire loaded from `additional_data/raw/`
- Filters by minimum text length (200 chars)
- Joins pre-calculated quality scores from `additional_data/classified/{collection}/classified_{year}.parquet`
- Same quality percentile filter and 10K cap

Quality scores for additional data are generated by `src/base_training/quality/classify_additional.py`, which applies the same corpus-trained Ridge models via BGE embeddings.

```bash
python -m src.post_training.prepare --period 1900_1949
python -m src.post_training.prepare --period 1900_1949 --source corpus       # corpus only
python -m src.post_training.prepare --period 1900_1949 --source additional   # news only
python -m src.post_training.prepare --period 1900_1949 --quality-percentile 75  # top 25%
```

### Step 2: Generate Synthetic Data

```bash
# Batch API (production) — 50% cost savings, ~24h turnaround
python -m src.post_training.generate submit  --period 1900_1949
python -m src.post_training.generate check   --period 1900_1949
python -m src.post_training.generate process --period 1900_1949

# Sync mode (testing) — instant results
python -m src.post_training.generate --period 1900_1949 --target 120 --sync

# Specific generators only
python -m src.post_training.generate submit --period 1900_1949 --generators A B D
```

**Output:** `synthetic/by_generator/gen_{name}_{fmt}.jsonl`

Each line is a nanochat CustomJSON conversation:
```jsonl
[{"role":"user","content":"What caused...?"},{"role":"assistant","content":"The main cause..."}]
```

### Step 3: Assemble

Merges all generator outputs into three files:

```bash
python -m src.post_training.assemble --period 1900_1949 --source raw
python -m src.post_training.assemble --period 1900_1949 --dry-run  # preview only
```

**Output:**
- `final/train/hist_synthetic_midtrain.jsonl` — all examples (1M)
- `final/train/hist_synthetic_sft.jsonl` — 10K proportional subsample
- `final/test/hist_synthetic_test.jsonl` — 5% holdout

---

## API Usage and Cost

Generation uses `gpt-4o-mini` via the OpenAI Batch API (50% discount).

At 1M target with 12 format slots and ~20,834 docs per generator:
- ~6 generators x ~20,834 docs x ~2 chunks = ~250,000 API calls
- At ~$0.00075/call (Batch API average) = **~$185/period**
- x 6 periods = **~$1,110 total**

For testing, use `--target 120 --sync` (< $0.01).

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
├── config.py                          # Central config: GENERATOR_SPEC, compute_plan(), paths
├── prepare.py                         # Step 1: Export corpus + additional → uniform parquets
├── generate.py                        # Step 2: Submit/check/process batch generation (+ --sync)
├── assemble.py                        # Step 3: Merge generators into train/test splits
├── utils.py                           # Shared: OpenAI client, JSONL I/O, Batch API helpers
│
├── generators/                        # Synthetic data generators (A-F)
│   ├── __init__.py                    # Registry: {"A": GenAFactual, ..., "F": GenFInstruct}
│   ├── base.py                        # BaseGenerator: run(), batch submit/process, render_mc
│   ├── prompts.py                     # 6 prompt templates (with distractor requests)
│   ├── gen_a_factual.py               # A: Factual QA (mc4, open)
│   ├── gen_b_cot.py                   # B: Chain-of-Thought (mc4, open, cot)
│   ├── gen_c_comprehension.py         # C: Reading Comprehension (mc4_passage, mc2_passage)
│   ├── gen_d_quantitative.py          # D: Quantitative / Math (open, cot)
│   ├── gen_e_completion.py            # E: Sentence Completion (mc4, mc2)
│   └── gen_f_instruct.py             # F: Instruction Following (mc4_passage)
│
├── quality/                           # Quality pipeline (deferred for v1)
│   ├── validate.py                    # Format + content validation
│   ├── dedup.py                       # 3-level dedup (exact, MinHash, cross-gen)
│   └── pipeline.py                    # Orchestrator: validate -> dedup
│
├── corpus/                            # Legacy scripts (deprecated, kept for reference)
│   ├── build_index.py                 # Deprecated → use prepare.py
│   ├── export.py                      # Deprecated → use prepare.py
│   └── export_additional.py           # Deprecated → use prepare.py
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
│   ├── input/                         # Source corpus (Step 1: prepare.py)
│   │   ├── {collection}.parquet       # Both corpus and additional data as parquets
│   │   │                              #   (columns: doc_name, text; 10K cap each)
│   │   └── {collection}/              # Legacy: individual txt files (still supported)
│   │       └── doc_00000.txt
│   │
│   ├── by_generator/                  # Per-generator raw output (Step 2: generate.py)
│   │   ├── gen_a_factual_mc4.jsonl
│   │   ├── gen_a_factual_open.jsonl
│   │   ├── gen_b_cot_mc4.jsonl
│   │   ├── gen_b_cot_open.jsonl
│   │   ├── gen_b_cot_cot.jsonl
│   │   ├── gen_c_comprehension_mc4_passage.jsonl
│   │   ├── gen_c_comprehension_mc2_passage.jsonl
│   │   ├── gen_d_quantitative_open.jsonl
│   │   ├── gen_d_quantitative_cot.jsonl
│   │   ├── gen_e_completion_mc4.jsonl
│   │   ├── gen_e_completion_mc2.jsonl
│   │   └── gen_f_instruct_mc4_passage.jsonl
│   │
│   └── batch_temp/                    # Batch API temporary files
│       ├── gen_*_requests.jsonl
│       ├── gen_*_manifest.jsonl
│       ├── gen_*_batch_id.txt
│       └── gen_*_results.jsonl
│
├── final/
│   ├── train/                         # Final train splits (Step 3: assemble.py)
│   │   ├── hist_synthetic_midtrain.jsonl  # All synthetic (1M target)
│   │   └── hist_synthetic_sft.jsonl       # 1% proportional subsample (10K)
│   └── test/
│       └── hist_synthetic_test.jsonl      # 5% holdout
│
└── quality/                           # Quality pipeline outputs (deferred for v1)
    ├── validated/
    └── deduped/
```

---

## nanochat Integration

All output files are in nanochat's **CustomJSON** format:

```jsonl
[{"role":"user","content":"What caused...?"},{"role":"assistant","content":"The main cause..."}]
[{"role":"user","content":"Multiple Choice question: ...\n- Option A=A\n- Option B=B\n..."},{"role":"assistant","content":"B"}]
[{"role":"user","content":"Explain..."},{"role":"assistant","content":"<think>\nStep 1...\n</think>\nThe answer is..."}]
```

---

## Dependencies

- **Required:** `openai`, `pandas`, `pyarrow`
- **Optional:** `datasketch` (for MinHash near-duplicate detection)
