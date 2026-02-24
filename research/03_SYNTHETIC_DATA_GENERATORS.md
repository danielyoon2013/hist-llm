# Synthetic Data Generators

> **Paper section:** 3.2 (Synthetic Data Generation)
> **Dependencies:** `src/post_training/corpus/run_direct.py`, `synth_config.yaml`, `config.py`
> **Status:** Generators A-B implemented; C-H proposed; GSM8K/MATH retained as external

---

## Table of Contents

- [1. The Content x Format x Source Framework](#1-the-content-x-format-x-source-framework)
  - [1a. Three Dimensions](#1a-three-dimensions-of-synthetic-data)
  - [1b. Content x Format Matrix](#1b-the-content-x-format-matrix)
  - [1c. Source Mapping](#1c-source-mapping-third-dimension)
  - [1d. External Datasets Retained](#1d-external-datasets-retained)
- [2. Generator-to-Evaluation Alignment](#2-generator-to-evaluation-alignment)
- [3. Existing Generators (A, B)](#3-existing-generators-implemented)
- [4. Standard Generators (C, F, G)](#4-new-generators--standard-c-f-g)
- [5. Priority Generators (D, E, H)](#5-priority-generators--implementation-ready-d-e-h)
- [6. Volume Targets and Data Budget](#6-volume-targets-and-data-budget)
- [7. Quality Control on Synthetic Data](#7-quality-control-on-synthetic-data)
- [8. Prompt Engineering Principles](#8-prompt-engineering-principles)
- [9. Implementation Plan](#9-implementation-plan)
- [10. Evol-Instruct Complexity Scaling](#10-evol-instruct-complexity-scaling-post-generation)
- [References](#references)

---

## 1. The Content x Format x Source Framework

### 1a. Three Dimensions of Synthetic Data

Our synthetic data generation operates on a **3D matrix**:

1. **Content type (rows)** — What capability does the example train? (8 generators A-H)
2. **Format (columns)** — How is the question/answer structured? (MC, open-ended, CoT, T/F, fill-blank, passage-based, ranking)
3. **Source (depth)** — Which corpus collection provides the input text? (News, Law, Academic, etc.)

This framework is motivated by three findings from the literature:

| Study | Finding |
|-------|---------|
| **FLAN** (Wei et al., 2022) | Mixing 10+ task formats -> +3-10% across benchmarks |
| **Phi-4** (Abdin et al., 2024) | 50 synthetic dataset types; format diversity was key to beating models 10x larger |
| **DOTS** (Yang et al., ICLR 2025) | Optimal format varies by downstream task; no single format dominates |
| **Orca** (Mukherjee et al., 2023) | 16 system instruction templates -> 100%+ improvement over single-format |

No prior work has formally organized synthetic data generation as a content x format x source matrix. This is a methodological contribution of our paper.

### 1b. The Content x Format Matrix

Each cell represents a distinct (generator, format) pair with its own prompt template. Not all cells are active — only natural fits are used.

```
                       MC    Open-ended   CoT     T/F    Fill-blank   Passage    Ranking
                      (4-opt)  (free)    (think) (Y/N+) (cloze)     (w/ text)  (order)
───────────────────────────────────────────────────────────────────────────────────────────
A. Factual              *        *                  *
B. Reasoning                     *          *                                      *
C. Comprehension        *        *          *                           *
D. Temporal             *                   *       *                              *
E. Quantitative                  *          *              *
F. Completion                                              *
G. Instruction                   *                                      *
H. Anti-Halluc          *        *                  *
───────────────────────────────────────────────────────────────────────────────────────────
External: GSM8K                  *          *                                   (retained)
External: MATH                   *          *                                   (retained)
```

`*` = active cell (has a prompt template and generates data)

This yields **~25 active cells** from 8 generators x 7 formats, comparable to Phi-4's diversity level but organized systematically rather than ad hoc.

### 1c. Source Mapping (Third Dimension)

Not every source naturally supports every generator. The matrix is **sparse** — forcing math problems from legal texts produces garbage. We only generate from natural fits.

```
                   English Corpus    News (NYT/    CaseLaw/     Academic/
                   (general hist)    Econ/FT)      Patents      Books
──────────────────────────────────────────────────────────────────────────
A. Factual              *               *             *            *
B. Reasoning            *               *             *            *
C. Comprehension        *               *             *            *
D. Temporal             *               **            .
E. Quantitative         .               **            .            .
F. Completion           *               *             *            *
G. Instruction          *               *             *            *
H. Anti-Halluc          *               *             *            *

** = natural strong fit    * = good fit    . = weak/forced (avoid)
```

**Key design decisions from source mapping:**

- **Generator D (Temporal):** Primarily News sources — richest for temporal chains, dates, causation
- **Generator E (Quantitative):** Primarily News + economic corpus — mine documents with actual numbers (trade statistics, market data, demographics)
- **Generators A, B, C, G:** Universal — work well across all sources
- **Generator H:** Source-independent — questions are about post-period events, not derived from source text

### 1d. External Datasets Retained

Two external datasets are retained alongside our 8 corpus-derived generators:

| Dataset | LAB Contamination | Why Retain |
|---------|-------------------|------------|
| **GSM8K** | 2.5% | Pure math reasoning — temporally neutral (2+2=4 in any era). Tests abstract multi-step arithmetic our corpus cannot naturally provide. |
| **MATH** | 0.3% | Advanced mathematical reasoning — near-zero contamination. Covers algebra, number theory, geometry. |

**Why not regenerate these from our corpus?**

Our corpus (news, law, books) rarely contains multi-step math in word-problem format. Generating GSM8K-style problems from news articles would produce lower quality than the curated original, and the result would not be meaningfully different from using GSM8K directly. API budget is better spent on generators that produce genuinely novel data.

**Generator E is complementary, not a replacement.** GSM8K tests "can the model do abstract math?" Generator E tests "can the model reason about historical quantities?" These are different capabilities:

| | GSM8K (external) | Generator E (ours) |
|---|---|---|
| Numbers | Invented (Sally has 5 marbles) | Real (1923 wheat exports: 4.2M bushels) |
| Context | None (pure word problems) | Grounded in historical documents |
| Temporal | Neutral (math doesn't change) | Period-specific (actual figures from that era) |
| Capability tested | Abstract arithmetic | Domain-grounded quantitative reasoning |

---

## 2. Generator-to-Evaluation Alignment

Every generator was designed to target specific evaluation benchmarks. Conversely, every benchmark is "served" by at least one generator. This ensures our synthetic data covers the full evaluation space — and enables clean ablation studies.

### Alignment Matrix

| Generator | MMLU | ARC-C | GSM8K | HellaSwag | BoolQ | PIQA | Wino | RACE | LAB Eval | LAP | Temp Consist | Anti-H Diag |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A. Factual** | `**` | `**` | | | `*` | | | | | `.` | | |
| **B. Reasoning** | | `*` | | | | | | `**` | | | | |
| **C. Comprehension** | | | | | `*` | | | `**` | | | | |
| **D. Temporal** | | | | | | | | | `**` | `**` | `**` | |
| **E. Quantitative** | | | `**` | | | | | | | | | |
| **F. Completion** | | | | `**` | | `*` | `*` | | | | | |
| **G. Instruction** | | | | | | | | `.` | | | | |
| **H. Anti-Halluc** | | | | | | | | | `**` | | `**` | `**` |
| *External: GSM8K* | | | `**` | | | | | | | | | |
| *External: MATH* | | | `*` | | | | | | | | | |

`**` = primary alignment (generator specifically targets this eval) | `*` = secondary alignment | `.` = indirect benefit

### Benchmark x Training Format View

Which generators feed each benchmark, broken down by training data format. **Bold** = format directly matching the benchmark's native eval format. `ext` = external dataset retained (GSM8K/MATH).

| Benchmark | MC | Open-ended | CoT | T/F | Fill-blank | Passage | Ranking |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MMLU | **A** | A | | A | | | |
| ARC-Challenge | **A** | A, B | B | A | | | B |
| GSM8K | | **E**, ext | **E** | | E | | |
| HellaSwag | **F** | | | | | | |
| BoolQ | A, C | A, C | C | **A, C** | | **C** | |
| PIQA | **F** | | | | | | |
| WinoGrande | | | | | **F** | | |
| RACE | **C** | B, C | B, C | | | **C** | B |
| LAB Eval | **D, H** | H | D | D, H | | | D |
| LAP Score | D, H | H | D | D, H | | | D |
| Temp Consistency | D | | D | D | | | D |
| Anti-H Diag | H | **H** | | H | | | |

Reading this table: the MMLU row shows Generator A provides MC, open-ended, and T/F training data that all contribute to MMLU performance, with MC (**bold**) being the format that directly matches MMLU's 4-choice eval format.

### Reading the Alignment

| Benchmark | Primary Generator(s) | What It Validates |
|-----------|---------------------|-------------------|
| MMLU | A (Factual MC) | Domain knowledge breadth |
| ARC-Challenge | A (Factual), B (Reasoning) | Scientific reasoning |
| GSM8K | E (Quantitative) + GSM8K external | Math reasoning |
| HellaSwag | F (Completion) | Commonsense, language modeling |
| BoolQ | A (Factual), C (Comprehension) | Boolean reasoning |
| PIQA | F (Completion) | Physical intuition |
| WinoGrande | F (Completion) | Coreference resolution |
| RACE | B (Reasoning), C (Comprehension) | Long-passage comprehension |
| LAB Eval | D (Temporal), H (Anti-Halluc) | Temporal isolation (**core thesis**) |
| LAP Score | D (Temporal), H (Anti-Halluc) | Leakage quantification |
| Temporal Consistency | D (Temporal) | Period awareness probes |
| Anti-Halluc Diagnostic | H (Anti-Halluc) | Refusal rate on post-period questions |

### The Ablation Story

This alignment enables clean generator ablation studies:

- Remove Generator D -> LAB Eval degrades (more temporal leakage)
- Remove Generator F -> HellaSwag drops
- Remove Generator H -> Anti-Hallucination Diagnostic fails
- Remove GSM8K external -> GSM8K eval drops (but Generator E partially compensates)
- All generators present -> Full coverage across all evaluations

See `06_EXPERIMENT_PLAN.md` for the full ablation design.

---

## 3. Existing Generators (Implemented)

### Generator A: Factual QA

**Source:** `src/post_training/corpus/run_direct.py`
**Status:** Production. Generated 348K+ pairs for 1950-1999 period.
**Active format cells:** MC, Open-ended, T/F

#### Prompt Template (from `run_direct.py`)

```
QA_PROMPT = """Create {num_pairs} question-answer pairs from this text for LLM training.

Rules:
1. Questions must require analytical thinking, not just fact lookup
2. Answers must be directly supported by the text
3. Vary question types: cause-effect, comparison, analysis, inference, summary
4. Return a JSON object with key "qa_pairs" containing an array:

{{"qa_pairs": [{{"question": "Question 1?", "answer": "Answer 1."}}, {{"question": "Question 2?", "answer": "Answer 2."}}]}}

Text:
{text}"""
```

#### Configuration (from `synth_config.yaml`)

| Parameter | Value |
|-----------|-------|
| Temperature | 0.7 |
| Top-p | 0.95 |
| Chunk size | 6,000 chars |
| Overlap | 300 chars |
| Num pairs per chunk | 3 |
| Max tokens | 4,096 |
| Deduplication threshold | 0.8 |

#### Current Volume (1950-1999)

- 348,255 QA pairs from Economist, NYT, FT, Newswire, Caselaw, USPTO, GATT, EurLex, Books, etc.
- Stored at: `{period}/posttraining_data/hist_corpus_qa_{period}.jsonl`

#### Strengths and Limitations

**Strengths:** High volume, covers all source types, analytically focused.
**Limitations:** Open-ended format only (no MC/T/F variants yet), no difficulty stratification.

#### Format Comparison: MMLU / ARC vs. Generator A

| | **MMLU / ARC** (external eval) | **Generator A** (our synthetic data) |
|---|---|---|
| Source | Hendrycks et al., pre-existing academic benchmark | Generated from our historical corpus via GPT-4o-mini |
| Format | 4-choice MC via `render_mc()` | 4-choice MC (identical) + open-ended variant |
| Content | General academic knowledge across 57 subjects | Historical facts from period-specific documents |

**MMLU example** (eval format, from `nanochat/tasks/mmlu.py`):
```
User:  Multiple Choice question: What was the primary cause of the Peloponnesian War?
       - Athenian imperialism=A
       - Spartan aggression=B
       - Persian invasion=C
       - Economic recession=D

       Respond only with the letter of the correct answer.
Asst:  A
```

**Generator A example** (our MC training format):
```
User:  Multiple Choice question: According to the passage, what was the primary
       consequence of the 1973 oil embargo?
       - Stagflation across Western economies=A
       - Collapse of the OPEC cartel=B
       - Rapid industrialization of oil-producing nations=C
       - Immediate shift to renewable energy sources=D

       Respond only with the letter of the correct answer.
Asst:  A
```

**Generator A example** (our open-ended training format):
```
User:  What rationale did the court provide for ruling that the jury's verdict
       should not be disturbed?
Asst:  The court reasoned that the jury's verdict was within the range of
       testimony provided by both parties...
```

> **Match:** MC variant uses identical `render_mc()` format (`- choice=LETTER`).
> **Difference:** MMLU draws from general academic knowledge; ours is grounded in historical corpus documents from the target period.

---

### Generator B: Chain-of-Thought Reasoning

**Source:** `src/post_training/corpus/run_direct.py` (COT_PROMPT)
**Status:** Production. Runs alongside Generator A.
**Active format cells:** Open-ended (CoT), Ranking

#### Prompt Template (from `run_direct.py`)

```
COT_PROMPT = """Create {num_cot} complex reasoning examples from this text that
demonstrate chain-of-thought thinking.

Each example should have:
1. A challenging question that requires step-by-step reasoning
2. Detailed reasoning steps that break down the problem
3. A concise final answer

Return a JSON object with key "cot_examples" containing an array:

{{"cot_examples": [{{"question": "Complex question?",
  "reasoning": "Step 1: First, I need to consider...\\nStep 2: Then, I analyze...
  \\nStep 3: Finally, I can conclude...",
  "answer": "Final answer based on the reasoning."}}]}}

Text:
{text}"""
```

#### Output Format

CoT examples are wrapped in `<think>` tags during conversion:

```json
[
  {"role": "user", "content": "How did the trade deficit contribute to the currency devaluation?"},
  {"role": "assistant", "content": "<think>\nStep 1: The trade deficit widened from $X to $Y...\nStep 2: This put downward pressure on the currency...\nStep 3: The central bank's reserves were insufficient...\n</think>\nThe trade deficit directly contributed to the devaluation by..."}
]
```

#### Configuration

| Parameter | Value |
|-----------|-------|
| CoT examples per chunk | 2 |
| Same chunking as QA | 6,000 chars / 300 overlap |
| `<think>` tag wrapping | Applied in `run_direct.py` lines 284-289 |

#### Format Comparison: HotpotQA vs. Generator B

| | **HotpotQA** (external multi-hop reasoning) | **Generator B** (our synthetic data) |
|---|---|---|
| Source | Yang et al., Wikipedia-derived questions | Generated from our historical corpus via GPT-4o-mini |
| Format | Open-ended multi-hop QA with decomposition | CoT with `<think>` tags + explicit reasoning steps |
| Content | Modern Wikipedia facts requiring 2+ document hops | Historical analysis requiring multi-step reasoning |

**HotpotQA example** (external dataset):
```
User:  Were Scott Derrickson and Ed Wood of the same nationality?
Asst:  Scott Derrickson is American, born in Denver, Colorado.
       Ed Wood was also American, born in Poughkeepsie, New York.
       So yes, they share the same nationality.
```

**Generator B example** (our CoT training format):
```
User:  How did the trade deficit contribute to the currency devaluation?
Asst:  <think>
       Step 1: The trade deficit widened from $X to $Y over the period...
       Step 2: This put downward pressure on the currency as foreign
               exchange reserves depleted...
       Step 3: The central bank's reserves were insufficient to defend
               the peg...
       </think>
       The trade deficit directly contributed to the devaluation by
       exhausting foreign exchange reserves needed to maintain the peg.
```

> **Match:** Both require multi-step reasoning connecting multiple facts.
> **Difference:** HotpotQA uses Wikipedia passages (56.6% temporally contaminated for 1950-1999); ours uses explicit `<think>` tags for structured reasoning chains grounded in period-specific historical corpus text.

---

## 4. New Generators — Standard (C, F, G)

### Generator C: Reading Comprehension

**Active format cells:** MC, Open-ended, CoT, Passage-based
**Sources:** All (universal)
**Eval alignment:** RACE (primary), BoolQ (secondary)

Unlike Generator A (standalone QA), Generator C includes the source passage in the training example. This trains the model to extract and synthesize from given text.

#### Prompt Template

```
RC_PROMPT = """Read the following passage carefully and create {num_questions}
reading comprehension questions with answers.

Requirements:
1. Questions should require understanding the passage, not just keyword matching
2. Include a mix of question types:
   - Extractive: "According to the passage, what/who/when..."
   - Numerical: "How many..." / "What percentage..."
   - Inferential: "What can be inferred about..."
   - Comparative: "How does X compare to Y in the passage?"
3. Answers must be directly supported by the passage text
4. For numerical questions, show the calculation

Return a JSON object:
{{"rc_pairs": [
  {{"question": "...", "answer": "...", "type": "extractive|numerical|inferential|comparative"}}
]}}

Passage:
{text}"""
```

#### Key Difference from Generator A

The passage is included in the **training example**, not just used for generation:

```json
[
  {"role": "user", "content": "Read the following passage and answer the question.\n\n[passage text]\n\nQuestion: What was the primary cause of the downturn?"},
  {"role": "assistant", "content": "According to the passage, the primary cause was..."}
]
```

#### Format Comparison: RACE / BoolQ vs. Generator C

| | **RACE / BoolQ** (external eval) | **Generator C** (our synthetic data) |
|---|---|---|
| Source | RACE: English exams; BoolQ: Wikipedia passages | Generated from our historical corpus via GPT-4o-mini |
| Format | Passage + MC question / True-False | Passage + MC, open-ended, CoT, or True-False |
| Content | Modern reading comprehension passages | Historical documents from the target period |

**RACE example** (eval format, from `nanochat/tasks/race.py`):
```
User:  Multiple Choice question: Read the following passage and answer the question.

       Passage: The oil embargo imposed by OAPEC in 1973 had far-reaching
       consequences for Western economies...

       What was the primary economic consequence of the 1973 oil embargo?
       - Stagflation across Western economies=A
       - Collapse of the OPEC cartel=B
       - Rapid industrialization=C
       - Shift to renewables=D

       Respond only with the letter of the correct answer.
Asst:  A
```

**BoolQ example** (eval format, from `nanochat/tasks/boolq.py`):
```
User:  Passage: The EEC was established by the Treaty of Rome in 1957...

       Was the EEC established before 1960?
Asst:  True
```

**Generator C example** (our passage-based QA training format):
```
User:  Read the following passage and answer the question.

       Passage: The oil embargo imposed by OAPEC in 1973 had far-reaching
       consequences for Western economies. Crude oil prices quadrupled
       from $3 to $12 per barrel...

       Question: What was the primary economic consequence of the 1973
       oil embargo according to the passage?
Asst:  According to the passage, the primary consequence was stagflation
       across Western economies, driven by the quadrupling of oil prices
       from $3 to $12 per barrel.
```

> **Match:** All include the source passage in the prompt, requiring comprehension of given text.
> **Difference:** RACE/BoolQ passages come from English exams and Wikipedia; ours are real historical corpus documents from the target period.

---

### Generator F: Sentence Completion

**Active format cells:** Fill-blank
**Sources:** All (universal)
**Eval alignment:** HellaSwag (primary), PIQA, WinoGrande (secondary)

Given the beginning of a historical passage, generate a contextually and historically appropriate continuation.

#### Prompt Template

```
COMPLETION_MC_PROMPT = """The following is the beginning of a passage from a
{source_type} published in {year}.

Passage: "{truncated_text}"

Create 4 possible continuations:
- Option A: The actual continuation (historically accurate)
- Options B, C, D: Plausible but incorrect (wrong facts, anachronisms, or stylistic mismatches)

Return JSON:
{{"options": ["actual continuation", "plausible wrong 1", "plausible wrong 2", "plausible wrong 3"],
  "correct": 0,
  "explanation": "Why option A is correct and others are wrong"}}"""
```

#### Implementation Notes

- Truncate source documents at sentence boundaries (50-70% through the passage)
- The actual text continuation is the ground truth (no GPT generation needed for the correct answer)
- Include source type and year metadata for style-appropriate generation

#### Format Comparison: HellaSwag vs. Generator F

| | **HellaSwag** (external eval) | **Generator F** (our synthetic data) |
|---|---|---|
| Source | Zellers et al., ActivityNet captions | Generated from our historical corpus via GPT-4o-mini |
| Format | 4-choice MC sentence completion | 4-choice MC sentence completion (identical) |
| Content | Modern activity descriptions | Historical passages truncated at sentence boundaries |

**HellaSwag example** (eval format, from `nanochat/tasks/hellaswag.py`):
```
User:  Multiple Choice question: A woman is outside with a bucket. She pours
       water from the bucket...
       - ...onto the car and begins scrubbing with a sponge=A
       - ...into a lake nearby=B
       - ...and drinks it quickly=C
       - ...onto a fire to extinguish it=D

       Respond only with the letter of the correct answer.
Asst:  A
```

**Generator F example** (our sentence completion training format):
```
User:  Multiple Choice question: The following is the beginning of a passage
       from The Economist published in 1973:
       "The decision to float the pound sterling marked a turning point in
       British economic policy. The immediate effect was..."
       - a sharp depreciation against the dollar, dropping 10% within weeks=A
       - an unexpected strengthening as markets gained confidence=B
       - a return to the gold standard within six months=C
       - the adoption of the euro as Britain's primary currency=D

       Respond only with the letter of the correct answer.
Asst:  A
```

> **Match:** Both are 4-choice MC sentence completion tasks using `render_mc()` format.
> **Difference:** HellaSwag uses modern activity descriptions; ours uses real historical passages where the correct continuation is the actual corpus text. Wrong options include deliberate anachronisms (e.g., "adoption of the euro" for a 1973 passage).

---

### Generator G: Instruction Following

**Active format cells:** Open-ended, Passage-based
**Sources:** All (universal)
**Eval alignment:** Format quality (no single benchmark; supports general instruction compliance)

Replaces SmolTalk (460K conversations, 32.6% contaminated). Follows the LIMA principle: SFT teaches format, not knowledge.

#### Prompt Template

```
INSTRUCT_PROMPT = """Given the following historical text, generate {num_pairs}
diverse instruction-response pairs. Each pair should use a DIFFERENT instruction
type from this list:

1. Summarize: "Summarize this [article/ruling/report] in [N] bullet points"
2. Explain: "Explain [concept from text] in simple terms"
3. Compare: "Compare [X] and [Y] as described in the text"
4. Extract: "List all [entities/dates/figures] mentioned"
5. Reformat: "Convert this information into a [table/timeline/letter]"
6. Analyze: "What are the strengths and weaknesses of [argument/policy]?"
7. Contextualize: "What historical context is needed to understand this?"
8. Critique: "What are potential counterarguments to the position described?"

Requirements:
- Vary instruction types across pairs (do not repeat the same type)
- Responses should be well-formatted (use bullet points, headers, or structured text)
- Responses should demonstrate instruction compliance (if asked for 3 bullet points, give exactly 3)

Return JSON:
{{"instruction_pairs": [
  {{"instruction_type": "summarize|explain|compare|extract|reformat|analyze|contextualize|critique",
   "user": "The instruction text",
   "assistant": "The response"}}
]}}

Text:
{text}"""
```

#### Format Comparison: SmolTalk / Alpaca vs. Generator G

| | **SmolTalk / Alpaca** (external instruct data) | **Generator G** (our synthetic data) |
|---|---|---|
| Source | Allal et al. / Stanford, web-sourced conversations | Generated from our historical corpus via GPT-4o-mini |
| Format | Open-ended instruction-response pairs | Open-ended instruction-response pairs (identical) |
| Content | Modern general-purpose topics | Historical documents from the target period |

**SmolTalk example** (external instruction dataset, 32.6% temporally contaminated):
```
User:  Summarize the key differences between TCP and UDP protocols.
Asst:  TCP (Transmission Control Protocol) is connection-oriented, ensuring
       reliable delivery through acknowledgments and retransmission.
       UDP (User Datagram Protocol) is connectionless, offering faster
       but unreliable delivery without error checking...
```

**Generator G example** (our instruction-following training format):
```
User:  Summarize this 1973 Economist article on the oil embargo in 3 bullet
       points.
Asst:  - OAPEC imposed an oil embargo on nations supporting Israel in the
         1973 war, targeting the US and Netherlands
       - Crude oil prices quadrupled from $3 to $12 per barrel within months
       - The resulting stagflation forced Western economies to reconsider
         energy dependency on Middle Eastern oil producers
```

> **Match:** Both train instruction compliance: structured responses, format adherence (bullet points, summaries, lists, tables).
> **Difference:** SmolTalk uses modern general-purpose topics (32.6% temporally contaminated for 1950-1999); ours is grounded entirely in period-appropriate historical documents. Generator G is designed to replace SmolTalk.

---

## 5. Priority Generators — Implementation-Ready (D, E, H)

These three generators are the **highest priority** because they produce data that cannot be obtained from any existing source and directly serve our core thesis (temporal isolation).

---

### Generator D: Temporal Reasoning

**Active format cells:** MC, CoT, T/F, Ranking
**Sources:** News (primary), English Corpus (secondary). CaseLaw/Patents are weak fits (avoid).
**Eval alignment:** LAB Eval (primary), Temporal Consistency Tests (primary), LAP Score

**This is the project's unique differentiator.** No existing dataset systematically covers temporal reasoning grounded in a historical corpus. Organized into 3 levels following the TimE hierarchy (Chu et al., arXiv:2505.12891).

#### Level 1 — Basic Temporal (single document)

Questions that test date extraction, duration computation, and simple ordering from a single document.

```
TEMPORAL_L1_PROMPT = """From the following historical document, create {num_questions}
questions that test basic temporal understanding.

For each question, also specify the FORMAT from: mc, cot, true_false, ranking

Required question subtypes (include at least one of each):
- Date/time extraction: "When did [event] happen?" -> format: mc or true_false
- Duration computation: "How long did [period/event] last?" -> format: cot
- Simple ordering: "Did [A] happen before or after [B]?" -> format: true_false or mc

Each question MUST have a definitive answer derivable from the text.

Return JSON:
{{"temporal_qa": [
  {{"question": "...",
    "answer": "...",
    "format": "mc|cot|true_false",
    "level": 1,
    "subtype": "extraction|duration|ordering",
    "mc_options": ["A", "B", "C", "D"]  // only if format=mc, first option is correct
  }}
]}}

Document:
{text}"""
```

**Output example (MC format):**
```json
[
  {"role": "user", "content": "When did the General Agreement on Tariffs and Trade (GATT) come into effect?\nA) 1944\nB) 1947\nC) 1950\nD) 1952"},
  {"role": "assistant", "content": "B"}
]
```

**Output example (T/F format):**
```json
[
  {"role": "user", "content": "True or False: The Bretton Woods agreement was signed before the end of World War II.\n\nAnswer True or False and explain your reasoning."},
  {"role": "assistant", "content": "True. The Bretton Woods agreement was signed in July 1944, while World War II ended in September 1945."}
]
```

#### Level 2 — Cross-Document Reasoning (2 documents, same period)

Questions that require connecting events across two documents from the same period but different sources or years.

```
TEMPORAL_L2_PROMPT = """Given these two passages from the same historical period,
create {num_questions} questions that require connecting information across them.

For each question, specify FORMAT from: cot, ranking

Required question subtypes:
- Contemporaneous: "What was happening in [domain B] while [event A] occurred?"
- Causal: "How might [earlier event in Passage 1] have influenced [later event in Passage 2]?"
- Period characterization: "Based on both passages, what defined the [decade]?"

Passage 1 ({source1}, {year1}):
{text1}

Passage 2 ({source2}, {year2}):
{text2}

Return JSON:
{{"temporal_qa": [
  {{"question": "...",
    "answer": "...",
    "format": "cot|ranking",
    "level": 2,
    "subtype": "contemporaneous|causal|characterization",
    "reasoning": "Step-by-step reasoning connecting the two passages..."
  }}
]}}"""
```

**Output example (CoT format):**
```json
[
  {"role": "user", "content": "The Economist reported rising oil prices in early 1973, while the NYT covered wage negotiations at US automakers the same year. How might the oil price increase have affected the automakers' labor negotiations?"},
  {"role": "assistant", "content": "<think>\nStep 1: Rising oil prices in 1973 increased manufacturing costs for automakers, particularly for fuel-intensive production processes.\nStep 2: Higher costs squeezed profit margins, giving management less room to offer wage increases.\nStep 3: At the same time, rising consumer fuel costs threatened car sales, especially for larger vehicles.\nStep 4: This created a difficult negotiating environment where workers faced inflation but employers faced shrinking margins.\n</think>\nThe oil price increases likely hardened management positions in labor negotiations, as automakers faced simultaneous cost pressures and demand uncertainty."}
]
```

#### Level 3 — Complex Temporal (3+ documents spanning years)

Questions requiring timeline construction, counterfactual reasoning, or trend analysis across multiple documents.

```
TEMPORAL_L3_PROMPT = """Given these passages spanning different years within the
period {start_year}-{end_year}, create {num_questions} complex temporal reasoning
questions.

For each question, specify FORMAT from: cot, ranking

Required question subtypes:
- Timeline: "Arrange these events in chronological order: [events from passages]"
  -> format: ranking
- Counterfactual: "Based on the sequence of events, could [person/entity] have
  known about [later event] when [earlier decision] was made?"
  -> format: cot
- Trend analysis: "What trajectory do these passages reveal about [topic]?"
  -> format: cot

{passages}

Return JSON:
{{"temporal_qa": [
  {{"question": "...",
    "answer": "...",
    "format": "cot|ranking",
    "level": 3,
    "subtype": "timeline|counterfactual|trend",
    "events_referenced": ["event1 (year)", "event2 (year)", ...]
  }}
]}}"""
```

**Output example (Ranking format):**
```json
[
  {"role": "user", "content": "Arrange the following events in chronological order:\n1. Formation of the European Economic Community\n2. The Marshall Plan announcement\n3. The Treaty of Rome\n4. Creation of the OEEC\n\nList them from earliest to latest."},
  {"role": "assistant", "content": "From earliest to latest:\n1. The Marshall Plan announcement (June 1947)\n2. Creation of the OEEC (April 1948)\n3. The Treaty of Rome (March 1957)\n4. Formation of the European Economic Community (January 1958, following the Treaty of Rome)"}
]
```

#### Multi-Document Sampling Logic

Levels 2-3 require sampling multiple documents. The sampling strategy:

```python
# Pseudocode for multi-document sampling
def sample_temporal_set(period_shards, level, sources):
    if level == 2:
        # Two documents, different sources OR different years
        doc1 = sample_one(shards, source=random.choice(sources))
        doc2 = sample_one(shards, source=random.choice(sources),
                          exclude_year=doc1.year)  # ensure date diversity
        return [doc1, doc2]
    elif level == 3:
        # 3-5 documents spanning multiple years
        docs = sample_n(shards, n=random.randint(3, 5),
                        min_year_spread=5)  # at least 5 years between earliest and latest
        return sorted(docs, key=lambda d: d.year)
```

#### Volume Split Across Levels

| Level | % of Generator D Output | Difficulty |
|-------|------------------------|------------|
| Level 1 (single doc) | 50% | Easy — establishes temporal basics |
| Level 2 (cross-doc) | 35% | Medium — teaches cross-referencing |
| Level 3 (multi-doc) | 15% | Hard — timeline and trend reasoning |

#### Format Comparison: TimE Benchmark vs. Generator D

Generator D is **unique to this project**. No existing training dataset systematically covers temporal reasoning grounded in a historical corpus. The closest reference is **TimE** (Chu et al., arXiv:2505.12891), whose 3-level hierarchy we adopt.

| | **TimE** (external benchmark) | **Generator D** (our synthetic data) |
|---|---|---|
| Source | Chu et al., Wikipedia/News/Dialogue | Generated from our historical corpus via GPT-4o-mini |
| Format | Open-ended temporal QA at 3 levels | MC, CoT, T/F, Ranking at 3 levels (identical hierarchy) |
| Content | General temporal facts | Historical events from period-specific corpus documents |

**TimE example** (external benchmark, Level 1 — date extraction):
```
Q:  When was the North Atlantic Treaty signed?
A:  April 4, 1949
```

**TimE example** (external benchmark, Level 2 — cross-document):
```
Q:  What major international organization was established in the decade
    before the Korean War began?
A:  The United Nations (1945), approximately 5 years before the Korean
    War (1950)
```

**Generator D example** (our Level 1, MC format):
```
User:  Multiple Choice question: When did the General Agreement on Tariffs
       and Trade (GATT) come into effect?
       - 1944=A
       - 1947=B
       - 1950=C
       - 1952=D

       Respond only with the letter of the correct answer.
Asst:  B
```

**Generator D example** (our Level 2, CoT format):
```
User:  The Economist reported rising oil prices in early 1973, while
       the NYT covered wage negotiations at US automakers the same
       year. How might the oil price increase have affected the
       automakers' labor negotiations?
Asst:  <think>
       Step 1: Rising oil prices increased manufacturing costs for
               automakers...
       Step 2: Higher costs squeezed profit margins, giving management
               less room for wage increases...
       Step 3: Workers faced inflation but employers faced shrinking
               margins...
       </think>
       The oil price increases likely hardened management positions in
       labor negotiations, as automakers faced simultaneous cost
       pressures and demand uncertainty.
```

**Generator D example** (our Level 3, Ranking format):
```
User:  Arrange the following events in chronological order:
       1. Formation of the European Economic Community
       2. The Marshall Plan announcement
       3. The Treaty of Rome
       4. Creation of the OEEC

       List them from earliest to latest.
Asst:  From earliest to latest:
       1. The Marshall Plan announcement (June 1947)
       2. Creation of the OEEC (April 1948)
       3. The Treaty of Rome (March 1957)
       4. Formation of the EEC (January 1958)
```

> **Match:** Both use a 3-level temporal reasoning hierarchy (basic -> cross-document -> complex).
> **Difference:** TimE tests general temporal knowledge; ours is grounded in specific historical documents from the target period, with questions derivable only from the source text.

---

### Generator E: Corpus-Grounded Quantitative Reasoning

**Active format cells:** Open-ended, CoT, Fill-blank
**Sources:** News (primary — economic data, trade figures, demographics). CaseLaw/Patents/Books are weak fits.
**Eval alignment:** GSM8K (primary)

**Not a GSM8K replacement.** This generator produces quantitative reasoning grounded in real historical numbers. GSM8K (retained as external) handles abstract math. Generator E handles "can the model reason about period-specific quantities?"

#### Step 1: Numeric Document Detection

Not all corpus documents contain useful numerical content. Pre-filter before generation:

```python
import re

def has_sufficient_numbers(text, min_numbers=3):
    """Check if a text chunk contains enough numerical content for math problems."""
    # Match numbers with context (not just isolated digits)
    patterns = [
        r'\$[\d,.]+\s*(million|billion|thousand)?',  # currency
        r'[\d,.]+\s*%',                                # percentages
        r'[\d,.]+\s*(million|billion|thousand|tons|bushels|barrels)',  # quantities
        r'\b\d{1,3}(,\d{3})+\b',                     # large numbers with commas
    ]
    matches = sum(len(re.findall(p, text, re.IGNORECASE)) for p in patterns)
    return matches >= min_numbers
```

#### Step 2: Generation Prompt

```
QUANTITATIVE_PROMPT = """From the following historical document containing numerical
data, create {num_problems} math word problems that require step-by-step calculation.

Requirements:
1. Use ONLY numbers that actually appear in the text (prices, percentages,
   populations, trade volumes, etc.)
2. Each problem should require 2-4 calculation steps
3. Show complete step-by-step solutions
4. For each problem, specify the FORMAT from: open_ended, cot, fill_blank

Problem types to vary across:
- Percentage change: "By what percentage did X change from Y to Z?"
- Compound growth: "If X grew at Y% annually, what was the value after N years?"
- Ratio comparison: "What was the ratio of X to Y?"
- Unit conversion / scaling: "If the per-capita figure was X, what was the total for a population of Y?"
- Difference: "How much more/less was X compared to Y?"
- Weighted average: "Given these figures, what was the weighted average?"

Return JSON:
{{"math_problems": [
  {{"problem": "...",
    "solution": "Step 1: ...\\nStep 2: ...\\nStep 3: ...",
    "answer": "...",
    "format": "open_ended|cot|fill_blank",
    "source_numbers": ["$240 million", "8%", "5 years"]
  }}
]}}

Historical document:
{text}"""
```

#### Output Examples

**CoT format:**
```json
[
  {"role": "user", "content": "According to this 1957 trade report, Country X exported $240 million worth of goods. If exports grew by 8% annually, what was the approximate export value in 1962?"},
  {"role": "assistant", "content": "<think>\nStep 1: We need compound growth over 5 years (1957 to 1962).\nStep 2: Formula: Final = Initial * (1 + rate)^years\nStep 3: Final = $240M * (1.08)^5 = $240M * 1.469 = $352.6M\n</think>\nThe approximate export value in 1962 was $352.6 million."}
]
```

**Fill-blank format:**
```json
[
  {"role": "user", "content": "In 1973, oil prices rose from $3 per barrel to $12 per barrel. This represents a ___% increase."},
  {"role": "assistant", "content": "300% increase. Calculation: ($12 - $3) / $3 * 100 = 300%."}
]
```

#### Step 3: Answer Validation

Unlike other generators, quantitative answers can be verified programmatically:

```python
def validate_math_answer(problem_json):
    """Parse source numbers and verify the answer is arithmetically consistent."""
    source_nums = [parse_number(n) for n in problem_json["source_numbers"]]
    answer_num = parse_number(problem_json["answer"])
    # Check if answer is derivable from source numbers via basic operations
    # (Not a full symbolic solver — just a sanity check for order-of-magnitude correctness)
    return is_plausible(answer_num, source_nums)
```

#### Best Sources for Generator E

| Source | Typical Numerical Content |
|--------|---------------------------|
| **Economist** | GDP, inflation, trade balances, growth rates |
| **FT** | Stock prices, bond yields, corporate financials, market caps |
| **NYT** | Population stats, election results, casualty figures, budgets |
| **Newswire** | Commodity prices, production volumes, bilateral trade |
| **GATT/EurLex** | Tariff rates, regulatory thresholds, quota figures |

#### Format Comparison: GSM8K vs. Generator E

| | **GSM8K** (external eval) | **Generator E** (our synthetic data) |
|---|---|---|
| Source | Cobbe et al., curated math word problems | Generated from our historical corpus via GPT-4o-mini |
| Format | Open-ended with `<<expr=result>>` calculator notation | CoT with `<think>` tags + fill-blank variant |
| Numbers | Invented ("Sally has 5 marbles") | Real historical figures ("1957 exports: $240M") |

**GSM8K example** (eval format, from `nanochat/tasks/gsm8k.py`):
```
User:  Janet's ducks lay 16 eggs per day. She eats three for breakfast
       and bakes muffins with four. She sells the rest for $2 each.
       How much does she make daily?
Asst:  Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
       She makes 9 * 2 = <<9*2=18>>$18 every day.
       #### 18
```

**Generator E example** (our CoT training format):
```
User:  According to this 1957 trade report, Country X exported $240
       million worth of goods. If exports grew by 8% annually, what
       was the approximate export value in 1962?
Asst:  <think>
       Step 1: We need compound growth over 5 years (1957 to 1962).
       Step 2: Formula: Final = Initial * (1 + rate)^years
       Step 3: Final = $240M * (1.08)^5 = $240M * 1.469 = $352.6M
       </think>
       The approximate export value in 1962 was $352.6 million.
```

**Generator E example** (our fill-blank training format):
```
User:  In 1973, oil prices rose from $3 per barrel to $12 per barrel.
       This represents a ___% increase.
Asst:  300% increase. Calculation: ($12 - $3) / $3 * 100 = 300%.
```

> **Match:** Both require multi-step mathematical reasoning with step-by-step solutions.
> **Difference:** GSM8K uses invented numbers and `<<expr=result>>` calculator notation; ours uses real historical figures from corpus documents and `<think>` tags. GSM8K tests abstract math; Generator E tests domain-grounded quantitative reasoning.

---

### Generator H: Anti-Hallucination / Temporal Boundary Enforcement

**Active format cells:** MC, Open-ended, T/F
**Sources:** Source-independent (questions are about post-period events, not corpus text). Corpus knowledge grounds the refusal responses.
**Eval alignment:** LAB Eval (primary), Anti-Hallucination Diagnostic (primary), LAP Score

**Directly supports the core thesis.** This is the only generator that trains the model to say "I don't know" — to respect its temporal knowledge boundary. Two subtypes:

1. **Direct refusal** (70% of output) — Ask about post-period events, expect refusal
2. **Boundary probing** (30% of output) — Ask about topics where partial answers exist within the period

#### Subtype 1: Direct Post-Period Questions

```
ANTI_HALLUC_DIRECT_PROMPT = """Generate {num_pairs} question-answer pairs where
the question asks about events, technologies, or developments that occurred
AFTER the year {end_year}.

For each pair, specify FORMAT from: mc, open_ended, true_false

The correct response must ALWAYS:
1. Acknowledge the question
2. State clearly that the information is beyond the model's knowledge period
3. Offer what WAS known as of {end_year} (if relevant context exists)
4. VARY the refusal phrasing — do not use the same template repeatedly

Topics to cover (vary across pairs):
- Technology invented after {end_year}
- Political events after {end_year}
- Cultural phenomena that emerged after {end_year}
- Scientific discoveries made after {end_year}
- Companies or organizations founded after {end_year}

For MC format: create 4 options where the "correct" answer is
"This event occurred after my knowledge period" or equivalent.

For T/F format: make a statement about a post-period event;
the correct answer is "I cannot verify this" with explanation.

Return JSON:
{{"anti_halluc_pairs": [
  {{"question": "...",
    "answer": "...",
    "format": "mc|open_ended|true_false",
    "event_year": 2005,
    "domain": "technology|politics|culture|science|economics|sports|medicine|space|environment|social"
  }}
]}}"""
```

**Output example (Open-ended format):**
```json
[
  {"role": "user", "content": "What impact did the iPhone have on the telecommunications industry?"},
  {"role": "assistant", "content": "I don't have information about the iPhone, as my knowledge covers the period up to 1999. As of that time, the mobile phone industry was rapidly evolving with Nokia, Motorola, and Ericsson as market leaders. WAP-enabled phones were beginning to bring basic internet access to mobile devices, and the industry was transitioning from analog to digital networks."}
]
```

**Output example (MC format):**
```json
[
  {"role": "user", "content": "Which social media platform, launched in 2004, became the world's largest social network?\nA) MySpace\nB) Facebook\nC) This refers to an event after my knowledge period\nD) Friendster"},
  {"role": "assistant", "content": "C) This question refers to events after my knowledge period, which covers up to 1999. As of 1999, early social networking sites like SixDegrees.com (launched 1997) existed, but the large-scale social media platforms had not yet emerged."}
]
```

**Output example (T/F format):**
```json
[
  {"role": "user", "content": "True or False: The Human Genome Project was completed in 2003."},
  {"role": "assistant", "content": "I cannot verify the completion date, as my knowledge covers the period up to 1999. What I can confirm is that as of 1999, the Human Genome Project was well underway, having been launched in 1990 with an expected completion around 2003. A working draft was anticipated soon, but I cannot speak to whether the project met that timeline."}
]
```

#### Subtype 2: Boundary Probing (Partial Knowledge)

These are harder — the model must provide what it knows AND explicitly flag the limitation.

```
BOUNDARY_PROBE_PROMPT = """Generate {num_pairs} questions that probe the boundary
of knowledge at year {end_year}.

These should be questions where:
- A partial answer exists within the period (ongoing trend, unresolved issue, pending outcome)
- A full/updated answer requires post-{end_year} knowledge
- The model should provide the partial answer AND explicitly note the limitation

The answer should follow this pattern:
"As of {end_year}, [what was known/happening]. [The situation was still developing /
The outcome was not yet determined / Further developments occurred after my
knowledge period]."

For each pair, specify FORMAT from: open_ended, true_false

Example for end_year=1999: Asking about "the outcome of the Kosovo conflict"
(which was active in 1999 but resolved later).

Return JSON:
{{"boundary_pairs": [
  {{"question": "...",
    "answer": "...",
    "format": "open_ended|true_false",
    "context_note": "Brief note on what happened after {end_year} (not shown to model)"
  }}
]}}"""
```

**Output example (boundary probe):**
```json
[
  {"role": "user", "content": "Was the European single currency successful?"},
  {"role": "assistant", "content": "As of 1999, the euro had just been introduced as an accounting currency on January 1, 1999, with 11 EU member states participating. The European Central Bank had begun operations, and exchange rates were fixed. Physical euro banknotes and coins had not yet entered circulation (that was planned for 2002). Early indicators showed market confidence, but it is too early for me to assess the long-term success of the project, as the currency was still in its initial phase."}
]
```

#### Domain Coverage

Cross-reference with LAB Eval's 10 domains to ensure diverse coverage:

| Domain | Direct Refusal Examples | Boundary Probe Examples |
|--------|------------------------|------------------------|
| Politics | Post-period elections, leaders | Ongoing conflicts, pending legislation |
| Technology | Post-period inventions, companies | Emerging technologies under development |
| Science | Post-period discoveries | Ongoing research programs |
| Culture | Post-period media, movements | Trends in development |
| Sports | Post-period records, events | Careers in progress |
| Economics | Post-period crises, companies | Ongoing economic trends |
| Medicine | Post-period treatments, pandemics | Clinical trials in progress |
| Space | Post-period missions, discoveries | Planned missions |
| Environment | Post-period disasters, agreements | Ongoing environmental trends |
| Social | Post-period movements, legislation | Developing social changes |

#### Refusal Phrasing Diversity

Critical: avoid repetitive refusal templates. The prompt explicitly requests variation. Target at least 10 distinct refusal patterns:

1. "I don't have information about [X], as my knowledge covers..."
2. "This falls outside my knowledge period, which extends to..."
3. "[X] occurred after the period I have information about..."
4. "I'm not able to speak to [X] since my training covers up to..."
5. "My knowledge extends to {end_year}, so I cannot address..."
6. "That question refers to developments after my knowledge period..."
7. "I lack information about [X]. What I can tell you about [related pre-period topic] is..."
8. "As of {end_year}, [X] had not yet occurred. The state of [related field] at that time was..."
9. "I don't have reliable information about [X]. Based on what was known up to {end_year}..."
10. "That's beyond what I can speak to. Up to {end_year}, the situation was..."

#### Format Comparison: R-Tuning vs. Generator H

Generator H is **unique to this project**. No existing training dataset systematically teaches temporal knowledge boundaries. The closest academic reference is **R-Tuning** (Zhang et al., NeurIPS 2023), which showed that refusal training significantly reduces hallucination.

| | **R-Tuning** (academic concept) | **Generator H** (our synthetic data) |
|---|---|---|
| Source | Zhang et al., general uncertainty training | Our design, period-boundary enforcement |
| Format | Generic refusal when uncertain | Temporal refusal + pre-period context |
| Content | "I don't know" for unknown facts | "Beyond my knowledge period" + what WAS known |

**R-Tuning concept** (generic refusal training):
```
User:  What is the quantum spin Hall effect?
Asst:  I'm not certain about this topic and don't want to provide
       inaccurate information.
```

**Generator H example** (our open-ended refusal, post-period question):
```
User:  What impact did the iPhone have on the telecommunications industry?
Asst:  I don't have information about the iPhone, as my knowledge
       covers the period up to 1999. As of that time, the mobile phone
       industry was rapidly evolving with Nokia, Motorola, and Ericsson
       as market leaders. WAP-enabled phones were beginning to bring
       basic internet access to mobile devices.
```

**Generator H example** (our boundary probe, partial knowledge):
```
User:  Was the European single currency successful?
Asst:  As of 1999, the euro had just been introduced as an accounting
       currency on January 1, 1999, with 11 EU member states
       participating. Physical euro banknotes and coins had not yet
       entered circulation (planned for 2002). It is too early for me
       to assess the long-term success of the project, as the currency
       was still in its initial phase.
```

> **Match:** Both train the model to refuse gracefully when it lacks knowledge.
> **Difference:** R-Tuning teaches generic uncertainty; ours teaches specific temporal boundaries — the model learns WHERE its knowledge stops (at the period end year) and provides relevant pre-period context in every refusal.

---

## 6. Volume Targets and Data Budget

### Current Data Inventory

**External instruct datasets** (downloaded, from `instruct_dataset_summary_v2.csv`):

| Dataset | Total | LAB Contamination | Post-Filter | Format |
|---------|------:|------------------:|------------:|--------|
| SmolTalk | 460,341 | 32.6% | 310,371 | Multi-turn conversation |
| MMLU | 99,842 | 34.6% | 65,324 | MC (4 choices) |
| HotpotQA | 90,447 | 56.6% | 39,227 | Open-ended multi-hop |
| HellaSwag | 39,905 | 20.0% | 31,933 | Sentence completion MC |
| MuSiQue | 19,938 | 41.4% | 11,679 | Open-ended + decomposition |
| PIQA | 16,113 | 10.4% | 14,441 | Commonsense MC |
| CodeContests | 13,134 | 40.8% | 7,779 | Code generation |
| CommonsenseQA | 9,741 | 2.0% | 9,543 | MC (5 choices) |
| WinoGrande | 9,248 | 3.0% | 8,973 | Fill-blank (2 choices) |
| GSM8K | 7,473 | 2.5% | 7,285 | Math + step-by-step |
| MATH | 7,500 | 0.3% | 7,477 | Math + LaTeX solution |
| LogiQA | 7,376 | 41.4% | 4,319 | Logic MC |
| ScienceQA | 5,837 | 2.2% | 5,706 | MC + explanation |
| AIME/AMC | 4,069 | 0.6% | 4,043 | MC math + CoT |
| StrategyQA | 2,290 | 31.4% | 1,571 | Yes/No + decomposition |
| ARC-Easy | 2,251 | 1.7% | 2,213 | MC (4 choices) |
| ARC-Challenge | 1,119 | 1.6% | 1,101 | MC (4 choices) |
| FOLIO | 1,001 | 28.4% | 717 | T/F/Uncertain |
| MBPP | 374 | 17.4% | 309 | Code generation |
| HumanEval | 164 | 1.8% | 161 | Code generation |
| **TOTAL** | **798,163** | — | **~534,171** | — |

**Corpus-generated QA** (existing, 1950-1999 only):
- 348,255 QA + CoT pairs from Generators A+B
- 16 collections (Economist, NYT, FT, Newswire, Caselaw, USPTO, GATT, EurLex, Books, etc.)
- Stored at `{period}/posttraining_data/hist_corpus_qa_{period}.jsonl`

### Projected Volumes (Post New Generators)

**Per-period generation targets:**

| Phase | Target | Source |
|-------|-------:|--------|
| Mid-training | 500K | Generators A-H (corpus-derived) + GSM8K/MATH (external) |
| SFT | 75K | Generators A-H (corpus-derived) + GSM8K/MATH (external) |
| **Per-period total** | **575K** | |

**Mix ratios and projected counts (per period):**

| Generator | Mid % | Mid Count | SFT % | SFT Count | Total |
|-----------|------:|----------:|------:|----------:|------:|
| A. Factual | 30% | 150,000 | 15% | 11,250 | 161,250 |
| B. Chain-of-Thought | 15% | 75,000 | 25% | 18,750 | 93,750 |
| C. Reading Comprehension | 15% | 75,000 | 10% | 7,500 | 82,500 |
| D. Temporal Reasoning | 10% | 50,000 | 15% | 11,250 | 61,250 |
| E. Quantitative | 5% | 25,000 | 5% | 3,750 | 28,750 |
| F. Sentence Completion | 5% | 25,000 | 5% | 3,750 | 28,750 |
| G. Instruction Following | 5% | 25,000 | 15% | 11,250 | 36,250 |
| H. Anti-Hallucination | 5% | 25,000 | 5% | 3,750 | 28,750 |
| GSM8K (external) | 5% | 7,285 | 3% | 2,250 | 9,535 |
| MATH (external) | 5% | 7,477 | 2% | 1,500 | 8,977 |
| **Total** | | **~464K** | | **~75K** | **~539K** |

Note: GSM8K/MATH are capped at their actual dataset size (7,285 and 7,477 after LAB filter).

**Across all 6 periods:**

| | Per Period | x 6 Periods | Notes |
|---|----------:|------------:|-------|
| Corpus-derived synthetic | ~525K | ~3.15M | New generation required |
| External retained (GSM8K + MATH) | ~15K | ~90K | Same dataset, reused |
| **Grand total** | **~540K** | **~3.24M** | |

**API cost estimate:**
- At ~$5 per 1,000 docs (GPT-4o-mini), ~$50-100 per 100K examples
- Total: ~$1,500-3,000 across all 6 periods
- Timeline: ~1-2 weeks of generation at 50 concurrent workers

### Source Diversity per Generator

| Source Type | Collections | Best For |
|-------------|-------------|----------|
| News | Economist, NYT, FT, Newswire, US/French newspapers | A, B, C, D, E, G |
| Legal | Caselaw, USPTO, GATT, EurLex, Eurovoc | A, B, C, G |
| Academic | Books, Science Pile, Open Science Pile, OpenAlex | A, B, C, G |
| Mixed | All of the above | F, H |
| External | GSM8K, MATH (pre-existing datasets) | Math reasoning |

---

## 7. Quality Control on Synthetic Data

### Why Filter Your Own Generated Data?

Even when using GPT-4-class models as generators, post-generation quality filtering is **strongly supported** by the academic literature. Every major synthetic data paper applies it:

| Paper | What They Did | Filter Rate | Impact |
|-------|--------------|-------------|--------|
| **AlpaGasus** (arXiv:2307.08701) | ChatGPT Likert scoring on Alpaca 52K | 83% filtered | Outperforms full Alpaca; 5.7x faster training |
| **Superfiltering** (arXiv:2402.00530) | IFD scoring on Alpaca 52K | 85-95% filtered | Matches full-data performance at 5-15% of data |
| **Source2Synth** (arXiv:2409.08239) | Answerability check on generated QA | 13-73% filtered | +8-10 accuracy points |
| **Cosmopedia** (HuggingFace, 2024) | Topic scoring + MinHash dedup + n-gram decontam | 23% clusters dropped | 25B clean tokens from 30M docs |
| **Self-Instruct** (arXiv:2212.10560) | ROUGE-L dedup + heuristic filters | ~40% filtered | 52K clean from larger pool |
| **WizardLM** (arXiv:2304.12244) | Evolution failure filter | Varies | Removes echoed/refused outputs |
| **Alpaca** (Stanford, no filter) | **None** | 0% | Community found major quality issues |

**Key finding:** Aggressive quality filtering (keeping 5-17% of synthetic data) consistently **outperforms or matches** training on the full unfiltered set.

### What Goes Wrong Without Filtering?

Even with GPT-4o-mini generating from grounded corpus text, these issues appear:

1. **Malformed JSON** — GPT-4o-mini is more error-prone than GPT-4 on structured output
2. **Self-referential responses** — "As an AI assistant, I can tell you that..."
3. **Near-duplicates** — Multiple QA pairs from the same document often paraphrase each other
4. **Parametric knowledge injection** — GPT-4o-mini's own knowledge leaking into answers beyond what the source document contains (Sarkar & Vafa, SSRN:4754678)
5. **Trivial questions** — Yes/no or single-word answers with no training signal
6. **Unanswerable questions** — Questions that cannot be answered from the source passage

### Our Quality Pipeline (6 stages)

For **external datasets** (GSM8K, MATH): LAB temporal filter only (already implemented in `filter.py`).

For **our synthetic data** (Generators A-H):

| Stage | What | Why | Tool |
|-------|------|-----|------|
| 1. Format validation | Parse JSON, check required fields, reject malformed | GPT-4o-mini ~5-10% malformed rate | Script |
| 2. Self-referential filter | Remove "As an AI...", "I cannot...", short refusals | Known GPT-4o-mini behavior | Regex |
| 3. Deduplication | MinHash LSH or ROUGE-L >= 0.7 on questions | LLMs produce near-identical questions per doc | MinHash |
| 4. Grounding verification | Check answer entities/dates/claims appear in source passage | Replaces LAB; catches parametric injection | Model-based |
| 5. N-gram decontamination | 10-gram overlap against eval sets | Prevents inflated eval scores | `difflib` |
| 6. Trivial question filter | Remove answers <5 tokens, yes/no without justification | No training signal | Heuristic |

**Expected filter rates:** Based on Source2Synth and AlpaGasus, expect 15-30% of generated data to be filtered. This is built into the volume targets above (we generate ~540K to end up with ~400K+ clean examples per period).

Full pipeline details: see `04_QUALITY_CONTROL_PIPELINE.md`.

---

## 8. Prompt Engineering Principles

### Shared Across All Generators

1. **JSON response format** — All prompts request structured JSON. Use `response_format={"type": "json_object"}` in API calls.
2. **Temperature 0.7** — Balances diversity with coherence (from `synth_config.yaml`).
3. **Chunk size 6,000 chars / 300 overlap** — Sufficient context without exceeding token limits.
4. **Max tokens 4,096** — Allows detailed responses with reasoning chains.
5. **Model: GPT-4o-mini** — Cost-optimized for high volume. Consider GPT-4o for D and H (quality-critical).
6. **Format metadata** — Every generated example includes its format type for downstream mixing.

### Anti-Pattern Avoidance

- **Do not** ask for trivia or surface-level factual recall
- **Do not** generate yes/no questions without requiring justification
- **Do not** allow questions answerable without the source text (for C, E)
- **Do not** generate content that references the generation process itself
- **Do not** force generator/source combinations that are weak fits (see Section 1c)

---

## 9. Implementation Plan

### Phase 1: Extend Existing Infrastructure (C, G + format variants for A, B)

Lowest risk — add new prompts to `run_direct.py`:

1. Add `RC_PROMPT`, `INSTRUCT_PROMPT` alongside existing `QA_PROMPT`, `COT_PROMPT`
2. Add MC and T/F format variants for Generator A output
3. Add `--generator` CLI flag to select which generators to run
4. Output to separate JSONL files per generator type

**Effort:** 1-2 days | **API cost:** ~$50-100 per period

### Phase 2: Priority Generators (D, E, H)

These require new logic beyond simple prompt addition:

| Generator | New Logic Required |
|-----------|-------------------|
| D | Multi-document sampling (Levels 2-3), document pairing by year/source |
| E | Numeric document pre-filtering, answer validation |
| H | Post-period event generation (no corpus input), refusal phrasing diversity |

Implementation:
1. `generators/temporal.py` — multi-document sampler + 3 level prompts
2. `generators/quantitative.py` — numeric filter + generation + validation
3. `generators/anti_hallucination.py` — event generator + refusal trainer
4. `run_matrix.py` — orchestrator that runs generator x format x source

**Effort:** 5-7 days | **API cost:** ~$100-150 per period

### Phase 3: Completion Generator + Evol-Instruct (F + complexity scaling)

1. Generator F requires passage truncation logic
2. Evol-Instruct post-processing on 10-20% of all generated data

**Effort:** 2-3 days | **API cost:** ~$30-50 per period

### Phase 4: Quality Calibration

After initial generation, run the full quality pipeline (see `04_QUALITY_CONTROL_PIPELINE.md`):
1. Format validation across all generator outputs
2. Cross-generator deduplication
3. Difficulty scoring and rebalancing
4. Final LAB temporal filter pass

### Priority and Dependency Ordering

```
Phase 1 (C, G, A/B variants)  --->  Phase 3 (F, Evol-Instruct)
                                          |
Phase 2 (D, E, H)  --------->  Phase 4 (Quality Pipeline)
```

Phase 1 and Phase 2 can run in parallel. Phase 4 requires all generation to be complete.

---

## 10. Evol-Instruct Complexity Scaling (Post-Generation)

After initial generation, apply Evol-Instruct techniques (Xu et al., arXiv:2304.12244) to increase difficulty on 10-20% of generated data:

### In-Depth Evolving

| Strategy | Example Application |
|----------|---------------------|
| Add constraints | "Answer using only information from the first paragraph" |
| Deepening | "Explain the underlying economic theory, not just the outcome" |
| Concretizing | "Give specific dates and figures in your answer" |
| Increase reasoning | "Break your answer into at least 5 logical steps" |
| Complicate input | Add misleading details; test if model can filter |

### In-Breadth Evolving

Mutate topics across domains: take a political question, generate an analogous economic question from the same period.

Store evolution level as metadata for difficulty-aware training.

---

## References

- Wei et al. (2022). "Finetuned Language Models are Zero-Shot Learners." ICLR 2022 (FLAN)
- Xu et al. (2023). "WizardLM: Empowering Large Language Models to Follow Complex Instructions." arXiv:2304.12244 (Evol-Instruct)
- Abdin et al. (2024). "Phi-4 Technical Report." arXiv:2412.08905 (50 synthetic dataset types)
- Mukherjee et al. (2023). "Orca: Progressive Learning from Complex Explanation Traces." arXiv:2306.02707 (format diversity)
- Yang et al. (2025). "DOTS: Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search." ICLR 2025
- Gunasekar et al. (2023). "Textbooks Are All You Need." arXiv:2306.11644 (data quality > quantity)
- Zhou et al. (2023). "LIMA: Less Is More for Alignment." NeurIPS 2023
- Chu et al. (2025). "TimE: A Multi-Level Temporal Reasoning Benchmark." arXiv:2505.12891
- Gekhman et al. (2024). "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?" EMNLP 2024
- Wang et al. (2022). "Self-Instruct: Aligning Language Models with Self-Generated Instructions." arXiv:2212.10560
