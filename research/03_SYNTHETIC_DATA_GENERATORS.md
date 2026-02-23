# Synthetic Data Generators

> **Paper section:** 3.2 (Synthetic Data Generation)
> **Dependencies:** `src/post_training/corpus/run_direct.py`, `synth_config.yaml`, `config.py`
> **Status:** Generators A-B implemented; C-H proposed; GSM8K/MATH retained as external

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

```
                        MMLU  ARC-C  GSM8K  Hella  BoolQ  PIQA  Wino  RACE  LAB   LAP   Temp   Anti-H
                                                   Swag                       Eval        Consist Diag
────────────────────────────────────────────────────────────────────────────────────────────────────────
A. Factual               **    **                   *                               .
B. Reasoning                    *                                           **
C. Comprehension                                     *                      **
D. Temporal                                                                       **    **     **
E. Quantitative                        **
F. Completion                                 **            *     *
G. Instruction                                                              .
H. Anti-Halluc                                                                    **           **     **
External: GSM8K                        **
External: MATH                         *
```

`**` = primary alignment (generator specifically targets this eval)
`*` = secondary alignment (generator contributes to this capability)
`.` = indirect benefit

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

---

## 6. Volume Targets and Mixing Ratios

### Per-Period Targets

| Phase | Total Examples | Purpose |
|-------|---------------|---------|
| Mid-training | 500K-1M | Heavy on factual QA + reading comprehension |
| SFT | 50K-100K | Heavy on CoT + instruction following (LIMA principle) |

### Mix Ratios

| Generator | Mid-Training % | SFT % | Rationale |
|-----------|---------------|-------|-----------|
| A. Factual MC | 30% | 15% | Core domain knowledge, high volume |
| B. Chain-of-Thought | 15% | 25% | Reasoning capability, critical for SFT |
| C. Reading Comprehension | 15% | 10% | Passage understanding |
| D. Temporal Reasoning | 10% | 15% | **Unique differentiator** |
| E. Quantitative | 5% | 5% | Corpus-grounded numbers (GSM8K external handles abstract math) |
| F. Sentence Completion | 5% | 5% | Language modeling signal |
| G. Instruction Following | 5% | 15% | Format learning (LIMA) |
| H. Anti-Hallucination | 5% | 5% | **Temporal boundary enforcement** |
| GSM8K (external) | 5% | 3% | Pure math reasoning |
| MATH (external) | 5% | 2% | Advanced math reasoning |

### Source Diversity per Generator

| Source Type | Collections | Best For |
|-------------|-------------|----------|
| News | Economist, NYT, FT, Newswire, US/French newspapers | A, B, C, D, E, G |
| Legal | Caselaw, USPTO, GATT, EurLex, Eurovoc | A, B, C, G |
| Academic | Books, Science Pile, Open Science Pile, OpenAlex | A, B, C, G |
| Mixed | All of the above | F, H |
| External | GSM8K, MATH (pre-existing datasets) | Math reasoning |

---

## 7. Prompt Engineering Principles

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

## 8. Implementation Plan

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

## 9. Evol-Instruct Complexity Scaling (Post-Generation)

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
