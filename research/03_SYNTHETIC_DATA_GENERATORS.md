# Synthetic Data Generators

> **Paper section:** 3.2 (Synthetic Data Generation)
> **Dependencies:** `src/post_training/corpus/run_direct.py`, `synth_config.yaml`, `config.py`
> **Status:** Generators A-B implemented; C-H proposed

---

## 1. Design Philosophy

### Why Synthetic-Only?

Our initial pipeline mixed corpus-derived QA with external instruction datasets (SmolTalk, MMLU, ARC, GSM8K, etc.). This created two problems:

1. **Temporal contamination** — External datasets contain post-period knowledge. Even after LAB filtering, residual leakage is hard to eliminate entirely (e.g., SmolTalk had 32.6% contamination, MMLU had 34.6%).
2. **Domain mismatch** — External datasets teach capabilities (code generation, modern science) that are anachronistic for historical periods.

Going fully synthetic from our own corpus eliminates both problems. The model learns only from period-appropriate material, and we control every aspect of data quality and format.

### Why 8 Generator Types?

Each generator targets a specific capability. Together they replace every external dataset while adding new capabilities (temporal reasoning, anti-hallucination) that no external dataset provides.

| Generator | Target Capability | Replaces |
|-----------|-------------------|----------|
| A. Factual MC | Knowledge recall + discrimination | MMLU, ARC |
| B. Chain-of-Thought | Multi-step reasoning | LogiQA, StrategyQA, FOLIO |
| C. Reading Comprehension | Passage understanding + extraction | DROP, SQuAD-style |
| D. Temporal Reasoning | Chronology, causation, ordering | Nothing (new) |
| E. Math/Quantitative | Numerical reasoning from corpus data | GSM8K, AIME/AMC |
| F. Sentence Completion | Language modeling, period-appropriate style | HellaSwag-style |
| G. Instruction Following | Format compliance, summarization, comparison | SmolTalk |
| H. Anti-Hallucination | Temporal boundary enforcement, refusal | Nothing (new) |

### Capability Coverage Matrix

```
                        Factual  Reasoning  Comprehension  Temporal  Quant  Style  Format  Safety
Generator A (MC)          +++                    +
Generator B (CoT)                  +++           +
Generator C (Reading)       +        +          +++
Generator D (Temporal)      +        +                       +++
Generator E (Math)                   ++                               +++
Generator F (Completion)    +                    +                     +++
Generator G (Instruct)                           +                           +++
Generator H (Anti-Halluc)                                    ++                      +++
```

---

## 2. Existing Generators (Implemented)

### Generator A: Factual QA

**Source:** `src/post_training/corpus/run_direct.py`
**Status:** Production. Generated 348K+ pairs for 1950-1999 period.

#### Prompt Template (from `run_direct.py`)

```
QA_PROMPT = """Create {num_pairs} question-answer pairs from this text for LLM training.

Rules:
1. Questions must require analytical thinking, not just fact lookup
2. Answers must be directly supported by the text
3. Vary question types: cause-effect, comparison, analysis, inference, summary
4. Return a JSON object with key "qa_pairs" containing an array:

{{"qa_pairs": [{{"question": "Question 1?", "answer": "Answer 1."}},
               {{"question": "Question 2?", "answer": "Answer 2."}}]}}

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

#### Output Format (nanochat CustomJSON)

```json
[
  {"role": "user", "content": "What rationale did the court provide for ruling that the jury's verdict should not be disturbed?"},
  {"role": "assistant", "content": "The court reasoned that the jury's verdict, which awarded the property owners $267,083.33, was within the range of testimony provided by both parties..."}
]
```

#### Pipeline

```
Corpus parquet → chunk_text(6000, 300) → GPT-4o-mini → JSON parse → JSONL
```

Concurrency: ThreadPoolExecutor with 50 workers (configurable). Rate: ~5-10x faster than subprocess-based approach.

#### Current Volume (1950-1999)

- 348,255 QA pairs from Economist, NYT, FT, Newswire, Caselaw, USPTO, GATT, EurLex, Books, etc.
- Stored at: `{period}/posttraining_data/hist_corpus_qa_{period}.jsonl`

#### Strengths and Limitations

**Strengths:** High volume, covers all source types, analytically focused questions.
**Limitations:** No multiple-choice format (open-ended only), no difficulty stratification, question types not explicitly controlled.

---

### Generator B: Chain-of-Thought Reasoning

**Source:** `src/post_training/corpus/run_direct.py` (COT_PROMPT)
**Status:** Production. Runs alongside Generator A.

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
  {"role": "assistant", "content": "<think>\nStep 1: The trade deficit widened from $X to $Y between 1975-1980...\nStep 2: This put downward pressure on the currency because...\nStep 3: The central bank's reserves were insufficient to...\n</think>\nThe trade deficit directly contributed to the devaluation by..."}
]
```

#### Configuration

| Parameter | Value |
|-----------|-------|
| CoT examples per chunk | 2 |
| Same chunking as QA | 6,000 chars / 300 overlap |
| `<think>` tag wrapping | Applied in `run_direct.py` lines 284-289 |

---

## 3. New Generators (Proposed)

### Generator C: Reading Comprehension

**Replaces:** DROP, SQuAD-style capabilities
**Priority:** High — teaches passage-level understanding beyond fact extraction

#### Design

Unlike Generator A (which generates standalone QA), Generator C includes the source passage in the prompt, training the model to extract and synthesize information from given text.

#### Prompt Template (Proposed)

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

#### Output Format

```json
[
  {"role": "user", "content": "Read the following passage and answer the question.\n\n[passage text]\n\nQuestion: According to the report, what was the primary cause of the economic downturn in 1973?"},
  {"role": "assistant", "content": "According to the passage, the primary cause of the economic downturn in 1973 was the oil embargo imposed by OPEC nations, which quadrupled crude oil prices from $3 to $12 per barrel within months..."}
]
```

#### Implementation

- **Extend:** `run_direct.py` — add `RC_PROMPT` alongside existing QA_PROMPT and COT_PROMPT
- **Input:** Same corpus chunks (6,000 chars)
- **Key difference from Generator A:** The passage is included in the training example, not just used for generation
- **Volume target:** 2 RC pairs per chunk (in addition to 3 QA + 2 CoT)

---

### Generator D: Temporal Reasoning

**Replaces:** Nothing — this is entirely new and critical for our project
**Priority:** Very High — unique differentiator for historical domain

#### Design

Generates questions that require understanding chronological relationships, causal chains across time, duration computation, and event ordering. This is where our project can contribute something no existing benchmark covers well.

Following the TimE benchmark's 3-level hierarchy (arXiv:2505.12891):

| Level | Subtask | Example |
|-------|---------|---------|
| 1 — Basic | Date extraction | "When did [event] occur?" |
| 1 — Basic | Duration computation | "How many years between [X] and [Y]?" |
| 1 — Basic | Order comparison | "Which happened first: [A] or [B]?" |
| 2 — Expression | Explicit reasoning | "What was happening in [country] at the time of [event]?" |
| 2 — Expression | Relative reasoning | "In the decade before [event], what trend..." |
| 3 — Complex | Co-temporality | "What else was occurring when [event] happened?" |
| 3 — Complex | Timeline ordering | "Arrange these events chronologically: [A], [B], [C], [D]" |
| 3 — Complex | Counterfactual | "Could [person] have witnessed [event]?" |

#### Prompt Templates (Proposed)

**Level 1 — Basic Temporal:**
```
TEMPORAL_L1_PROMPT = """From the following historical text, create {num_questions}
questions that test basic temporal understanding.

Question types to include:
- Date/time extraction: "When did [event] happen?"
- Duration computation: "How long did [period/event] last?"
- Simple ordering: "Did [A] happen before or after [B]?"

Each question must have a definitive answer derivable from the text.

Return JSON:
{{"temporal_qa": [
  {{"question": "...", "answer": "...", "level": 1, "subtype": "extraction|duration|ordering"}}
]}}

Text:
{text}"""
```

**Level 2 — Expression Reasoning (requires 2 document chunks):**
```
TEMPORAL_L2_PROMPT = """Given these two passages from the same historical period,
create {num_questions} questions that require connecting events across them.

Question types:
- Contemporaneous events: "What was happening in [domain B] while [event A] occurred?"
- Cause-effect across time: "How might [earlier event] have influenced [later event]?"
- Period characterization: "Based on both passages, what defined the [decade]?"

Passage 1 ({source1}):
{text1}

Passage 2 ({source2}):
{text2}

Return JSON:
{{"temporal_qa": [
  {{"question": "...", "answer": "...", "level": 2, "subtype": "contemporaneous|causal|characterization"}}
]}}"""
```

**Level 3 — Complex Temporal (requires 3+ documents):**
```
TEMPORAL_L3_PROMPT = """Given these passages spanning different years within the
same historical period, create {num_questions} complex temporal reasoning questions.

Question types:
- Timeline construction: "Arrange these events in chronological order: [list events from passages]"
- Counterfactual: "Based on the sequence of events, could [person/entity] have known about [later event]?"
- Trend analysis: "What trajectory do these passages reveal about [topic] over the period?"

{passages}

Return JSON:
{{"temporal_qa": [
  {{"question": "...", "answer": "...", "level": 3, "subtype": "timeline|counterfactual|trend",
   "events_referenced": ["event1", "event2", ...]}}
]}}"""
```

#### Implementation Notes

- Level 1 uses single documents (same as Generators A/B)
- Levels 2-3 require **multi-document sampling** — select 2-3 documents from the same period but different sources or years
- Multi-document sampling logic: sample from the same period's parquet shards, ensuring date diversity
- **Volume target:** 15% of total mix (see Section 5)

---

### Generator E: Math/Quantitative Reasoning

**Replaces:** GSM8K, AIME/AMC
**Priority:** Medium — maintains quantitative reasoning capability

#### Design

Extracts real numbers from historical documents (economic data, statistics, trade figures, population counts, financial data) and generates word problems around them. This keeps math grounded in period-appropriate data rather than using anachronistic modern datasets.

#### Prompt Template (Proposed)

```
MATH_PROMPT = """From the following historical text containing numerical data,
create {num_problems} math word problems that require step-by-step calculation.

Requirements:
1. Use actual numbers from the text (prices, percentages, populations, etc.)
2. Problems should require 2-4 calculation steps
3. Show complete step-by-step solutions
4. Problem types: percentage change, compound growth, ratio comparison,
   unit conversion, weighted average, difference calculation

Return JSON:
{{"math_problems": [
  {{"problem": "In 1957, exports totaled $X million. If exports grew by Y% annually...",
   "solution": "Step 1: Calculate annual growth...\nStep 2: Apply to base...\nStep 3: ...",
   "answer": "The export value in 1962 was approximately $Z million.",
   "source_numbers": ["X million", "Y%"]}}
]}}

Text:
{text}"""
```

#### Output Format

```json
[
  {"role": "user", "content": "In 1957, Country X exported $240 million worth of goods. If exports grew by 8% annually, what was the approximate export value in 1962?"},
  {"role": "assistant", "content": "<think>\nStep 1: We need to calculate compound growth over 5 years (1957 to 1962).\nStep 2: Using the compound growth formula: Final = Initial × (1 + rate)^years\nStep 3: Final = $240M × (1.08)^5 = $240M × 1.469 = $352.6M\n</think>\nThe approximate export value in 1962 was $352.6 million."}
]
```

#### Source Selection

Best sources for numerical content:
- **Economist** — Economic data, GDP, trade figures, inflation rates
- **FT (Financial Times)** — Stock prices, bond yields, corporate financials
- **Newswire** — Population stats, election results, casualty figures
- **USPTO** — Patent statistics, filing numbers
- **GATT/EurLex** — Trade tariff rates, regulatory thresholds

#### Implementation Notes

- Pre-filter corpus chunks: only send chunks containing 3+ numeric tokens to the math generator
- Validate answers by running a simple arithmetic check (parse numbers, verify final answer is computable)
- **Volume target:** 10% of total mix

---

### Generator F: Sentence Completion

**Replaces:** HellaSwag-style capabilities
**Priority:** Medium — trains language modeling and period-appropriate style

#### Design

Given the beginning of a historical passage, the model must produce a contextually and historically appropriate continuation. Also generates MC variant with plausible but incorrect alternatives.

#### Prompt Templates (Proposed)

**Open-ended completion:**
```
COMPLETION_PROMPT = """The following is the beginning of a passage from a
{source_type} published in {year}. Generate a historically accurate and
stylistically appropriate continuation of approximately 2-3 sentences.

Passage beginning:
"{truncated_text}"

Return JSON:
{{"continuation": "The continuation text...",
  "style_notes": "Brief note on why this continuation fits the period and source type"}}"""
```

**MC variant:**
```
COMPLETION_MC_PROMPT = """The following is the beginning of a passage from a
{source_type} published in {year}.

Passage: "{truncated_text}"

Create 4 possible continuations:
- Option A: The actual continuation (historically accurate)
- Options B, C, D: Plausible but incorrect continuations (wrong facts, anachronisms, or stylistic mismatches)

Return JSON:
{{"options": ["actual continuation", "plausible wrong 1", "plausible wrong 2", "plausible wrong 3"],
  "correct": 0,
  "explanation": "Why option A is correct and others are wrong"}}"""
```

#### Implementation Notes

- Truncate source documents at sentence boundaries (50-70% through the passage)
- Include source type and year metadata for style-appropriate generation
- The actual text continuation serves as the ground truth (no GPT generation needed for the correct answer — just for distractors)
- **Volume target:** 5% of total mix

---

### Generator G: Instruction Following

**Replaces:** SmolTalk (460K conversations, 32.6% contaminated)
**Priority:** High — teaches format and instruction compliance

#### Design

Generates diverse instruction-response pairs grounded in the historical corpus. Follows the LIMA principle: SFT teaches format, not knowledge (see `../context_free_qa_writeup.md`).

#### Prompt Template (Proposed)

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
- Responses should be well-formatted (use bullet points, headers, or structured text where appropriate)
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

#### Output Format

```json
[
  {"role": "user", "content": "Summarize the following trade agreement developments in exactly 3 bullet points:\n\n[passage]"},
  {"role": "assistant", "content": "- The 1967 Kennedy Round reduced industrial tariffs by an average of 35% across participating nations\n- Developing countries secured special provisions exempting them from full reciprocity requirements\n- The agreement established a new framework for resolving anti-dumping disputes through multilateral consultation"}
]
```

#### Implementation Notes

- Explicitly randomize instruction types per batch to ensure diversity
- Include formatting constraints (numbered lists, tables, word limits) to train format compliance
- Generate multi-turn conversations for 20% of examples (follow-up questions)
- **Volume target:** 10% of total mix

---

### Generator H: Anti-Hallucination / Temporal Boundary Enforcement

**Replaces:** Nothing — entirely new, critical for temporal isolation
**Priority:** Very High — directly supports the project's core thesis

#### Design

Generates questions about events that occurred AFTER the training period's end year. The correct response is refusal or hedging. This teaches the model to respect its knowledge boundaries.

Two subtypes:
1. **Direct post-period questions** — Ask about known post-period events
2. **Ambiguous boundary questions** — Ask about topics where the answer changed after the period

#### Prompt Templates (Proposed)

**Subtype 1 — Direct post-period questions:**
```
ANTI_HALLUC_PROMPT = """Generate {num_pairs} question-answer pairs where the
question asks about events, technologies, or developments that occurred AFTER
the year {end_year}.

The correct answer should ALWAYS be a polite refusal or hedging response that:
1. Acknowledges the question
2. States that the information is beyond the model's knowledge period
3. Optionally offers what WAS known as of {end_year}

Topics to ask about (vary across pairs):
- Technology invented after {end_year}
- Political events after {end_year}
- Cultural phenomena that emerged after {end_year}
- Scientific discoveries made after {end_year}
- Companies or organizations founded after {end_year}

Return JSON:
{{"anti_halluc_pairs": [
  {{"question": "What was the impact of [post-period event]?",
   "answer": "I don't have information about [post-period event]. My knowledge covers the period up to {end_year}. As of that time, [relevant context if any].",
   "event_year": 2005,
   "domain": "technology"}}
]}}"""
```

**Subtype 2 — Boundary probing:**
```
BOUNDARY_PROBE_PROMPT = """Generate {num_pairs} questions that probe the boundary
of knowledge at year {end_year}.

These should be questions where:
- A partial answer exists within the period (e.g., an ongoing trend, an unresolved issue)
- A full/updated answer requires post-{end_year} knowledge
- The model should provide the partial answer and explicitly note the limitation

Example: For end_year=1999, asking about "the outcome of the Kosovo conflict"
(which was ongoing in 1999 but resolved in later years).

Return JSON:
{{"boundary_pairs": [
  {{"question": "...",
   "answer": "As of {end_year}, [what was known]. The situation was still developing and I cannot speak to subsequent developments.",
   "context": "Brief note on what happened after {end_year}"}}
]}}"""
```

#### Output Format

```json
[
  {"role": "user", "content": "What impact did the iPhone have on the telecommunications industry?"},
  {"role": "assistant", "content": "I don't have information about the iPhone, as my knowledge covers the period up to 1999. As of that time, the mobile phone industry was rapidly evolving with companies like Nokia, Motorola, and Ericsson leading the market. The introduction of WAP-enabled phones was beginning to bring internet access to mobile devices."}
]
```

#### Implementation Notes

- Use the same LAB question generation approach as `generate_lab_questions.py` but produce full training conversations instead of MC eval questions
- Cross-reference with LAB eval domains (10 domains: politics, technology, science, culture, sports, economics, medicine, space, environment, social movements)
- Ensure variety in refusal phrasing — avoid repetitive templates
- Balance: 70% direct refusals, 30% boundary probing (partial answers)
- **Volume target:** 5% of total mix

---

## 4. Volume Targets and Mixing Ratios

### Per-Period Targets

| Phase | Total Examples | Purpose |
|-------|---------------|---------|
| Mid-training | 500K–1M | Heavy on factual QA + reading comprehension |
| SFT | 50K–100K | Heavy on CoT + instruction following (LIMA principle) |

### Mix Ratios

| Generator | Mid-Training % | SFT % | Rationale |
|-----------|---------------|-------|-----------|
| A. Factual MC | 35% | 20% | Core domain knowledge, high volume |
| B. Chain-of-Thought | 15% | 25% | Reasoning capability, critical for SFT |
| C. Reading Comprehension | 15% | 10% | Passage understanding |
| D. Temporal Reasoning | 10% | 15% | Unique differentiator |
| E. Math/Quantitative | 10% | 5% | Maintains quantitative skills |
| F. Sentence Completion | 5% | 5% | Language modeling signal |
| G. Instruction Following | 5% | 15% | Format learning (LIMA) |
| H. Anti-Hallucination | 5% | 5% | Temporal boundary enforcement |

### Source Diversity per Generator

Each generator should draw from all available corpus sources to avoid domain bias:

| Source Type | Collections | Best For |
|-------------|-------------|----------|
| News | Economist, NYT, FT, Newswire, US/French newspapers | A, B, C, D, G |
| Legal | Caselaw, USPTO, GATT, EurLex, Eurovoc | A, B, C, E |
| Academic | Books, Science Pile, Open Science Pile, OpenAlex | A, B, C, E |
| Mixed | All of the above | D (multi-source), F, G, H |

---

## 5. Prompt Engineering Principles

### Shared Across All Generators

1. **JSON response format** — All prompts request structured JSON output. Use `response_format={"type": "json_object"}` in OpenAI API calls.
2. **Temperature 0.7** — Balances diversity with coherence (from `synth_config.yaml`).
3. **Chunk size 6,000 chars / 300 overlap** — Established by existing pipeline; sufficient context without exceeding token limits.
4. **Max tokens 4,096** — Allows detailed responses with reasoning chains.
5. **Model: GPT-4o-mini** — Cost-optimized for high-volume generation. Consider GPT-4o for quality-critical generators (D, H).

### Quality Signals in Prompts

- Explicit instruction to "vary question types" prevents monotonous output
- Requiring answers to be "directly supported by the text" reduces hallucination
- Requesting metadata (question type, difficulty, domain) enables downstream filtering
- Specifying exact JSON schema prevents parsing errors

### Anti-Pattern Avoidance

- **Do not** ask for trivia or surface-level factual recall
- **Do not** generate yes/no questions without requiring justification
- **Do not** allow questions that could be answered without the source text (for Generators C, E)
- **Do not** generate content that references the generation process itself

---

## 6. Implementation Plan

### Phase 1: Extend Existing Infrastructure (Generators C, E, G)

These three can be added to `run_direct.py` with minimal changes:

1. Add new prompt constants (`RC_PROMPT`, `MATH_PROMPT`, `INSTRUCT_PROMPT`)
2. Add a `--generator` CLI flag to select which generators to run
3. Add task types in `process_task()` function
4. Output to separate JSONL files per generator type

**Estimated effort:** 1-2 days
**API cost:** ~$50-100 per period (at GPT-4o-mini rates for ~100K examples)

### Phase 2: Multi-Document Generators (D, F)

Temporal reasoning (Levels 2-3) and sentence completion require sampling multiple documents:

1. Build a document sampler that selects 2-3 documents from the same period with date diversity
2. Create a new `run_temporal.py` script for multi-document generation
3. Implement passage truncation logic for Generator F

**Estimated effort:** 3-5 days
**API cost:** ~$30-50 per period

### Phase 3: Refusal Training (H)

Anti-hallucination generator requires external knowledge about post-period events:

1. Leverage existing LAB question generation (`generate_lab_questions.py`) for event selection
2. Convert LAB-style MC questions into full conversational refusal training data
3. Generate boundary-probing questions using corpus documents near period end

**Estimated effort:** 2-3 days
**API cost:** ~$20-30 per period

### Phase 4: Quality Calibration

After initial generation, run the full quality pipeline (see `04_QUALITY_CONTROL_PIPELINE.md`):

1. Format validation across all generator outputs
2. Cross-generator deduplication
3. Difficulty scoring and rebalancing
4. Final LAB temporal filter pass

### Priority Ordering

```
Phase 1 (C, E, G) → Phase 2 (D, F) → Phase 3 (H) → Phase 4 (Quality)
         ↑                    ↑                ↑
    Low risk,            Medium risk,     Requires design,
    extends existing     new sampling      new approach
```

---

## 7. Evol-Instruct Complexity Scaling (Post-Generation)

After initial generation, apply Evol-Instruct techniques (Xu et al., arXiv:2304.12244) to increase difficulty:

### In-Depth Evolving (5 strategies)

| Strategy | How It Applies |
|----------|----------------|
| Add constraints | "Answer using only information from the first paragraph" |
| Deepening | "Explain the underlying economic theory, not just the outcome" |
| Concretizing | "Give specific dates and figures in your answer" |
| Increase reasoning steps | "Break your answer into at least 5 logical steps" |
| Complicate input | Add misleading details to the passage, test if model can filter |

### In-Breadth Evolving

Mutate topics across domains: take a political question and generate an analogous economic question from the same period.

### Application

Run Evol-Instruct as a **post-processing pass** on 10-20% of generated data to create a difficulty gradient. Store the evolution level as metadata for difficulty-aware training.

---

## References

- Wang et al. (2022). "Self-Instruct: Aligning Language Models with Self-Generated Instructions." arXiv:2212.10560
- Xu et al. (2023). "WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions." arXiv:2304.12244 (Evol-Instruct)
- Abdin et al. (2024). "Phi-4 Technical Report." arXiv:2412.08905 (50 synthetic dataset types)
- Gunasekar et al. (2023). "Textbooks Are All You Need." arXiv:2306.11644 (data quality > quantity)
- Zhou et al. (2023). "LIMA: Less Is More for Alignment." NeurIPS 2023 (superficial alignment hypothesis)
- Chu et al. (2025). "TimE: A Multi-Level Temporal Reasoning Benchmark." arXiv:2505.12891
- Gekhman et al. (2024). "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?" EMNLP 2024
