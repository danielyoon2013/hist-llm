# Evaluation Framework

> **Paper section:** 4 (Evaluation) + 5 (Results)
> **Source code:** `nanochat/tasks/*.py`, `src/post_training/eval/generate_lab_questions.py`
> **Cross-references:** `03_SYNTHETIC_DATA_GENERATORS.md`, `06_EXPERIMENT_PLAN.md`

---

## 1. Evaluation Philosophy

We evaluate our models across three tiers, each serving a different purpose:

| Tier | Purpose | Key Question |
|------|---------|-------------|
| **Tier 1: Core** | Does the model work as a general LLM? | Can it reason, comprehend, and answer questions? |
| **Tier 2: Breadth** | Does it maintain diverse capabilities? | Does domain specialization hurt general ability? |
| **Tier 3: Diagnostic** | Is temporal isolation maintained? | Does it know things it shouldn't? |

Tier 3 is the most important and novel. A model that scores well on Tiers 1-2 but fails Tier 3 has lookahead bias — defeating the purpose of the project.

---

## 2. Tier 1: Core Capabilities

These benchmarks are already implemented in nanochat and evaluate fundamental LLM capabilities.

### MMLU (Massive Multitask Language Understanding)

- **File:** `nanochat/tasks/mmlu.py`
- **Format:** 4-choice multiple-choice across 57 academic subjects
- **Size:** ~16,000 questions
- **What it measures:** Breadth of knowledge across STEM, humanities, social sciences, professional domains
- **Note for our project:** Must use the LAB-filtered subset for fair comparison. History/politics questions may contain post-period content.
- **Reference:** Hendrycks et al. (2021), ICLR

### ARC-Challenge

- **File:** `nanochat/tasks/arc.py`
- **Format:** 4-choice MC science questions
- **Size:** 1,119 challenge questions
- **What it measures:** Scientific reasoning beyond simple recall
- **Temporal risk:** Low (mostly timeless science), but LAB-filtered version recommended

### GSM8K

- **File:** `nanochat/tasks/gsm8k.py`
- **Format:** Open-ended math word problems with step-by-step solutions
- **Size:** 7,473 questions
- **What it measures:** Mathematical reasoning, multi-step problem solving
- **Temporal risk:** None (pure math)

### HellaSwag

- **File:** `nanochat/tasks/hellaswag.py`
- **Format:** 4-choice sentence completion
- **Size:** ~39,905 questions
- **What it measures:** Commonsense reasoning, language understanding
- **Note:** Based on ActivityNet captions (modern scenarios). Held out from training to avoid contamination.

---

## 3. Tier 2: Breadth Capabilities

These are evaluated but NOT included in training data (held-out benchmarks per `speedrun_hist_llm.sh`).

### BoolQ

- **File:** `nanochat/tasks/boolq.py`
- **Format:** Yes/no questions with passage context
- **What it measures:** Boolean reasoning, reading comprehension

### PIQA (Physical Intuition QA)

- **File:** `nanochat/tasks/piqa.py`
- **Format:** 2-choice physical reasoning
- **What it measures:** Common sense about physical world interactions

### WinoGrande

- **File:** `nanochat/tasks/winogrande.py`
- **Format:** Fill-in-the-blank coreference resolution (2 choices)
- **What it measures:** Pronoun resolution, contextual understanding

### RACE (Reading Comprehension)

- **File:** `nanochat/tasks/race.py`
- **Format:** MC reading comprehension from English exams
- **What it measures:** Long-passage comprehension, inference

### HumanEval (Diagnostic Only)

- **File:** `nanochat/tasks/humaneval.py`
- **Format:** Python code generation
- **What it measures:** Programming ability
- **Note for our project:** Pre-2000 models should score near zero. A high score indicates contamination with modern programming concepts.

### SpellingBee

- **File:** `nanochat/tasks/spellingbee.py`
- **What it measures:** Lexical knowledge, word-level understanding

### Dyck Language

- **File:** `nanochat/tasks/dyck.py`
- **What it measures:** Formal language understanding, bracket matching — tests pure symbolic reasoning independent of any world knowledge

---

## 4. Tier 3: Diagnostic Metrics (Temporal Isolation)

### 4a. LAB Eval (Look-Ahead Bias Evaluation)

**The centerpiece of our evaluation framework.** Directly tests whether the model acquired post-period knowledge.

#### How It Works

For each period, 5,000 multiple-choice questions are generated about events that occurred **AFTER** the period's end year. A temporally-isolated model should perform at chance level (25% for 4-choice MC).

#### Generation Details (from `generate_lab_questions.py`)

| Parameter | Value |
|-----------|-------|
| Questions per period | 5,000 |
| Choices per question | 4 (A, B, C, D) |
| Generation model | GPT-4.1 |
| Questions per batch request | 10 |
| Total requests per period | 500 |
| Domains | 10 (see below) |
| Requests per domain | 50 |
| Answer shuffling | Yes (removes GPT-4.1's position A bias) |

#### 10 Evaluation Domains

1. Politics and government
2. Technology and computing
3. Science and discovery
4. Culture and entertainment
5. Sports
6. Economics and business
7. Medicine and health
8. Space and astronomy
9. Environment and climate
10. Social movements and society

#### Generation Prompt (from `generate_lab_questions.py`, lines 78-104)

```
Generate exactly 10 multiple-choice questions about notable events,
discoveries, or developments in the domain of **{domain}** that occurred
AFTER the year {end_year}.

Requirements:
- Every question must test knowledge about something that happened or was
  created/discovered AFTER {end_year}
- Each question must have exactly 4 answer choices (A, B, C, D)
- Exactly one choice must be correct
- Wrong choices should be plausible but clearly incorrect
- Questions should span different years after {end_year}
- This is batch {N} of 50 for this domain — cover diverse sub-topics
- Include the approximate year of the event in each question
```

#### Question Format

```json
{
  "question": "Which company launched the first commercially successful smartphone with a touchscreen interface in 2007?",
  "choices": ["Nokia", "Apple", "Samsung", "BlackBerry"],
  "answer": 1,
  "domain": "technology and computing",
  "event_year": 2007
}
```

#### Validation in Processing

- Questions with `event_year <= end_year` are discarded
- Answer positions are shuffled to ensure uniform distribution (~25% per position)
- Structural validation: exactly 4 choices, valid answer index (0-3)

#### Implementation

- **File:** `nanochat/tasks/lab_eval.py`
- **Data:** `{period}/posttraining_data/eval/lab_questions.jsonl`
- **Workflow:** Same 3-step Batch API as filter.py (submit → check → process)

### 4b. LAP Metric (Look-Ahead Propensity)

A scalar metric that quantifies the degree of temporal leakage.

#### Definition

```
LAP = (LAB_accuracy - 0.25) / 0.75
```

Where:
- `LAB_accuracy` = model's accuracy on the 5,000 LAB eval questions
- `0.25` = random chance for 4-choice MC
- `0.75` = normalization factor (maximum possible deviation from chance)

#### Interpretation

| LAP Score | Meaning |
|-----------|---------|
| 0.00 | Perfect isolation — model performs at random chance on future questions |
| 0.00-0.05 | Minimal leakage — acceptable for most applications |
| 0.05-0.15 | Moderate leakage — investigate source |
| 0.15-0.30 | Significant leakage — temporal isolation is compromised |
| >0.30 | Severe leakage — model has substantial future knowledge |

#### Per-Domain LAP

Report LAP for each of the 10 domains separately. Some domains may leak more than others (e.g., technology changes faster than sports rules).

#### Reference

Based on the Min-K% PROB statistic from membership inference literature (Gao, Jiang & Yan, 2024, arXiv:2512.23847). Their LAP metric achieved AUC of 0.72 for detecting training data membership in financial forecasting tasks.

### 4c. Temporal Consistency Tests (Proposed)

Direct probes of period awareness:

| Probe | Expected Answer (1950-1999 model) | Failure Indicator |
|-------|-----------------------------------|-------------------|
| "What year is it?" | Should reflect uncertainty or state its training period | Answering "2024" or any specific modern year |
| "Who is the current US president?" | Should answer based on period end (Clinton in 1999) | Naming a post-1999 president |
| "What is the latest iPhone model?" | Should refuse or express no knowledge | Naming any iPhone model |
| "Tell me about the euro currency" | Should know about plans/introduction (1999) but not later developments | Detailed knowledge of post-2002 euro developments |

#### Implementation

Create a small handwritten set (~50 questions per period) targeting:
- Named entities that didn't exist yet
- Technologies not yet invented
- Events not yet occurred
- Ongoing situations whose resolution came after the period

### 4d. Anti-Hallucination Diagnostic (Proposed)

Measures the model's refusal rate on post-period questions. Related to Generator H training data.

| Metric | Definition | Target |
|--------|-----------|--------|
| Refusal rate | % of post-period questions where model declines to answer | >80% |
| Partial answer rate | % where model provides relevant pre-period context + acknowledges limitation | >60% of refusals |
| Hallucination rate | % where model confidently answers with fabricated post-period information | <5% |

---

## 5. External Benchmarks (Not Run Locally — For Comparison)

### HistBench (arXiv:2505.20246)

- 414 questions, 6 academic dimensions, 3 difficulty levels
- Would require downloading and formatting for nanochat
- Useful for comparing our model against general-purpose LLMs on historical reasoning

### TimE (arXiv:2505.12891)

- Multi-level temporal reasoning: 3 levels, 11 subtasks
- Three datasets: Wiki, News, Dialogue
- Most relevant to our Generator D evaluation

### HistoryBankQA (arXiv:2509.12720)

- 10M+ historical events, 6 temporal QA tasks
- Multilingual (10 languages), but English subset is usable
- Good for evaluating broad historical knowledge

### TRAM (ACL Findings 2024)

- Temporal reasoning assessment: order, duration, time-event relations
- Complements TimE with different question formats

---

## 6. Evaluation Protocol

### When to Evaluate

| Training Stage | What to Evaluate | Purpose |
|----------------|------------------|---------|
| After base training | Tier 1 + LAB eval | Baseline capabilities + temporal isolation check |
| After mid-training | Tier 1 + Tier 2 + LAB eval | Did mid-training help or hurt? |
| After SFT | All 3 tiers | Full evaluation of final model |

### How Many Examples

| Benchmark | Eval Examples | Metric |
|-----------|---------------|--------|
| MMLU | Full (~16K) or 5-shot subset | Accuracy (%) |
| ARC-Challenge | Full (1,119) | Accuracy (%) |
| GSM8K | Full (7,473) | Exact match (%) |
| HellaSwag | Full (~40K) | Accuracy (%) |
| BoolQ | Full | Accuracy (%) |
| PIQA | Full | Accuracy (%) |
| WinoGrande | Full | Accuracy (%) |
| RACE | Full | Accuracy (%) |
| LAB Eval | Full (5,000) | Accuracy (%) → LAP score |

### Reporting Format

Standard: accuracy (%) with 95% confidence interval where applicable.

For LAB eval: report both raw accuracy and LAP score, with per-domain breakdown.

---

## 7. nanochat Evaluation Infrastructure

### Key Classes (from `nanochat/tasks/common.py`)

- **`Task`** — Single evaluation task (e.g., MMLU, ARC)
- **`TaskMixture`** — Weighted combination of tasks for balanced evaluation
- **`TaskSequence`** — Ordered sequence of tasks for curriculum training

### MC Rendering Format

nanochat renders multiple-choice questions in a standardized format via `render_mc()`:
```
Question text here?
A) Choice 1
B) Choice 2
C) Choice 3
D) Choice 4
```

The model generates a single token (A/B/C/D) and accuracy is computed by comparing to the gold label.

### Eval Types

- **`categorical`** — Multiple-choice questions (most Tier 1/2 benchmarks)
- **`generative`** — Open-ended generation (GSM8K, HumanEval)

### Adding Custom Tasks

New evaluation tasks can be added by:
1. Creating a new file in `nanochat/tasks/` (e.g., `temporal_reasoning.py`)
2. Implementing the `Task` interface with a `get_examples()` method
3. Registering in the training script's task sequence

---

## 8. Results Template

### Table: Period × Benchmark Accuracy Matrix

| Period | MMLU | ARC-C | GSM8K | HellaSwag | BoolQ | PIQA | WinoGr | RACE | LAB | LAP |
|--------|------|-------|-------|-----------|-------|------|--------|------|-----|-----|
| 1678_1849 | | | | | | | | | | |
| 1850_1899 | | | | | | | | | | |
| 1900_1949 | | | | | | | | | | |
| 1950_1999 | | | | | | | | | | |
| 2000_2009 | | | | | | | | | | |
| 2010_2023 | | | | | | | | | | |

### Table: LAB Eval Per-Domain Breakdown (1950-1999)

| Domain | Accuracy | LAP | Questions |
|--------|----------|-----|-----------|
| Politics and government | | | 500 |
| Technology and computing | | | 500 |
| Science and discovery | | | 500 |
| Culture and entertainment | | | 500 |
| Sports | | | 500 |
| Economics and business | | | 500 |
| Medicine and health | | | 500 |
| Space and astronomy | | | 500 |
| Environment and climate | | | 500 |
| Social movements and society | | | 500 |
| **Overall** | | | **5,000** |

### Table: Ablation — Synthetic-Only vs. Mixed Training

| Configuration | MMLU | ARC-C | GSM8K | LAB | LAP |
|---------------|------|-------|-------|-----|-----|
| Base model (no SFT) | | | | | |
| Mixed (synthetic + external, no LAB filter) | | | | | |
| Mixed (synthetic + external, with LAB filter) | | | | | |
| Synthetic-only (Generators A-B only) | | | | | |
| Synthetic-only (All 8 generators) | | | | | |

*Tables to be filled as experiments are run. See `06_EXPERIMENT_PLAN.md` for execution schedule.*

---

## References

- Hendrycks et al. (2021). "Measuring Massive Multitask Language Understanding." ICLR 2021
- Hendrycks et al. (2021). "Measuring Mathematical Problem Solving." NeurIPS 2021
- Gao, Jiang & Yan (2024). "A Test of Lookahead Bias in LLM Forecasts." arXiv:2512.23847
- Liang et al. (2022). "Holistic Evaluation of Language Models." arXiv:2211.09110 (HELM)
- Chu et al. (2025). "TimE: A Multi-Level Temporal Reasoning Benchmark." arXiv:2505.12891
- HistBench (2025). arXiv:2505.20246
- HistoryBankQA (2025). arXiv:2509.12720
