# Paper Outline

> **Purpose:** Skeleton of the research paper. Each section references the research file it draws from.

---

## Title Options

1. **"Temporally-Isolated Language Models: Training Domain-Adapted LLMs with Strict Period Boundaries on Historical Corpora"**
2. **"No Lookahead: Synthetic Data Pipelines for Period-Specific Historical Language Models"**
3. **"Preventing Temporal Leakage in Domain-Adapted LLMs: A Historical Corpus Approach"**
4. **"Historical LLMs Without Hindsight: Temporal Isolation Through Synthetic Data"**

---

## Abstract Template

> We present a methodology for training temporally-isolated language models on a historical English corpus spanning 1678-2023 (~125M documents). Each model variant is constrained to knowledge from a specific time period through temporal isolation at every pipeline stage: period-bounded pre-training data, corpus-derived synthetic instruction tuning, and LAB (Look-Ahead Bias) filtering. We introduce eight synthetic data generator types that produce diverse training data exclusively from period-appropriate documents, eliminating the need for external instruction datasets that carry temporal contamination. We define the LAP (Look-Ahead Propensity) metric to quantify temporal leakage and evaluate across [N] benchmarks. Our fully synthetic pipeline achieves [result on MMLU/ARC/GSM8K] while maintaining a LAP score of [X], compared to [Y] for the unfiltered baseline. We release our evaluation framework, synthetic data pipeline, and period-specific models for [6] historical periods.

---

## Section 1: Introduction (~1.5 pages)

**Source:** `01_PROBLEM_AND_MOTIVATION.md`, Sections 1-3

**Paragraph 1 — The problem:**
Large language models encode knowledge from their training data without temporal boundaries. A model trained on web text spanning decades cannot distinguish what was known in 1980 from what was discovered in 2020. This creates lookahead bias — the model "knows the future" relative to any historical reference point.

**Paragraph 2 — Why it matters:**
Applications in historical analysis, counterfactual reasoning, and point-in-time knowledge retrieval require models that respect temporal boundaries. [2-3 sentences on specific applications from 01_PROBLEM_AND_MOTIVATION.md Section 2]

**Paragraph 3 — Current approaches fall short:**
Existing domain-adapted LLMs (BloombergGPT, BioMedLM, Galactica) specialize to a domain but do not enforce temporal boundaries. Standard SFT datasets (SmolTalk, MMLU) contain 30-57% temporal contamination when filtered for a 1950-1999 period. [Reference instruct_dataset_summary_v2.csv]

**Paragraph 4 — Our approach:**
We propose temporal isolation at every pipeline stage: period-bounded pre-training, corpus-derived synthetic data generation (8 generator types), and LAB filtering. We introduce the LAP metric to quantify leakage.

**Paragraph 5 — Contributions:**
[Numbered list of 5 contributions from 01_PROBLEM_AND_MOTIVATION.md Section 5]

---

## Section 2: Related Work (~2 pages)

**Source:** `01_PROBLEM_AND_MOTIVATION.md`, Section 4

### 2.1 Domain-Adapted Language Models
BloombergGPT, BioMedLM, Galactica, SaulLM — continued pretraining on domain corpora. None enforce temporal boundaries. [Table: Model, Domain, Approach, Temporal Isolation?]

### 2.2 Temporal Reasoning in NLP
TimE benchmark (3-level hierarchy), TRAM, Test of Time, temporal knowledge graphs. These measure temporal reasoning ability but not temporal isolation.

### 2.3 Historical NLP
HistBench (414 questions, 6 dimensions), HistoryBankQA (10M+ events), ArchivalQA, MacBERTh. Evaluate historical knowledge in general models; none train period-specific models.

### 2.4 Synthetic Data for LLM Training
Phi series ("Textbooks Are All You Need"), Self-Instruct, Evol-Instruct, Cosmopedia. Our approach differs: synthetic data derived exclusively from temporally-bounded corpus.

### 2.5 Data Contamination and Decontamination
N-gram decontamination, Min-K% PROB, dynamic benchmarks. Our LAB filtering is a form of temporal decontamination that goes beyond benchmark overlap.

### 2.6 Instruction Tuning and the Alignment Hypothesis
LIMA, Gekhman et al. — SFT teaches format, not knowledge. Justifies context-free QA training. [Reference context_free_qa_writeup.md]

---

## Section 3: Methodology (~4 pages)

### 3.1 Corpus and Pre-Training (~1.5 pages)

**Source:** `02_CORPUS_AND_PRETRAINING.md`

- Corpus description: 125M documents, 1678-2023, source types
- Period definitions (6 analysis periods, 14 quality periods)
- Quality filtering pipeline: cleaning → BGE embeddings → GPT-4o-mini labeling → Ridge classifiers → sharding
- Temporal isolation in pre-training: how period boundaries are enforced
- **Table 1:** Corpus statistics per period (documents, tokens, quality thresholds)

### 3.2 Synthetic Data Generation (~2 pages)

**Source:** `03_SYNTHETIC_DATA_GENERATORS.md`

- Design philosophy: why 8 generator types, capability mapping
- Generator specifications: for each type (A-H), describe input format, prompt design, output format
- Volume targets and mixing ratios
- **Table 2:** Generator specifications (type, target capability, replaces, volume)
- **Figure 1:** Synthetic data generation pipeline diagram

### 3.3 Quality Control Pipeline (~1 page)

**Source:** `04_QUALITY_CONTROL_PIPELINE.md`

- 7-stage pipeline overview
- LAB temporal filtering: classification prompt, batch API workflow, results
- Deduplication, consistency checking, difficulty calibration
- N-gram decontamination
- **Table 3:** LAB filtering results per dataset (contamination rates)
- **Table 4:** Quality pipeline statistics (input → output per stage)

---

## Section 4: Evaluation Framework (~1.5 pages)

**Source:** `05_EVALUATION_FRAMEWORK.md`

### 4.1 Evaluation Design
- 3-tier philosophy: Core (capability), Breadth (generalization), Diagnostic (temporal isolation)
- Benchmark selection rationale

### 4.2 LAB Eval and LAP Metric
- 5,000 MC questions per period across 10 domains
- LAP = (accuracy - 0.25) / 0.75
- Interpretation scale (0 = perfect isolation, >0.3 = severe leakage)

### 4.3 Experimental Setup
- Training configuration (model size, learning rate, etc.)
- Ablation study design
- Baselines (no SFT, no filter, mixed data, generic model)
- **Table 5:** Experimental configurations

---

## Section 5: Results (~2 pages)

**Source:** `05_EVALUATION_FRAMEWORK.md` (results tables) + `06_EXPERIMENT_PLAN.md`

### 5.1 Main Results
- Period × Benchmark accuracy matrix
- Comparison with baselines
- **Table 6:** Main results (all periods × all benchmarks)

### 5.2 Temporal Isolation
- LAP scores across periods and domains
- Per-domain breakdown
- Comparison: synthetic-only vs. mixed vs. unfiltered
- **Figure 2:** LAP scores across experimental configurations
- **Figure 3:** Accuracy vs. LAP tradeoff curve

### 5.3 Ablation Studies
- Generator ablation: contribution of each type
- Volume ablation: how much synthetic data is needed?
- Quality threshold ablation: filtering strictness vs. capability
- **Table 7:** Generator ablation results
- **Table 8:** Volume ablation results

---

## Section 6: Analysis and Discussion (~1.5 pages)

### 6.1 The Purity-Capability Tradeoff
Do we sacrifice general capability for temporal purity? How much?

### 6.2 Which Generators Matter Most?
Analysis of ablation results — which generator types have the largest impact?

### 6.3 Cross-Period Patterns
Do earlier periods (sparse data, high OCR noise) show different behavior? How does document density affect model quality?

### 6.4 Failure Cases
Where does temporal isolation break down? Which domains leak most?

### 6.5 Comparison with External Benchmarks
How do our models compare on HistBench, TimE (if run)?

---

## Section 7: Conclusion (~0.5 pages)

### Summary of Contributions
[Restate 5 contributions concisely]

### Limitations
- Shared tokenizer across periods
- English-only corpus
- OCR noise in early periods
- GPT-4o-mini as the generation and filtering model (potential biases)

### Future Work
- Multilingual extension
- Period-specific tokenizers
- Reinforcement learning from temporal feedback
- Application to other temporal domains (financial, legal, scientific)

---

## Appendices

### Appendix A: Full Prompt Templates
All 8 generator prompts + LAB classification prompt + LAB eval generation prompt

### Appendix B: Per-Subject MMLU Breakdown
Accuracy per MMLU subject, highlighting which subjects are most affected by temporal filtering

### Appendix C: Per-Domain LAB Results
Full 10-domain LAB eval breakdown for every period

### Appendix D: Complete Dataset Statistics
Document counts, token counts, quality thresholds, filtering rates for all 6 periods

### Appendix E: Sample Generated Data
5 examples from each of the 8 generator types

---

## Figure and Table Plan

| # | Type | Description | Source |
|---|------|-------------|--------|
| Table 1 | Corpus stats | Documents, tokens, quality thresholds per period | `02_CORPUS_AND_PRETRAINING.md` |
| Table 2 | Generator specs | Type, capability, replaces, volume, format | `03_SYNTHETIC_DATA_GENERATORS.md` |
| Table 3 | LAB filtering | Contamination rates per external dataset | `04_QUALITY_CONTROL_PIPELINE.md` |
| Table 4 | Quality pipeline | Input → output per stage | `04_QUALITY_CONTROL_PIPELINE.md` |
| Table 5 | Experimental configs | Ablation study configurations | `06_EXPERIMENT_PLAN.md` |
| Table 6 | Main results | Period × Benchmark accuracy matrix | `05_EVALUATION_FRAMEWORK.md` |
| Table 7 | Generator ablation | Results with different generator subsets | `06_EXPERIMENT_PLAN.md` |
| Table 8 | Volume ablation | Results at different training sizes | `06_EXPERIMENT_PLAN.md` |
| Figure 1 | Pipeline diagram | End-to-end: corpus → generators → quality → training → eval | `00_INDEX.md` |
| Figure 2 | LAP scores | Bar chart: LAP across configurations | `05_EVALUATION_FRAMEWORK.md` |
| Figure 3 | Purity-capability tradeoff | Scatter: MMLU accuracy vs. LAP score | Experiment results |
| Figure 4 | Temporal gradient | Line: accuracy vs. event_year (should drop at cutoff) | LAB eval detailed results |
| Figure 5 | Quality filtering | Cumulative token curves per period | `02_CORPUS_AND_PRETRAINING.md` |

---

## Target Venue / Format

- **Length:** 8-10 pages (main) + appendices
- **Format:** Standard NLP/ML conference (ACL, EMNLP, NeurIPS) or journal (TACL)
- **Audience:** NLP researchers interested in domain adaptation, temporal reasoning, synthetic data
