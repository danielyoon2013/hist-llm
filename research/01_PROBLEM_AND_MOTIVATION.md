# Problem and Motivation

> **Paper section:** 1 (Introduction) + 2 (Related Work)
> **Cross-references:** `../context_free_qa_writeup.md`, `03_SYNTHETIC_DATA_GENERATORS.md`

---

## 1. Research Question

**Can we train temporally-isolated language models on historical corpora spanning 1678-2023 such that each model's knowledge is strictly bounded to its training period, with no lookahead bias — and does a fully synthetic, corpus-derived instruction-tuning pipeline achieve this more reliably than external datasets?**

Sub-questions:
1. How do we measure lookahead bias in a fine-tuned LLM? (See `05_EVALUATION_FRAMEWORK.md`)
2. What types of synthetic data are sufficient to produce a "reasonable" domain-adapted model without any external instruction datasets? (See `03_SYNTHETIC_DATA_GENERATORS.md`)
3. How does temporal contamination in SFT data affect model behavior, and does LAB filtering eliminate it? (See `04_QUALITY_CONTROL_PIPELINE.md`)

---

## 2. Why This Matters

### Applications of Temporally-Isolated LLMs

1. **Historical NLP and Digital Humanities** — Models that reason about historical periods without anachronistic knowledge. Useful for analyzing historical documents, assisting historians, and powering digital humanities tools.

2. **Counterfactual Analysis** — If a model truly knows only what was known at time T, it can be used for "what would you have predicted?" analyses without hindsight bias. Relevant to economic forecasting, policy analysis, and financial research.

3. **Temporal Bias Studies** — Quantifying how much modern LLMs leak future knowledge into ostensibly historical contexts. Foundation for studying how training data composition affects temporal awareness.

4. **Point-in-Time Knowledge Retrieval** — A suite of period-specific models can answer "what was known about X in 1975?" without the user needing to manually filter for temporal relevance.

5. **Training Data Research** — Our pipeline contributes methodology for synthetic data generation, temporal decontamination, and domain adaptation that generalizes beyond the historical domain.

---

## 3. The Lookahead Bias Problem

### Definition

**Lookahead bias** occurs when a model trained for a specific historical period exhibits knowledge of events, technologies, people, or concepts that did not exist or were not publicly known until after that period. For example, a model trained on 1950-1999 data that can describe the iPhone (released 2007) or reference the 2008 financial crisis has lookahead bias.

### Sources of Lookahead Bias in Standard LLM Pipelines

| Source | Mechanism | Example |
|--------|-----------|---------|
| Pre-training data | Web crawl contains documents from all time periods | Wikipedia article updated in 2023 appears in pre-training data for all models |
| Instruction tuning datasets | SmolTalk, MMLU, etc. contain modern knowledge | SmolTalk conversation about COVID-19 vaccines included in 1950-1999 model's SFT |
| Tokenizer | BPE vocabularies encode modern word frequencies | Tokens like "blockchain" or "COVID" have dedicated tokens from modern training |
| Evaluation benchmarks | Benchmark questions leak into training | MMLU questions about 21st-century events |

### Our Approach: Temporal Isolation at Every Stage

| Stage | How Isolation Is Enforced |
|-------|--------------------------|
| Base training | Only documents from within the period (e.g., 1950-1999) are included in pre-training data. Quality filtering is period-specific. |
| Synthetic data generation | QA pairs are generated exclusively from period-appropriate corpus documents. No external knowledge sources. |
| LAB filtering | Every instruction-tuning example is classified for temporal contamination via GPT-4o-mini. Items requiring post-period knowledge are removed. |
| Evaluation | LAB eval questions (5,000 per period) test whether the model acquired post-period knowledge. LAP metric quantifies leakage. |

---

## 4. Related Work

### 4a. Domain-Adapted LLMs

Pre-trained LLMs adapted to specific domains through continued pre-training and/or domain-specific instruction tuning:

| Model | Domain | Approach | Temporal Isolation? |
|-------|--------|----------|---------------------|
| **BloombergGPT** (Wu et al., 2023) | Finance | 363B token financial corpus + general data | No |
| **BioMedLM** (Bolton et al., 2024) | Biomedical | PubMed-only pre-training (2.7B params) | No |
| **Galactica** (Taylor et al., 2022) | Science | 106B scientific tokens | No |
| **Legal-BERT** variants | Legal | Legal corpus continued pre-training | No |
| **SaulLM-7B** (Colombo et al., 2024) | Legal | 30B English legal corpus | No |
| **Ours** | Historical | Period-specific corpus with temporal isolation | **Yes** |

**Gap:** All existing domain-adapted LLMs treat their corpus as a single, time-agnostic collection. None enforce period boundaries or measure lookahead bias. Our work is the first to apply temporal isolation to domain-adapted LLM training.

### 4b. Temporal Reasoning in NLP

Benchmarks and methods for evaluating temporal understanding:

- **TimE** (Chu et al., 2025, arXiv:2505.12891) — Multi-level temporal reasoning benchmark with 3 hierarchical levels and 11 subtasks. Found that even advanced models (o3-mini) achieve only ~50% on implicit temporal reasoning. Timeline ordering is the weakest area across all models (<30%).

- **TRAM** (ACL Findings 2024) — Temporal Reasoning Assessment covering order, duration, and time-event relations. Notes that temporal narratives and temporal causality remain under-explored.

- **Test of Time (ToT)** (arXiv:2406.09170) — Synthetically generated temporal reasoning tasks designed to prevent knowledge shortcuts. Categories: time arithmetic, chronological comparison, date differences.

- **TGB 2.0** — Framework for temporal knowledge graph evaluation with 8 datasets spanning 5 domains and up to 53M edges.

**Relevance to our work:** These benchmarks measure temporal reasoning ability but do not address temporal isolation — the question of whether a model has been exposed to future information. We build on their task taxonomies (particularly TimE's 3-level hierarchy) for Generator D.

### 4c. Historical NLP

Prior work specifically on NLP for historical text:

- **HistBench** (2025, arXiv:2505.20246) — First comprehensive benchmark for historical reasoning in AI. 414 questions from 40+ contributors across 6 academic dimensions (rarity, linguistic complexity, format heterogeneity, perceptual accessibility, interdisciplinary scope, reasoning complexity). 3 difficulty levels with carefully designed distractors.

- **HistoryBankQA** (2025, arXiv:2509.12720) — Multilingual database of 10M+ historical events from Wikipedia timeline pages. 10 languages, 6 temporal QA reasoning tasks. Found GPT-4o outperforms all models; among small models, Gemma-2-9b leads.

- **ArchivalQA** (Wang et al.) — Large-scale QA dataset for temporal news archives. Classifies questions by difficulty and temporal expressions.

- **MacBERTh** (Manjavacas & Fonteyn, 2022) — BERT model pre-trained on historical English text (1500-1950). Focused on NER and text classification, not generative tasks.

**Gap:** HistBench and HistoryBankQA evaluate historical knowledge in general-purpose LLMs. Neither trains period-specific models or addresses the question of what happens when you restrict a model's training data to a specific time window.

### 4d. Synthetic Data for LLMs

The field has established that high-quality synthetic data can replace or augment organic training data:

- **"Textbooks Are All You Need" / Phi-1** (Gunasekar et al., 2023, arXiv:2306.11644) — Demonstrated that a 1.3B model trained on textbook-quality synthetic data outperforms much larger models. Key insight: data quality > data quantity.

- **Phi-4** (Abdin et al., 2024, arXiv:2412.08905) — 14B model with 40% synthetic data (50 broad types, ~400B tokens). Core techniques: multi-agent prompting, self-revision workflows, instruction reversal. Achieved state-of-the-art on MATH (80.4) and GPQA (56.1).

- **Self-Instruct** (Wang et al., 2022, arXiv:2212.10560) — Pipeline for generating instruction data from 175 seed tasks. 33% improvement over vanilla GPT-3 on Super-NaturalInstructions. Foundation for Alpaca, Code Alpaca.

- **Evol-Instruct / WizardLM** (Xu et al., 2023, arXiv:2304.12244) — Two evolution mechanisms: in-depth (add constraints, deepen, concretize, increase steps, complicate) and in-breadth (topic diversification). 4 evolution rounds from 52K initial Alpaca instructions to 250K.

- **Cosmopedia** (HuggingFace, 2024) — Largest open synthetic dataset: 30M files, 25B tokens generated by Mixtral-8x7B. Content types: textbooks, blog posts, stories, WikiHow.

- **Meta Synthetic Data Kit** (2025) — Open-source tooling for synthetic dataset generation. Our initial pipeline used this before switching to direct API calls (5-10x faster).

**Our approach differs** from all of the above in one critical way: our synthetic data is exclusively derived from a corpus that is itself temporally bounded. We do not use web data, Wikipedia, or any source that might contain future information. This is a strict constraint that none of the above papers consider.

### 4e. Contamination and Decontamination

The problem of benchmark data appearing in training sets:

- **Min-K% PROB** — Membership inference attack that computes average log-probability of the bottom K% of tokens. Used as the basis for the LAP (Lookahead Propensity) metric (Gao, Jiang & Yan, 2024, arXiv:2512.23847).

- **N-gram decontamination** — Phi-4 uses 13-gram and 7-gram matching to detect and remove benchmark data. Standard practice for responsible benchmark reporting.

- **LiveCodeBench / LiveBench / AntiLeak-Bench** — Dynamic benchmarks that use timestamps to ensure post-training test instances. Address contamination by continuously generating new test data.

- **Clean-Eval / MMLU-CF** — Contamination-free evaluation through paraphrasing, back-translation, and surface-level permutations.

- **"A Survey on Data Contamination for LLMs"** (arXiv:2502.14425) — Comprehensive survey distinguishing exact contamination (duplicate) from syntactic contamination (duplicates after transformation).

**Our LAB filtering** is a form of temporal decontamination that goes beyond standard n-gram overlap. Rather than detecting whether a specific benchmark appears in training data, we detect whether any piece of information from the future has leaked into the training set.

### 4f. SFT Alignment and the LIMA Hypothesis

Our pipeline generates QA pairs without including the source document at training time. This is justified by:

- **LIMA** (Zhou et al., 2023, NeurIPS) — 1,000 context-free examples achieve competitive performance with GPT-4. Core finding: "A model's knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used."

- **Gekhman et al.** (EMNLP 2024) — SFT on facts the model does NOT already know linearly increases hallucination. Our QA pairs come from the same corpus used for pre-training, so this risk is mitigated.

- **Alpaca** (52K synthetic pairs), **Llama 2** (27K human QA pairs) — Both use context-free instruction tuning as standard practice.

See `../context_free_qa_writeup.md` for our full analysis of this design decision.

---

## 5. Our Contribution

1. **Temporal isolation methodology** — First systematic approach to training LLMs with strict period boundaries at every pipeline stage (pre-training, synthetic data generation, filtering, evaluation).

2. **LAB metric and filtering pipeline** — Language Acquisition Boundary evaluation: 5,000 MC questions per period across 10 domains, with the LAP (Lookahead Propensity) metric for quantifying temporal leakage.

3. **Corpus-derived synthetic data pipeline** — 8 generator types producing diverse training data exclusively from period-appropriate historical documents, eliminating the need for external instruction datasets.

4. **Multi-period evaluation framework** — 3-tier evaluation (core capabilities, breadth, diagnostics) applied across 6 historical periods spanning 1678-2023.

5. **Empirical analysis** — Quantitative comparison of synthetic-only vs. mixed (synthetic + external) training, measuring the tradeoff between temporal purity and general capability.

---

## 6. References

### Domain-Adapted LLMs
- Wu et al. (2023). "BloombergGPT: A Large Language Model for Finance." arXiv:2303.17564
- Bolton et al. (2024). "BioMedLM: A 2.7B Parameter Language Model Trained On Biomedical Text." arXiv:2403.18421
- Taylor et al. (2022). "Galactica: A Large Language Model for Science." arXiv:2211.09085
- Colombo et al. (2024). "SaulLM-7B: A Pioneering Large Language Model for Law." arXiv:2403.03883

### Temporal Reasoning
- Chu et al. (2025). "TimE: A Multi-Level Temporal Reasoning Benchmark." arXiv:2505.12891
- Fatemi et al. (2024). "Test of Time: A Benchmark for Evaluating LLMs on Temporal Reasoning." arXiv:2406.09170
- TRAM. ACL Findings 2024. "Temporal Reasoning Assessment."

### Historical NLP
- HistBench (2025). "On Path to Multimodal Historical Reasoning." arXiv:2505.20246
- HistoryBankQA (2025). arXiv:2509.12720 (Microsoft Research)
- Manjavacas & Fonteyn (2022). "MacBERTh: Development and Evaluation of a Historically Pre-trained Language Model for English."

### Synthetic Data
- Gunasekar et al. (2023). "Textbooks Are All You Need." arXiv:2306.11644
- Abdin et al. (2024). "Phi-4 Technical Report." arXiv:2412.08905
- Wang et al. (2022). "Self-Instruct." arXiv:2212.10560
- Xu et al. (2023). "WizardLM / Evol-Instruct." arXiv:2304.12244
- Li et al. (2023). "Textbooks Are All You Need II: phi-1.5." arXiv:2309.05463

### Contamination
- Gao, Jiang & Yan (2024). "A Test of Lookahead Bias in LLM Forecasts." arXiv:2512.23847
- Sarkar & Vafa (2024). "Lookahead Bias in Pretrained Language Models." SSRN
- "A Survey on Data Contamination for LLMs." arXiv:2502.14425

### SFT Alignment
- Zhou et al. (2023). "LIMA: Less Is More for Alignment." NeurIPS 2023
- Gekhman et al. (2024). "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?" EMNLP 2024
- Touvron et al. (2023). "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv:2307.09288

### Evaluation
- HELM: Liang et al. (2022). "Holistic Evaluation of Language Models." arXiv:2211.09110
- Hendrycks et al. (2021). "Measuring Massive Multitask Language Understanding." ICLR 2021 (MMLU)
- Hendrycks et al. (2021). "Measuring Mathematical Problem Solving." NeurIPS 2021 (MATH)
