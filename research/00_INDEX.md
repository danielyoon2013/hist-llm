# Historical LLM Research — Index

> Training temporally-isolated language models on a 1678-2023 English historical corpus (~125M documents), with fully synthetic instruction-tuning data and strict period boundaries to eliminate lookahead bias.

---

## Research Files

| # | File | Description | Paper Section |
|---|------|-------------|---------------|
| 01 | [Problem and Motivation](01_PROBLEM_AND_MOTIVATION.md) | Research question, related work, contributions | Sec 1-2 (Intro + Related Work) |
| 02 | [Corpus and Pretraining](02_CORPUS_AND_PRETRAINING.md) | 125M-doc corpus, quality filtering, base training | Sec 3.1 |
| 03 | [Synthetic Data Generators](03_SYNTHETIC_DATA_GENERATORS.md) | 8 generator types: specs, prompts, formats | Sec 3.2 |
| 04 | [Quality Control Pipeline](04_QUALITY_CONTROL_PIPELINE.md) | LAB filtering, dedup, decontamination | Sec 3.3 |
| 05 | [Evaluation Framework](05_EVALUATION_FRAMEWORK.md) | 3-tier eval, LAB/LAP metrics, benchmarks | Sec 4-5 |
| 06 | [Experiment Plan](06_EXPERIMENT_PLAN.md) | Ablations, baselines, timeline, compute budget | Sec 4.2 + Appendix |
| 07 | [Paper Outline](07_PAPER_OUTLINE.md) | Section-by-section skeleton, figure/table plan | Full paper |

---

## Key Terminology

| Term | Definition |
|------|-----------|
| **LAB** | Look-Ahead Bias — knowledge of events that occurred after a model's training period |
| **LAP** | Look-Ahead Propensity — scalar metric: `(LAB_accuracy - 0.25) / 0.75`. 0 = perfect isolation |
| **Temporal isolation** | Ensuring a model for period X sees only data from period X, at every pipeline stage |
| **Analysis periods** | The 6 time windows for model training: `1678_1849`, `1850_1899`, `1900_1949`, `1950_1999`, `2000_2009`, `2010_2023` |
| **Quality periods** | 14 finer-grained 25-year windows for Ridge quality models |
| **Generator A-H** | The 8 synthetic data generator types (see `03_SYNTHETIC_DATA_GENERATORS.md`) |
| **Content x Format x Source** | The 3D matrix organizing synthetic data: content type (A-H) x question format (MC, CoT, T/F, etc.) x corpus source (News, Law, etc.) |
| **Generator-Eval Alignment** | Mapping from each generator to the benchmarks it targets, enabling ablation studies |
| **LAB filtering** | GPT-4o-mini classification of each example for post-period knowledge (see `filter.py`) |
| **nanochat** | The training framework (Karpathy's nanoGPT lineage) — model, tokenizer, training scripts |

---

## Pipeline Architecture

```
                         HISTORICAL CORPUS (1678-2023, ~125M docs)
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
              [Cleaning]          [BGE Embeddings]    [Quality Scoring]
                    │                   │                   │
                    └───────────────────┼───────────────────┘
                                        │
                              [Period-Specific Shards]
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
           BASE TRAINING                         SYNTHETIC DATA GENERATION
        (continued pretraining)          (Content x Format x Source matrix)
                    │                       (8 generators + GSM8K/MATH ext.)
                    │                                       │
                    │                              [Quality Pipeline]
                    │                          (dedup, LAB filter, decontam)
                    │                                       │
                    ▼                                       ▼
              Base Checkpoint ──► MID-TRAINING ──► SFT ──► Final Model
                                                              │
                                                        [EVALUATION]
                                                    (Tier 1 + 2 + 3)
                                                    (LAB eval → LAP score)
```

---

## Existing Artifacts (Not in research/)

| File | Location | What It Contains |
|------|----------|------------------|
| Context-free QA justification | `../context_free_qa_writeup.md` | Why SFT without source docs is valid (LIMA, Alpaca, Llama 2) |
| Dataset contamination analysis | `../instruct_dataset_summary_v2.csv` | Per-dataset LAB filtering results with contamination rates |
| Technical pipeline docs | `../src/README.md` | Full engineering documentation of both training stages |
| Training script | `../nanochat/runs/speedrun_hist_llm.sh` | Curriculum training with 36 datasets |
| Post-training config | `../src/post_training/config.py` | Period definitions, path conventions, API settings |

---

## Key Source Code

| Script | Purpose |
|--------|---------|
| `src/post_training/corpus/run_direct.py` | QA + CoT generation (Generators A-B) |
| `src/post_training/corpus/synth_config.yaml` | Generation parameters (temp, chunk size, dedup) |
| `src/post_training/corpus/convert.py` | Convert to nanochat JSONL format |
| `src/post_training/instruct/filter.py` | LAB temporal filtering |
| `src/post_training/instruct/split.py` | Train/test splitting (95/5) |
| `src/post_training/eval/generate_lab_questions.py` | LAB eval question generation (5K/period) |
| `nanochat/scripts/base_train.py` | Base model training |
| `nanochat/scripts/mid_train.py` | Mid-training |
| `nanochat/scripts/chat_sft.py` | SFT fine-tuning |
| `nanochat/scripts/chat_eval.py` | Model evaluation |

---

## Open Questions

*Living list — update as decisions are made.*

1. **Model size:** What parameter count to target? nanochat default (~770M) or larger?
2. **Tokenizer isolation:** Should we train period-specific BPE tokenizers, or accept the shared tokenizer limitation?
3. **Evol-Instruct budget:** How many evolution rounds for complexity scaling? (Affects API cost and diversity)
4. **Generator H validation:** How do we verify that refusal training generalizes beyond the specific post-period events used in training?
5. **Cross-period transfer:** Can a model trained on 1950-1999 benefit from pre-training on earlier periods first (cumulative training)?
6. **Evaluation on external benchmarks:** Should we run HistBench / TimE, or focus only on internal evaluation?
7. **Paper venue:** Conference (ACL/EMNLP/NeurIPS) or journal (TACL)?

---

## Reading Order

For someone new to the project:

1. This file (`00_INDEX.md`) — orientation
2. `01_PROBLEM_AND_MOTIVATION.md` — why we're doing this
3. `02_CORPUS_AND_PRETRAINING.md` — what data we have
4. `03_SYNTHETIC_DATA_GENERATORS.md` — how we create training data (core contribution)
5. `04_QUALITY_CONTROL_PIPELINE.md` — how we ensure quality
6. `05_EVALUATION_FRAMEWORK.md` — how we measure success
7. `06_EXPERIMENT_PLAN.md` — what experiments to run
8. `07_PAPER_OUTLINE.md` — how it all becomes a paper
