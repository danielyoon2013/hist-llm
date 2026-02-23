# Experiment Plan

> **Paper section:** 4.2 (Experimental Setup) + Appendix
> **Cross-references:** `03_SYNTHETIC_DATA_GENERATORS.md`, `05_EVALUATION_FRAMEWORK.md`

---

## 1. Training Schedule

### Three-Stage Pipeline (Per Period)

```
Stage 1: Base Training (Continued Pretraining)
    Input: Quality-filtered corpus shards
    Script: nanochat/scripts/base_train.py
    Output: Base checkpoint

         ▼

Stage 2: Mid-Training
    Input: Base checkpoint + synthetic data (heavy on factual QA + comprehension)
    Script: nanochat/scripts/mid_train.py
    Output: Mid-training checkpoint

         ▼

Stage 3: SFT (Supervised Fine-Tuning)
    Input: Mid-training checkpoint + synthetic data (heavy on CoT + instruction following)
    Script: nanochat/scripts/chat_sft.py
    Output: Final chat model
```

### Per-Stage Data Composition

| Stage | Data Source | Volume | Mix Focus |
|-------|-----------|--------|-----------|
| Base | Quality-filtered corpus shards | All available tokens | Raw text only |
| Mid-training | Generators A, C, D, E | 500K-1M examples | Factual QA (35%), Reading Comp (15%), Temporal (10%), Math (10%) |
| SFT | Generators B, D, G, H (+ smaller portions of A, C, E, F) | 50K-100K examples | CoT (25%), Temporal (15%), Instruct (15%), Anti-Halluc (5%) |

### Priority Period

Start with **1950-1999** — the most complete period with:
- 4.3M pre-training documents
- 348K existing synthetic QA pairs
- LAB eval questions already generated
- Most external dataset filtering already complete

---

## 2. Ablation Studies

### 2a. Generator Ablation

**Question:** How much does each generator type contribute to model quality?

| Experiment | Generators Included | What It Tests |
|------------|-------------------|---------------|
| Baseline | A only (existing QA) | Current pipeline performance |
| + CoT | A + B | Value of reasoning traces |
| + Reading | A + B + C | Value of passage-grounded comprehension |
| + Temporal | A + B + C + D | Value of temporal reasoning (unique) |
| + Math | A + B + C + D + E | Value of quantitative reasoning |
| + All | A + B + C + D + E + F + G + H | Full pipeline |

**Metrics:** MMLU, ARC-C, GSM8K, LAB/LAP (compare across configurations)

### 2b. Volume Ablation

**Question:** How much synthetic data is needed?

| Experiment | Mid-training | SFT | Total |
|------------|-------------|-----|-------|
| Minimal | 50K | 10K | 60K |
| Small | 100K | 25K | 125K |
| Medium | 250K | 50K | 300K |
| Large | 500K | 100K | 600K |
| Full | 1M | 100K | 1.1M |

**Hypothesis:** Following the LIMA principle, SFT benefits plateau quickly (diminishing returns past ~50K). Mid-training volume should scale more linearly with corpus size.

### 2c. Quality Threshold Ablation

**Question:** How strict should LAB filtering be?

| Experiment | Filtering Level | Expected Contamination |
|------------|----------------|----------------------|
| No filter | Skip LAB entirely | High (baseline contamination rate) |
| Standard | Current GPT-4o-mini binary classification | Low (~2-5% residual) |
| Strict | GPT-4o-mini + manual review of borderline cases | Very low (<1%) |

**Metric:** LAP score across configurations. Plot LAP vs. MMLU to visualize the purity-capability tradeoff.

### 2d. Period Length Ablation

**Question:** Do narrower time windows produce better temporal isolation?

| Experiment | Period | Width |
|------------|--------|-------|
| Wide | 1900-1999 | 100 years |
| Standard | 1950-1999 | 50 years |
| Narrow | 1975-1999 | 25 years |
| Very narrow | 1990-1999 | 10 years |

**Hypothesis:** Narrower periods should have lower LAP scores but also lower general capability (less training data). There is an optimal width.

---

## 3. Baselines

### 3a. Base Model Only (No SFT)

- Train: Base training on period corpus only
- Eval: All Tier 1 benchmarks + LAB
- Purpose: Shows raw pre-training quality. The LAB score here indicates whether base training data itself has temporal leakage.

### 3b. Standard SFT (No Temporal Filtering)

- Train: Base + mid-training + SFT using ALL external datasets (no LAB filter)
- Eval: All 3 tiers
- Purpose: The "naive" approach. Shows the cost of ignoring temporal contamination. Expect high Tier 1 scores but poor LAP.

### 3c. Generic Model (No Domain Adaptation)

- Train: Off-the-shelf nanochat model (or equivalent small LLM) without historical corpus pre-training
- Eval: All 3 tiers
- Purpose: Shows the value of domain-specific pre-training. Expect moderate Tier 1 but random LAB performance (since no historical specialization).

### 3d. Mixed Synthetic + External (Current Pipeline)

- Train: Current setup — corpus QA + LAB-filtered external datasets (SmolTalk, MMLU, ARC, GSM8K, etc.)
- Eval: All 3 tiers
- Purpose: The current best configuration. The synthetic-only pipeline should match or exceed this while achieving better LAP scores.

---

## 4. Compute Budget

### API Costs (GPT-4o-mini for Synthetic Data Generation)

| Generator | Examples/Period | API Calls/Period | Est. Cost/Period |
|-----------|----------------|------------------|------------------|
| A (Factual MC) | ~350K | ~120K | $30-50 |
| B (CoT) | ~200K | ~100K | $25-40 |
| C (Reading Comp) | ~150K | ~75K | $20-30 |
| D (Temporal) | ~100K | ~50K | $15-25 |
| E (Math) | ~80K | ~40K | $10-20 |
| F (Completion) | ~50K | ~25K | $5-10 |
| G (Instruct) | ~80K | ~40K | $10-20 |
| H (Anti-Halluc) | ~50K | ~25K | $5-10 |
| **Total/period** | **~1M** | **~475K** | **$120-205** |

- 6 periods × $120-205 = **$720-1,230 total API cost**
- LAB filtering (Batch API, 50% discount): ~$50-100 per period
- LAB eval generation (GPT-4.1): ~$30 per period
- Quality scoring: ~$30-50 per period

**Total estimated API cost: $1,500-2,500** across all periods

### GPU Costs (Training)

Depends on model size and available hardware. Estimates based on nanochat's $100 speedrun benchmark (770M params on 8xH100):
- Base training: 4-8 GPU-hours per period
- Mid-training: 2-4 GPU-hours per period
- SFT: 1-2 GPU-hours per period
- Evaluation: <1 GPU-hour per period

**Total per period: ~10-15 GPU-hours**
**Total across 6 periods: ~60-90 GPU-hours**

---

## 5. Priority Order

### Phase 1: Foundation (Weeks 1-2)

1. **Generate synthetic data for 1950-1999** using existing Generators A+B
2. **Implement Generators C, E, G** (extend `run_direct.py`)
3. **Run base training** for 1950-1999 period
4. **Evaluate base model** on all Tier 1 + LAB

### Phase 2: New Generators (Weeks 3-4)

5. **Implement Generator D** (temporal reasoning — requires multi-document sampling)
6. **Implement Generator F** (sentence completion)
7. **Implement Generator H** (anti-hallucination)
8. **Generate full synthetic dataset** for 1950-1999 using all 8 generators
9. **Run quality pipeline** (dedup, LAB filter, difficulty scoring)

### Phase 3: Training and Ablation (Weeks 5-6)

10. **Run mid-training + SFT** with synthetic-only data
11. **Run generator ablation** (A-only → A+B → ... → all 8)
12. **Run volume ablation** (50K → 1M)
13. **Compare against baselines** (no SFT, no filter, mixed data)

### Phase 4: Multi-Period and Analysis (Weeks 7-8)

14. **Repeat pipeline for remaining 5 periods**
15. **Cross-period analysis** (how do results vary across periods?)
16. **Period length ablation**
17. **Compile results for paper**

### Dependency Graph

```
[Generate A+B data] ──► [Implement C,E,G] ──► [Generate full dataset] ──► [Train + Eval]
        │                                                                       │
        ▼                                                                       ▼
[Base training] ──► [Base eval (Tier 1 + LAB)] ──────────────────────► [Ablation studies]
        │
        ▼
[Implement D,F,H] ──► [Multi-document sampling] ──► [Generate D,F,H data]
```

---

## 6. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **API rate limits** | Medium | Delays data generation | Use Batch API (higher limits), generate off-peak hours |
| **OCR noise in early periods** | High | Lower quality for 1678-1849 | Accept lower quality; report per-period metrics honestly |
| **Insufficient diversity** | Medium | Model overfits to narrow question types | Evol-Instruct post-processing; explicit diversity controls in prompts |
| **LAB filter false positives** | Low | Remove good training examples | Conservative default (keep on parse error); inspect removed items |
| **LAB filter false negatives** | Medium | Leave contaminated examples | Cross-check with n-gram decontamination; manual spot-checks |
| **Insufficient training data for narrow periods** | High for 2000-2009, 2010-2023 | Model underfits | Adjust quality thresholds; consider wider period boundaries |
| **Generator H quality** | Medium | Refusal training doesn't generalize | Test with diverse post-period probes; iterate on prompts |
| **Cost overrun** | Low | Budget exceeded | Monitor API usage; start with 1 period, scale after validation |

---

## 7. Timeline

| Week | Focus | Deliverables |
|------|-------|-------------|
| 1 | Generator implementation (C, E, G) | Extended `run_direct.py` with 3 new generators |
| 2 | Generator implementation (D, F, H) + base training | All 8 generators working; base model checkpoint |
| 3 | Full data generation (1950-1999) | ~1M synthetic examples across all 8 types |
| 4 | Quality pipeline + mid-training | Filtered/deduplicated dataset; mid-training checkpoint |
| 5 | SFT + evaluation | Final model; full Tier 1-3 evaluation |
| 6 | Ablation studies | Generator ablation, volume ablation results |
| 7 | Multi-period generation | Data generated for remaining 5 periods |
| 8 | Multi-period training + paper writing | Cross-period results; draft paper sections |

---

## References

- `03_SYNTHETIC_DATA_GENERATORS.md` — Generator specifications and implementation plan
- `05_EVALUATION_FRAMEWORK.md` — Evaluation benchmarks and metrics
- `nanochat/runs/speedrun_hist_llm.sh` — Training script with curriculum configuration
- Zhou et al. (2023). LIMA. NeurIPS 2023 (SFT volume scaling)
