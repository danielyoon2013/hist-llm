# Corpus and Pretraining

> **Paper section:** 3.1 (Corpus and Pretraining)
> **Source code:** `src/base_training/`, `src/post_training/config.py`
> **Technical reference:** `../src/README.md`

---

## 1. Corpus Overview

The English Historical Corpus spans **1678-2023** and contains approximately **125 million documents**. It is sourced from digitized archives of newspapers, periodicals, legal filings, books, patents, and academic publications.

### Source Breakdown

| Source Type | Collections | Coverage |
|-------------|-------------|----------|
| **News (periodicals)** | The Economist, New York Times (filtered), Financial Times | Varies by collection; strongest post-1850 |
| **News (wire services)** | Newswire archives | 1878-1926+ |
| **Newspapers** | US Newspapers, French Newspapers | Broad historical coverage |
| **Legal** | Caselaw Access Project, USPTO (patents), GATT, EurLex, Eurovoc | Primarily 20th century+ |
| **Academic** | Books, Science Pile, Open Science Pile, OpenAlex | Varies |

### Data Locations

- **Raw corpus:** `D:\hist_LLM\corpus\raw\{year}\subset_*.parquet`
- **Code:** `C:\Users\danielyoon\Dropbox\hist_LLM\src\` (synced via Dropbox)
- **Processing intermediates:** `D:\hist_LLM\processing\`
- **Final training shards:** `D:\hist_LLM\periods\{period}\base_data\`

---

## 2. Period Definitions

### Analysis Periods (6 total — used for model training)

These define the temporal boundaries for each model variant. A model trained for a given period sees **only** documents from within that period.

| Period | Start | End | Approximate Span |
|--------|-------|-----|------------------|
| `1678_1849` | 1678 | 1849 | 171 years |
| `1850_1899` | 1850 | 1899 | 50 years |
| `1900_1949` | 1900 | 1949 | 50 years |
| `1950_1999` | 1950 | 1999 | 50 years |
| `2000_2009` | 2000 | 2009 | 10 years |
| `2010_2023` | 2010 | 2023 | 14 years |

Source: `src/post_training/config.py`, lines 25-32

### Quality Periods (14 total — used for Ridge quality models)

Finer-grained 25-year windows for training period-specific quality classifiers. Each period gets its own Ridge regression model because document quality characteristics (OCR quality, writing style, format conventions) change over time.

```
1678_1700, 1701_1725, 1726_1750, 1751_1775, 1776_1800,
1801_1825, 1826_1850, 1851_1875, 1876_1900, 1901_1925,
1926_1950, 1951_1975, 1976_2000, 2001_2023
```

### Period Boundary Justification

- **1678-1849:** Sparse digitized text, high OCR noise, combined into a single large period
- **1850-1899 / 1900-1949 / 1950-1999:** 50-year windows balance document density with manageable period scope
- **2000-2009 / 2010-2023:** Shorter periods due to exponential growth in document volume and the research focus on avoiding modern data contamination

---

## 3. Cleaning Pipeline

### Heuristic Text Cleaning (Step 0)

Three-stage filter applied to all raw documents. Each stage produces a binary keep/reject mask.

| Stage | Filter | Threshold | What It Catches |
|-------|--------|-----------|-----------------|
| 1 | Symbol ratio | >25% non-word characters | Garbled OCR, encoding errors |
| 2 | Punctuation density | <12% lines ending in `.?!` | Tables, lists, structured data, metadata |
| 3 | Stopword count | <3 common English stopwords | Non-English text, boilerplate, short fragments |

**Implementation:** `src/base_training/cleaning/clean_data_helper.py` → `compute_clean_mask()`
**Output:** Per-year cleaning masks at `D:\hist_LLM\corpus\cleaning_masks\{year}\*_mask.parquet`

### Design Decisions

- Conservative thresholds to minimize false negatives (better to include marginal text than lose good content)
- Applied uniformly across all years — no period-specific tuning
- Cleaning masks are stored separately from raw data, allowing re-runs with different thresholds without re-downloading

---

## 4. Quality Filtering Pipeline

### Overview

A 7-step pipeline that transforms raw text into quality-scored, sharded training data.

```
Raw corpus → Clean → Embed → Sample → Label → Train Ridge → Classify → Shard
  (Step 0)  (Step 0.5) (Step 1) (Step 2) (Steps 3-4)   (Step 5)  (Steps 6-7)
```

### Step 0.5: BGE Embedding Generation

- Model: BGE (BAAI General Embedding), 1024-dimensional
- Applied to all clean documents
- Requires GPU for efficient processing
- Output: `D:\hist_LLM\corpus\embeddings\embeddings_{year}.parquet`
- Implementation: `src/base_training/embeddings/run_embeddings_fast.py`

### Step 1: Sample Documents for Labeling

- **10,000 documents** sampled per 25-year quality period
- Stratified sampling to ensure representation across the period
- Implementation: `src/base_training/quality/Sample_Data.ipynb`

### Step 2: GPT-4o-mini Quality Labeling

- Each sampled document rated on a **1-5 quality scale** by GPT-4o-mini
- Uses OpenAI Batch API for 50% cost savings
- Rating criteria: coherence, informativeness, writing quality, factual density
- Output: `D:\hist_LLM\processing\label_data\labeled_data_{period}.parquet`

### Steps 3-4: Train Ridge Quality Models

- **Step 3:** Join GPT quality labels with BGE embeddings to create training data
- **Step 4:** Train one Ridge regression model per 25-year period (14 models total)
  - Input: 1024-dim BGE embeddings
  - Output: Predicted quality score (continuous, 1-5 scale)
  - Ridge chosen for speed and robustness with high-dimensional embeddings
- Implementation: `src/base_training/quality/train_ridge_models.py`
- Models saved: `D:\hist_LLM\processing\quality_models\{ridge,scaler}_{period}.pkl`

### Step 5: Classify All Documents

- Apply trained Ridge models to every clean, embedded document in the corpus
- Each document receives a predicted quality score
- Output: `D:\hist_LLM\corpus\classified\classified_{year}.parquet`

### Step 6: Cumulative Token Analysis

- For each analysis period, compute the cumulative token count as a function of quality threshold
- Determines the optimal quality cutoff: the threshold that retains enough tokens for training while excluding low-quality text
- Produces period-specific graphs stored in `D:\hist_LLM\processing\quality_graphs\`
- Implementation: `src/base_training/analysis/plot_cumulative_tokens.py`

### Step 7: Shard Training Data

- Quality-filtered documents (above the determined threshold) are packed into Parquet shards
- Each shard: ~250M characters, zstd compressed
- Output: `D:\hist_LLM\periods\{period}\base_data\shard_{NNNNN}.parquet`
- Implementation: `src/base_training/sharding/prepare_training_data.py`

---

## 5. Base Training Configuration

### Model Architecture

The model uses the **nanochat** framework (based on Karpathy's nanoGPT lineage):

- Architecture: Dense decoder-only Transformer (GPT-2 family)
- Key modules: `nanochat/nanochat/gpt.py` (model), `optim.py` (AdamW + Muon optimizer), `dataloader.py` (distributed tokenizing dataloader)
- Training scripts: `nanochat/scripts/base_train.py`

### Training Approach: Continued Pretraining

Rather than training from scratch, we continue pre-training from an existing checkpoint on our period-specific corpus. This leverages general language understanding while specializing to historical text.

### Curriculum Training

The training script (`nanochat/runs/speedrun_hist_llm.sh`) uses a single TaskSequence curriculum with 36 datasets ordered by type:

1. Historical Corpus News (6 collections)
2. Historical Corpus Legal (5 collections)
3. Academic sources (4 collections)
4. Math (2 datasets)
5. Logic (3 datasets)
6. Science (3 datasets)
7. Programming (1 dataset)
8. Additional datasets

Eval + checkpoint saves occur at each dataset boundary.

---

## 6. Temporal Isolation in Base Training

### How Period Boundaries Are Enforced

1. **Document selection:** Only documents with publication dates within the period's `[start_year, end_year]` range are included in training shards
2. **Quality model isolation:** Ridge models are trained per 25-year period, so quality assessment itself is period-appropriate
3. **No cross-period data mixing:** Each model variant gets its own set of shards, completely disjoint from other periods

### What Is NOT Isolated

- **Tokenizer:** The BPE tokenizer is shared across all periods (trained on a mixed corpus). This means modern tokens exist in the vocabulary even for early periods. This is a known limitation — training period-specific tokenizers would add significant complexity for marginal benefit.
- **Model initialization:** If using continued pre-training from a general checkpoint, the initial weights encode knowledge from the pre-training corpus. The continued pre-training is intended to shift the distribution toward period-specific content.

---

## 7. Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **OCR noise in early periods** (pre-1850) | Garbled text, inconsistent character encoding | Heuristic cleaning filters; quality scoring penalizes noisy text |
| **Uneven document density** | 1678-1849 has far fewer documents than 1950-1999 | Longer period spans for sparse eras; quality thresholds adjusted per period |
| **Publication date accuracy** | Some documents may have incorrect date metadata | Cross-reference multiple metadata fields; LAB eval catches temporal leakage |
| **Source bias** | English-language sources over-represent UK/US perspectives | Acknowledged limitation; not addressable without multilingual corpus |
| **Shared tokenizer** | Modern terms have BPE tokens even in early period models | Accepted tradeoff; tokenizer vocabulary has minimal impact on generated content |
| **Quality label subjectivity** | GPT-4o-mini's quality ratings may not align with human judgment | Ridge models smooth individual rating noise; validated by cumulative token analysis |

---

## 8. Key Statistics (to be filled per period)

| Period | Raw Docs | After Cleaning | After Quality Filter | Training Tokens | Shards |
|--------|----------|----------------|---------------------|-----------------|--------|
| 1678_1849 | | | | | |
| 1850_1899 | | | | | |
| 1900_1949 | | | | | |
| 1950_1999 | ~4.3M | | | | |
| 2000_2009 | | | | | |
| 2010_2023 | | | | | |

*Table to be populated with actual pipeline output statistics as each period is processed.*

---

## References

- `../src/README.md` — Full technical pipeline documentation
- `src/base_training/` — All base training code
- `src/post_training/config.py` — Period definitions and path conventions
