# Corpus Summary: Document Counts and Generation Budget

Generated from `check_collection_counts.ipynb` data. Shows how many documents are available per collection, how many survive quality filtering, and how many are selected for synthetic data generation.

## Pipeline: Raw -> Quality Filter -> Cap -> Export -> Generate

1. **Raw**: Total documents in the corpus for this period
2. **After QF**: Documents passing the top-50% quality filter (embedding classifier)
3. **Cap 10K**: Capped at 10,000 per collection (current default in `export.py`)
4. **Exported**: Actually written to `synthetic/input/` (0 = not yet exported for this period)

---

## Per-Period Summary

| Period | Raw Docs | After QF | Cap 10K (Selected) | Exported | Est. Items (at cap) |
|--------|----------:|---------:|-------------------:|---------:|--------------------:|
| 1678_1849 | 3,562,142 | 1,781,071 | 69,977 | 0 | ~2,519,172 |
| 1850_1899 | 10,238,352 | 5,119,176 | 97,510 | 0 | ~3,510,360 |
| 1900_1949 | 14,449,815 | 7,244,911 | 150,025 | 139,929 | ~5,400,900 |
| 1950_1999 | 6,033,969 | 3,036,986 | 66,767 | 56,267 | ~2,403,612 |
| 2000_2009 | 10,922,692 | 5,461,351 | 35,302 | 0 | ~1,270,872 |
| 2010_2023 | 39,230,164 | 19,615,087 | 109,649 | 0 | ~3,947,364 |
| **Total** | **84,437,134** | **42,258,582** | **529,230** | **196,196** | **~19,052,280** |

**Est. Items** = selected docs x 2 chunks/doc x 6 corpus-based generators x 3 items/chunk. Metadata-based generators (D, H) add ~300 items per period on top.

---

## Generation Budget Analysis

| | Per Period Target | Current Cap (10K/collection) | Overproduction |
|---|---:|---:|---:|
| **Target items** | 575,000 | ~2.5M-5.4M | 4x-9x |
| **Docs needed** | ~16,000 | 35K-150K | 2x-9x |

The research doc targets **575K items per period** (500K mid-train + 75K SFT). At ~36 items per document (2 chunks x 6 generators x 3 items/chunk), only **~16K documents** are needed. The current 10K-per-collection cap significantly overproduces.

### Recommended `--max-docs` Settings

| Setting | Docs Used | Est. Items | Est. Cost (Batch API) | Notes |
|---------|----------:|-----------:|----------------------:|-------|
| `--max-docs 3` | 3 | ~108 | < $0.01 | Testing only |
| `--max-docs 500` | ~500 | ~18,000 | ~$4 | Quick validation |
| `--max-docs 5000` | ~5,000 | ~180,000 | ~$44 | Lean run (1/3 target) |
| `--max-docs 16000` | ~16,000 | ~575,000 | ~$140 | Hits 575K target |
| Full cap (10K/coll) | 35K-150K | 1.3M-5.4M | ~$150-$1,300 | Massive oversupply |

Costs assume GPT-4o-mini via **Batch API** (50% discount): $0.075/1M input, $0.30/1M output.

---

## Detailed Breakdown: 1900_1949 (Reference Period)

### Corpus Collections (sorted by selected count)

| Collection | Type | Raw | After QF | Selected (Cap 10K) |
|------------|------|----:|--------:|---------:|
| Caselaw Access Project | corpus | 1,031,666 | 993,136 | 10,000 |
| English-PD | corpus | 2,009,159 | 1,762,503 | 10,000 |
| LoC-PD-Books | corpus | 345,602 | 283,889 | 10,000 |
| NewZealand-PD-Newspapers | corpus | 63,729 | 37,666 | 10,000 |
| Open-Science-Pile | corpus | 13,125 | 10,618 | 10,000 |
| US-PD-Books | corpus | 582,485 | 487,732 | 10,000 |
| US-PD-Newspapers | corpus | 8,539,758 | 3,038,031 | 10,000 |
| USPTO | corpus | 1,784,114 | 561,311 | 10,000 |
| German-PD | corpus | 9,348 | 8,133 | 8,133 |
| French-PD-diverse | corpus | 10,031 | 6,255 | 6,255 |
| Multilingual-PD | corpus | 6,037 | 5,123 | 5,123 |
| Greek-PD | corpus | 5,205 | 4,622 | 4,622 |
| GATT_library | corpus | 2,009 | 1,279 | 1,279 |
| Wikisource | corpus | 1,275 | 1,134 | 1,134 |
| French-Science-Pile | corpus | 1,570 | 1,083 | 1,083 |
| Italian-PD | corpus | 1,265 | 1,065 | 1,065 |
| Spanish-PD-Books | corpus | 825 | 451 | 451 |
| French-PD-Newspapers | corpus | 1,518 | 434 | 434 |
| Spanish-Science-Pile | corpus | 260 | 193 | 193 |
| OpenAlex | corpus | 154 | 112 | 112 |
| Spanish-PD-Newspapers | corpus | 219 | 70 | 70 |
| Danish-PD | corpus | 49 | 32 | 32 |
| German-PD-Newspapers | corpus | 360 | 27 | 27 |
| Portuguese-PD | corpus | 17 | 9 | 9 |
| Arabic-PD | corpus | 2 | 2 | 2 |
| Europeana | corpus | 28 | 1 | 1 |
| **Corpus subtotal** | | **14,409,815** | **7,204,911** | **110,025** |

### News Archives

| Dataset | Raw | Selected (Cap 10K) |
|---------|----:|---------:|
| economist | 10,000 | 10,000 |
| ft | 10,000 | 10,000 |
| newswire | 10,000 | 10,000 |
| nyt_filtered | 10,000 | 10,000 |
| **News subtotal** | **40,000** | **40,000** |

### Grand Total (1900_1949)

| | Count |
|---|---:|
| Corpus collections selected | 110,025 |
| News archives selected | 40,000 |
| **Total selected** | **150,025** |
| Est. chunks (~2/doc) | ~300,050 |
| Est. items (6 gens x 3/chunk) | ~5,400,900 |
| + Metadata items (D + H) | ~300 |

---

## Mid-Training / SFT Allocation (Research Plan)

Target mix ratios from `research/03_SYNTHETIC_DATA_GENERATORS.md`:

| Generator | Mid-Train % | Mid-Train Count | SFT % | SFT Count | Total |
|-----------|------:|----------:|------:|----------:|------:|
| A. Factual QA | 30% | 150,000 | 15% | 11,250 | 161,250 |
| B. Chain-of-Thought | 15% | 75,000 | 25% | 18,750 | 93,750 |
| C. Reading Comprehension | 15% | 75,000 | 10% | 7,500 | 82,500 |
| D. Temporal Reasoning | 10% | 50,000 | 15% | 11,250 | 61,250 |
| E. Quantitative | 5% | 25,000 | 5% | 3,750 | 28,750 |
| F. Sentence Completion | 5% | 25,000 | 5% | 3,750 | 28,750 |
| G. Instruction Following | 5% | 25,000 | 15% | 11,250 | 36,250 |
| H. Historical Facts | 5% | 25,000 | 5% | 3,750 | 28,750 |
| GSM8K (external) | 5% | 7,285 | 3% | 2,250 | 9,535 |
| MATH (external) | 5% | 7,477 | 2% | 1,500 | 8,977 |
| **Total** | | **~464K** | | **~75K** | **~539K** |

**Notes:**
- Gen H (Historical Facts) is **train-only** — all items go to mid-train + SFT, none held out for test
- Gen D and H are metadata-based — their output is fixed by the year range, not corpus size
- GSM8K/MATH are capped at their actual dataset size after LAB temporal filtering
- The test split (5% of non-H generators) is used for training loss monitoring, not evaluation
- Evaluation uses external benchmarks: MMLU, ARC, HellaSwag, PIQA, WinoGrande, GSM8K, RACE, BoolQ, LAB Eval

---

## API Cost Estimate (All 6 Periods)

Using `--max-docs 16000` to hit the 575K target per period:

| | Per Period | x 6 Periods |
|---|---:|---:|
| Regular API (GPT-4o-mini) | ~$280 | ~$1,680 |
| **Batch API (50% off)** | **~$140** | **~$840** |

At full 10K/collection cap (current default):

| | Per Period (avg) | x 6 Periods |
|---|---:|---:|
| Regular API | ~$500-2,600 | ~$6,000-15,600 |
| **Batch API** | **~$250-1,300** | **~$3,000-7,800** |
