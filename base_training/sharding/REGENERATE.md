# Regenerating `periods/*/base_data` (and related derived data)

`D:\hist_LLM\periods\{period}\base_data\shard_*.parquet` is a **derived intermediate**
— the quality-filtered, text-joined, sharded training input for nanochat base training.
It is **safe to delete to free disk** because it can be regenerated from inputs that are
kept, and because the trained **models** (the real deliverables) are saved separately in
`periods\{period}\model\`. Deleting base_data costs only a re-run, never a retrain.

## Regenerate
```bash
cd src/base_training/sharding
python prepare_base_data.py                     # all 6 periods (uses period_summary.csv cutoffs)
python prepare_base_data.py --period 1900_1949  # a single period
```
Output: `D:\hist_LLM\periods\{period}\base_data\shard_{NNNNN}.parquet` (text column only).
Time: roughly tens of minutes per period (reads raw text for the selected docs + shards).

## Required inputs (all currently kept on D:)
- `D:\hist_LLM\corpus\classified\classified_{year}.parquet` — per-doc quality scores (clean docs).
- `D:\hist_LLM\corpus\raw\{year}\subset_*.parquet` — the source text (joined by `identifier`).
- `D:\hist_LLM\processing\quality_graphs\period_summary.csv` — the per-period `cutoff_score`
  (the 20B-token quality threshold) that defines which docs are selected.
- Script: `src/base_training/sharding/prepare_base_data.py` (committed).

If `period_summary.csv` is ever lost, it is itself regenerable:
`python src/base_training/analysis/compute_quality_cutoffs.py` (from `classified\`).
If `classified\` is lost, re-run `src/base_training/quality/check_and_classify.py` (needs
`corpus\embeddings\` + the Ridge models in `processing\quality_models\`).

## Reproducibility notes
- **Document selection is deterministic** (cutoff-based) → same docs given the same
  `period_summary.csv` cutoffs.
- **Shard assignment/order** use `np.random.seed(42)` with deterministic load order
  (sorted files, ordered reads) → same content per shard.
- **One known difference:** `ROW_GROUP_SIZE` was changed 1024 → 128 (DDP row-group fix),
  so regenerated shards have finer internal row groups. Same documents and shard
  membership; functionally identical for training (and required for 8-GPU DDP).
- Not guaranteed *byte-identical* across pyarrow/pandas versions, but **content-identical**.

## Dependency chain (what regenerates what)
```
corpus\raw + corpus\embeddings
        └─ quality_models (Ridge)  ──► classified\  ──► period_summary.csv (cutoffs)
                                                  └────────────┬──────────────┘
                                                               ▼
                                              periods\{period}\base_data\  (this file's subject)
```
Keep `raw\` (source, not regenerable) and either `embeddings\`+`quality_models\` OR
`classified\` (cheaper). With `classified\` + `period_summary.csv` + `raw\`, base_data
regenerates directly.
