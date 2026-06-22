r"""
Prepare PER-YEAR training data for continual point-in-time pretraining (1850-2023).

For each year:
  1. Rank ALL docs by expected quality (corpus/classified_all/classified_{year}.parquet)
  2. Select the top docs until cumulative tokens reach min(year_total_tokens, --cap)  [cap default 1B]
  3. Join their text from corpus/raw/{year}/ (reuses prepare_base_data helpers)
  4. Write YEAR-PREFIXED shards `shard_{year}_{NNNNN}.parquet` into ONE base_data dir,
     so the (filename-sorted, non-shuffling) nanochat dataloader streams them in
     temporal order with zero dataloader changes.

Also emits:
  - manifest.csv          : per-year n_docs, tokens_used, cutoff_quality, n_shards, cum_tokens
  - tok_sample/base_data/ : year-balanced sample for training ONE tokenizer
  - base_data/zz_val.parquet : small held-out val shard (sorts last -> dataloader's val split)

Memory-safe: streams each year's texts straight into shards (never holds a whole year).

Usage:
  python prepare_yearly_data.py                         # all years 1850-2023
  python prepare_yearly_data.py --years 1850,1851       # subset (testing)
  python prepare_yearly_data.py --cap 1000000000        # tokens/year cap
"""
import os, sys, gc, argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Reuse the proven helpers/constants from prepare_base_data.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_base_data import _read_matching_texts, RAW_DIR, ROW_GROUP_SIZE, SHARD_SIZE_CHARS, FILE_READ_WORKERS

# --- CONFIG ---
CLASSIFIED_ALL_DIR = Path(r"D:\hist_LLM\corpus\classified_all")
OUTPUT_ROOT = Path(r"D:\hist_LLM\continual")
DATA_DIR = OUTPUT_ROOT / "data" / "base_data"          # single dir, year-prefixed shards
TOK_DIR = OUTPUT_ROOT / "tok_sample" / "base_data"     # tokenizer-training sample
MANIFEST = OUTPUT_ROOT / "manifest.csv"

DEFAULT_CAP = 1_000_000_000        # tokens per year
START_YEAR, END_YEAR = 1850, 2023
TOK_CHARS_PER_YEAR = 12_000_000    # ~12M chars/yr * 174 yrs ~= 2.1B chars for tokenizer
VAL_DOCS_PER_YEAR = 30             # ~30 * 174 ~= 5200 val docs (held-out health metric)


def select_top_ids(year: int, cap: int):
    """Rank all docs by quality; return (id_set, tokens_used, n_docs, cutoff_q, total_tokens)."""
    path = CLASSIFIED_ALL_DIR / f"classified_{year}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["identifier", "predicted_quality", "token_count"])
    df = df.dropna(subset=["token_count"])
    if len(df) == 0:
        return None
    df = df.sort_values("predicted_quality", ascending=False, kind="mergesort").reset_index(drop=True)
    total = int(df["token_count"].sum())
    cum = df["token_count"].cumsum().values
    if total <= cap:
        n = len(df)
    else:
        n = int((cum >= cap).argmax()) + 1
    sel = df.iloc[:n]
    id_set = set(sel["identifier"].astype(str).tolist())
    tokens_used = int(cum[n - 1])
    cutoff_q = float(sel["predicted_quality"].iloc[-1])
    return id_set, tokens_used, n, cutoff_q, total


class YearSharder:
    """Stream texts into year-prefixed shards of ~SHARD_SIZE_CHARS, shuffled within shard."""
    def __init__(self, year, data_dir):
        self.year = year; self.dir = data_dir
        self.buf = []; self.chars = 0; self.idx = 0; self.nshards = 0
    def add(self, texts):
        for t in texts:
            if not t:
                continue
            self.buf.append(t); self.chars += len(t)
            if self.chars >= SHARD_SIZE_CHARS:
                self._flush()
    def _flush(self):
        if not self.buf:
            return
        np.random.shuffle(self.buf)
        path = self.dir / f"shard_{self.year}_{self.idx:05d}.parquet"
        pq.write_table(pa.Table.from_pydict({"text": self.buf}), path,
                       row_group_size=ROW_GROUP_SIZE, use_dictionary=False,
                       compression="zstd", compression_level=3, write_statistics=False)
        self.idx += 1; self.nshards += 1; self.buf = []; self.chars = 0
    def close(self):
        self._flush()
        return self.nshards


def process_year(year, cap, read_workers=FILE_READ_WORKERS):
    """Select + shard one year. Writes its shards + tok_{year}.parquet (unique names, so
    workers don't conflict). Returns (manifest_dict_or_None, val_texts_list)."""
    np.random.seed(42 + int(year))  # deterministic within-shard shuffle, distinct per year
    sel = select_top_ids(year, cap)
    if sel is None:
        print(f"  [SKIP] {year}: no classified_all data", flush=True); return None, []
    id_set, tokens_used, n_docs, cutoff_q, total = sel

    year_dir = RAW_DIR / str(year)
    files = sorted(year_dir.glob("*.parquet")) if year_dir.exists() else []
    if not files:
        print(f"  [SKIP] {year}: no raw text", flush=True); return None, []

    sharder = YearSharder(year, DATA_DIR)
    tok_texts = []; tok_chars = 0; val_texts = []; val_count = 0
    args = [(f, id_set) for f in files]
    with ThreadPoolExecutor(max_workers=read_workers) as ex:
        for texts in ex.map(_read_matching_texts, args):
            sharder.add(texts)
            # sample for tokenizer (year-balanced) and val (small) without a 2nd pass
            for t in texts:
                if not t:
                    continue
                if tok_chars < TOK_CHARS_PER_YEAR:
                    tok_texts.append(t); tok_chars += len(t)
                if val_count < VAL_DOCS_PER_YEAR:
                    val_texts.append(t); val_count += 1
            del texts
    n_shards = sharder.close()
    write_sample_shards(TOK_DIR, tok_texts, f"tok_{year}")  # worker-local, unique name
    gc.collect()
    print(f"  {year}: {n_docs:,} docs, {tokens_used/1e6:.0f}M tok "
          f"(avail {total/1e6:.0f}M), cutoff_q={cutoff_q:.3f}, {n_shards} shards", flush=True)
    return dict(year=year, n_docs=n_docs, tokens_used=tokens_used, total_tokens=total,
                cutoff_quality=round(cutoff_q, 4), n_shards=n_shards), val_texts


def _worker(arg):
    """Top-level worker for ProcessPoolExecutor (must be picklable/importable)."""
    year, cap, read_workers = arg
    return process_year(year, cap, read_workers)


def write_sample_shards(out_dir: Path, texts: list, prefix: str, row_group_size=ROW_GROUP_SIZE):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not texts:
        return
    pq.write_table(pa.Table.from_pydict({"text": texts}), out_dir / f"{prefix}.parquet",
                   row_group_size=row_group_size, use_dictionary=False,
                   compression="zstd", compression_level=3, write_statistics=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=START_YEAR)
    ap.add_argument("--end", type=int, default=END_YEAR)
    ap.add_argument("--years", type=str, default=None, help="comma-sep subset for testing")
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP, help="tokens per year cap")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel YEAR workers (ProcessPool). USB-SSD bound, 4-6 is a good range.")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TOK_DIR.mkdir(parents=True, exist_ok=True)
    years = ([int(y) for y in args.years.split(",")] if args.years
             else list(range(args.start, args.end + 1)))

    rows, val_texts = [], []
    if args.workers > 1:
        # split the per-year read threads across workers so we don't oversubscribe the disk
        rpw = max(2, FILE_READ_WORKERS // args.workers)
        print(f"Parallel: {args.workers} year-workers x {rpw} read-threads", flush=True)
        tasks = [(y, args.cap, rpw) for y in years]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for d, v in ex.map(_worker, tasks):
                if d is not None:
                    rows.append(d); val_texts.extend(v)
    else:
        for y in tqdm(years, desc="Years"):
            d, v = process_year(y, args.cap)
            if d is not None:
                rows.append(d); val_texts.extend(v)

    # held-out val shard: name sorts AFTER all shard_{year}_* so it becomes the val split.
    # The val loader also strides row groups across ranks, so it needs >= num_gpus row
    # groups -> pick a small row_group_size to guarantee ~16 groups even for few docs.
    np.random.shuffle(val_texts)
    val_rgsize = min(ROW_GROUP_SIZE, max(1, len(val_texts) // 16)) if val_texts else ROW_GROUP_SIZE
    write_sample_shards(DATA_DIR, val_texts, "zz_val", row_group_size=val_rgsize)

    # manifest with cumulative tokens (used later to derive year-boundary checkpoint steps)
    if rows:
        mdf = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
        mdf["cum_tokens"] = mdf["tokens_used"].cumsum()
        # merge-preserve if re-running a subset
        if MANIFEST.exists() and args.years:
            old = pd.read_csv(MANIFEST)
            mdf = (pd.concat([old, mdf]).drop_duplicates("year", keep="last")
                   .sort_values("year").reset_index(drop=True))
            mdf["cum_tokens"] = mdf["tokens_used"].cumsum()
        mdf.to_csv(MANIFEST, index=False)
        print(f"\nTotal training tokens: {mdf['tokens_used'].sum()/1e9:.1f}B | "
              f"years: {len(mdf)} | manifest: {MANIFEST}")
        print(f"val docs: {len(val_texts):,} -> {DATA_DIR / 'zz_val.parquet'}")


if __name__ == "__main__":
    main()
