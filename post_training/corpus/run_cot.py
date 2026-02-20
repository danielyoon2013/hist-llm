"""
Run CoT generation only on already-parsed documents.

Use this when QA was already generated (e.g., with --skip-cot) and you
want to add CoT without re-doing ingest + QA.

Usage:
    python -m src.post_training.corpus.run_cot --period 1950_1999 --collection nyt_filtered --workers 8
"""

import os
import sys
import argparse
import subprocess
import time
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.post_training.config import PERIODS, get_paths, load_api_key, PROJECT_ROOT
from src.post_training.corpus.export import get_collection_folders


CONFIG_PATH = PROJECT_ROOT / "src" / "post_training" / "corpus" / "synth_config.yaml"
BATCH_SIZE = 10


def process_cot_batch(args):
    """Generate CoT for a single batch of parsed documents."""
    batch_id, collection_name, batch_parsed_dir, batch_generated_dir, \
        synthetic_dir, config, api_key, num_cot = args

    env = os.environ.copy()
    env["API_ENDPOINT_KEY"] = api_key
    env["PYTHONIOENCODING"] = "utf-8"

    batch_name = f"{collection_name}_cot_batch{batch_id}"

    num_docs = len(list(Path(batch_parsed_dir).glob("*.txt")))
    if num_docs == 0:
        return {"batch": batch_name, "success": True, "num_docs": 0, "time": 0}

    t0 = time.time()
    os.makedirs(batch_generated_dir, exist_ok=True)

    cmd = [
        "synthetic-data-kit", "-c", str(config),
        "create", str(batch_parsed_dir),
        "--type", "cot", "--num-pairs", str(num_cot),
        "--output-dir", str(batch_generated_dir),
    ]

    result = subprocess.run(cmd, cwd=str(synthetic_dir), env=env, capture_output=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        return {"batch": batch_name, "success": False, "num_docs": num_docs, "time": elapsed}
    return {"batch": batch_name, "success": True, "num_docs": num_docs, "time": elapsed}


def run_cot_for_collection(collection_name, synthetic_dir, config, api_key,
                           num_cot=2, workers=4, batch_size=BATCH_SIZE):
    """Run CoT generation on already-parsed files for a collection."""
    parsed_dir = synthetic_dir / "parsed" / collection_name
    generated_dir = synthetic_dir / "generated" / collection_name

    if not parsed_dir.exists():
        print(f"Error: Parsed directory not found: {parsed_dir}")
        print("Run the full pipeline first to generate parsed files.")
        return False

    txt_files = sorted(parsed_dir.glob("*.txt"))
    num_docs = len(txt_files)
    print(f"\nCollection: {collection_name}")
    print(f"Parsed dir: {parsed_dir}")
    print(f"Documents: {num_docs}")
    print(f"Workers: {workers}, Batch size: {batch_size}")

    if num_docs == 0:
        print("No parsed documents found.")
        return True

    # Split parsed files into batches
    batch_temp_dir = synthetic_dir / "_cot_batches" / collection_name
    if batch_temp_dir.exists():
        shutil.rmtree(batch_temp_dir)

    batch_parsed_base = batch_temp_dir / "parsed"
    batch_generated_base = batch_temp_dir / "generated"

    batches = []
    for i in range(0, len(txt_files), batch_size):
        batch_files = txt_files[i:i + batch_size]
        batch_id = i // batch_size
        batch_parsed = batch_parsed_base / f"batch_{batch_id:04d}"
        batch_generated = batch_generated_base / f"batch_{batch_id:04d}"
        os.makedirs(batch_parsed, exist_ok=True)

        for f in batch_files:
            shutil.copy(f, batch_parsed / f.name)

        batches.append((
            batch_id, collection_name, str(batch_parsed), str(batch_generated),
            str(synthetic_dir), str(config), api_key, num_cot
        ))

    print(f"Created {len(batches)} batches")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Process in parallel
    start_time = time.time()
    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_cot_batch, args): args[0] for args in batches}

        for future in as_completed(futures):
            batch_id = futures[future]
            try:
                result = future.result()
                if result["success"]:
                    completed += 1
                    docs_done = completed * batch_size
                    docs_done = min(docs_done, num_docs)
                    elapsed = time.time() - start_time
                    rate = docs_done / elapsed if elapsed > 0 else 0
                    eta = (num_docs - docs_done) / rate if rate > 0 else 0
                    eta_time = datetime.now() + timedelta(seconds=eta)

                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] Batch {batch_id:3d} done: "
                          f"{result['num_docs']} docs in {result.get('time', 0):.1f}s | "
                          f"Progress: {docs_done:,}/{num_docs:,} ({100*docs_done/num_docs:.1f}%) | "
                          f"Rate: {rate:.2f} docs/s | ETA: {eta/60:.1f}m ({eta_time.strftime('%H:%M')})")
                else:
                    failed += 1
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] Batch {batch_id} FAILED")
            except Exception as e:
                failed += 1
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] Batch {batch_id} ERROR: {e}")

    # Merge CoT files to main generated directory
    print(f"Merging CoT results...")
    os.makedirs(generated_dir, exist_ok=True)
    total_copied = 0
    for i in range(len(batches)):
        batch_gen = batch_generated_base / f"batch_{i:04d}"
        if batch_gen.exists():
            for f in sorted(batch_gen.glob("*_cot_*.json")):
                shutil.copy(f, generated_dir / f.name)
                total_copied += 1

    # Cleanup
    shutil.rmtree(batch_temp_dir, ignore_errors=True)

    total_time = time.time() - start_time
    print(f"\nCoT generation complete!")
    print(f"  {completed}/{len(batches)} batches successful, {failed} failed")
    print(f"  {total_copied} CoT files written to {generated_dir}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")

    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CoT generation only on already-parsed documents"
    )
    parser.add_argument("--period", type=str, required=True,
                        choices=list(PERIODS.keys()))
    parser.add_argument("--collection", type=str, required=True,
                        help="Collection to process")
    parser.add_argument("--num-cot", type=int, default=2,
                        help="CoT examples per chunk (default: 2)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Documents per batch (default: {BATCH_SIZE})")
    args = parser.parse_args()

    paths = get_paths(args.period)
    api_key = load_api_key()

    print(f"Period: {args.period} ({paths['start_year']}-{paths['end_year']})")

    ok = run_cot_for_collection(
        args.collection,
        paths["synthetic_dir"],
        CONFIG_PATH,
        api_key,
        num_cot=args.num_cot,
        workers=args.workers,
        batch_size=args.batch_size,
    )

    if not ok:
        sys.exit(1)
