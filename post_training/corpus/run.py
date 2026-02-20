"""
End-to-end runner for the synthetic-data-kit pipeline.

Runs all stages: export docs → ingest → create QA → create CoT → curate → convert

Supports parallel processing via --workers flag for faster execution.

Usage:
    # Test with small sample (2 docs per collection)
    python -m src.post_training.corpus.run --period 1950_1999 --max-per-collection 2

    # Production run with parallelization
    python -m src.post_training.corpus.run --period 1950_1999 --workers 8

    # Run single collection
    python -m src.post_training.corpus.run --period 1950_1999 --collection USPTO --workers 4
"""

import os
import sys
import argparse
import subprocess
import time
import shutil
import json
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.post_training.config import PERIODS, get_paths, load_api_key, PROJECT_ROOT
from src.post_training.corpus.export import (
    export_documents_stratified,
    get_collection_folders,
    sanitize_collection_name,
)


CONFIG_PATH = PROJECT_ROOT / "src" / "post_training" / "corpus" / "synth_config.yaml"
BATCH_SIZE = 10  # docs per batch for parallel processing (smaller = more parallelism)


def write_progress(progress_file, data):
    """Write progress to a JSON file for monitoring."""
    with open(progress_file, "w") as f:
        json.dump(data, f, indent=2, default=str)


def format_eta(seconds):
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def run_cmd(cmd, cwd=None, env=None, quiet=False):
    """Run a shell command. Returns True on success."""
    if not quiet:
        print(f"Running: {' '.join(cmd[:5])}...")
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=quiet)
    return result.returncode == 0


def run_curate_batch(args):
    """Run curation on a single batch. Called by parallel workers."""
    import sys
    batch_input_dir, batch_output_dir, config, api_key, threshold = args

    curate_script = PROJECT_ROOT / "src" / "post_training" / "corpus" / "run_curate.py"

    env = os.environ.copy()
    env["API_ENDPOINT_KEY"] = api_key
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        sys.executable, "-X", "utf8", str(curate_script),
        "-c", str(config),
        "curate", str(batch_input_dir),
        "--threshold", str(threshold),
        "--output", str(batch_output_dir),
    ]

    result = subprocess.run(cmd, env=env, capture_output=True)
    return result.returncode == 0


def run_curate_parallel(input_dir, output_dir, config, api_key, threshold=7.0, workers=8, batch_size=50):
    """Run curation in parallel by splitting files into batches."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Get all JSON files to curate
    json_files = sorted(input_dir.glob("*_qa_*.json"))
    if not json_files:
        return True

    # Split into batches
    temp_dir = input_dir.parent / "_curate_batches"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    batches = []
    for i in range(0, len(json_files), batch_size):
        batch_files = json_files[i:i + batch_size]
        batch_id = i // batch_size
        batch_in = temp_dir / f"batch_{batch_id:04d}" / "input"
        batch_out = temp_dir / f"batch_{batch_id:04d}" / "output"
        os.makedirs(batch_in, exist_ok=True)
        os.makedirs(batch_out, exist_ok=True)

        for f in batch_files:
            shutil.copy(f, batch_in / f.name)

        batches.append((str(batch_in), str(batch_out), str(config), api_key, threshold))

    # Process batches in parallel
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = list(executor.map(run_curate_batch, batches))
        completed = sum(futures)

    # Merge results
    for i in range(len(batches)):
        batch_out = temp_dir / f"batch_{i:04d}" / "output"
        if batch_out.exists():
            for f in batch_out.glob("*.json"):
                shutil.copy(f, output_dir / f.name)

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

    return completed == len(batches)


def process_batch(args):
    """Process a single batch of documents. Called by parallel workers."""
    (batch_id, collection_name, batch_input_dir, batch_parsed_dir,
     batch_generated_dir, synthetic_dir, config, api_key,
     num_qa, num_cot, skip_cot) = args

    env = os.environ.copy()
    env["API_ENDPOINT_KEY"] = api_key
    env["PYTHONIOENCODING"] = "utf-8"

    cwd = str(synthetic_dir)
    batch_name = f"{collection_name}_batch{batch_id}"

    # Count docs
    num_docs = len(list(Path(batch_input_dir).glob("*.txt")))
    if num_docs == 0:
        return {"batch": batch_name, "success": True, "num_docs": 0, "time": 0}

    t0 = time.time()

    # Ensure directories exist
    os.makedirs(batch_parsed_dir, exist_ok=True)
    os.makedirs(batch_generated_dir, exist_ok=True)

    # Ingest
    ok = run_cmd([
        "synthetic-data-kit", "-c", str(config),
        "ingest", str(batch_input_dir),
        "--output-dir", str(batch_parsed_dir),
    ], cwd=cwd, env=env, quiet=True)
    if not ok:
        return {"batch": batch_name, "success": False, "stage": "ingest", "num_docs": num_docs}

    # Generate QA
    ok = run_cmd([
        "synthetic-data-kit", "-c", str(config),
        "create", str(batch_parsed_dir),
        "--type", "qa", "--num-pairs", str(num_qa),
        "--output-dir", str(batch_generated_dir),
    ], cwd=cwd, env=env, quiet=True)
    if not ok:
        return {"batch": batch_name, "success": False, "stage": "qa", "num_docs": num_docs}

    # Generate CoT (optional)
    if not skip_cot:
        ok = run_cmd([
            "synthetic-data-kit", "-c", str(config),
            "create", str(batch_parsed_dir),
            "--type", "cot", "--num-pairs", str(num_cot),
            "--output-dir", str(batch_generated_dir),
        ], cwd=cwd, env=env, quiet=True)
        if not ok:
            return {"batch": batch_name, "success": False, "stage": "cot", "num_docs": num_docs}

    elapsed = time.time() - t0
    return {"batch": batch_name, "success": True, "num_docs": num_docs, "time": elapsed}


def split_collection_into_batches(collection_input_dir, batch_base_dir, batch_size=BATCH_SIZE):
    """Split a collection's input files into batch subdirectories."""
    collection_input_dir = Path(collection_input_dir)
    batch_base_dir = Path(batch_base_dir)

    # Get all txt files
    txt_files = sorted(collection_input_dir.glob("*.txt"))
    if not txt_files:
        return []

    # Split into batches
    batches = []
    for i in range(0, len(txt_files), batch_size):
        batch_files = txt_files[i:i + batch_size]
        batch_id = i // batch_size
        batch_dir = batch_base_dir / f"batch_{batch_id:04d}"
        os.makedirs(batch_dir, exist_ok=True)

        # Copy files to batch directory
        for f in batch_files:
            shutil.copy(f, batch_dir / f.name)

        batches.append((batch_id, batch_dir))

    return batches


def run_collection_parallel(
    collection_name,
    synthetic_dir,
    config,
    api_key,
    num_qa=3,
    num_cot=2,
    skip_cot=False,
    skip_curate=False,
    workers=4,
    batch_size=BATCH_SIZE,
    progress_file=None,
):
    """Run pipeline for a collection using parallel batch processing."""
    print(f"\n{'#'*70}")
    print(f"# Processing collection: {collection_name} (parallel, {workers} workers)")
    print(f"{'#'*70}")

    input_dir = synthetic_dir / "input" / collection_name

    if not input_dir.exists():
        print(f"  Input directory not found: {input_dir}")
        return False, {}

    num_docs = len(list(input_dir.glob("*.txt")))
    mode = "QA only" if skip_cot else "QA + CoT"
    print(f"  Input: {num_docs} documents ({mode})")

    if num_docs == 0:
        return True, {"collection": collection_name, "num_docs": 0, "total": 0}

    # For small collections, process directly without batching
    if num_docs <= batch_size:
        print(f"  Small collection, processing directly...")
        return run_collection_sequential(
            collection_name, synthetic_dir, config, api_key,
            num_qa, num_cot, skip_cot
        )

    # Split into batches
    batch_temp_dir = synthetic_dir / "_batches" / collection_name
    if batch_temp_dir.exists():
        shutil.rmtree(batch_temp_dir)

    batch_input_base = batch_temp_dir / "input"
    batch_parsed_base = batch_temp_dir / "parsed"
    batch_generated_base = batch_temp_dir / "generated"

    print(f"  Splitting into batches of {batch_size}...")
    batches = split_collection_into_batches(input_dir, batch_input_base, batch_size)
    print(f"  Created {len(batches)} batches")

    # Prepare batch arguments
    batch_args = []
    for batch_id, batch_input_dir in batches:
        batch_parsed_dir = batch_parsed_base / f"batch_{batch_id:04d}"
        batch_generated_dir = batch_generated_base / f"batch_{batch_id:04d}"

        batch_args.append((
            batch_id, collection_name, str(batch_input_dir),
            str(batch_parsed_dir), str(batch_generated_dir),
            str(synthetic_dir), str(config), api_key,
            num_qa, num_cot, skip_cot
        ))

    # Process batches in parallel
    collection_start = time.time()
    results = []
    completed = 0
    failed = 0

    print(f"  Processing {len(batches)} batches with {workers} workers...")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_batch, args): args[0] for args in batch_args}

        for future in as_completed(futures):
            batch_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                if result["success"]:
                    completed += 1
                    docs_done = sum(r.get("num_docs", 0) for r in results if r["success"])
                    elapsed = time.time() - collection_start
                    rate = docs_done / elapsed if elapsed > 0 else 0
                    eta = (num_docs - docs_done) / rate if rate > 0 else 0
                    eta_time = datetime.now() + timedelta(seconds=eta)

                    print(f"    [{datetime.now().strftime('%H:%M:%S')}] Batch {batch_id:3d} done: "
                          f"{result['num_docs']} docs in {result.get('time', 0):.1f}s | "
                          f"Progress: {docs_done:,}/{num_docs:,} ({100*docs_done/num_docs:.1f}%) | "
                          f"Rate: {rate:.2f} docs/s | ETA: {format_eta(eta)} ({eta_time.strftime('%H:%M')})")

                    # Update progress file
                    if progress_file:
                        write_progress(progress_file, {
                            "collection": collection_name,
                            "docs_done": docs_done,
                            "docs_total": num_docs,
                            "batches_done": completed,
                            "batches_total": len(batches),
                            "rate_docs_per_sec": round(rate, 2),
                            "eta_seconds": round(eta),
                            "eta_finish": eta_time.isoformat(),
                            "elapsed_seconds": round(elapsed),
                            "updated_at": datetime.now().isoformat(),
                        })
                else:
                    failed += 1
                    print(f"    [{datetime.now().strftime('%H:%M:%S')}] Batch {batch_id} FAILED at {result.get('stage', 'unknown')}")
            except Exception as e:
                failed += 1
                print(f"    [{datetime.now().strftime('%H:%M:%S')}] Batch {batch_id} ERROR: {e}")

    # Merge generated files to main generated directory
    print(f"  Merging results...")
    generated_dir = synthetic_dir / "generated" / collection_name
    parsed_dir = synthetic_dir / "parsed" / collection_name
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(parsed_dir, exist_ok=True)

    # Copy all files - they already have unique names from the original input
    total_copied = 0
    for batch_id, _ in batches:
        batch_gen_dir = batch_generated_base / f"batch_{batch_id:04d}"
        batch_parse_dir = batch_parsed_base / f"batch_{batch_id:04d}"

        # Copy generated JSON files (keep original names - already unique)
        if batch_gen_dir.exists():
            for f in sorted(batch_gen_dir.glob("*.json")):
                shutil.copy(f, generated_dir / f.name)
                total_copied += 1

        # Copy parsed TXT files (keep original names - already unique)
        if batch_parse_dir.exists():
            for f in sorted(batch_parse_dir.glob("*.txt")):
                shutil.copy(f, parsed_dir / f.name)

    print(f"  Copied {total_copied} generated files")

    # Cleanup temp batches
    shutil.rmtree(batch_temp_dir, ignore_errors=True)

    # Run curation if not skipped
    curate_time = 0
    if not skip_curate:
        print(f"  Running curation (parallel, {workers} workers)...")
        curate_start = time.time()
        curated_dir = synthetic_dir / "curated" / collection_name
        os.makedirs(curated_dir, exist_ok=True)
        ok = run_curate_parallel(generated_dir, curated_dir, config, api_key, threshold=7.0, workers=workers)
        curate_time = time.time() - curate_start
        if ok:
            curated_count = len(list(curated_dir.glob("*.json")))
            print(f"  Curation done: {curated_count} files in {curate_time:.1f}s")
        else:
            print(f"  Curation failed (continuing with uncurated output)")

    collection_time = time.time() - collection_start
    timings = {
        "collection": collection_name,
        "num_docs": num_docs,
        "batches": len(batches),
        "completed": completed,
        "failed": failed,
        "curate_time": curate_time,
        "total": collection_time,
    }

    print(f"\n  Collection {collection_name} complete!")
    print(f"    {completed}/{len(batches)} batches successful")
    print(f"    {num_docs} docs in {collection_time:.1f}s ({num_docs/collection_time:.2f} docs/s)")

    return failed == 0, timings


def run_collection_sequential(
    collection_name,
    synthetic_dir,
    config,
    api_key,
    num_qa=3,
    num_cot=2,
    skip_cot=False,
    skip_curate=False,
):
    """Run pipeline for a collection sequentially (for small collections)."""
    env = os.environ.copy()
    env["API_ENDPOINT_KEY"] = api_key
    env["PYTHONIOENCODING"] = "utf-8"

    input_dir = synthetic_dir / "input" / collection_name
    parsed_dir = synthetic_dir / "parsed" / collection_name
    generated_dir = synthetic_dir / "generated" / collection_name
    curated_dir = synthetic_dir / "curated" / collection_name

    num_docs = len(list(input_dir.glob("*.txt")))
    steps = 2 if skip_cot else 3
    if not skip_curate:
        steps += 1

    os.makedirs(parsed_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)

    cwd = str(synthetic_dir)
    t0 = time.time()

    # Ingest
    print(f"    [1/{steps}] Ingesting...")
    ok = run_cmd([
        "synthetic-data-kit", "-c", str(config),
        "ingest", str(input_dir),
        "--output-dir", str(parsed_dir),
    ], cwd=cwd, env=env, quiet=True)
    if not ok:
        return False, {"collection": collection_name, "num_docs": num_docs, "stage": "ingest"}

    # Generate QA
    print(f"    [2/{steps}] Generating QA...")
    ok = run_cmd([
        "synthetic-data-kit", "-c", str(config),
        "create", str(parsed_dir),
        "--type", "qa", "--num-pairs", str(num_qa),
        "--output-dir", str(generated_dir),
    ], cwd=cwd, env=env, quiet=True)
    if not ok:
        return False, {"collection": collection_name, "num_docs": num_docs, "stage": "qa"}

    # Generate CoT (optional)
    if not skip_cot:
        print(f"    [3/{steps}] Generating CoT...")
        ok = run_cmd([
            "synthetic-data-kit", "-c", str(config),
            "create", str(parsed_dir),
            "--type", "cot", "--num-pairs", str(num_cot),
            "--output-dir", str(generated_dir),
        ], cwd=cwd, env=env, quiet=True)
        if not ok:
            return False, {"collection": collection_name, "num_docs": num_docs, "stage": "cot"}

    # Curate (optional) - use parallel curation even for sequential generation
    curate_time = 0
    if not skip_curate:
        curate_step = steps
        print(f"    [{curate_step}/{steps}] Curating (parallel)...")
        curate_start = time.time()
        os.makedirs(curated_dir, exist_ok=True)
        ok = run_curate_parallel(generated_dir, curated_dir, config, api_key, threshold=7.0, workers=8)
        curate_time = time.time() - curate_start
        if ok:
            curated_count = len(list(curated_dir.glob("*.json")))
            print(f"    Curated: {curated_count} files in {curate_time:.1f}s")
        else:
            print(f"    Curation failed (continuing with uncurated output)")

    elapsed = time.time() - t0
    print(f"    Done: {num_docs} docs in {elapsed:.1f}s")

    return True, {"collection": collection_name, "num_docs": num_docs, "curate_time": curate_time, "total": elapsed}


def main():
    parser = argparse.ArgumentParser(
        description="Run the synthetic-data-kit pipeline for corpus Q&A generation"
    )
    parser.add_argument("--period", type=str, required=True,
                        choices=list(PERIODS.keys()),
                        help="Time period to process")
    parser.add_argument("--collection", type=str, default=None,
                        help="Process only a specific collection")
    parser.add_argument("--max-per-collection", type=int, default=10000,
                        help="Max docs per collection (default: 10000)")
    parser.add_argument("--quality-percentile", type=int, default=50,
                        help="Quality floor percentile (default: 50 = top 50%%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--num-qa", type=int, default=3,
                        help="QA pairs per document chunk (default: 3)")
    parser.add_argument("--num-cot", type=int, default=2,
                        help="CoT examples per document chunk (default: 2)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for batch processing (default: 1)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Documents per batch (default: {BATCH_SIZE})")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip document export (reuse existing)")
    parser.add_argument("--skip-cot", action="store_true",
                        help="Skip CoT generation (2x faster, QA only)")
    parser.add_argument("--skip-curate", action="store_true",
                        help="Skip curation step")
    args = parser.parse_args()

    paths = get_paths(args.period)
    synthetic_dir = paths["synthetic_dir"]
    config = CONFIG_PATH
    api_key = load_api_key()

    mode_parts = ["QA"]
    if not args.skip_cot:
        mode_parts.append("CoT")
    if not args.skip_curate:
        mode_parts.append("Curate")
    mode = " + ".join(mode_parts)

    print(f"Period: {args.period} ({paths['start_year']}-{paths['end_year']})")
    print(f"Synthetic dir: {synthetic_dir}")
    print(f"Workers: {args.workers}, Batch size: {args.batch_size}")
    print(f"Mode: {mode}")
    print(f"Max per collection: {args.max_per_collection}")

    # Progress file for monitoring
    progress_file = synthetic_dir / "progress.json"

    # Check metadata index exists
    if not paths["metadata_index"].exists():
        print(f"\nError: Metadata index not found: {paths['metadata_index']}")
        print(f"Run first: python -m src.post_training.corpus.build_index --period {args.period}")
        sys.exit(1)

    # Ensure synthetic directory exists
    os.makedirs(synthetic_dir, exist_ok=True)

    # Step 1: Export documents per collection
    if not args.skip_export:
        print("\n[Export] Exporting documents per collection...")
        export_documents_stratified(
            args.period,
            max_per_collection=args.max_per_collection,
            quality_percentile=args.quality_percentile,
            seed=args.seed,
        )
    else:
        print("\n[Export] Skipping (--skip-export)")

    # Get collection folders
    collections = get_collection_folders(args.period)
    if not collections:
        print("No collection folders found in synthetic/input/")
        sys.exit(1)

    # Filter to specific collection if requested
    if args.collection:
        target = sanitize_collection_name(args.collection)
        if target not in collections:
            print(f"Collection '{args.collection}' (folder: {target}) not found.")
            print(f"Available: {collections}")
            sys.exit(1)
        collections = [target]

    print(f"\nCollections to process: {len(collections)}")
    for c in collections:
        num_docs = len(list((synthetic_dir / "input" / c).glob("*.txt")))
        print(f"  - {c}: {num_docs} docs")

    # Process each collection
    results = {}
    all_timings = []
    pipeline_start = time.time()

    for i, collection in enumerate(collections, 1):
        print(f"\n[{i}/{len(collections)}] Starting {collection}...")

        if args.workers > 1:
            ok, timings = run_collection_parallel(
                collection, synthetic_dir, config, api_key,
                num_qa=args.num_qa, num_cot=args.num_cot,
                skip_cot=args.skip_cot, skip_curate=args.skip_curate,
                workers=args.workers, batch_size=args.batch_size,
                progress_file=progress_file,
            )
        else:
            ok, timings = run_collection_sequential(
                collection, synthetic_dir, config, api_key,
                num_qa=args.num_qa, num_cot=args.num_cot,
                skip_cot=args.skip_cot, skip_curate=args.skip_curate,
            )
        results[collection] = ok
        all_timings.append(timings)

    pipeline_total = time.time() - pipeline_start

    # Summary
    print(f"\n{'='*70}")
    print("Pipeline Summary:")
    print(f"{'='*70}")
    for coll, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {coll}: {status}")

    failed = [c for c, ok in results.items() if not ok]
    if failed:
        print(f"\n{len(failed)} collection(s) failed: {failed}")

    # Timing summary
    print(f"\n{'='*70}")
    print("Timing Summary:")
    print(f"{'='*70}")
    total_docs = sum(t.get("num_docs", 0) for t in all_timings)
    print(f"  Total documents: {total_docs:,}")
    print(f"  Pipeline total: {pipeline_total:.1f}s ({pipeline_total/60:.1f}m)")
    if total_docs > 0:
        print(f"  Throughput: {total_docs/pipeline_total:.2f} docs/s")

    # Convert all to nanochat format
    print(f"\n[Final] Converting all output to nanochat format...")
    from src.post_training.corpus.convert import convert_pipeline_output
    convert_pipeline_output(args.period)

    # Clean up spurious 'data' folder created by synthetic-data-kit
    spurious_data_dir = synthetic_dir / "data"
    if spurious_data_dir.exists():
        shutil.rmtree(spurious_data_dir, ignore_errors=True)

    print(f"\nPipeline complete! Output: {paths['hist_corpus_qa_output']}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
