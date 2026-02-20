"""
Identify and manage years that need reprocessing.

This script:
1. Identifies corrupted/missing embedding files
2. Checks which years are already queued on Lambda
3. Helps prioritize and batch reprocessing

Usage:
    python reprocess_years.py                  # Show status
    python reprocess_years.py --upload YEAR   # Upload specific year for reprocessing
    python reprocess_years.py --upload-batch N # Upload next N years that need processing
"""

import subprocess
import argparse
from pathlib import Path

# --- CONFIG ---
LOCAL_DATA_DIR = Path(r"D:\English")
LOCAL_RESULT_DIR = Path(r"D:\English_Results")

REMOTE_USER_IP = "ubuntu@129.146.2.145"
REMOTE_INBOX_PATH = "/home/ubuntu/hist-llm-data/input_data/"
REMOTE_OUTBOX_PATH = "/home/ubuntu/hist-llm-data/output_data/"
SSH_KEY = r"C:\Users\danielyoon\Documents\lambda_nanochat_openssh"

def verify_parquet_file(filepath: Path) -> bool:
    """Check if parquet file is complete (has magic bytes at start and end)."""
    if not filepath.exists():
        return False

    try:
        with open(filepath, 'rb') as f:
            start_magic = f.read(4)
            if start_magic != b'PAR1':
                return False
            f.seek(-4, 2)
            end_magic = f.read(4)
            if end_magic != b'PAR1':
                return False
        return True
    except:
        return False


def get_all_years() -> list:
    """Get all years from raw data directory."""
    return sorted([int(d.name) for d in LOCAL_DATA_DIR.iterdir()
                   if d.is_dir() and d.name.isdigit()])


def get_valid_embeddings() -> set:
    """Get years that have valid (non-corrupted) embedding files."""
    valid = set()
    for f in LOCAL_RESULT_DIR.glob("embeddings_*.parquet"):
        try:
            year = int(f.stem.split('_')[1])
            if verify_parquet_file(f):
                valid.add(year)
        except:
            pass
    return valid


def get_corrupted_years() -> list:
    """Get years with corrupted (truncated) embedding files."""
    corrupted = []
    for f in LOCAL_RESULT_DIR.glob("embeddings_*.parquet"):
        try:
            year = int(f.stem.split('_')[1])
            if not verify_parquet_file(f):
                size_mb = f.stat().st_size / 1024 / 1024
                corrupted.append((year, size_mb))
        except:
            pass
    return sorted(corrupted)


def check_remote_status() -> dict:
    """Check what's currently on Lambda (inbox and outbox)."""
    status = {'inbox': [], 'outbox': []}

    # Check inbox (queued for processing)
    try:
        result = subprocess.run(
            ["ssh", "-i", SSH_KEY, REMOTE_USER_IP, f"ls {REMOTE_INBOX_PATH}"],
            capture_output=True, text=True, timeout=30
        )
        for line in result.stdout.strip().split('\n'):
            if line.startswith('year_') and line.endswith('.tar'):
                try:
                    year = int(line.replace('year_', '').replace('.tar', ''))
                    status['inbox'].append(year)
                except:
                    pass
    except:
        pass

    # Check outbox (completed, ready for download)
    try:
        result = subprocess.run(
            ["ssh", "-i", SSH_KEY, REMOTE_USER_IP, f"ls {REMOTE_OUTBOX_PATH}"],
            capture_output=True, text=True, timeout=30
        )
        for line in result.stdout.strip().split('\n'):
            if line.startswith('embeddings_') and line.endswith('.parquet'):
                try:
                    year = int(line.replace('embeddings_', '').replace('.parquet', ''))
                    status['outbox'].append(year)
                except:
                    pass
    except:
        pass

    return status


def print_status():
    """Print comprehensive status of all years."""
    all_years = set(get_all_years())
    valid = get_valid_embeddings()
    corrupted = get_corrupted_years()
    corrupted_years = {y for y, _ in corrupted}
    missing = all_years - valid - corrupted_years

    print("=" * 70)
    print("REPROCESSING STATUS")
    print("=" * 70)

    print(f"\nTotal years in raw data: {len(all_years)}")
    print(f"  Valid embeddings:    {len(valid)} ({100*len(valid)/len(all_years):.1f}%)")
    print(f"  Corrupted files:     {len(corrupted)}")
    print(f"  Missing files:       {len(missing)}")

    # Check remote
    print("\nChecking Lambda server status...")
    remote = check_remote_status()

    if remote['inbox']:
        print(f"\nQueued on Lambda (inbox): {len(remote['inbox'])}")
        print(f"  Years: {sorted(remote['inbox'])[:10]}{'...' if len(remote['inbox']) > 10 else ''}")

    if remote['outbox']:
        print(f"\nReady for download (outbox): {len(remote['outbox'])}")
        print(f"  Years: {sorted(remote['outbox'])}")

    # List corrupted files
    if corrupted:
        print(f"\nCorrupted files (need re-download or reprocess):")
        for year, size in corrupted[:20]:
            in_outbox = " [READY ON LAMBDA]" if year in remote['outbox'] else ""
            print(f"  {year}: {size:.1f} MB{in_outbox}")
        if len(corrupted) > 20:
            print(f"  ... and {len(corrupted) - 20} more")

    # List missing (prioritize recent years)
    if missing:
        missing_list = sorted(missing)
        print(f"\nMissing embeddings (need full reprocess):")
        print(f"  First 15: {missing_list[:15]}")
        print(f"  Last 15:  {missing_list[-15:]}")

    # Years that need action
    needs_action = corrupted_years | missing
    already_handled = set(remote['inbox']) | set(remote['outbox'])
    needs_upload = needs_action - already_handled

    print(f"\n" + "=" * 70)
    print("ACTION REQUIRED")
    print("=" * 70)

    if remote['outbox']:
        print(f"\n1. Download ready files from Lambda ({len(remote['outbox'])} years):")
        print(f"   python download_embeddings_safe.py --year YEAR")
        print(f"   Or download all: process each year in {sorted(remote['outbox'])}")

    if needs_upload:
        print(f"\n2. Upload for reprocessing ({len(needs_upload)} years):")
        print(f"   python reprocess_years.py --upload YEAR")
        print(f"   Or batch: python reprocess_years.py --upload-batch 10")
        priority = sorted(needs_upload)
        print(f"   Next years to upload: {priority[:15]}")

    print()
    return {
        'valid': valid,
        'corrupted': corrupted_years,
        'missing': missing,
        'needs_upload': needs_upload,
        'remote': remote
    }


def upload_year(year: int) -> bool:
    """Upload a year's data to Lambda for reprocessing."""
    year_dir = LOCAL_DATA_DIR / str(year)
    if not year_dir.exists():
        print(f"[ERROR] Year directory not found: {year_dir}")
        return False

    archive_path = Path(r"D:\temp") / f"year_{year}.tar"
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    archive_name = f"year_{year}.tar"
    temp_remote_path = f"{REMOTE_INBOX_PATH}{archive_name}.tmp"
    final_remote_path = f"{REMOTE_INBOX_PATH}{archive_name}"

    print(f"Uploading year {year}...")

    # Create archive
    print(f"  Creating archive...")
    result = subprocess.run(
        ["tar", "-cf", str(archive_path), "-C", str(year_dir), "."],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [ERROR] Failed to create archive: {result.stderr}")
        return False

    # Upload to temp location
    print(f"  Uploading to Lambda...")
    result = subprocess.run(
        ["scp", "-i", SSH_KEY, str(archive_path), f"{REMOTE_USER_IP}:{temp_remote_path}"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [ERROR] SCP failed: {result.stderr}")
        return False

    # Atomic move to final location
    print(f"  Finalizing...")
    result = subprocess.run(
        ["ssh", "-i", SSH_KEY, REMOTE_USER_IP, f"mv {temp_remote_path} {final_remote_path}"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [ERROR] Move failed: {result.stderr}")
        return False

    # Clean up local archive
    archive_path.unlink()

    print(f"  [OK] Year {year} uploaded to Lambda inbox")
    return True


def main():
    parser = argparse.ArgumentParser(description="Manage reprocessing of embedding years")
    parser.add_argument('--upload', type=int, help='Upload specific year for reprocessing')
    parser.add_argument('--upload-batch', type=int, help='Upload next N years that need processing')
    parser.add_argument('--delete-corrupted', action='store_true', help='Delete corrupted local files')
    args = parser.parse_args()

    if args.delete_corrupted:
        corrupted = get_corrupted_years()
        print(f"Deleting {len(corrupted)} corrupted files...")
        for year, size in corrupted:
            f = LOCAL_RESULT_DIR / f"embeddings_{year}.parquet"
            if f.exists():
                f.unlink()
                print(f"  Deleted: {f.name} ({size:.1f} MB)")
        return

    if args.upload:
        upload_year(args.upload)
        return

    if args.upload_batch:
        status = print_status()
        needs_upload = sorted(status['needs_upload'])

        if not needs_upload:
            print("No years need uploading!")
            return

        batch = needs_upload[:args.upload_batch]
        print(f"\nUploading batch of {len(batch)} years: {batch}")

        for year in batch:
            upload_year(year)

        print(f"\nBatch upload complete!")
        return

    # Default: show status
    print_status()


if __name__ == "__main__":
    main()
