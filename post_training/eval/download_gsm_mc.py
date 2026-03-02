"""
Download GSM-MC dataset from github.com/Geralt-Targaryen/MC-Evaluation.

Downloads data.tar.gz, extracts gsm8k-mc/test.jsonl, and saves it
as gsm_mc.jsonl in the specified output directory.

Usage:
    python -m src.post_training.eval.download_gsm_mc --output D:/hist_LLM/eval_data
"""

import argparse
import io
import json
import os
import tarfile
import urllib.request


ARCHIVE_URL = "https://github.com/Geralt-Targaryen/MC-Evaluation/raw/main/data.tar.gz"
MEMBER_PATH = "data/gsm8k-mc/test.jsonl"


def download_gsm_mc(output_dir):
    output_file = os.path.join(output_dir, "gsm_mc.jsonl")
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            n = sum(1 for line in f if line.strip())
        print(f"Already exists: {output_file} ({n} problems)")
        return output_file

    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading {ARCHIVE_URL} ...")
    response = urllib.request.urlopen(ARCHIVE_URL)
    archive_bytes = response.read()
    print(f"Downloaded {len(archive_bytes) / 1024 / 1024:.1f} MB")

    # Extract the test JSONL from the tar.gz archive
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode='r:gz') as tar:
        member = tar.getmember(MEMBER_PATH)
        f = tar.extractfile(member)
        raw_lines = f.read().decode('utf-8').strip().split('\n')

    # Validate and write
    records = []
    for line in raw_lines:
        row = json.loads(line)
        assert all(k in row for k in ("Question", "A", "B", "C", "D", "Answer")), \
            f"Unexpected GSM-MC format: {list(row.keys())}"
        records.append(row)

    with open(output_file, 'w', encoding='utf-8') as f:
        for row in records:
            f.write(json.dumps(row) + '\n')

    print(f"Wrote {len(records)} problems to {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GSM-MC dataset")
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory (e.g., D:/hist_LLM/eval_data)')
    args = parser.parse_args()
    download_gsm_mc(args.output)
