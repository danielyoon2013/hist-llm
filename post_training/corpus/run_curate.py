"""
Wrapper to run synthetic-data-kit curate command with UTF-8 encoding.
This works around the Windows emoji encoding bug.

Usage:
    python run_curate.py -c config.yaml curate input_dir --threshold 7.0 --output output_dir
"""
import sys
import os

# Force UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Ensure API key is available - load from key.txt
from pathlib import Path

# Load API key from key.txt in project root
project_root = Path(__file__).parent.parent.parent.parent
key_file = project_root / 'key.txt'
if key_file.exists():
    with open(key_file) as f:
        api_key = f.read().strip()
    os.environ['API_ENDPOINT_KEY'] = api_key
    print(f"API_ENDPOINT_KEY loaded from key.txt")
else:
    print(f"Warning: key.txt not found at: {key_file}")

# Patch rich console to handle encoding errors
from rich.console import Console
original_print = Console.print

def safe_print(self, *args, **kwargs):
    try:
        return original_print(self, *args, **kwargs)
    except UnicodeEncodeError:
        # Replace emoji with text equivalent
        new_args = []
        for arg in args:
            if isinstance(arg, str):
                arg = arg.encode('ascii', 'replace').decode('ascii')
            new_args.append(arg)
        return original_print(self, *new_args, **kwargs)

Console.print = safe_print

# Now run the CLI
from synthetic_data_kit.cli import app

if __name__ == "__main__":
    app()
