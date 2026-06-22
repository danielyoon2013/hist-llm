"""
Create cumulative token count graphs for analysis periods.

Loads classified data from the English corpus to determine quality cutoff
scores for each analysis period.

Note: Classification uses 14 models (one per 25-year period).
      This script aggregates classified data into larger analysis periods.

For each analysis period:
1. Load all classified files for years in that period
2. Sort by predicted_quality (high to low)
3. Calculate cumulative token count
4. Plot with 20B threshold line

Analysis periods (6 non-overlapping, historically meaningful eras):
    1678-1849: Early Modern + Industrial Revolution
    1850-1899: Victorian Era
    1900-1949: World Wars Era
    1950-1999: Cold War + Digital Age
    2000-2009: Early Internet
    2010-2023: Social Media Era

To change analysis periods, modify PERIOD_RANGES below.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
CLASSIFIED_DIR = Path(r"D:\hist_LLM\corpus\classified")
ALL_CLASSIFIED_DIR = Path(r"D:\hist_LLM\corpus\classified_all")  # all-docs audit output
OUTPUT_DIR = Path(r"D:\hist_LLM\processing\quality_graphs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 25-year periods (the granularity of the Ridge quality models)
PERIODS_25YR = [
    (1678, 1700, "1678_1700"), (1701, 1725, "1701_1725"), (1726, 1750, "1726_1750"),
    (1751, 1775, "1751_1775"), (1776, 1800, "1776_1800"), (1801, 1825, "1801_1825"),
    (1826, 1850, "1826_1850"), (1851, 1875, "1851_1875"), (1876, 1900, "1876_1900"),
    (1901, 1925, "1901_1925"), (1926, 1950, "1926_1950"), (1951, 1975, "1951_1975"),
    (1976, 2000, "1976_2000"), (2001, 2023, "2001_2023"),
]

THRESHOLD = 20_000_000_000  # 20 billion tokens

# Analysis periods (can be changed without retraining models)
# Option A: 6 non-overlapping periods (historically meaningful eras)
PERIOD_RANGES = [
    (1678, 1849, "1678_1849"),  # Early Modern + Industrial Revolution
    (1850, 1899, "1850_1899"),  # Victorian Era
    (1900, 1949, "1900_1949"),  # World Wars Era
    (1950, 1999, "1950_1999"),  # Cold War + Digital Age
    (2000, 2009, "2000_2009"),  # Early Internet
    (2010, 2023, "2010_2023"),  # Social Media Era
]


def load_period_data(start_year: int, end_year: int) -> pd.DataFrame:
    """Load and combine all classified files for a period."""
    dfs = []

    for year in range(start_year, end_year + 1):
        path = CLASSIFIED_DIR / f"classified_{year}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df['year'] = year
            dfs.append(df)

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


def create_cumulative_plot(period_name: str, df: pd.DataFrame):
    """Create and save cumulative token plot for a period."""

    # Sort by predicted_quality descending (highest quality first)
    df_sorted = df.sort_values('predicted_quality', ascending=False).reset_index(drop=True)

    # Calculate cumulative token count
    df_sorted['cumulative_tokens'] = df_sorted['token_count'].cumsum()

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot cumulative tokens vs predicted quality
    ax.plot(df_sorted['predicted_quality'], df_sorted['cumulative_tokens'] / 1e9,
            linewidth=0.5, alpha=0.8, color='blue')

    # Add 20B threshold line
    ax.axhline(y=THRESHOLD / 1e9, color='red', linestyle='--', linewidth=2, label=f'20B threshold')

    # Find cutoff point
    cutoff_idx = (df_sorted['cumulative_tokens'] >= THRESHOLD).idxmax() if (df_sorted['cumulative_tokens'] >= THRESHOLD).any() else None

    if cutoff_idx is not None:
        cutoff_score = df_sorted.loc[cutoff_idx, 'predicted_quality']
        cutoff_tokens = df_sorted.loc[cutoff_idx, 'cumulative_tokens']
        ax.axvline(x=cutoff_score, color='green', linestyle=':', linewidth=2,
                   label=f'Cutoff: score={cutoff_score:.2f}')
        ax.scatter([cutoff_score], [cutoff_tokens / 1e9], color='green', s=100, zorder=5)

    # Labels and formatting
    ax.set_xlabel('Predicted Quality Score (High → Low)', fontsize=12)
    ax.set_ylabel('Cumulative Tokens (Billions)', fontsize=12)
    ax.set_title(f'Cumulative Token Count vs Quality Score\nPeriod: {period_name}', fontsize=14)
    ax.invert_xaxis()  # High quality on left, low on right
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add stats text
    total_tokens = df['token_count'].sum()
    total_docs = len(df)
    stats_text = f'Total: {total_docs:,} docs, {total_tokens/1e9:.2f}B tokens'
    if cutoff_idx is not None:
        docs_above = cutoff_idx + 1
        stats_text += f'\nAbove 20B: {docs_above:,} docs ({100*docs_above/total_docs:.1f}%)'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'cumulative_tokens_{period_name}.png', dpi=150)
    plt.close()

    return {
        'period': period_name,
        'total_docs': total_docs,
        'total_tokens': total_tokens,
        'cutoff_score': cutoff_score if cutoff_idx is not None else None,
        'docs_above_threshold': cutoff_idx + 1 if cutoff_idx is not None else total_docs
    }


# ---------------------------------------------------------------------------
# Expected-quality report over ALL docs (audit of the hardcoded heuristic filter)
# Reads the all-docs scores (classified_all/) and summarizes expected quality per
# period, split into kept vs dropped. Writes a NEW csv; never touches the cutoff
# logic or period_summary.csv above.
# ---------------------------------------------------------------------------

def _load_period_all(input_dir: Path, start_year: int, end_year: int) -> pd.DataFrame:
    """Load classified_{year}.parquet for a period from an arbitrary input dir."""
    dfs = []
    for year in range(start_year, end_year + 1):
        path = input_dir / f"classified_{year}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    # classified_all has is_clean (True/False); be robust if a file lacks it
    if 'is_clean' not in df.columns:
        df['is_clean'] = True
    df['token_count'] = df['token_count'].fillna(0)
    return df


def _tok_weighted(d: pd.DataFrame) -> float:
    t = d['token_count'].sum()
    return float((d['predicted_quality'] * d['token_count']).sum() / t) if t > 0 else float('nan')


def expected_quality_report(input_dir: Path, period_ranges: list, output_csv: Path):
    """Expected quality over ALL docs per period, split kept vs dropped."""
    rows = []
    for start, end, name in tqdm(period_ranges, desc=f"Expected-quality ({output_csv.name})"):
        df = _load_period_all(input_dir, start, end)
        if df is None or len(df) == 0:
            print(f"  [SKIP] {name}: no data in {input_dir}")
            continue
        kept = df[df['is_clean']]
        drop = df[~df['is_clean']]
        kmed = kept['predicted_quality'].median() if len(kept) else float('nan')
        pct_above = (100 * (drop['predicted_quality'] > kmed).mean()
                     if len(drop) and len(kept) else float('nan'))
        rows.append({
            'period': name,
            'total_docs': len(df),
            'total_tokens': int(df['token_count'].sum()),
            'kept_frac_docs_pct': round(100 * len(kept) / len(df), 2),
            'mean_quality_all': round(df['predicted_quality'].mean(), 4),
            'tokwt_quality_all': round(_tok_weighted(df), 4),
            'kept_docs': len(kept),
            'kept_mean_quality': round(kept['predicted_quality'].mean(), 4) if len(kept) else None,
            'kept_tokwt_quality': round(_tok_weighted(kept), 4) if len(kept) else None,
            'dropped_docs': len(drop),
            'dropped_tokens': int(drop['token_count'].sum()),
            'dropped_mean_quality': round(drop['predicted_quality'].mean(), 4) if len(drop) else None,
            'dropped_tokwt_quality': round(_tok_weighted(drop), 4) if len(drop) else None,
            'pct_dropped_above_kept_median': round(pct_above, 2) if pct_above == pct_above else None,
        })
        print(f"  {name}: {len(df):,} docs | all mean q={df['predicted_quality'].mean():.2f} "
              f"(kept {kept['predicted_quality'].mean():.2f} vs dropped {drop['predicted_quality'].mean():.2f}) "
              f"| {pct_above:.0f}% of dropped >= kept-median")
    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)
    print(f"\nExpected-quality summary saved to: {output_csv}")
    print("NOTE: Ridge was trained on CLEAN docs only; scores on the dropped/garbage "
          "tail are out-of-distribution (indicative, not calibrated).")
    return out


def run_expected_quality(input_dir: Path):
    """Emit expected-quality summaries at both 6-era and 25-yr granularity."""
    expected_quality_report(input_dir, PERIOD_RANGES, OUTPUT_DIR / 'expected_quality_all.csv')
    expected_quality_report(input_dir, PERIODS_25YR, OUTPUT_DIR / 'expected_quality_all_25yr.csv')


def main():
    print("Creating cumulative token graphs for each period...")

    results = []

    for start, end, period_name in tqdm(PERIOD_RANGES, desc="Processing periods"):
        df = load_period_data(start, end)

        if df is None or len(df) == 0:
            print(f"  [SKIP] {period_name}: No data")
            continue

        # Check for missing token counts
        missing_tokens = df['token_count'].isna().sum()
        if missing_tokens > 0:
            print(f"  [WARN] {period_name}: {missing_tokens} rows missing token_count")
            df = df.dropna(subset=['token_count'])

        result = create_cumulative_plot(period_name, df)
        results.append(result)

        print(f"  {period_name}: {result['total_docs']:,} docs, {result['total_tokens']/1e9:.2f}B tokens, cutoff={result['cutoff_score']}")

    # Save summary
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(OUTPUT_DIR / 'period_summary.csv', index=False)

    print(f"\nGraphs saved to: {OUTPUT_DIR}")
    print(f"Summary saved to: {OUTPUT_DIR / 'period_summary.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quality-cutoff plots, or expected-quality audit report")
    parser.add_argument('--expected-quality', action='store_true',
                        help='Report expected quality over ALL docs (kept vs dropped) from classified_all/; '
                             'writes expected_quality_all*.csv. Does not touch the cutoff/plot path.')
    parser.add_argument('--input-dir', type=str, default=str(ALL_CLASSIFIED_DIR),
                        help='Input dir of classified_{year}.parquet for --expected-quality')
    args = parser.parse_args()

    if args.expected_quality:
        run_expected_quality(Path(args.input_dir))
    else:
        main()
