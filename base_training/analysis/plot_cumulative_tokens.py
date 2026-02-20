"""
Create cumulative token count graphs for analysis periods.

Loads classified data from both the English corpus and additional news
collections (NYT, Economist, FT, Newswire) to determine combined quality
cutoff scores for each analysis period.

Note: Classification uses 14 models (one per 25-year period).
      This script aggregates classified data into larger analysis periods.

For each analysis period:
1. Load all classified files for years in that period (English + additional)
2. Sort by predicted_quality (high to low)
3. Calculate cumulative token count
4. Plot with 20B threshold line

Analysis periods (Option A - 6 non-overlapping, historically meaningful eras):
    1678-1849: Early Modern + Industrial Revolution
    1850-1899: Victorian Era
    1900-1949: World Wars Era
    1950-1999: Cold War + Digital Age
    2000-2009: Early Internet
    2010-2023: Social Media Era

To change analysis periods, modify PERIOD_RANGES below.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
CLASSIFIED_DIR = Path(r"D:\hist_LLM\corpus\classified")
ADDITIONAL_CLASSIFIED_DIR = Path(r"D:\hist_LLM\additional_data\classified")
ADDITIONAL_COLLECTIONS = ["nyt", "economist", "ft", "newswire"]

OUTPUT_DIR = Path(r"D:\hist_LLM\processing\quality_graphs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    """Load and combine all classified files for a period (English + additional)."""
    dfs = []

    # English corpus
    for year in range(start_year, end_year + 1):
        path = CLASSIFIED_DIR / f"classified_{year}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df['year'] = year
            df['source'] = 'english'
            dfs.append(df)

    # Additional collections
    for collection in ADDITIONAL_COLLECTIONS:
        coll_dir = ADDITIONAL_CLASSIFIED_DIR / collection
        if not coll_dir.exists():
            continue
        for year in range(start_year, end_year + 1):
            path = coll_dir / f"classified_{year}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                df['year'] = year
                df['source'] = collection
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
    english_docs = (df['source'] == 'english').sum() if 'source' in df.columns else total_docs
    additional_docs = total_docs - english_docs
    stats_text = f'Total: {total_docs:,} docs, {total_tokens/1e9:.2f}B tokens'
    stats_text += f'\nEnglish: {english_docs:,} | Additional: {additional_docs:,}'
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
        'english_docs': english_docs,
        'additional_docs': additional_docs,
        'total_tokens': total_tokens,
        'cutoff_score': cutoff_score if cutoff_idx is not None else None,
        'docs_above_threshold': cutoff_idx + 1 if cutoff_idx is not None else total_docs
    }


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
    main()
