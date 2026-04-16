"""
Generate 7 per-benchmark Jupyter notebooks for error analysis.

Each notebook follows the same structure, parameterized by benchmark name.
Run once:
    python -m src.post_training.analysis.build_notebooks
"""

import json
from pathlib import Path


NOTEBOOKS_DIR = Path(__file__).resolve().parents[2] / "notebooks"

BENCHMARKS = {
    "ARC-Challenge":   {"random": 0.25, "desc": "Grade-school science; multiple-choice 1-of-4", "targets": "Gen A (Factual QA), targeted science content"},
    "HellaSwag":       {"random": 0.25, "desc": "Commonsense completion of everyday scenes; 1-of-4", "targets": "Gen E (Historical Completion — period-appropriate physical scenes)"},
    "RACE-Middle":     {"random": 0.25, "desc": "Reading comprehension of English passages for Chinese middle-school learners; 1-of-4", "targets": "Gen C (Reading Comprehension)"},
    "RACE-High":       {"random": 0.25, "desc": "Reading comprehension, high-school difficulty; 1-of-4", "targets": "Gen C (Reading Comprehension)"},
    "Winogrande":      {"random": 0.50, "desc": "Pronoun resolution using commonsense; 1-of-2", "targets": "Gen F (Winogrande-style pronoun resolution, GPT-4o)"},
    "PIQA":            {"random": 0.50, "desc": "Physical commonsense; pick the sensible method; 1-of-2", "targets": "Gen B (Physical Commonsense)"},
    "GSM-MC":          {"random": 0.25, "desc": "Math word problems in MC form; 1-of-4", "targets": "Gen D (Quantitative)"},
}


def cell(src, cell_type="code"):
    lines = src.split("\n")
    source = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    if cell_type == "code":
        return {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": source,
        }
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build_notebook(benchmark: str, meta: dict) -> dict:
    random = meta["random"]
    desc = meta["desc"]
    targets = meta["targets"]

    cells = []

    # Title
    cells.append(cell(f"""# Error Analysis — {benchmark}

**Benchmark:** {benchmark}
**Description:** {desc}
**Random baseline:** {random*100:.0f}%
**Our generator(s) targeting this:** {targets}

This notebook drills into why the model isn't getting a higher score. It:
1. Splits every question into `answerable` / `unanswerable` / `ambiguous` using a hybrid keyword + LLM-judge classifier (cached).
2. Reports accuracy stratified by answerability — so we see real model capability, not just raw score.
3. Examines confidence on wrong answers (high-confidence wrong = learned wrong pattern).
4. Surfaces the highest-confidence wrong answers on the *answerable* subset — these are the diagnostic gold.
5. Compares Mid vs SFT phases to see where the gap closed or opened.
6. Delivers a one-line verdict: content gap, skill gap, or both.""", "markdown"))

    # Setup
    cells.append(cell("""import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "src" / "post_training").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.post_training.analysis import (
    load_details, load_classifications, classify_batch,
    accuracy_triple, confidence_breakdown, high_confidence_wrong,
    plot_confidence_histogram, extract_stem_and_choices,
)
from src.post_training.analysis.answerability import _hash_question

plt.rcParams['figure.dpi'] = 110
plt.rcParams['figure.figsize'] = (12, 5)

BENCHMARK = \"""" + benchmark + """\"
RANDOM_BASELINE = """ + str(random) + """

BASE = Path(r'D:/hist_LLM/periods/1900_1949/error_analysis_new')
CACHE = BASE / '.answerability_cache' / f'{BENCHMARK}.jsonl'

STAGES = ['mid_final', 'sft_final', 'sft_final_debiased']
dfs = {}
for stage in STAGES:
    p = BASE / stage / f'{BENCHMARK}_details.jsonl'
    dfs[stage] = load_details(p)
    print(f'{stage}: {len(dfs[stage])} rows')"""))

    # Classifications
    cells.append(cell("""## 1. Answerability Classifications

Load the cached classifications. If the cache is missing, build it now.
(Classifier is hybrid: keyword filter first, then LLM judge for the ambiguous middle.)"""
                      , "markdown"))

    cells.append(cell("""classes = load_classifications(CACHE)
print(f'Cached classifications: {len(classes)}')

if len(classes) < len(dfs['mid_final']):
    print('Cache incomplete — building now...')
    questions = []
    for i, row in dfs['mid_final'].iterrows():
        questions.append({'index': int(row['index']), 'question': row['question']})
    classify_batch(questions, CACHE, start_year=1900, end_year=1949)
    classes = load_classifications(CACHE)
    print(f'Now cached: {len(classes)}')

from collections import Counter
label_counts = Counter(r['label'] for r in classes.values())
print(f'\\nLabel distribution: {dict(label_counts)}')"""))

    # Accuracy stratification
    cells.append(cell("""## 2. Accuracy Stratified by Answerability

The headline question: **on the subset the model COULD answer with period knowledge, how well does it do?**""", "markdown"))

    cells.append(cell(f"""rows = []
for stage in STAGES:
    if dfs[stage].empty: continue
    trip = accuracy_triple(dfs[stage], classes)
    for bucket, stats in trip.items():
        if stats['acc'] is None: continue
        rows.append({{
            'stage': stage, 'bucket': bucket,
            'n': stats['n'], 'correct': stats['correct'],
            'acc_%': round(100 * stats['acc'], 1),
            'vs_random_%': round(100 * (stats['acc'] - RANDOM_BASELINE), 1),
        }})
summary = pd.DataFrame(rows)
print(f'Random baseline: {{100*RANDOM_BASELINE:.0f}}%')
summary"""))

    cells.append(cell("""## 3. Answerability Sanity Check

A sample of how the classifier labeled individual questions.""", "markdown"))

    cells.append(cell("""# Show 4 examples from each label
df_with_labels = dfs['mid_final'].copy()
df_with_labels['qhash'] = df_with_labels['question'].map(_hash_question)
df_with_labels['label'] = df_with_labels['qhash'].map(lambda h: classes.get(h,{}).get('label','ambiguous'))
df_with_labels['reason'] = df_with_labels['qhash'].map(lambda h: classes.get(h,{}).get('reason',''))
for label in ['answerable','unanswerable','ambiguous']:
    subset = df_with_labels[df_with_labels['label'] == label]
    if len(subset) == 0: continue
    print(f'=== {label.upper()} ({len(subset)} total) — sampling 3 ===')
    for _, row in subset.sample(min(3, len(subset)), random_state=42).iterrows():
        parsed = extract_stem_and_choices(row['question'])
        print(f'  Q: {parsed[\"stem\"][:200]}')
        if row['reason']: print(f'     reason: {row[\"reason\"][:200]}')
    print()"""))

    # Confidence histogram
    cells.append(cell("""## 4. Confidence Distribution

Right vs wrong, split by answerability. High-confidence wrong on the *answerable* subset = model learned a wrong pattern.""", "markdown"))

    cells.append(cell("""fig, axes = plt.subplots(len([s for s in STAGES if not dfs[s].empty]), 1, figsize=(12, 4.5*len([s for s in STAGES if not dfs[s].empty])))
axes = [axes] if not hasattr(axes, '__iter__') else axes
ax_iter = iter(axes)
for stage in STAGES:
    if dfs[stage].empty: continue
    ax = next(ax_iter)
    plot_confidence_histogram(dfs[stage], classes, title=f'{BENCHMARK} — {stage}', ax=ax)
plt.tight_layout()
plt.show()"""))

    # High-confidence wrong
    cells.append(cell("""## 5. High-Confidence Wrong Answers (answerable subset, Mid phase)

The diagnostic gold. These are questions the model should be able to answer using 1900-1949 knowledge, but got confidently wrong. Review these to identify patterns.""", "markdown"))

    cells.append(cell("""wrong = high_confidence_wrong(dfs['mid_final'], classes, label_filter='answerable', n=20, min_conf=0.5)
print(f'{len(wrong)} high-confidence wrong answers on the answerable subset\\n')

for i, (_, row) in enumerate(wrong.iterrows(), 1):
    parsed = extract_stem_and_choices(row['question'])
    print(f'[{i:02d}] conf={row[\"confidence\"]:.0%} pred={row[\"predicted\"]} expected={row[\"expected\"]}')
    print(f'     Q: {parsed[\"stem\"][:280]}')
    for L in sorted(parsed['choices'].keys()):
        mark = ' ← PREDICTED' if L == row['predicted'] else (' ← CORRECT' if L == row['expected'] else '')
        print(f'       {L}: {parsed[\"choices\"][L][:160]}{mark}')
    print()"""))

    # Mid vs SFT on answerable
    cells.append(cell("""## 6. Mid vs SFT on Answerable Subset

Did SFT help on the answerable slice, or did it just overfit to internal distribution?""", "markdown"))

    cells.append(cell("""if not dfs['sft_final'].empty:
    mid_trip = accuracy_triple(dfs['mid_final'], classes)
    sft_trip = accuracy_triple(dfs['sft_final'], classes)
    deb_trip = accuracy_triple(dfs['sft_final_debiased'], classes) if not dfs['sft_final_debiased'].empty else None
    rows = []
    for bucket in ['overall','answerable','unanswerable']:
        m = mid_trip.get(bucket,{}).get('acc')
        s = sft_trip.get(bucket,{}).get('acc')
        d = (deb_trip.get(bucket,{}).get('acc') if deb_trip else None)
        if m is None or s is None: continue
        rows.append({
            'bucket': bucket, 'n': mid_trip[bucket]['n'],
            'Mid_%': round(100*m,1), 'SFT_%': round(100*s,1),
            'Debiased_%': round(100*d,1) if d is not None else None,
            'SFT-Mid_delta': round(100*(s-m),1),
        })
    pd.DataFrame(rows)"""))

    cells.append(cell("""## 7. Error-Type Clustering

What kinds of questions (within the answerable subset) does the model miss most?""", "markdown"))

    # Per-benchmark error clustering
    cluster_code = _build_cluster_cell(benchmark)
    cells.append(cell(cluster_code))

    # Verdict
    cells.append(cell("""## 8. Verdict

Interpretation based on the answerable-subset accuracy:

- **answerable_acc > random + 15pp**: Real skill transfer. Model has the capability.
- **answerable_acc within ±5pp of random**: Skill gap. The model can't reason about even period-compatible content.
- **answerable_acc >> overall_acc**: Content gap dominates. Targeted synthetic data on modern vocabulary could help.
- **high-conf wrong clusters around one error type**: Targeted fix possible — rewrite the corresponding generator.""", "markdown"))

    cells.append(cell("""# Verdict computation
trip = accuracy_triple(dfs['mid_final'], classes)
overall = trip.get('overall', {}).get('acc', 0) or 0
answerable = trip.get('answerable', {}).get('acc', 0) or 0
unanswerable = trip.get('unanswerable', {}).get('acc', 0) or 0

verdict = []
if answerable > RANDOM_BASELINE + 0.15:
    verdict.append(f'✅ Real skill transfer: answerable={100*answerable:.1f}% vs random={100*RANDOM_BASELINE:.0f}%')
elif abs(answerable - RANDOM_BASELINE) < 0.05:
    verdict.append(f'⚠️ Skill gap: answerable subset at {100*answerable:.1f}% (near random).')
else:
    verdict.append(f'➖ Modest transfer: answerable={100*answerable:.1f}%, +{100*(answerable-RANDOM_BASELINE):.1f}pp vs random.')

if answerable - overall > 0.05:
    verdict.append(f'📈 Content gap contributes: answerable ({100*answerable:.1f}%) beats overall ({100*overall:.1f}%) by {100*(answerable-overall):.1f}pp.')

if unanswerable > RANDOM_BASELINE + 0.05:
    verdict.append(f'⚠️ Unanswerable subset at {100*unanswerable:.1f}% — model may be leaking modern knowledge.')

print('\\n'.join(verdict))"""))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _build_cluster_cell(benchmark: str) -> str:
    """Return benchmark-specific error-type clustering code."""
    if benchmark == "ARC-Challenge":
        return """# ARC-Challenge: cluster by science topic keyword
import re
SCIENCE_TOPICS = {
    'physics/energy': r'\\b(energy|force|motion|gravity|electric|current|voltage|circuit|magnet|heat|temperature|light|sound|wave|mass|velocity|friction|pressure|power|watt|volt)\\b',
    'biology/life': r'\\b(cell|plant|animal|organism|ecosystem|food chain|photosynthesis|respiration|reproduction|dna|gene|species|bacteria|evolution|skeleton|muscle|organ|nerve|blood)\\b',
    'chemistry/matter': r'\\b(atom|molecule|element|compound|acid|base|solution|mixture|chemical|reaction|metal|liquid|solid|gas|crystal|salt|oxygen|hydrogen|carbon)\\b',
    'earth/weather': r'\\b(weather|climate|storm|rain|snow|wind|cloud|earth|rock|soil|mountain|ocean|river|volcano|earthquake|planet|moon|sun|star|season)\\b',
}

wrong_df = dfs['mid_final'][dfs['mid_final']['correct'] == False].copy()
wrong_df['qhash'] = wrong_df['question'].map(_hash_question)
wrong_df['label'] = wrong_df['qhash'].map(lambda h: classes.get(h,{}).get('label','ambiguous'))
ans_wrong = wrong_df[wrong_df['label']=='answerable']

for topic, pat in SCIENCE_TOPICS.items():
    hits = ans_wrong[ans_wrong['question'].str.contains(pat, case=False, regex=True, na=False)]
    print(f'{topic}: {len(hits)} answerable wrong (of {len(ans_wrong)})')
    if len(hits):
        sample = hits.iloc[0]
        parsed = extract_stem_and_choices(sample['question'])
        print(f'  example: {parsed[\"stem\"][:180]}')
    print()"""

    if benchmark == "HellaSwag":
        return """# HellaSwag: cluster by scene-type keyword
import re
SCENE_TOPICS = {
    'sports/fitness': r'\\b(gym|workout|treadmill|weight|ball|court|field|track|run|swim|bike|athlete|soccer|basketball|football|tennis|yoga)\\b',
    'food/cooking': r'\\b(cook|recipe|ingredient|kitchen|oven|stove|pan|pot|bowl|plate|dough|bake|fry|boil|steam|mix|chop|season)\\b',
    'grooming/beauty': r'\\b(hair|shampoo|makeup|nail|polish|razor|shave|skin|lotion|face|lipstick|eye|manicure|brush|comb)\\b',
    'crafts/DIY': r'\\b(paint|tool|drill|hammer|nail|glue|tape|cut|measure|sew|knit|fabric|thread|needle|yarn)\\b',
    'home/chores': r'\\b(clean|vacuum|wash|laundry|dishwasher|mop|sweep|dust|garden|lawn|mow)\\b',
}

wrong_df = dfs['mid_final'][dfs['mid_final']['correct'] == False].copy()
wrong_df['qhash'] = wrong_df['question'].map(_hash_question)
wrong_df['label'] = wrong_df['qhash'].map(lambda h: classes.get(h,{}).get('label','ambiguous'))
ans_wrong = wrong_df[wrong_df['label']=='answerable']

for topic, pat in SCENE_TOPICS.items():
    hits = ans_wrong[ans_wrong['question'].str.contains(pat, case=False, regex=True, na=False)]
    print(f'{topic}: {len(hits)} answerable wrong (of {len(ans_wrong)})')
    if len(hits):
        sample = hits.iloc[0]
        parsed = extract_stem_and_choices(sample['question'])
        print(f'  example: {parsed[\"stem\"][:180]}')
    print()"""

    if benchmark in ("RACE-Middle", "RACE-High"):
        return """# RACE: cluster by question type
import re
Q_TYPES = {
    'main_idea_or_title': r'\\b(main idea|best title|mainly about|passage mainly|theme of|best summary)\\b',
    'detail_recall': r'\\b(who|when|where|how many|how much|which of the following|according to)\\b',
    'inference': r'\\b(infer|imply|suggest|probably|most likely|conclude|author believes|author.*would agree)\\b',
    'character_motivation': r'\\b(why did|why does|how does .* feel|what kind of person|attitude|motivation)\\b',
    'vocabulary': r'\\b(underlined|mean in the passage|most nearly means|probably means|refers to)\\b',
}

wrong_df = dfs['mid_final'][dfs['mid_final']['correct'] == False].copy()
wrong_df['qhash'] = wrong_df['question'].map(_hash_question)
wrong_df['label'] = wrong_df['qhash'].map(lambda h: classes.get(h,{}).get('label','ambiguous'))
ans_wrong = wrong_df[wrong_df['label']=='answerable']

for qtype, pat in Q_TYPES.items():
    hits = ans_wrong[ans_wrong['question'].str.contains(pat, case=False, regex=True, na=False)]
    print(f'{qtype}: {len(hits)} answerable wrong (of {len(ans_wrong)})')
    if len(hits):
        sample = hits.iloc[0]
        parsed = extract_stem_and_choices(sample['question'])
        print(f'  example Q: {parsed[\"stem\"][:180]}')
    print()"""

    if benchmark == "Winogrande":
        return """# Winogrande: cluster by commonsense anchor (heuristic)
import re
ANCHORS = {
    'physical_property_attribute': r'\\b(too (big|small|heavy|light|hot|cold|hard|soft|fast|slow|long|short|tall|wide|narrow|thick|thin))\\b',
    'role_profession': r'\\b(doctor|teacher|student|lawyer|judge|farmer|miller|chef|cook|mechanic|nurse|builder|carpenter|engineer|manager|officer|clerk)\\b',
    'motivation_feeling': r'\\b(wanted|needed|hoped|feared|loved|hated|thought|believed|worried|because _ was feeling)\\b',
    'possession_ownership': r'\\b(_ had|_ owned|_ brought|_ kept|_.s)\\b',
}

wrong_df = dfs['mid_final'][dfs['mid_final']['correct'] == False].copy()
wrong_df['qhash'] = wrong_df['question'].map(_hash_question)
wrong_df['label'] = wrong_df['qhash'].map(lambda h: classes.get(h,{}).get('label','ambiguous'))
ans_wrong = wrong_df[wrong_df['label']=='answerable']

for anchor, pat in ANCHORS.items():
    hits = ans_wrong[ans_wrong['question'].str.contains(pat, case=False, regex=True, na=False)]
    print(f'{anchor}: {len(hits)} answerable wrong (of {len(ans_wrong)})')
    if len(hits):
        sample = hits.iloc[0]
        parsed = extract_stem_and_choices(sample['question'])
        print(f'  example: {parsed[\"stem\"][:180]}')
    print()"""

    if benchmark == "PIQA":
        return """# PIQA: cluster by physical-task domain
import re
DOMAINS = {
    'food/cooking/baking': r'\\b(cook|bake|fry|boil|steam|grill|mix|chop|food|meat|vegetable|fruit|dough|recipe)\\b',
    'cleaning/household': r'\\b(clean|wash|dust|vacuum|stain|dirt|laundry|soap|bleach|detergent|sponge|mop)\\b',
    'tools/DIY/repair': r'\\b(tool|nail|screw|hammer|saw|drill|fix|repair|glue|tape|paint|sand|cut)\\b',
    'crafts/sewing': r'\\b(sew|knit|fabric|thread|needle|yarn|button|crochet|stitch|quilt)\\b',
    'body/grooming/medical': r'\\b(hair|skin|nail|razor|bandage|wound|injur|medicin|first aid|body|face)\\b',
}

wrong_df = dfs['mid_final'][dfs['mid_final']['correct'] == False].copy()
wrong_df['qhash'] = wrong_df['question'].map(_hash_question)
wrong_df['label'] = wrong_df['qhash'].map(lambda h: classes.get(h,{}).get('label','ambiguous'))
ans_wrong = wrong_df[wrong_df['label']=='answerable']

for dom, pat in DOMAINS.items():
    hits = ans_wrong[ans_wrong['question'].str.contains(pat, case=False, regex=True, na=False)]
    print(f'{dom}: {len(hits)} answerable wrong (of {len(ans_wrong)})')
    if len(hits):
        sample = hits.iloc[0]
        parsed = extract_stem_and_choices(sample['question'])
        print(f'  example: {parsed[\"stem\"][:180]}')
    print()"""

    if benchmark == "GSM-MC":
        return """# GSM-MC: cluster by operation count (proxy for step count)
import re
wrong_df = dfs['mid_final'][dfs['mid_final']['correct'] == False].copy()
wrong_df['qhash'] = wrong_df['question'].map(_hash_question)
wrong_df['label'] = wrong_df['qhash'].map(lambda h: classes.get(h,{}).get('label','ambiguous'))
ans_wrong = wrong_df[wrong_df['label']=='answerable']

# Count numbers in the question as a step-count proxy
def num_count(q):
    return len(re.findall(r'\\b\\d+(?:\\.\\d+)?\\b', q))

ans_wrong['num_count'] = ans_wrong['question'].apply(num_count)
print('Step-count distribution on answerable wrong answers (number of numeric values in question):')
print(ans_wrong['num_count'].value_counts().sort_index())
print()

# Operation detection
OPS = {
    'addition': r'\\b(total|sum|altogether|combined|plus|add)\\b',
    'subtraction': r'\\b(remaining|left|difference|fewer|less|minus|subtract)\\b',
    'multiplication': r'\\b(each|per|every|twice|double|triple|times)\\b',
    'division': r'\\b(divide|share|split|each|per|average|half|quarter|third)\\b',
    'percent/fraction': r'\\b(percent|%|half|quarter|third|fraction|tenth)\\b',
}

for op, pat in OPS.items():
    hits = ans_wrong[ans_wrong['question'].str.contains(pat, case=False, regex=True, na=False)]
    print(f'{op}: {len(hits)} answerable wrong (of {len(ans_wrong)})')"""

    return "# (no benchmark-specific clustering defined)"


def main():
    NOTEBOOKS_DIR.mkdir(exist_ok=True)
    for benchmark, meta in BENCHMARKS.items():
        nb = build_notebook(benchmark, meta)
        safe_name = benchmark.lower().replace('-', '_')
        out = NOTEBOOKS_DIR / f"error_analysis_{safe_name}.ipynb"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
