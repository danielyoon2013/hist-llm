"""
Shuffle answer positions in LAB evaluation question sets.

The original GPT-4.1 generation has a strong bias toward placing the correct
answer at position 0 (A) — often 65-70% of the time. This invalidates the
LAB evaluation because a model with a positional preference for "A" would
score ~65% without any actual look-ahead knowledge.

This script randomly shuffles the 4 choices for each question so the correct
answer is uniformly distributed across positions A/B/C/D.

Usage:
    # Shuffle all periods (overwrites lab_questions.jsonl in-place)
    python -m src.post_training.eval.shuffle_lab_answers

    # Shuffle one period
    python -m src.post_training.eval.shuffle_lab_answers --period 1950_1999

    # Just check the current distribution (no modification)
    python -m src.post_training.eval.shuffle_lab_answers --verify
    python -m src.post_training.eval.shuffle_lab_answers --verify --period 1950_1999
"""

import json
import argparse
import random
from collections import Counter

from src.post_training.config import PERIODS, get_paths


SEED = 42


def load_questions(path):
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def save_questions(questions, path):
    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


def get_answer_distribution(questions):
    """Return Counter of correct answer positions (0-3)."""
    return Counter(q["answer"] for q in questions)


def print_distribution(dist, total, label=""):
    if label:
        print(f"  {label}")
    for pos in range(4):
        count = dist.get(pos, 0)
        pct = count / total * 100 if total else 0
        bar = "#" * int(pct / 2)
        print(f"    Position {pos} ({chr(65+pos)}): {count:>5} ({pct:5.1f}%) {bar}")


def shuffle_questions(questions, seed=SEED):
    """Shuffle answer positions so correct answer is uniformly distributed."""
    rng = random.Random(seed)
    shuffled = []
    for q in questions:
        choices = q["choices"]
        answer = q["answer"]

        # Create a random permutation of [0,1,2,3]
        indices = list(range(4))
        rng.shuffle(indices)

        # Apply permutation to choices
        new_choices = [choices[i] for i in indices]
        # Find where the original correct answer index ended up
        new_answer = indices.index(answer)

        shuffled.append({
            **q,
            "choices": new_choices,
            "answer": new_answer,
        })
    return shuffled


def process_period(period, verify_only=False):
    paths = get_paths(period)
    lab_file = paths["lab_eval_dir"] / "lab_questions.jsonl"

    if not lab_file.exists():
        print(f"\n{period}: no lab_questions.jsonl found, skipping")
        return

    questions = load_questions(lab_file)
    total = len(questions)
    dist_before = get_answer_distribution(questions)

    print(f"\n{'='*60}")
    print(f"{period} ({total:,} questions)")
    print(f"{'='*60}")
    print_distribution(dist_before, total, label="Current distribution:")

    # Check if already uniform (within tolerance)
    expected = total / 4
    max_deviation = max(abs(dist_before.get(i, 0) - expected) for i in range(4))
    is_uniform = max_deviation < (total * 0.02)  # within 2%

    if is_uniform:
        print(f"  -> Already uniform (max deviation: {max_deviation:.0f} from expected {expected:.0f})")
        if verify_only:
            return
        print(f"  -> Shuffling anyway for consistency (seed={SEED})")

    if verify_only:
        return

    # Shuffle
    shuffled = shuffle_questions(questions, seed=SEED)

    # Verify correctness: the correct choice text should be the same
    for orig, shuf in zip(questions, shuffled):
        orig_correct = orig["choices"][orig["answer"]]
        shuf_correct = shuf["choices"][shuf["answer"]]
        assert orig_correct == shuf_correct, (
            f"Shuffle error: '{orig_correct}' != '{shuf_correct}' "
            f"for question: {orig['question'][:60]}..."
        )

    # Check new distribution
    dist_after = get_answer_distribution(shuffled)
    print_distribution(dist_after, total, label="After shuffle:")

    # Save
    save_questions(shuffled, lab_file)
    print(f"  -> Saved to {lab_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle LAB question answer positions for uniform distribution",
    )
    parser.add_argument("--period", type=str, default=None, choices=list(PERIODS.keys()),
                        help="Period to process (default: all)")
    parser.add_argument("--verify", action="store_true",
                        help="Only show current distribution, don't modify files")
    args = parser.parse_args()

    periods = [args.period] if args.period else list(PERIODS.keys())

    print(f"Mode: {'VERIFY ONLY' if args.verify else 'SHUFFLE'} (seed={SEED})")
    for period in periods:
        process_period(period, verify_only=args.verify)

    if not args.verify:
        print(f"\nDone. All files shuffled with seed={SEED}.")
        print("Re-run with --verify to confirm distributions.")


if __name__ == "__main__":
    main()
