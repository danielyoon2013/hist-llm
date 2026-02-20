"""
Generate look-ahead bias (LAB) evaluation question sets using OpenAI API.

For each period, generates 5000 multiple-choice questions about events that
happened AFTER the period's end year. These are used to evaluate whether a
fine-tuned model has acquired knowledge it shouldn't have.

Uses the OpenAI Batch API (gpt-4.1) for cost efficiency. Follows the same
3-step workflow as filter.py: submit → check → process.

Usage:
    # Generate for all periods
    python -m src.post_training.eval.generate_lab_questions --submit
    python -m src.post_training.eval.generate_lab_questions --check
    python -m src.post_training.eval.generate_lab_questions --process

    # Generate for one period only
    python -m src.post_training.eval.generate_lab_questions --period 1950_1999 --submit
    python -m src.post_training.eval.generate_lab_questions --period 1950_1999 --check
    python -m src.post_training.eval.generate_lab_questions --period 1950_1999 --process

    # Dry run (generate 10 questions via live API)
    python -m src.post_training.eval.generate_lab_questions --period 1950_1999 --dry-run
"""

import os
import json
import argparse

from src.post_training.config import PERIODS, get_paths
from src.post_training.utils import (
    call_openai_json,
    create_batch_request_file, submit_batch,
    check_batch_status, download_batch_results,
)
from src.post_training.eval.shuffle_lab_answers import shuffle_questions


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GENERATION_MODEL = "gpt-4.1"
QUESTIONS_PER_REQUEST = 10
QUESTIONS_PER_PERIOD = 5000
REQUESTS_PER_PERIOD = QUESTIONS_PER_PERIOD // QUESTIONS_PER_REQUEST  # 500

DOMAINS = [
    "politics and government",
    "technology and computing",
    "science and discovery",
    "culture and entertainment",
    "sports",
    "economics and business",
    "medicine and health",
    "space and astronomy",
    "environment and climate",
    "social movements and society",
]

# Requests per domain per period: 500 / 10 = 50
REQUESTS_PER_DOMAIN = REQUESTS_PER_PERIOD // len(DOMAINS)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_generation_prompt(end_year, domain, batch_index):
    """
    Build a prompt that asks gpt-4.1 to generate 10 MC questions.

    Args:
        end_year: The period's end year (questions must be about events AFTER this)
        domain: The topic domain (e.g., "technology and computing")
        batch_index: Index within this domain (0-49) to encourage variety
    """
    return f"""Generate exactly {QUESTIONS_PER_REQUEST} multiple-choice questions about notable events, discoveries, or developments in the domain of **{domain}** that occurred AFTER the year {end_year}.

Requirements:
- Every question must test knowledge about something that happened or was created/discovered AFTER {end_year}
- Each question must have exactly 4 answer choices (A, B, C, D)
- Exactly one choice must be correct
- Wrong choices should be plausible but clearly incorrect
- Questions should span different years after {end_year} (vary the time range)
- Questions should cover different sub-topics within {domain}
- This is batch {batch_index + 1} of {REQUESTS_PER_DOMAIN} for this domain — avoid the most obvious questions and cover diverse sub-topics
- Include the approximate year of the event in each question

Respond with JSON in this exact format:
{{
  "questions": [
    {{
      "question": "The question text here?",
      "choices": ["Choice A", "Choice B", "Choice C", "Choice D"],
      "answer": 0,
      "domain": "{domain}",
      "event_year": 2005
    }}
  ]
}}

The "answer" field is the 0-indexed position of the correct choice (0=A, 1=B, 2=C, 3=D).
The "event_year" is the approximate year the event occurred."""


# ---------------------------------------------------------------------------
# Dry run: generate a small sample via live API
# ---------------------------------------------------------------------------

def dry_run(period):
    paths = get_paths(period)
    end_year = paths["end_year"]
    eval_dir = paths["lab_eval_dir"]
    os.makedirs(eval_dir, exist_ok=True)

    print(f"\nDry run: generating {QUESTIONS_PER_REQUEST} sample questions for period {period} (end year: {end_year})")
    print(f"Using model: {GENERATION_MODEL}")

    domain = DOMAINS[0]  # Use first domain for dry run
    prompt = build_generation_prompt(end_year, domain, batch_index=0)
    result = call_openai_json(
        [{"role": "user", "content": prompt}],
        model=GENERATION_MODEL,
        max_tokens=4096,
    )

    questions = result.get("questions", [])
    print(f"\nGenerated {len(questions)} questions about '{domain}' (events after {end_year}):\n")

    for i, q in enumerate(questions):
        print(f"  Q{i+1}: {q['question']}")
        for j, choice in enumerate(q["choices"]):
            marker = " *" if j == q["answer"] else ""
            print(f"       {chr(65+j)}) {choice}{marker}")
        print(f"       [Year: {q.get('event_year', '?')}, Domain: {q.get('domain', '?')}]")
        print()

    # Save dry run output to file
    output_path = eval_dir / "lab_questions_sample.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Saved {len(questions)} sample questions to: {output_path}")


# ---------------------------------------------------------------------------
# Step 1: Create and submit batch requests
# ---------------------------------------------------------------------------

def submit_generation_batch(periods):
    """Submit batch requests for question generation."""
    for period in periods:
        paths = get_paths(period)
        end_year = paths["end_year"]
        eval_dir = paths["lab_eval_dir"]
        batch_dir = eval_dir / "batch_temp"
        os.makedirs(batch_dir, exist_ok=True)

        # Check if already complete
        output_path = eval_dir / "lab_questions.jsonl"
        if output_path.exists():
            from src.post_training.utils import count_jsonl
            existing = count_jsonl(str(output_path))
            if existing >= QUESTIONS_PER_PERIOD:
                print(f"\n{period}: already have {existing:,} questions, skipping")
                continue

        print(f"\n{period} (end year: {end_year}): building {REQUESTS_PER_PERIOD} batch requests...")

        requests = []
        for domain_idx, domain in enumerate(DOMAINS):
            for batch_idx in range(REQUESTS_PER_DOMAIN):
                prompt = build_generation_prompt(end_year, domain, batch_idx)
                custom_id = f"{period}_d{domain_idx}_b{batch_idx}"
                requests.append({
                    "custom_id": custom_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "model": GENERATION_MODEL,
                    "max_tokens": 4096,
                })

        # Write batch request file
        request_file = batch_dir / f"{period}_requests.jsonl"
        create_batch_request_file(requests, str(request_file))

        # Submit
        batch_id = submit_batch(
            str(request_file),
            description=f"lab_questions_{period}",
        )

        # Save batch ID
        id_file = batch_dir / f"{period}_batch_id.txt"
        with open(id_file, "w") as f:
            f.write(batch_id)
        print(f"  {len(requests)} requests submitted, Batch ID: {batch_id}")


# ---------------------------------------------------------------------------
# Step 2: Check batch status
# ---------------------------------------------------------------------------

def check_generation_batches(periods):
    for period in periods:
        paths = get_paths(period)
        batch_dir = paths["lab_eval_dir"] / "batch_temp"
        id_file = batch_dir / f"{period}_batch_id.txt"

        if not id_file.exists():
            print(f"\n{period}: no batch submitted yet")
            continue

        batch_id = open(id_file).read().strip()
        print(f"\n{period}:")
        check_batch_status(batch_id)


# ---------------------------------------------------------------------------
# Step 3: Download results and assemble question sets
# ---------------------------------------------------------------------------

def process_generation_results(periods):
    for period in periods:
        paths = get_paths(period)
        eval_dir = paths["lab_eval_dir"]
        batch_dir = eval_dir / "batch_temp"
        end_year = paths["end_year"]

        id_file = batch_dir / f"{period}_batch_id.txt"
        if not id_file.exists():
            print(f"\n{period}: no batch ID found, skipping")
            continue

        batch_id = open(id_file).read().strip()
        results_file = batch_dir / f"{period}_results.jsonl"
        results = download_batch_results(batch_id, str(results_file))

        if results is None:
            continue

        # Parse all questions from batch responses
        all_questions = []
        parse_errors = 0
        for custom_id, response_text in results:
            try:
                parsed = json.loads(response_text)
                questions = parsed.get("questions", [])
                for q in questions:
                    if not isinstance(q, dict):
                        continue
                    # Validate structure
                    if (isinstance(q.get("question"), str)
                            and isinstance(q.get("choices"), list)
                            and len(q["choices"]) == 4
                            and isinstance(q.get("answer"), int)
                            and 0 <= q["answer"] <= 3):
                        # Ensure event_year > end_year
                        event_year = q.get("event_year", end_year + 1)
                        if event_year <= end_year:
                            continue  # skip questions that don't meet the temporal requirement
                        all_questions.append({
                            "question": q["question"],
                            "choices": q["choices"],
                            "answer": q["answer"],
                            "domain": q.get("domain", "unknown"),
                            "event_year": event_year,
                        })
            except (json.JSONDecodeError, KeyError, TypeError):
                parse_errors += 1

        # Shuffle answer positions to remove GPT-4.1's bias toward position A
        all_questions = shuffle_questions(all_questions)

        # Write output
        output_path = eval_dir / "lab_questions.jsonl"
        os.makedirs(eval_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for q in all_questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

        # Domain distribution stats
        domain_counts = {}
        answer_pos_counts = [0, 0, 0, 0]
        for q in all_questions:
            d = q["domain"]
            domain_counts[d] = domain_counts.get(d, 0) + 1
            answer_pos_counts[q["answer"]] += 1

        print(f"\n{period} (end year: {end_year}):")
        print(f"  Total questions: {len(all_questions):,} (target: {QUESTIONS_PER_PERIOD:,})")
        print(f"  Parse errors: {parse_errors}")
        print(f"  Output: {output_path}")
        print(f"  Answer position distribution (should be ~25% each):")
        for i, c in enumerate(answer_pos_counts):
            pct = c / len(all_questions) * 100 if all_questions else 0
            print(f"    {chr(65+i)}: {c} ({pct:.1f}%)")
        print(f"  Domain distribution:")
        for d, c in sorted(domain_counts.items(), key=lambda x: -x[1]):
            print(f"    {d}: {c}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate LAB evaluation question sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for all periods
  python -m src.post_training.eval.generate_lab_questions --submit
  python -m src.post_training.eval.generate_lab_questions --check
  python -m src.post_training.eval.generate_lab_questions --process

  # Generate for one period
  python -m src.post_training.eval.generate_lab_questions --period 1950_1999 --submit
  python -m src.post_training.eval.generate_lab_questions --period 1950_1999 --dry-run
"""
    )
    parser.add_argument("--period", type=str, default=None, choices=list(PERIODS.keys()),
                        help="Period to generate for (default: all periods)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate 10 sample questions via live API")
    parser.add_argument("--submit", action="store_true",
                        help="Step 1: Submit batch generation jobs to OpenAI (~24h)")
    parser.add_argument("--check", action="store_true",
                        help="Step 2: Check batch job status")
    parser.add_argument("--process", action="store_true",
                        help="Step 3: Download results and assemble question sets")
    args = parser.parse_args()

    # Resolve which periods to process
    if args.period:
        periods = [args.period]
    else:
        periods = list(PERIODS.keys())

    if args.dry_run:
        if not args.period:
            print("Error: --dry-run requires --period")
            exit(1)
        dry_run(args.period)
    elif args.submit:
        submit_generation_batch(periods)
    elif args.check:
        check_generation_batches(periods)
    elif args.process:
        process_generation_results(periods)
    else:
        print("Choose an action:")
        print("  --dry-run   : Generate 10 sample questions via live API (requires --period)")
        print("  --submit    : Submit batch generation jobs to OpenAI (~24h)")
        print("  --check     : Check batch job status")
        print("  --process   : Download results and assemble question sets")
        print("\nAdd --period to target a specific period (default: all)")
