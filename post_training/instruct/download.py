"""
Download instruct and reasoning datasets from HuggingFace and save as JSONL.

These are saved in data/instruct_data/ as browseable files and serve as
input for the temporal filtering step.

Usage:
    python -m src.post_training.instruct.download
    python -m src.post_training.instruct.download --dataset smoltalk
    python -m src.post_training.instruct.download --dataset math
"""

import os
import re
import json
import argparse
from pathlib import Path
from datasets import load_dataset

from src.post_training.config import PROJECT_ROOT


OUTPUT_DIR = PROJECT_ROOT / "data" / "instruct_data"


# ---------------------------------------------------------------------------
# Rendering helpers (match nanochat's format exactly)
# ---------------------------------------------------------------------------

def render_mc(question, letters, choices):
    """Render multiple choice question in nanochat's format."""
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query


def flatten_gsm8k_answer(answer_text):
    """
    Flatten GSM8K's step-by-step answer with <<expr=result>> tool calls
    into a plain text string suitable for CustomJSON format.
    """
    # GSM8K answers use <<expression=result>> for calculator calls.
    # We keep these as-is since they're readable text.
    # Just clean up any extra whitespace.
    return answer_text.strip()


# ---------------------------------------------------------------------------
# Dataset downloaders
# ---------------------------------------------------------------------------

def download_smoltalk():
    """Download SmolTalk train split and save as JSONL."""
    print("Downloading SmolTalk (train)...")
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    print(f"  Loaded {len(ds)} rows")

    output_path = OUTPUT_DIR / "smoltalk.jsonl"
    count = 0
    skipped = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            messages = row["messages"]
            # SmolTalk may have a system message — strip it for CustomJSON compatibility
            if messages and messages[0]["role"] == "system":
                messages = messages[1:]
            # Validate: need at least 2 messages, alternating user/assistant
            if len(messages) < 2:
                skipped += 1
                continue
            valid = True
            for i, msg in enumerate(messages):
                expected = "user" if i % 2 == 0 else "assistant"
                if msg["role"] != expected or not isinstance(msg.get("content", ""), str):
                    valid = False
                    break
            if not valid:
                skipped += 1
                continue
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Saved {count} conversations to {output_path} (skipped {skipped})")


def download_mmlu():
    """Download MMLU auxiliary_train split and save as JSONL."""
    print("Downloading MMLU (auxiliary_train)...")
    ds = load_dataset("cais/mmlu", "auxiliary_train", split="train")
    # auxiliary_train has a 'train' wrapper
    ds = ds.map(lambda row: row['train'], remove_columns=['train'])
    print(f"  Loaded {len(ds)} rows")

    letters = ('A', 'B', 'C', 'D')
    output_path = OUTPUT_DIR / "mmlu.jsonl"
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            question = row["question"]
            choices = row["choices"]
            answer_idx = row["answer"]
            user_msg = render_mc(question, letters, choices)
            assistant_msg = letters[answer_idx]
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Saved {count} conversations to {output_path}")


def download_arc(subset):
    """Download ARC (Easy or Challenge) train split and save as JSONL."""
    name = "arc_easy" if subset == "ARC-Easy" else "arc_challenge"
    print(f"Downloading ARC ({subset}, train)...")
    ds = load_dataset("allenai/ai2_arc", subset, split="train")
    print(f"  Loaded {len(ds)} rows")

    output_path = OUTPUT_DIR / f"{name}.jsonl"
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            question = row["question"]
            choices = row["choices"]["text"]
            letters = row["choices"]["label"]
            answer_key = row["answerKey"]
            user_msg = render_mc(question, letters, choices)
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": answer_key}
            ]
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Saved {count} conversations to {output_path}")


def download_gsm8k():
    """Download GSM8K main train split and save as JSONL."""
    print("Downloading GSM8K (main, train)...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    print(f"  Loaded {len(ds)} rows")

    output_path = OUTPUT_DIR / "gsm8k.jsonl"
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            question = row["question"]
            answer = flatten_gsm8k_answer(row["answer"])
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Saved {count} conversations to {output_path}")


# ---------------------------------------------------------------------------
# Reasoning dataset downloaders
# ---------------------------------------------------------------------------

def download_math():
    """Download MATH competition problems with step-by-step solutions."""
    print("Downloading MATH (competition_math, train)...")
    ds = load_dataset("EleutherAI/hendrycks_math", revision="refs/convert/parquet", split="train")
    print(f"  Loaded {len(ds)} rows")

    output_path = OUTPUT_DIR / "math.jsonl"
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            messages = [
                {"role": "user", "content": row["problem"]},
                {"role": "assistant", "content": row["solution"]},
            ]
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Saved {count} conversations to {output_path}")


def download_aime_amc():
    """Download AIME/AMC math competition problems."""
    print("Downloading AIME/AMC (camel-ai distilled)...")
    ds = load_dataset("camel-ai/amc_aime_distilled", split="train")
    print(f"  Loaded {len(ds)} rows")

    output_path = OUTPUT_DIR / "aime_amc.jsonl"
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            problem = row.get("problem", "")
            solution = row.get("reasoning_solution", row.get("groud_truth_solution", ""))
            if not problem or not solution:
                continue
            messages = [
                {"role": "user", "content": str(problem)},
                {"role": "assistant", "content": str(solution)},
            ]
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Saved {count} conversations to {output_path}")


def download_commonsenseqa():
    """Download CommonsenseQA multiple-choice questions."""
    print("Downloading CommonsenseQA (train)...")
    ds = load_dataset("tau/commonsense_qa", split="train")
    print(f"  Loaded {len(ds)} rows")

    output_path = OUTPUT_DIR / "commonsenseqa.jsonl"
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            question = row["question"]
            labels = row["choices"]["label"]
            texts = row["choices"]["text"]
            answer_key = row["answerKey"]
            user_msg = render_mc(question, labels, texts)
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": answer_key},
            ]
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Saved {count} conversations to {output_path}")


def download_hellaswag():
    """Download HellaSwag sentence completion dataset."""
    print("Downloading HellaSwag (train)...")
    ds = load_dataset("Rowan/hellaswag", split="train")
    print(f"  Loaded {len(ds)} rows")

    output_path = OUTPUT_DIR / "hellaswag.jsonl"
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            ctx = row["ctx"]
            endings = row["endings"]
            label = int(row["label"])
            user_msg = f"Complete the following:\n\n{ctx}"
            assistant_msg = endings[label]
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Saved {count} conversations to {output_path}")


def download_piqa():
    """Download PIQA physical intuition questions."""
    print("Downloading PIQA (train)...")
    ds = load_dataset("ybisk/piqa", revision="refs/convert/parquet", split="train")
    print(f"  Loaded {len(ds)} rows")

    output_path = OUTPUT_DIR / "piqa.jsonl"
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            goal = row["goal"]
            answer = row["sol1"] if row["label"] == 0 else row["sol2"]
            messages = [
                {"role": "user", "content": goal},
                {"role": "assistant", "content": answer},
            ]
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Saved {count} conversations to {output_path}")


def download_winogrande():
    """Download WinoGrande pronoun resolution dataset."""
    print("Downloading WinoGrande (winogrande_debiased, train)...")
    ds = load_dataset("allenai/winogrande", "winogrande_debiased", split="train")
    print(f"  Loaded {len(ds)} rows")

    output_path = OUTPUT_DIR / "winogrande.jsonl"
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            sentence = row["sentence"]
            option1 = row["option1"]
            option2 = row["option2"]
            answer = row["answer"]  # "1" or "2"
            correct = option1 if answer == "1" else option2
            resolved = sentence.replace("_", correct)
            user_msg = f"Fill in the blank:\n\n{sentence}\n\nOption 1: {option1}\nOption 2: {option2}"
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": resolved},
            ]
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Saved {count} conversations to {output_path}")


def download_logiqa():
    """Download LogiQA logical reasoning questions."""
    print("Downloading LogiQA (train)...")
    ds = load_dataset("lucasmccabe/logiqa", revision="refs/convert/parquet", split="train")
    print(f"  Loaded {len(ds)} rows")

    letters = ('A', 'B', 'C', 'D')
    output_path = OUTPUT_DIR / "logiqa.jsonl"
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            context = row["context"]
            query = row["query"]
            options = row["options"]
            answer_idx = row["correct_option"]
            question_text = f"{context}\n\nQuestion: {query}"
            user_msg = render_mc(question_text, letters[:len(options)], options)
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": letters[answer_idx]},
            ]
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Saved {count} conversations to {output_path}")


def download_folio():
    """Download FOLIO first-order logic reasoning dataset."""
    print("Downloading FOLIO (train)...")
    ds = load_dataset("tasksource/folio", split="train")
    print(f"  Loaded {len(ds)} rows")

    output_path = OUTPUT_DIR / "folio.jsonl"
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            premises = row["premises"]
            conclusion = row["conclusion"]
            label = row["label"]
            user_msg = f"Given the following premises:\n{premises}\n\nIs the following conclusion True, False, or Uncertain?\n{conclusion}"
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": label},
            ]
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Saved {count} conversations to {output_path}")


def download_hotpotqa():
    """Download HotpotQA multi-hop questions."""
    print("Downloading HotpotQA (distractor, train)...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    print(f"  Loaded {len(ds)} rows")

    output_path = OUTPUT_DIR / "hotpotqa.jsonl"
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            question = row["question"]
            answer = row["answer"]
            if not answer or not question:
                continue
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Saved {count} conversations to {output_path}")


def download_musique():
    """Download MuSiQue multi-hop questions with decomposition."""
    print("Downloading MuSiQue (train)...")
    ds = load_dataset("dgslibisey/MuSiQue", split="train")
    print(f"  Loaded {len(ds)} rows")

    output_path = OUTPUT_DIR / "musique.jsonl"
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            question = row["question"]
            answer = row["answer"]
            if not answer or not question:
                continue
            # Include decomposition as reasoning in the assistant response
            decomp = row.get("question_decomposition", [])
            if decomp:
                steps = []
                for step in decomp:
                    steps.append(f"Step: {step['question']} → {step['answer']}")
                reasoning = "\n".join(steps) + f"\n\nFinal answer: {answer}"
            else:
                reasoning = answer
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": reasoning},
            ]
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Saved {count} conversations to {output_path}")


def download_strategyqa():
    """Download StrategyQA implicit reasoning questions."""
    print("Downloading StrategyQA...")
    ds = load_dataset("wics/strategy-qa", revision="refs/convert/parquet", split="test")  # only test split available
    print(f"  Loaded {len(ds)} rows")

    output_path = OUTPUT_DIR / "strategyqa.jsonl"
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            question = row["question"]
            answer = "Yes" if row["answer"] else "No"
            decomp = row.get("decomposition", [])
            facts = row.get("facts", [])
            # Build reasoning from decomposition and facts
            parts = []
            if decomp:
                for i, step in enumerate(decomp):
                    parts.append(f"Step {i+1}: {step}")
            if facts:
                parts.append("Supporting facts: " + "; ".join(facts))
            parts.append(f"Answer: {answer}")
            reasoning = "\n".join(parts)
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": reasoning},
            ]
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Saved {count} conversations to {output_path}")


def download_scienceqa():
    """Download ScienceQA text-only subset with step-by-step solutions."""
    print("Downloading ScienceQA (train, text-only)...")
    ds = load_dataset("derek-thomas/ScienceQA", split="train")
    print(f"  Loaded {len(ds)} rows total")

    output_path = OUTPUT_DIR / "scienceqa.jsonl"
    count = 0
    skipped_image = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            # Skip rows that require an image
            if row.get("image") is not None:
                skipped_image += 1
                continue
            question = row["question"]
            choices = row["choices"]
            solution = row.get("solution", "")
            answer_idx = row["answer"]
            if not solution:
                continue
            # Format as MC question + solution
            letters = [chr(65 + i) for i in range(len(choices))]
            user_msg = render_mc(question, letters, choices)
            assistant_msg = f"{solution}\n\nThe answer is {letters[answer_idx]}."
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Saved {count} conversations to {output_path} (skipped {skipped_image} image-based)")


def download_humaneval():
    """Download HumanEval code generation benchmark."""
    print("Downloading HumanEval (test)...")
    ds = load_dataset("openai/openai_humaneval", split="test")
    print(f"  Loaded {len(ds)} rows")

    output_path = OUTPUT_DIR / "humaneval.jsonl"
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            prompt = row["prompt"]
            solution = row["canonical_solution"]
            messages = [
                {"role": "user", "content": f"Write the following function:\n\n{prompt}"},
                {"role": "assistant", "content": solution},
            ]
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Saved {count} conversations to {output_path}")


def download_mbpp():
    """Download MBPP basic Python problems."""
    print("Downloading MBPP (full, train)...")
    ds = load_dataset("google-research-datasets/mbpp", "full", split="train")
    print(f"  Loaded {len(ds)} rows")

    output_path = OUTPUT_DIR / "mbpp.jsonl"
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = row["text"]
            code = row["code"]
            messages = [
                {"role": "user", "content": text},
                {"role": "assistant", "content": code},
            ]
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Saved {count} conversations to {output_path}")


def download_codecontests():
    """Download CodeContests competitive programming problems."""
    print("Downloading CodeContests (train)...")
    ds = load_dataset("deepmind/code_contests", split="train")
    print(f"  Loaded {len(ds)} rows")

    output_path = OUTPUT_DIR / "codecontests.jsonl"
    count = 0
    skipped = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            description = row["description"]
            solutions = row.get("solutions", {})
            solution_list = solutions.get("solution", [])
            if not solution_list or not description:
                skipped += 1
                continue
            # Use the first solution
            messages = [
                {"role": "user", "content": description},
                {"role": "assistant", "content": solution_list[0]},
            ]
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Saved {count} conversations to {output_path} (skipped {skipped} without solutions)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASETS = {
    # Original instruct datasets
    "smoltalk": download_smoltalk,
    "mmlu": download_mmlu,
    "arc_easy": lambda: download_arc("ARC-Easy"),
    "arc_challenge": lambda: download_arc("ARC-Challenge"),
    "gsm8k": download_gsm8k,
    # Reasoning datasets
    "math": download_math,
    "aime_amc": download_aime_amc,
    "commonsenseqa": download_commonsenseqa,
    "hellaswag": download_hellaswag,
    "piqa": download_piqa,
    "winogrande": download_winogrande,
    "logiqa": download_logiqa,
    "folio": download_folio,
    "hotpotqa": download_hotpotqa,
    "musique": download_musique,
    "strategyqa": download_strategyqa,
    "scienceqa": download_scienceqa,
    "humaneval": download_humaneval,
    "mbpp": download_mbpp,
    "codecontests": download_codecontests,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download instruct datasets from HuggingFace")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=list(DATASETS.keys()),
                        help="Download a specific dataset (default: all)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    if args.dataset:
        DATASETS[args.dataset]()
    else:
        for name, fn in DATASETS.items():
            fn()
            print()

    print("\nDone! You can browse the files in:")
    print(f"  {OUTPUT_DIR}")
