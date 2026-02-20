"""
Generate neutral identity conversations using GPT-4o-mini.

Replaces Karpathy's identity_conversations.jsonl with conversations that:
- Respond to greetings naturally
- Identify the model as trained on historical documents from the period

NOTE: This intentionally does NOT include refusal training (e.g., "I can't answer
questions about events after {end_year}"). For research on look-ahead bias, the
model's behavior on post-period questions should be a genuine experimental result,
not a coached response.

Usage:
    python -m src.post_training.generate_identity --period 1950_1999
    python -m src.post_training.generate_identity --period 1950_1999 --dry-run
    python -m src.post_training.generate_identity --period 1950_1999 --num-conversations 500
"""

import os
import json
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.post_training.config import PERIODS, get_paths
from src.post_training.utils import call_openai_json, write_jsonl


# ---------------------------------------------------------------------------
# Conversation starter templates
# ---------------------------------------------------------------------------

GREETING_STARTERS = [
    "hi", "hello", "hey", "hi there", "hello there", "hey there",
    "howdy", "greetings", "good morning", "good afternoon", "good evening",
    "hi!", "hello!", "hey!", "yo", "sup", "what's up",
    "hola", "bonjour", "konnichiwa", "guten tag", "ciao",
    "ola", "namaste", "salaam", "ni hao",
]

IDENTITY_STARTERS = [
    "Who are you?",
    "What are you?",
    "Tell me about yourself.",
    "What can you do?",
    "What do you know about?",
    "What kind of model are you?",
    "How were you trained?",
    "What is your purpose?",
    "What are your capabilities?",
    "What data were you trained on?",
    "What time period do you cover?",
    "Are you like ChatGPT?",
    "Can you help me with research?",
]



def build_system_prompt(start_year, end_year):
    return f"""You are generating training data for a language model.
This model was trained on historical documents from {start_year} to {end_year}.

Generate a realistic multi-turn conversation (2-6 messages, alternating user/assistant)
based on the user's first message. The assistant should:

1. For greetings: Respond warmly and naturally
2. For identity questions: Explain factually that it is a language model trained on
   historical documents (patents, legal cases, government proceedings, etc.) from {start_year}-{end_year}

Keep responses natural and concise. Do not mention any specific AI companies or modern AI systems.

Output a JSON object with a "messages" key containing a list of message objects.
Each message has "role" (user or assistant) and "content" (string).
Messages must alternate user/assistant starting with user. Minimum 2 messages."""


def generate_one(starter, start_year, end_year):
    """Generate one identity conversation from a starter message."""
    system_prompt = build_system_prompt(start_year, end_year)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'The user\'s first message is: "{starter}"\n\nGenerate the conversation as JSON.'},
    ]
    result = call_openai_json(messages)
    return result.get("messages", [])


def build_starter_list(start_year, end_year, num_conversations):
    """Build a list of conversation starters: greetings + identity questions."""
    starters = []

    # Greetings (60%)
    n_greetings = int(num_conversations * 0.60)
    for i in range(n_greetings):
        starters.append(GREETING_STARTERS[i % len(GREETING_STARTERS)])

    # Identity questions (40%)
    n_identity = num_conversations - n_greetings
    for i in range(n_identity):
        starters.append(IDENTITY_STARTERS[i % len(IDENTITY_STARTERS)])

    return starters[:num_conversations]


def main():
    parser = argparse.ArgumentParser(description="Generate identity conversations")
    parser.add_argument("--period", type=str, required=True, choices=list(PERIODS.keys()))
    parser.add_argument("--num-conversations", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Generate 5 examples and print them")
    args = parser.parse_args()

    paths = get_paths(args.period)
    start_year, end_year = paths["start_year"], paths["end_year"]
    num = 5 if args.dry_run else args.num_conversations

    print(f"Period: {args.period} ({start_year}-{end_year})")
    print(f"Generating {num} identity conversations...")

    starters = build_starter_list(start_year, end_year, num)
    conversations = []
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(generate_one, starter, start_year, end_year): starter
            for starter in starters
        }
        for i, future in enumerate(as_completed(futures)):
            try:
                msgs = future.result()
                if msgs and len(msgs) >= 2:
                    conversations.append(msgs)
            except Exception as e:
                errors += 1
                print(f"  Error: {e}")
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{num} ({len(conversations)} valid, {errors} errors)")

    print(f"\nGenerated {len(conversations)} valid conversations ({errors} errors)")

    if args.dry_run:
        for i, conv in enumerate(conversations[:5]):
            print(f"\n--- Conversation {i+1} ---")
            for msg in conv:
                print(f"  [{msg['role']}]: {msg['content'][:200]}")
    else:
        # Write to posttraining_data/identity.jsonl
        output_path = str(paths["identity_output"])
        write_jsonl(conversations, output_path)

        # Copy to identity_conversations.jsonl (where nanochat expects it)
        nanochat_path = str(paths["identity_nanochat"])
        shutil.copy2(output_path, nanochat_path)
        print(f"Copied to {nanochat_path}")


if __name__ == "__main__":
    main()
