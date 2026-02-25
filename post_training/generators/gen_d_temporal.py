"""Generator D: Temporal Reasoning — period-aware questions (metadata-based, no corpus)."""

import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.post_training.generators.base import BaseGenerator, call_api
from src.post_training.generators.prompts import TEMPORAL_PROMPT
from src.post_training.config import MODEL
from src.post_training.utils import validate_conversation, write_jsonl


MC_TEMPLATE = """{question}
- choice=A: {a}
- choice=B: {b}
- choice=C: {c}
- choice=D: {d}

Respond only with the letter of the correct answer."""


class GenDTemporal(BaseGenerator):

    name = "gen_d_temporal"
    items_per_chunk = 5
    needs_corpus = False

    def build_prompt(self, chunk, period, start_year, end_year):
        # chunk is actually batch_num for metadata-based generators
        return TEMPORAL_PROMPT.format(
            num_items=self.items_per_chunk,
            start_year=start_year,
            end_year=end_year,
            batch_num=chunk,
        )

    def parse_response(self, response):
        return response.get("questions", [])

    def format_conversation(self, item):
        choices = item.get("choices", {})
        user_msg = MC_TEMPLATE.format(
            question=item.get("question", ""),
            a=choices.get("A", ""),
            b=choices.get("B", ""),
            c=choices.get("C", ""),
            d=choices.get("D", ""),
        )
        correct = item.get("correct", "A")
        return [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": correct},
        ]

    def _run_metadata_based(self, client, period, start_year, end_year,
                            max_workers, output_path):
        """Generate temporal reasoning questions from period metadata."""
        num_batches = 10  # 10 batches x 5 items = 50 questions (small for testing)

        print(f"Generating {num_batches} batches x {self.items_per_chunk} items")
        print(f"Output: {output_path}")

        tasks = []
        for batch_num in range(1, num_batches + 1):
            tasks.append((client, batch_num, period, start_year, end_year))

        conversations = []
        completed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=min(max_workers, num_batches)) as executor:
            futures = {}
            for task in tasks:
                client, batch_num, period, sy, ey = task
                prompt = self.build_prompt(batch_num, period, sy, ey)
                futures[executor.submit(call_api, client, prompt)] = batch_num

            for future in as_completed(futures):
                try:
                    response = future.result()
                    if response:
                        items = self.parse_response(response)
                        for item in items:
                            conv = self.format_conversation(item)
                            valid, _ = validate_conversation(conv)
                            if valid:
                                conversations.append(conv)
                    completed += 1
                except Exception as e:
                    failed += 1
                    print(f"  [ERROR] {type(e).__name__}: {e}", flush=True)

        write_jsonl(conversations, str(output_path))
        print(f"Complete. {len(conversations):,} conversations. Failed: {failed}")
        return output_path
