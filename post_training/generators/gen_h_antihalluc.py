"""Generator H: Anti-Hallucination — refusal examples for post-period questions."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.post_training.generators.base import BaseGenerator, call_api
from src.post_training.generators.prompts import ANTIHALLUC_PROMPT
from src.post_training.config import MODEL
from src.post_training.utils import validate_conversation, write_jsonl


class GenHAntiHalluc(BaseGenerator):

    name = "gen_h_antihalluc"
    items_per_chunk = 5
    needs_corpus = False

    def build_prompt(self, chunk, period, start_year, end_year):
        # chunk is actually batch_num for metadata-based generators
        return ANTIHALLUC_PROMPT.format(
            num_items=self.items_per_chunk,
            end_year=end_year,
            batch_num=chunk,
        )

    def parse_response(self, response):
        return response.get("examples", [])

    def format_conversation(self, item):
        return [
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["response"]},
        ]

    def _run_metadata_based(self, client, period, start_year, end_year,
                            max_workers, output_path):
        """Generate anti-hallucination refusal examples."""
        num_batches = 10  # 10 batches x 5 items = 50 questions (small for testing)

        print(f"Generating {num_batches} batches x {self.items_per_chunk} items")
        print(f"Output: {output_path}")

        conversations = []
        completed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=min(max_workers, num_batches)) as executor:
            futures = {}
            for batch_num in range(1, num_batches + 1):
                prompt = self.build_prompt(batch_num, period, start_year, end_year)
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
