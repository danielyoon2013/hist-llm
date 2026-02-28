"""Generator H: Historical Facts & Dates — multi-format (MC-4, Open-ended). Metadata-based, no corpus.

Generates per-year (multiple API calls per year if needed) to eliminate duplicates.
Train-only: factual recall is evaluated via external benchmarks, not held-out test set.

Default allocation: 2.5% of target (25,000 at 1M target).
Items-per-year computed dynamically from target_examples so output is
consistent across all periods regardless of year span.
"""

import math
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.post_training.generators.base import (
    BaseGenerator, FORMAT_MC4, FORMAT_OPEN, render_mc, make_mc_choices,
    call_api,
)
from src.post_training.generators.prompts import HIST_FACTS_PROMPT
from src.post_training.utils import validate_conversation, write_jsonl


class GenHHistFacts(BaseGenerator):

    name = "gen_h_histfacts"
    items_per_chunk = 10        # items per API call (reasonable for quality)
    needs_corpus = False
    train_only = True  # factual recall — no test split
    SUPPORTED_FORMATS = (FORMAT_MC4, FORMAT_OPEN)

    def build_prompt(self, year, period, start_year, end_year, num_items=None):
        return HIST_FACTS_PROMPT.format(
            num_items=num_items or self.items_per_chunk,
            year=year,
            start_year=start_year,
            end_year=end_year,
        )

    def parse_response(self, response):
        return response.get("facts", [])

    def format_conversation(self, item, fmt, source_chunk=None):
        question = item.get("question", "")
        answer = item.get("answer", "")

        if fmt == FORMAT_OPEN:
            return [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]

        if fmt == FORMAT_MC4:
            distractors = item.get("distractors", [])
            if len(distractors) < 3:
                return None
            letters, choices, correct = make_mc_choices(
                answer, distractors, num_choices=4,
                position_idx=next(self._mc_counters[fmt]),
            )
            user_msg = render_mc(question, letters, choices)
            return [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": correct},
            ]

        return None

    def _run_metadata_based(self, client, period, start_year, end_year,
                            max_workers, output_paths, target_examples=None):
        """Override: multiple API calls per year, scaled to hit target_examples.

        For a 50-year period targeting 25,000 examples (2 formats):
          raw_needed = 25,000 / 2 = 12,500 items
          items_per_year = ceil(12,500 / 50) = 250
          calls_per_year = ceil(250 / 10) = 25
          total_calls = 50 * 25 = 1,250

        For a 172-year period targeting 25,000 examples:
          items_per_year = ceil(12,500 / 172) = 73
          calls_per_year = ceil(73 / 10) = 8
          total_calls = 172 * 8 = 1,376
        """
        years = list(range(start_year, end_year + 1))
        num_years = len(years)
        num_formats = len(self.SUPPORTED_FORMATS)

        # Compute how many raw items we need total and per year
        if target_examples and num_formats > 0:
            raw_needed = target_examples // num_formats
        else:
            raw_needed = num_years * self.items_per_chunk  # legacy: 1 call/year

        items_per_year = math.ceil(raw_needed / num_years)
        calls_per_year = math.ceil(items_per_year / self.items_per_chunk)
        total_calls = num_years * calls_per_year

        print(f"Generating {num_years} years x {calls_per_year} calls/year "
              f"x {self.items_per_chunk} items/call = ~{raw_needed:,} raw items "
              f"({start_year}-{end_year})")
        print(f"Total API calls: {total_calls:,}")
        for fmt, path in output_paths.items():
            print(f"  {fmt} -> {path.name}")

        all_results = {fmt: [] for fmt in self.SUPPORTED_FORMATS}
        completed = 0
        failed = 0

        # Build task list: (year, call_index) pairs
        tasks = []
        for year in years:
            for call_idx in range(calls_per_year):
                # For last call of a year, request only remaining items
                remaining = items_per_year - call_idx * self.items_per_chunk
                items_this_call = min(self.items_per_chunk, remaining)
                if items_this_call <= 0:
                    break
                tasks.append((year, call_idx, items_this_call))

        with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as executor:
            futures = {}
            for year, call_idx, num_items in tasks:
                prompt = self.build_prompt(
                    year, period, start_year, end_year, num_items=num_items,
                )
                futures[executor.submit(call_api, client, prompt)] = (year, call_idx)

            for future in as_completed(futures):
                year, call_idx = futures[future]
                try:
                    response = future.result()
                    if response:
                        items = self.parse_response(response)
                        for item in items:
                            for fmt in self.SUPPORTED_FORMATS:
                                conv = self.format_conversation(item, fmt)
                                if conv is None:
                                    continue
                                valid, _ = validate_conversation(conv)
                                if valid:
                                    all_results[fmt].append(conv)
                    completed += 1
                except Exception as e:
                    failed += 1
                    print(f"  [ERROR] year {year} call {call_idx}: "
                          f"{type(e).__name__}: {e}", flush=True)

        total_convs = 0
        for fmt in self.SUPPORTED_FORMATS:
            write_jsonl(all_results[fmt], str(output_paths[fmt]))
            count = len(all_results[fmt])
            total_convs += count
            print(f"  {fmt}: {count:,} conversations")

        print(f"Complete. {total_convs:,} total conversations. "
              f"API calls: {completed:,} OK, {failed:,} failed")
        return output_paths
