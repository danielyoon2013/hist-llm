"""Generator H: Historical Facts & Dates — multi-format (MC-4, Open-ended). Metadata-based, no corpus.

Generates per-year (one API call per year in the period) to eliminate duplicates.
Train-only: factual recall is evaluated via external benchmarks, not held-out test set.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from src.post_training.generators.base import (
    BaseGenerator, FORMAT_MC4, FORMAT_OPEN, render_mc, make_mc_choices,
    call_api,
)
from src.post_training.generators.prompts import HIST_FACTS_PROMPT
from src.post_training.utils import validate_conversation, write_jsonl


class GenHHistFacts(BaseGenerator):

    name = "gen_h_histfacts"
    items_per_chunk = 5
    needs_corpus = False
    train_only = True  # factual recall — no test split
    SUPPORTED_FORMATS = (FORMAT_MC4, FORMAT_OPEN)

    def build_prompt(self, year, period, start_year, end_year):
        return HIST_FACTS_PROMPT.format(
            num_items=self.items_per_chunk,
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
                            max_workers, output_paths):
        """Override: one API call per year in the period to eliminate duplicates."""
        years = list(range(start_year, end_year + 1))
        print(f"Generating {len(years)} years x {self.items_per_chunk} items "
              f"({start_year}-{end_year})")
        for fmt, path in output_paths.items():
            print(f"  {fmt} -> {path.name}")

        all_results = {fmt: [] for fmt in self.SUPPORTED_FORMATS}
        completed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=min(max_workers, len(years))) as executor:
            futures = {}
            for year in years:
                prompt = self.build_prompt(year, period, start_year, end_year)
                futures[executor.submit(call_api, client, prompt)] = year

            for future in as_completed(futures):
                year = futures[future]
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
                    print(f"  [ERROR] year {year}: {type(e).__name__}: {e}",
                          flush=True)

        total_convs = 0
        for fmt in self.SUPPORTED_FORMATS:
            write_jsonl(all_results[fmt], str(output_paths[fmt]))
            count = len(all_results[fmt])
            total_convs += count
            print(f"  {fmt}: {count:,} conversations")

        print(f"Complete. {total_convs:,} total conversations. Failed: {failed}")
        return output_paths
