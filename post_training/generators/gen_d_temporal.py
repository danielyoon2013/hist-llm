"""Generator D: Temporal Reasoning — multi-format (MC-4, Open-ended). Metadata-based, no corpus.

Batch count computed dynamically from target_examples.
"""

from src.post_training.generators.base import (
    BaseGenerator, FORMAT_MC4, FORMAT_OPEN, render_mc, make_mc_choices,
)
from src.post_training.generators.prompts import TEMPORAL_PROMPT


class GenDTemporal(BaseGenerator):

    gen_key = "D"
    name = "gen_d_temporal"
    num_batches = 10  # legacy default; overridden by target_examples

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

    def format_conversation(self, item, fmt, source_chunk=None):
        question = item.get("question", "")
        choices_dict = item.get("choices", {})
        correct_letter = item.get("correct", "A")

        letters_4 = ("A", "B", "C", "D")
        correct_text = choices_dict.get(correct_letter, "")
        distractors = [choices_dict[l] for l in letters_4
                       if l != correct_letter and choices_dict.get(l)]

        if fmt == FORMAT_MC4:
            if len(distractors) < 3 or not correct_text:
                return None
            letters, choices, correct = make_mc_choices(
                correct_text, distractors, num_choices=4,
                position_idx=next(self._mc_counters[fmt]),
            )
            user_msg = render_mc(question, letters, choices)
            return [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": correct},
            ]

        if fmt == FORMAT_OPEN:
            if not correct_text:
                return None
            return [
                {"role": "user", "content": question},
                {"role": "assistant", "content": correct_text},
            ]

        return None
