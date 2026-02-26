"""Generator D: Temporal Reasoning — multi-format (MC-4, Open-ended). Metadata-based, no corpus."""

from src.post_training.generators.base import (
    BaseGenerator, FORMAT_MC4, FORMAT_OPEN, render_mc,
)
from src.post_training.generators.prompts import TEMPORAL_PROMPT


class GenDTemporal(BaseGenerator):

    name = "gen_d_temporal"
    items_per_chunk = 5
    needs_corpus = False
    num_batches = 10
    SUPPORTED_FORMATS = (FORMAT_MC4, FORMAT_OPEN)

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

        if fmt == FORMAT_MC4:
            letters = ("A", "B", "C", "D")
            choices = [choices_dict.get(l, "") for l in letters]
            user_msg = render_mc(question, letters, choices)
            return [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": correct_letter},
            ]

        if fmt == FORMAT_OPEN:
            # Extract the correct answer text
            correct_text = choices_dict.get(correct_letter, "")
            if not correct_text:
                return None
            return [
                {"role": "user", "content": question},
                {"role": "assistant", "content": correct_text},
            ]

        return None
