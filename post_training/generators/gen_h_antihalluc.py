"""Generator H: Anti-Hallucination — multi-format (MC-4, Open-ended). Metadata-based, no corpus."""

from src.post_training.generators.base import (
    BaseGenerator, FORMAT_MC4, FORMAT_OPEN, render_mc, make_mc_choices,
)
from src.post_training.generators.prompts import ANTIHALLUC_PROMPT


class GenHAntiHalluc(BaseGenerator):

    name = "gen_h_antihalluc"
    items_per_chunk = 5
    needs_corpus = False
    num_batches = 10
    SUPPORTED_FORMATS = (FORMAT_MC4, FORMAT_OPEN)

    def build_prompt(self, chunk, period, start_year, end_year):
        # chunk is actually batch_num for metadata-based generators
        return ANTIHALLUC_PROMPT.format(
            num_items=self.items_per_chunk,
            end_year=end_year,
            batch_num=chunk,
        )

    def parse_response(self, response):
        return response.get("examples", [])

    def format_conversation(self, item, fmt, source_chunk=None):
        question = item.get("question", "")
        response_text = item.get("response", "")

        if fmt == FORMAT_OPEN:
            return [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response_text},
            ]

        if fmt == FORMAT_MC4:
            # MC where correct answer is the refusal, distractors are hallucinations
            distractors = item.get("distractors", [])
            if len(distractors) < 3:
                return None
            letters, choices, correct = make_mc_choices(
                response_text, distractors, num_choices=4,
                position_idx=next(self._mc_counters[fmt]),
            )
            user_msg = render_mc(question, letters, choices)
            return [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": correct},
            ]

        return None
