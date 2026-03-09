"""Generator A: Factual QA — multi-format (MC-4, Open-ended)."""

from src.post_training.generators.base import (
    BaseGenerator, FORMAT_MC4, FORMAT_OPEN, render_mc, make_mc_choices,
)
from src.post_training.generators.prompts import QA_PROMPT


class GenAFactual(BaseGenerator):

    gen_key = "A"
    name = "gen_a_factual"

    def build_prompt(self, chunk, period, start_year, end_year):
        return QA_PROMPT.format(num_items=self.items_per_chunk, text=chunk,
                                start_year=start_year, end_year=end_year)

    def parse_response(self, response):
        return response.get("qa_pairs", [])

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
