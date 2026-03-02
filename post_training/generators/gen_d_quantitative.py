"""Generator D: Quantitative — multi-format (MC-4, Open-ended, CoT)."""

from src.post_training.generators.base import (
    BaseGenerator, FORMAT_MC4, FORMAT_OPEN, FORMAT_COT,
    render_mc, make_mc_choices,
)
from src.post_training.generators.prompts import QUANTITATIVE_PROMPT


class GenDQuantitative(BaseGenerator):

    gen_key = "D"
    name = "gen_d_quantitative"

    def build_prompt(self, chunk, period, start_year, end_year):
        return QUANTITATIVE_PROMPT.format(num_items=self.items_per_chunk, text=chunk)

    def parse_response(self, response):
        return response.get("problems", [])

    def format_conversation(self, item, fmt, source_chunk=None):
        question = item.get("question", "")
        answer = item.get("answer", "")
        reasoning = item.get("reasoning", "")

        if fmt == FORMAT_MC4:
            distractors = item.get("distractors", [])
            if not answer or len(distractors) < 3:
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

        if fmt == FORMAT_COT:
            content = f"<think>\n{reasoning}\n</think>\n{answer}" if reasoning else answer
            return [
                {"role": "user", "content": question},
                {"role": "assistant", "content": content},
            ]

        if fmt == FORMAT_OPEN:
            return [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]

        return None
