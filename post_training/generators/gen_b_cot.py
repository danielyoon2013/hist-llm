"""Generator B: Physical Commonsense (PIQA-style) — MC-2, CoT."""

from src.post_training.generators.base import (
    BaseGenerator, FORMAT_MC2, FORMAT_COT,
    render_mc, make_mc_choices,
)
from src.post_training.generators.prompts import PIQA_PROMPT


class GenBCoT(BaseGenerator):

    gen_key = "B"
    name = "gen_b_cot"

    def build_prompt(self, chunk, period, start_year, end_year):
        return PIQA_PROMPT.format(num_items=self.items_per_chunk, text=chunk,
                                  start_year=start_year, end_year=end_year)

    def parse_response(self, response):
        return response.get("piqa_items", [])

    def format_conversation(self, item, fmt, source_chunk=None):
        goal = item.get("goal", "")
        solution = item.get("solution", "")
        distractor = item.get("distractor", "")
        reasoning = item.get("reasoning", "")

        if not goal or not solution or not distractor:
            return None

        if fmt == FORMAT_MC2:
            letters, choices, correct = make_mc_choices(
                solution, [distractor], num_choices=2,
                position_idx=next(self._mc_counters[fmt]),
            )
            user_msg = render_mc(goal, letters, choices)
            return [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": correct},
            ]

        if fmt == FORMAT_COT:
            if not reasoning:
                return None
            content = f"<think>\n{reasoning}\n</think>\n{solution}"
            return [
                {"role": "user", "content": goal},
                {"role": "assistant", "content": content},
            ]

        return None
