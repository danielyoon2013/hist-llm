"""Generator F: Instruction Following — passage-based format (MC-4+Passage)."""

from src.post_training.generators.base import (
    BaseGenerator, FORMAT_MC4_PASSAGE,
    render_mc, make_mc_choices,
)
from src.post_training.generators.prompts import INSTRUCT_PROMPT


class GenFInstruct(BaseGenerator):

    gen_key = "F"
    name = "gen_f_instruct"

    def build_prompt(self, chunk, period, start_year, end_year):
        return INSTRUCT_PROMPT.format(
            num_items=self.items_per_chunk, text=chunk,
            start_year=start_year, end_year=end_year,
        )

    def parse_response(self, response):
        return response.get("tasks", [])

    def format_conversation(self, item, fmt, source_chunk=None):
        instruction = item.get("instruction", "")
        passage = item.get("passage", "")

        if fmt == FORMAT_MC4_PASSAGE:
            if not passage:
                return None
            short_answer = item.get("short_answer", "")
            distractors = item.get("distractors", [])
            if not short_answer or len(distractors) < 3:
                return None
            letters, choices, correct = make_mc_choices(
                short_answer, distractors, num_choices=4,
                position_idx=next(self._mc_counters[fmt]),
            )
            passage_question = (
                f"Read the following passage and answer the question.\n\n"
                f"Passage: {passage}\n\n{instruction}"
            )
            user_msg = render_mc(passage_question, letters, choices)
            return [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": correct},
            ]

        return None
