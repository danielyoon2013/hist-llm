"""Generator F: Instruction Following — multi-format (MC-4+Passage, Open-ended, CoT)."""

from src.post_training.generators.base import (
    BaseGenerator, FORMAT_MC4_PASSAGE, FORMAT_OPEN, FORMAT_COT,
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
        response = item.get("response", "")
        short_answer = item.get("short_answer", "")

        if fmt == FORMAT_MC4_PASSAGE:
            if not passage:
                return None
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

        if fmt == FORMAT_OPEN:
            if not passage or not response:
                return None
            user_msg = (
                f"Read the following passage and follow the instruction.\n\n"
                f"Passage: {passage}\n\n{instruction}"
            )
            return [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response},
            ]

        if fmt == FORMAT_COT:
            if not passage or not response or not short_answer:
                return None
            user_msg = (
                f"Read the following passage and follow the instruction.\n\n"
                f"Passage: {passage}\n\n{instruction}"
            )
            content = f"<think>\n{response}\n</think>\n{short_answer}"
            return [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": content},
            ]

        return None
