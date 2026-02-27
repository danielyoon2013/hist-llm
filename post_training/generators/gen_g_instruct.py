"""Generator G: Instruction Following — multi-format (Open-ended, MC-4+Passage)."""

from src.post_training.generators.base import (
    BaseGenerator, FORMAT_OPEN, FORMAT_MC4_PASSAGE,
    render_mc, make_mc_choices, truncate_passage,
)
from src.post_training.generators.prompts import INSTRUCT_PROMPT


class GenGInstruct(BaseGenerator):

    name = "gen_g_instruct"
    items_per_chunk = 2
    SUPPORTED_FORMATS = (FORMAT_OPEN, FORMAT_MC4_PASSAGE)

    def build_prompt(self, chunk, period, start_year, end_year):
        return INSTRUCT_PROMPT.format(num_items=self.items_per_chunk, text=chunk)

    def parse_response(self, response):
        return response.get("tasks", [])

    def format_conversation(self, item, fmt, source_chunk=None):
        instruction = item.get("instruction", "")
        response_text = item.get("response", "")

        if fmt == FORMAT_OPEN:
            return [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response_text},
            ]

        if fmt == FORMAT_MC4_PASSAGE:
            # RACE-style: passage + instruction as question, short answers as MC choices
            if not source_chunk:
                return None
            short_answer = item.get("short_answer", "")
            distractors = item.get("distractors", [])
            if not short_answer or len(distractors) < 3:
                return None
            letters, choices, correct = make_mc_choices(
                short_answer, distractors, num_choices=4,
                position_idx=next(self._mc_counters[fmt]),
            )
            passage = truncate_passage(source_chunk)
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
