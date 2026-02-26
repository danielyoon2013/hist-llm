"""Generator C: Reading Comprehension — multi-format (MC-4, MC-4+Passage, MC-2+Passage)."""

import random as _random

from src.post_training.generators.base import (
    BaseGenerator, FORMAT_MC4, FORMAT_MC4_PASSAGE, FORMAT_MC2_PASSAGE,
    render_mc, make_mc_choices, truncate_passage,
)
from src.post_training.generators.prompts import COMPREHENSION_PROMPT


class GenCComprehension(BaseGenerator):

    name = "gen_c_comprehension"
    items_per_chunk = 3
    SUPPORTED_FORMATS = (FORMAT_MC4, FORMAT_MC4_PASSAGE, FORMAT_MC2_PASSAGE)

    def build_prompt(self, chunk, period, start_year, end_year):
        return COMPREHENSION_PROMPT.format(num_items=self.items_per_chunk, text=chunk)

    def parse_response(self, response):
        return response.get("questions", [])

    def format_conversation(self, item, fmt, source_chunk=None):
        question = item.get("question", "")
        choices_dict = item.get("choices", {})
        correct_letter = item.get("correct", "A")

        # Extract ordered choices
        letters_4 = ("A", "B", "C", "D")
        choices_4 = [choices_dict.get(l, "") for l in letters_4]
        correct_text = choices_dict.get(correct_letter, "")

        if fmt == FORMAT_MC4:
            # Standard MC-4, no passage prefix (HellaSwag/MMLU style)
            user_msg = render_mc(question, letters_4, choices_4)
            return [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": correct_letter},
            ]

        if fmt == FORMAT_MC4_PASSAGE:
            # RACE-style: passage + MC-4
            if not source_chunk:
                return None
            passage = truncate_passage(source_chunk)
            passage_question = (
                f"Read the following passage and answer the question.\n\n"
                f"Passage: {passage}\n\n{question}"
            )
            user_msg = render_mc(passage_question, letters_4, choices_4)
            return [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": correct_letter},
            ]

        if fmt == FORMAT_MC2_PASSAGE:
            # BoolQ-style: passage + MC-2
            if not source_chunk:
                return None
            # Pick one distractor
            wrong_choices = [
                choices_dict[l] for l in letters_4
                if l != correct_letter and choices_dict.get(l)
            ]
            if not wrong_choices:
                return None
            rng = _random.Random(hash(question))
            wrong_choice = rng.choice(wrong_choices)
            letters_2, choices_2, correct_2 = make_mc_choices(
                correct_text, [wrong_choice], num_choices=2, seed=hash(question)
            )
            passage = truncate_passage(source_chunk)
            passage_question = (
                f"Passage: {passage}\n\nQuestion: {question}"
            )
            user_msg = render_mc(passage_question, letters_2, choices_2)
            return [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": correct_2},
            ]

        return None
