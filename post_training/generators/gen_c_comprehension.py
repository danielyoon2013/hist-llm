"""Generator C: Reading Comprehension — passage-based formats (MC-4+Passage, MC-2+Passage)."""

import random as _random

from src.post_training.generators.base import (
    BaseGenerator, FORMAT_MC4_PASSAGE, FORMAT_MC2_PASSAGE,
    render_mc, make_mc_choices, truncate_passage,
)
from src.post_training.generators.prompts import COMPREHENSION_PROMPT


class GenCComprehension(BaseGenerator):

    gen_key = "C"
    name = "gen_c_comprehension"

    def build_prompt(self, chunk, period, start_year, end_year):
        return COMPREHENSION_PROMPT.format(num_items=self.items_per_chunk, text=chunk)

    def parse_response(self, response):
        return response.get("questions", [])

    def format_conversation(self, item, fmt, source_chunk=None):
        question = item.get("question", "")
        choices_dict = item.get("choices", {})
        correct_letter = item.get("correct", "A")

        # Extract correct text and distractors from GPT's choices dict
        letters_4 = ("A", "B", "C", "D")
        correct_text = choices_dict.get(correct_letter, "")
        distractors = [choices_dict[l] for l in letters_4
                       if l != correct_letter and choices_dict.get(l)]

        if fmt == FORMAT_MC4_PASSAGE:
            if not source_chunk or len(distractors) < 3:
                return None
            letters, choices, correct = make_mc_choices(
                correct_text, distractors, num_choices=4,
                position_idx=next(self._mc_counters[fmt]),
            )
            passage = truncate_passage(source_chunk)
            passage_question = (
                f"Read the following passage and answer the question.\n\n"
                f"Passage: {passage}\n\n{question}"
            )
            user_msg = render_mc(passage_question, letters, choices)
            return [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": correct},
            ]

        if fmt == FORMAT_MC2_PASSAGE:
            if not source_chunk or not distractors:
                return None
            # Pick one distractor deterministically
            rng = _random.Random(hash(question))
            wrong_choice = rng.choice(distractors)
            letters_2, choices_2, correct_2 = make_mc_choices(
                correct_text, [wrong_choice], num_choices=2,
                position_idx=next(self._mc_counters[fmt]),
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
