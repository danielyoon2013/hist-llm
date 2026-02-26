"""Generator F: Sentence Completion — multi-format (MC-4, MC-2)."""

import random as _random

from src.post_training.generators.base import (
    BaseGenerator, FORMAT_MC4, FORMAT_MC2, render_mc, make_mc_choices,
)
from src.post_training.generators.prompts import COMPLETION_PROMPT


class GenFCompletion(BaseGenerator):

    name = "gen_f_completion"
    items_per_chunk = 3
    SUPPORTED_FORMATS = (FORMAT_MC4, FORMAT_MC2)

    def build_prompt(self, chunk, period, start_year, end_year):
        return COMPLETION_PROMPT.format(num_items=self.items_per_chunk, text=chunk)

    def parse_response(self, response):
        return response.get("completions", [])

    def format_conversation(self, item, fmt, source_chunk=None):
        context = item.get("context", "")
        choices_dict = item.get("choices", {})
        correct_letter = item.get("correct", "A")
        correct_text = choices_dict.get(correct_letter, "")

        if fmt == FORMAT_MC4:
            # HellaSwag-style: context + 4 completions
            letters = ("A", "B", "C", "D")
            choices = [choices_dict.get(l, "") for l in letters]
            user_msg = render_mc(context, letters, choices)
            return [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": correct_letter},
            ]

        if fmt == FORMAT_MC2:
            # WinoGrande-style: context + 2 completions
            wrong_choices = [
                choices_dict[l] for l in ("A", "B", "C", "D")
                if l != correct_letter and choices_dict.get(l)
            ]
            if not wrong_choices:
                return None
            rng = _random.Random(hash(context))
            wrong_choice = rng.choice(wrong_choices)
            letters_2, choices_2, correct_2 = make_mc_choices(
                correct_text, [wrong_choice], num_choices=2, seed=hash(context)
            )
            user_msg = render_mc(context, letters_2, choices_2)
            return [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": correct_2},
            ]

        return None
