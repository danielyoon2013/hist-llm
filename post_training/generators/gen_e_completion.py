"""Generator E: Sentence Completion — multi-format (MC-4, MC-2)."""

import random as _random

from src.post_training.generators.base import (
    BaseGenerator, FORMAT_MC4, FORMAT_MC2, render_mc, make_mc_choices,
)
from src.post_training.generators.prompts import COMPLETION_PROMPT


class GenECompletion(BaseGenerator):

    gen_key = "E"
    name = "gen_e_completion"

    def build_prompt(self, chunk, period, start_year, end_year):
        return COMPLETION_PROMPT.format(num_items=self.items_per_chunk, text=chunk)

    def parse_response(self, response):
        return response.get("completions", [])

    def format_conversation(self, item, fmt, source_chunk=None):
        context = item.get("context", "")
        choices_dict = item.get("choices", {})
        correct_letter = item.get("correct", "A")
        correct_text = choices_dict.get(correct_letter, "")
        distractors = [choices_dict[l] for l in ("A", "B", "C", "D")
                       if l != correct_letter and choices_dict.get(l)]

        if fmt == FORMAT_MC4:
            if len(distractors) < 3 or not correct_text:
                return None
            letters, choices, correct = make_mc_choices(
                correct_text, distractors, num_choices=4,
                position_idx=next(self._mc_counters[fmt]),
            )
            user_msg = render_mc(context, letters, choices)
            return [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": correct},
            ]

        if fmt == FORMAT_MC2:
            if not distractors or not correct_text:
                return None
            rng = _random.Random(hash(context))
            wrong_choice = rng.choice(distractors)
            letters_2, choices_2, correct_2 = make_mc_choices(
                correct_text, [wrong_choice], num_choices=2,
                position_idx=next(self._mc_counters[fmt]),
            )
            user_msg = render_mc(context, letters_2, choices_2)
            return [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": correct_2},
            ]

        return None
