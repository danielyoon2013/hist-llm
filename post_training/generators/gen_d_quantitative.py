"""Generator D: Quantitative — multi-format (MC-4, Open-ended, CoT).

Post-processing: one distractor is converted to a realistic decimal
("unfinished calculation") to teach the model to reject non-integer
answers — countering the GSM-MC decimal-bias failure mode.
"""

import re
import random as _random

from src.post_training.generators.base import (
    BaseGenerator, FORMAT_MC4, FORMAT_OPEN, FORMAT_COT,
    render_mc, make_mc_choices,
)
from src.post_training.generators.prompts import QUANTITATIVE_PROMPT


def _decimalize(integer_str: str) -> str | None:
    """Turn a wrong INTEGER distractor into a decimal that looks like an
    unfinished calculation. The result is far from the correct answer
    (matching GSM-MC's pattern where decimals come from wrong calc paths).
    Returns None if input isn't numeric."""
    clean = re.sub(r'[,$%]', '', integer_str).strip()
    try:
        val = float(clean)
    except ValueError:
        return None
    if val == 0:
        return None
    rng = _random.Random()
    # Add a fractional part that looks like an unfinished division
    divisor = rng.choice([3, 6, 7, 9, 11, 13, 17, 19, 23])
    frac_offset = val / divisor * rng.uniform(0.01, 0.1)
    sign = rng.choice([-1, 1])
    n_places = rng.choice([2, 2, 3, 3, 4])
    decimal_val = round(val + frac_offset * sign, n_places)
    # Ensure non-trivial fractional part
    if decimal_val == int(decimal_val):
        decimal_val = round(decimal_val + rng.uniform(0.01, 0.99), n_places)
    return str(decimal_val)


class GenDQuantitative(BaseGenerator):

    gen_key = "D"
    name = "gen_d_quantitative"

    def build_prompt(self, chunk, period, start_year, end_year):
        return QUANTITATIVE_PROMPT.format(num_items=self.items_per_chunk, text=chunk,
                                          start_year=start_year, end_year=end_year)

    def parse_response(self, response):
        return response.get("problems", [])

    def format_conversation(self, item, fmt, source_chunk=None):
        question = item.get("question", "")
        answer = item.get("answer", "")
        reasoning = item.get("reasoning", "")

        if fmt == FORMAT_MC4:
            distractors = list(item.get("distractors", []))
            if not answer or len(distractors) < 3:
                return None
            # Post-process: ~25% of items get one wrong-integer distractor
            # converted to a decimal (matching GSM-MC's ~27% rate).
            # The decimal is derived from the DISTRACTOR (not correct answer),
            # so it's far from the correct answer — matching GSM-MC's pattern
            # where decimals come from wrong calculation paths.
            if _random.random() < 0.25:
                idx = _random.randint(0, len(distractors) - 1)
                decimal = _decimalize(distractors[idx])
                if decimal is not None:
                    distractors[idx] = decimal
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
