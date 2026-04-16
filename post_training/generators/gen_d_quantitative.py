"""Generator D: Quantitative — multi-format (MC-4, Open-ended, CoT).

Post-processing strategy (A+B combined):

A. Decimal injection from correct answer (~25% of items). The decimal
   distractor is the full unrounded float representation of
   correct_answer × (p/q) for a rational ratio p/q that produces a
   non-terminating decimal (7/6, 5/3, 4/3, etc). This matches GSM-MC's
   distinctive signature of distractors like "64.2857142857143" that
   arise from real wrong-division paths (divide by wrong denominator,
   forget to round). Unlike the earlier random-offset approach, every
   decimal here is derived from REAL arithmetic applied to the correct
   value, not noise — so the model learns "full-unrounded-decimal = wrong
   calculation" rather than "decimal = more specific/correct-looking".

B. Sign-flip injection (~20% of items). The distractor is -correct_answer,
   simulating a student who subtracted in the wrong order. Mutually
   exclusive with A on the same item.

Dollar signs and commas are stripped from all choices before rendering
(model sometimes mixes "$1100" with "350" in the same item).
"""

import re
import random as _random

from src.post_training.generators.base import (
    BaseGenerator, FORMAT_MC4, FORMAT_OPEN, FORMAT_COT,
    render_mc, make_mc_choices,
)
from src.post_training.generators.prompts import QUANTITATIVE_PROMPT


_DECIMAL_PROB = 0.25  # fraction of MC4 items that get a long-unrounded-float distractor
_SIGNFLIP_PROB = 0.20  # fraction of MC4 items (disjoint from decimal) that get -correct as distractor

# Rational ratios p/q where p/q is in irreducible form and q has prime factors
# other than 2 and 5 — guarantees a non-terminating decimal expansion.
_WRONG_RATIOS = [(7, 6), (5, 3), (4, 3), (11, 7), (9, 7), (13, 9),
                 (5, 6), (2, 3), (7, 9), (8, 3), (11, 6), (13, 6)]


def _strip_currency(s):
    """Strip $ and thousands commas. Leaves digits, minus sign, decimal point, %."""
    if not isinstance(s, str):
        s = str(s)
    return re.sub(r'[\$,]', '', s).strip()


def _to_float(s):
    """Parse a cleaned numeric string to float; returns None if non-numeric."""
    clean = _strip_currency(str(s)).rstrip('%')
    try:
        return float(clean)
    except ValueError:
        return None


def _decimalize_from_correct(correct_str, rng):
    """Return a full-unrounded-float decimal = correct × p/q for a ratio that
    produces a non-terminating decimal. Returns None if correct can't be parsed."""
    val = _to_float(correct_str)
    if val is None or val == 0:
        return None
    num, den = rng.choice(_WRONG_RATIOS)
    result = val * num / den
    # repr() produces Python's full-precision float string ("46.666666666666664"),
    # matching GSM-MC's distinctive unrounded-decimal signature.
    return repr(result)


def _sign_flip(correct_str):
    """Return -correct_answer formatted as a string. Returns None if non-numeric."""
    val = _to_float(correct_str)
    if val is None or val == 0:
        return None
    flipped = -val
    if flipped == int(flipped):
        return str(int(flipped))
    return str(flipped)


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

            # Strip $ and commas from correct + distractors so the rendered
            # choices have consistent formatting.
            answer_clean = _strip_currency(answer)
            distractors_clean = [_strip_currency(d) for d in distractors]

            # Decimal OR sign-flip (mutually exclusive, both derived from correct).
            rng = _random.Random()
            roll = rng.random()
            if roll < _DECIMAL_PROB:
                injected = _decimalize_from_correct(answer_clean, rng)
            elif roll < _DECIMAL_PROB + _SIGNFLIP_PROB:
                injected = _sign_flip(answer_clean)
            else:
                injected = None

            if injected is not None and injected not in distractors_clean and injected != answer_clean:
                idx = rng.randint(0, len(distractors_clean) - 1)
                distractors_clean[idx] = injected

            letters, choices, correct = make_mc_choices(
                answer_clean, distractors_clean, num_choices=4,
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
