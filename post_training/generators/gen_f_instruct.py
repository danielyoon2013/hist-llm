"""Generator F: Pronoun Resolution (Winogrande-style) — MC-2, CoT."""

from src.post_training.generators.base import (
    BaseGenerator, FORMAT_MC2, FORMAT_COT,
    render_mc, make_mc_choices,
)
from src.post_training.generators.prompts import COMMONSENSE_REFERENCE_PROMPT


class GenFInstruct(BaseGenerator):

    gen_key = "F"
    name = "gen_f_instruct"

    def build_prompt(self, chunk, period, start_year, end_year):
        return COMMONSENSE_REFERENCE_PROMPT.format(
            num_items=self.items_per_chunk, text=chunk,
            start_year=start_year, end_year=end_year,
        )

    def parse_response(self, response):
        return response.get("winogrande_items", [])

    def format_conversation(self, item, fmt, source_chunk=None):
        sentence = item.get("sentence", "").strip()
        opt_a = item.get("option_a", "").strip()
        opt_b = item.get("option_b", "").strip()
        correct_letter = item.get("correct", "").strip().upper()
        reasoning = item.get("reasoning", "").strip()

        if not sentence or not opt_a or not opt_b or correct_letter not in ("A", "B"):
            return None
        if "_" not in sentence:
            return None

        sent_lower = sentence.lower()
        def _in_sentence(opt):
            o = opt.lower().strip()
            if o in sent_lower:
                return True
            for prefix in ("the ", "a ", "an "):
                if o.startswith(prefix) and o[len(prefix):] in sent_lower:
                    return True
            return False
        if not _in_sentence(opt_a) or not _in_sentence(opt_b):
            return None

        correct_filler = opt_a if correct_letter == "A" else opt_b
        distractor = opt_b if correct_letter == "A" else opt_a

        if fmt == FORMAT_MC2:
            letters, choices, correct = make_mc_choices(
                correct_filler, [distractor], num_choices=2,
                position_idx=next(self._mc_counters[fmt]),
            )
            user_msg = render_mc(f"Fill in the blank: {sentence}", letters, choices)
            return [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": correct},
            ]

        if fmt == FORMAT_COT:
            if not reasoning:
                return None
            content = f"<think>\n{reasoning}\n</think>\n{correct_filler}"
            return [
                {"role": "user", "content": f"Fill in the blank: {sentence}"},
                {"role": "assistant", "content": content},
            ]

        return None
