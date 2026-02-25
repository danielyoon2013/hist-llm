"""Generator C: Reading Comprehension — passage-based MC questions."""

from src.post_training.generators.base import BaseGenerator
from src.post_training.generators.prompts import COMPREHENSION_PROMPT


MC_TEMPLATE = """{passage}

{question}
- choice=A: {a}
- choice=B: {b}
- choice=C: {c}
- choice=D: {d}

Respond only with the letter of the correct answer."""


class GenCComprehension(BaseGenerator):

    name = "gen_c_comprehension"
    items_per_chunk = 3

    def build_prompt(self, chunk, period, start_year, end_year):
        return COMPREHENSION_PROMPT.format(num_items=self.items_per_chunk, text=chunk)

    def parse_response(self, response):
        return response.get("questions", [])

    def format_conversation(self, item):
        choices = item.get("choices", {})
        user_msg = MC_TEMPLATE.format(
            passage=item.get("question", "").split("?")[0] + "?" if "?" in item.get("question", "") else item.get("question", ""),
            question=item.get("question", ""),
            a=choices.get("A", ""),
            b=choices.get("B", ""),
            c=choices.get("C", ""),
            d=choices.get("D", ""),
        )
        correct = item.get("correct", "A")
        return [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": correct},
        ]
