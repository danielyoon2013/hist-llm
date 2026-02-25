"""Generator F: Sentence Completion — HellaSwag-style MC questions."""

from src.post_training.generators.base import BaseGenerator
from src.post_training.generators.prompts import COMPLETION_PROMPT


MC_TEMPLATE = """{context}
- choice=A: {a}
- choice=B: {b}
- choice=C: {c}
- choice=D: {d}

Respond only with the letter of the correct answer."""


class GenFCompletion(BaseGenerator):

    name = "gen_f_completion"
    items_per_chunk = 3

    def build_prompt(self, chunk, period, start_year, end_year):
        return COMPLETION_PROMPT.format(num_items=self.items_per_chunk, text=chunk)

    def parse_response(self, response):
        return response.get("completions", [])

    def format_conversation(self, item):
        choices = item.get("choices", {})
        user_msg = MC_TEMPLATE.format(
            context=item.get("context", ""),
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
