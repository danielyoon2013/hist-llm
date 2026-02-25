"""Generator E: Quantitative — math word problems from text with numbers."""

from src.post_training.generators.base import BaseGenerator
from src.post_training.generators.prompts import QUANTITATIVE_PROMPT


class GenEQuantitative(BaseGenerator):

    name = "gen_e_quantitative"
    items_per_chunk = 2

    def build_prompt(self, chunk, period, start_year, end_year):
        return QUANTITATIVE_PROMPT.format(num_items=self.items_per_chunk, text=chunk)

    def parse_response(self, response):
        return response.get("problems", [])

    def format_conversation(self, item):
        answer = item.get("answer", "")
        reasoning = item.get("reasoning", "")
        if reasoning:
            answer = f"<think>\n{reasoning}\n</think>\n{answer}"
        return [
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": answer},
        ]
