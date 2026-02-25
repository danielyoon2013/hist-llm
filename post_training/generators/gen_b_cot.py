"""Generator B: Chain-of-Thought — reasoning examples with <think> tags."""

from src.post_training.generators.base import BaseGenerator
from src.post_training.generators.prompts import COT_PROMPT


class GenBCoT(BaseGenerator):

    name = "gen_b_cot"
    items_per_chunk = 2

    def build_prompt(self, chunk, period, start_year, end_year):
        return COT_PROMPT.format(num_items=self.items_per_chunk, text=chunk)

    def parse_response(self, response):
        return response.get("cot_examples", [])

    def format_conversation(self, item):
        answer = item.get("answer", "")
        reasoning = item.get("reasoning", "")
        if reasoning:
            answer = f"<think>\n{reasoning}\n</think>\n{answer}"
        return [
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": answer},
        ]
