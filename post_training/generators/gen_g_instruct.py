"""Generator G: Instruction Following — instruction-response pairs from passages."""

from src.post_training.generators.base import BaseGenerator
from src.post_training.generators.prompts import INSTRUCT_PROMPT


class GenGInstruct(BaseGenerator):

    name = "gen_g_instruct"
    items_per_chunk = 2

    def build_prompt(self, chunk, period, start_year, end_year):
        return INSTRUCT_PROMPT.format(num_items=self.items_per_chunk, text=chunk)

    def parse_response(self, response):
        return response.get("tasks", [])

    def format_conversation(self, item):
        return [
            {"role": "user", "content": item["instruction"]},
            {"role": "assistant", "content": item["response"]},
        ]
