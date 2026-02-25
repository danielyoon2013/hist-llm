"""Generator A: Factual QA — open-ended question-answer pairs from corpus text."""

from src.post_training.generators.base import BaseGenerator
from src.post_training.generators.prompts import QA_PROMPT


class GenAFactual(BaseGenerator):

    name = "gen_a_factual"
    items_per_chunk = 3

    def build_prompt(self, chunk, period, start_year, end_year):
        return QA_PROMPT.format(num_items=self.items_per_chunk, text=chunk)

    def parse_response(self, response):
        return response.get("qa_pairs", [])

    def format_conversation(self, item):
        return [
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["answer"]},
        ]
