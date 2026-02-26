"""Generator E: Quantitative — multi-format (Open-ended, CoT)."""

from src.post_training.generators.base import (
    BaseGenerator, FORMAT_OPEN, FORMAT_COT,
)
from src.post_training.generators.prompts import QUANTITATIVE_PROMPT


class GenEQuantitative(BaseGenerator):

    name = "gen_e_quantitative"
    items_per_chunk = 2
    SUPPORTED_FORMATS = (FORMAT_OPEN, FORMAT_COT)

    def build_prompt(self, chunk, period, start_year, end_year):
        return QUANTITATIVE_PROMPT.format(num_items=self.items_per_chunk, text=chunk)

    def parse_response(self, response):
        return response.get("problems", [])

    def format_conversation(self, item, fmt, source_chunk=None):
        question = item.get("question", "")
        answer = item.get("answer", "")
        reasoning = item.get("reasoning", "")

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
