"""Generator G: Multi-format rephrasing (FinePhrase Appendix B reproduction).

Each source chunk produces FOUR SEPARATE API calls — one per format (tutorial,
FAQ, narrative, explanation). Plain-text output (not JSON). No topic field,
no word-count caps — matches FinePhrase paper verbatim.

Prompt bodies are verbatim from huggingface/finephrase prompts/format/*.md;
only the temporal preamble (1900-1949 period constraint) is our addition.

Conversation format: instruction-tuning style with a brief task hint as user
turn and the rephrased passage as assistant turn. Targets HellaSwag/Winogrande
(fluent generative prose) and RACE (multi-paragraph comprehension) where the
current MC-only training pipeline has left coverage gaps.
"""
from src.post_training.generators.base import BaseGenerator
from src.post_training.generators.prompts import (
    REPHRASE_TUTORIAL_PROMPT,
    REPHRASE_FAQ_PROMPT,
    REPHRASE_NARRATIVE_PROMPT,
    REPHRASE_EXPLANATION_PROMPT,
    REPHRASE_MATH_PROMPT,
)

# Format constants — must match entries in GENERATOR_SPEC["G"]["formats"]
FORMAT_REPHRASE_TUTORIAL    = "rephrase_tutorial"
FORMAT_REPHRASE_FAQ         = "rephrase_faq"
FORMAT_REPHRASE_NARRATIVE   = "rephrase_narrative"
FORMAT_REPHRASE_EXPLANATION = "rephrase_explanation"
FORMAT_REPHRASE_MATH        = "rephrase_math"

_PROMPT_BY_FORMAT = {
    FORMAT_REPHRASE_TUTORIAL:    REPHRASE_TUTORIAL_PROMPT,
    FORMAT_REPHRASE_FAQ:         REPHRASE_FAQ_PROMPT,
    FORMAT_REPHRASE_NARRATIVE:   REPHRASE_NARRATIVE_PROMPT,
    FORMAT_REPHRASE_EXPLANATION: REPHRASE_EXPLANATION_PROMPT,
    FORMAT_REPHRASE_MATH:        REPHRASE_MATH_PROMPT,
}

_USER_INSTRUCTION_BY_FORMAT = {
    FORMAT_REPHRASE_TUTORIAL:    "Rewrite the following as a step-by-step tutorial:",
    FORMAT_REPHRASE_FAQ:         "Rewrite the following as a comprehensive FAQ:",
    FORMAT_REPHRASE_NARRATIVE:   "Rewrite the following as a flowing narrative:",
    FORMAT_REPHRASE_EXPLANATION: "Rewrite the following as a clear explanation:",
    FORMAT_REPHRASE_MATH:        "Rewrite the following as a math word problem with step-by-step solution:",
}


class GenGRephrase(BaseGenerator):
    """Multi-format rephrasing generator. Each chunk → 4 API calls."""

    gen_key = "G"
    name = "gen_g_rephrase"

    # ---- Hooks required by BaseGenerator ----

    def build_prompt(self, chunk, period, start_year, end_year):
        """Unused for Gen G — expand_chunk_to_tasks handles prompt construction
        because each format needs its own prompt. build_prompt is declared
        abstract on the base class, so we provide a no-op that will raise if
        ever called (which it shouldn't be, since our task-expansion path
        bypasses it)."""
        raise NotImplementedError(
            "Gen G uses expand_chunk_to_tasks() per format; build_prompt is unused."
        )

    def expand_chunk_to_tasks(self, chunk, period, start_year, end_year):
        """Expand one chunk into 4 format-specific tasks."""
        return [
            (fmt, tmpl.format(
                text=chunk, start_year=start_year, end_year=end_year,
            ))
            for fmt, tmpl in _PROMPT_BY_FORMAT.items()
        ]

    def is_plaintext_format(self, fmt):
        """All Gen G formats return plain text, not JSON."""
        return fmt in _PROMPT_BY_FORMAT

    def parse_response(self, response):
        """Wrap the plain-text response as a single item dict.

        For Gen G, response is always plain text (not JSON). The base class
        pre-wraps it as {"text": ...} when is_plaintext_format returns True,
        so here we just pass through.
        """
        if isinstance(response, str):
            return [{"text": response.strip()}]
        if isinstance(response, dict) and "text" in response:
            return [{"text": (response.get("text") or "").strip()}]
        return []

    def format_conversation(self, item, fmt, source_chunk=None):
        """Build the assistant conversation for one rephrased passage.

        user:      "Rewrite the following as a [tutorial/faq/narrative/explanation]:"
        assistant: <the rephrased text>
        """
        text = (item.get("text") or "").strip()
        if not text:
            return None

        # Length sanity check: reject too-short (model refused) and too-long (runaway).
        if len(text) < 150 or len(text) > 8000:
            return None

        # Reject verbatim source copying (first 200 chars appear in source unchanged).
        if source_chunk and len(text) >= 200 and text[:200] in source_chunk:
            return None

        instruction = _USER_INSTRUCTION_BY_FORMAT.get(fmt)
        if instruction is None:
            return None

        return [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": text},
        ]
