"""
Prompt templates for synthetic data generators (A-F).

Generators A+B: extracted from run_direct.py
Generators C-F: new, designed per 03_SYNTHETIC_DATA_GENERATORS.md

Each prompt requests all fields needed for multi-format rendering:
- Generators needing MC format also request distractors
- This allows one API call to produce multiple format variants
"""

# ---------------------------------------------------------------------------
# Generator A: Factual QA
# Formats: MC-4, Open-ended
# ---------------------------------------------------------------------------

QA_PROMPT = """Create {num_items} question-answer pairs from this text for LLM training.

Rules:
1. Questions must require analytical thinking, not just fact lookup
2. Answers must be directly supported by the text
3. Vary question types: cause-effect, comparison, analysis, inference, summary
4. For each pair, provide 3 plausible but INCORRECT alternative answers as "distractors"
5. CRITICAL — Length matching: The correct answer and ALL distractors must be similar length (1-2 sentences each). Do NOT make the correct answer longer or more detailed than the distractors.
6. Each question must include specific historical context — names, dates, places, or events — so it is fully answerable on its own. BAD: "What happened during the battle?" GOOD: "What role did Commodore Tattnall play at the Battle of Taku Forts in 1859?"
7. Do NOT use phrases like "according to the text", "the passage states", "mentioned above", "during the period described". Include all necessary context in the question itself.
8. IMPORTANT — Each question must focus on a DIFFERENT topic, fact, or aspect of the text. Do NOT ask overlapping or rephrased versions of the same question.
9. Return a JSON object with key "qa_pairs" containing an array:

{{"qa_pairs": [{{"question": "Question 1?", "answer": "Answer 1.", "distractors": ["Wrong answer A.", "Wrong answer B.", "Wrong answer C."]}}]}}

Text:
{text}"""


# ---------------------------------------------------------------------------
# Generator B: Chain-of-Thought
# Formats: MC-4, CoT
# ---------------------------------------------------------------------------

COT_PROMPT = """Create {num_items} complex reasoning examples from this text that demonstrate chain-of-thought thinking.

Each example should have:
1. A challenging question that requires step-by-step reasoning
2. Detailed reasoning steps that break down the problem
3. A concise final answer
4. 3 plausible but INCORRECT alternative final answers as "distractors"
5. CRITICAL — Length matching: The correct answer and ALL distractors must be similar length (1-2 sentences). Do NOT make the correct answer longer or more detailed.
6. Questions must be SELF-CONTAINED — answerable without the source text. Do NOT reference "the text", "the passage", "the article", or "above". Include enough context in the question itself.
7. IMPORTANT — Each question must address a DIFFERENT aspect, argument, or theme from the text. Avoid overlapping questions.

Return a JSON object with key "cot_examples" containing an array:

{{"cot_examples": [{{"question": "Complex question?", "reasoning": "Step 1: First, I need to consider...\\nStep 2: Then, I analyze...\\nStep 3: Finally, I can conclude...", "answer": "Final answer based on the reasoning.", "distractors": ["Plausible wrong answer 1.", "Plausible wrong answer 2.", "Plausible wrong answer 3."]}}]}}

Text:
{text}"""


# ---------------------------------------------------------------------------
# Generator C: Reading Comprehension (MC)
# Formats: MC-4+Passage (GPT produces clean passage + MC questions)
# ---------------------------------------------------------------------------

COMPREHENSION_PROMPT = """You are given raw source text from a historical document published between {start_year} and {end_year}. The text may be poorly formatted (broken words, column artifacts, mid-sentence starts).

Your task has two parts:

PART 1 — Write a clean passage:
- Rewrite the source content into a well-structured, self-contained passage of 150-300 words
- The passage must start and end at natural sentence boundaries
- Faithfully represent the source content — do NOT invent facts, add modern context, or include any information beyond what is stated or clearly implied in the source text
- CRITICAL — TEMPORAL CONSTRAINT: Do NOT introduce any knowledge, events, terminology, or references from after {end_year}. The passage must read as if written during the {start_year}-{end_year} period.
- Fix OCR artifacts (broken words, garbled text) but preserve the original meaning
- Write in clear, coherent prose suitable for a reading comprehension exercise
- Write the passage as direct prose about the subject matter — do NOT begin with meta-references like "The text discusses", "The passage describes", "This document outlines", "This passage covers", etc. Start directly with the historical content.

PART 2 — Create {num_items} multiple-choice questions about your passage:
1. Each question should test understanding of the passage (main idea, inference, vocabulary in context, supporting detail)
2. Each question must have exactly 4 answer choices labeled A, B, C, D
3. Exactly one choice must be correct
4. Wrong choices should be plausible, clearly incorrect based on the passage, and similar in length to the correct choice
5. Include a mix of difficulty levels
6. Each question must test a DIFFERENT aspect of the passage — one about the main idea, one about a specific detail, one requiring inference, etc.

Return a JSON object:
{{"questions": [{{"passage": "Your clean 150-300 word passage here.", "question": "What does the passage suggest about...?", "choices": {{"A": "Choice 1", "B": "Choice 2", "C": "Choice 3", "D": "Choice 4"}}, "correct": "B"}}]}}

IMPORTANT: The "passage" field must be IDENTICAL across all items — write it once and repeat it in each item.

Source text:
{text}"""


# ---------------------------------------------------------------------------
# Generator D: Quantitative (from text with numbers)
# Formats: MC-4, Open-ended, CoT
# ---------------------------------------------------------------------------

QUANTITATIVE_PROMPT = """Create {num_items} math word problems inspired by the numerical data in this text.

Requirements:
1. Extract real numbers, dates, percentages, or quantities from the text
2. Create word problems that require mathematical reasoning (arithmetic, percentages, comparisons, rates, probability/combinatorics when applicable)
3. Each problem should have step-by-step reasoning and a final numerical answer
4. Problems should be grounded in the historical context of the text
5. Questions must be SELF-CONTAINED word problems — include all necessary numbers and context in the question itself. Do NOT reference "the text" or "the passage".
6. IMPORTANT — Each problem must use DIFFERENT numbers or calculations from the text. Vary the math type: percentage change, ratio, difference, probability, combinatorics, etc.
7. For each problem, provide 3 plausible but INCORRECT final answers as "distractors"
8. CRITICAL — Distractors must be the same format as the correct answer (e.g., if the answer is a number, distractors must be numbers). Make distractors plausible by using common arithmetic mistakes (off-by-one, wrong operation, rounding errors).

Return a JSON object:
{{"problems": [{{"question": "If production increased from X to Y between 1920 and 1930, what was the average annual increase?", "reasoning": "Step 1: Calculate total increase: Y - X = Z\\nStep 2: Divide by number of years: Z / 10 = W", "answer": "The average annual increase was W units.", "distractors": ["The average annual increase was V units.", "The average annual increase was U units.", "The average annual increase was T units."]}}]}}

Text:
{text}"""


# ---------------------------------------------------------------------------
# Generator E: Historical Completion (MC)
# Formats: MC-4
# Already produces 4 choices — no prompt change needed
# ---------------------------------------------------------------------------

COMPLETION_PROMPT = """Create {num_items} sentence completion questions from this text in HellaSwag style.

The goal is to test SITUATIONAL COMPREHENSION — whether the reader can identify which completion fits the current situation, without needing any external knowledge.

Requirements:
1. Take a sentence or short passage from the text and truncate it at a natural point
2. Create 4 possible completions (A, B, C, D): one correct (from the text) and three wrong
3. The correct completion should continue the SAME narrative thread as the stem
4. Wrong completions MUST be:
   - Grammatically correct and similar in length to the correct completion. CRITICAL: if the correct completion is one sentence, distractors must also be one sentence. Do NOT make the correct answer longer.
   - SITUATIONALLY OFF — they shift to a different topic, subject, or type of action that does not fit the current situation. Example: if the stem discusses a prisoner exchange, wrong completions should be about unrelated matters (tariff negotiations, troop deployments, treaty provisions) — NOT alternative prisoner outcomes.
   - The test must be answerable by comprehension alone: "which completion continues what the sentence is actually about?" A reader should NOT need historical knowledge to eliminate wrong answers.
   - NEVER contradict the stem directly (if stem says "not satisfied", do NOT say "expressed satisfaction")
   - NEVER be vague (no "faced many challenges" or "took a different approach")
5. The context must be SELF-CONTAINED — it should make sense on its own without the source text. Do NOT reference "the text" or "the passage".
6. IMPORTANT — Each completion must start from a DIFFERENT sentence or section of the text. Do NOT create multiple completions from the same or adjacent sentences.

Return a JSON object:
{{"completions": [{{"context": "The beginning of the sentence or passage...", "choices": {{"A": "completion 1", "B": "completion 2", "C": "completion 3", "D": "completion 4"}}, "correct": "C"}}]}}

Text:
{text}"""


# ---------------------------------------------------------------------------
# Generator F: Instruction Following
# Formats: MC-4+Passage (GPT produces clean passage + instruction pairs)
# ---------------------------------------------------------------------------

INSTRUCT_PROMPT = """You are given raw source text from a historical document published between {start_year} and {end_year}. The text may be poorly formatted (broken words, column artifacts, mid-sentence starts).

Your task has two parts:

PART 1 — Write a clean passage:
- Rewrite the source content into a well-structured, self-contained passage of 150-300 words
- The passage must start and end at natural sentence boundaries
- Faithfully represent the source content — do NOT invent facts, add modern context, or include any information beyond what is stated or clearly implied in the source text
- CRITICAL — TEMPORAL CONSTRAINT: Do NOT introduce any knowledge, events, terminology, or references from after {end_year}. The passage must read as if written during the {start_year}-{end_year} period.
- Fix OCR artifacts (broken words, garbled text) but preserve the original meaning
- Write in clear, coherent prose suitable for an instruction-following exercise
- Write the passage as direct prose about the subject matter — do NOT begin with meta-references like "The text discusses", "The passage describes", "This document outlines", "This passage covers", etc. Start directly with the historical content.

PART 2 — Create {num_items} instruction-response pairs about your passage:
1. Instructions should be diverse: summarize, explain, compare, analyze, list, describe
2. Responses must be directly supported by the passage content
3. Responses should be detailed and well-structured (2-4 paragraphs)
4. Do NOT include information beyond what the passage provides
5. For each pair, also provide:
   - "short_answer": a concise 1-2 sentence summary of the response
   - "distractors": 3 plausible but INCORRECT short answer alternatives
   - CRITICAL: Distractors must be similar in length and detail to the short_answer
6. Each instruction must use a DIFFERENT instruction type (e.g., one summarize, one analyze, one compare). Do NOT repeat the same type.

Return a JSON object:
{{"tasks": [{{"passage": "Your clean 150-300 word passage here.", "instruction": "Summarize the key developments described in this passage.", "response": "The passage describes...", "short_answer": "The passage covers X and Y developments.", "distractors": ["The passage focuses on Z.", "The main topic is W.", "The text describes Q."]}}]}}

IMPORTANT: The "passage" field must be IDENTICAL across all items — write it once and repeat it in each item.

Source text:
{text}"""
