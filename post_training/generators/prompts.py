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

QA_PROMPT = """You are given text from a historical document published between {start_year} and {end_year}. Create {num_items} question-answer pairs for LLM training.

Rules:
1. Questions must require analytical thinking, not just fact lookup
2. Answers must be directly supported by the text
3. Vary question types: cause-effect, comparison, analysis, inference, summary
4. For each pair, provide 3 plausible but INCORRECT alternative answers as "distractors"
5. CRITICAL — Length matching: The correct answer and ALL distractors must be similar length (1-2 sentences each). Do NOT make the correct answer longer or more detailed than the distractors.
6. Each question must include specific historical context — names, dates, places, or events — so it is fully answerable on its own. BAD: "What happened during the battle?" GOOD: "What role did Commodore Tattnall play at the Battle of Taku Forts in 1859?"
7. Do NOT use phrases like "according to the text", "the passage states", "mentioned above", "during the period described". Include all necessary context in the question itself.
8. CRITICAL — TEMPORAL CONSTRAINT: All questions and answers must be grounded ONLY in knowledge available during the {start_year}-{end_year} period. Do NOT introduce any facts, events, outcomes, terminology, or references from after {end_year}.
9. IMPORTANT — Each question must focus on a DIFFERENT topic, fact, or aspect of the text. Do NOT ask overlapping or rephrased versions of the same question.
10. For each pair, provide detailed step-by-step reasoning (3-5 sentences) explaining WHY the answer is correct. Show the thought process: what the question asks, what evidence supports the answer, and why alternatives are wrong.
11. Return a JSON object with key "qa_pairs" containing an array:

{{"qa_pairs": [{{"question": "Question 1?", "answer": "Answer 1.", "reasoning": "First, the question asks about X. The text states Y, which directly supports this answer. Alternative interpretations such as Z are incorrect because...", "distractors": ["Wrong answer A.", "Wrong answer B.", "Wrong answer C."]}}]}}

Text:
{text}"""


# ---------------------------------------------------------------------------
# Generator B: Chain-of-Thought
# Formats: MC-4, CoT
# ---------------------------------------------------------------------------

COT_PROMPT = """You are given text from a historical document published between {start_year} and {end_year}. Create {num_items} complex reasoning examples that demonstrate chain-of-thought thinking.

Each example should have:
1. A challenging question that requires step-by-step reasoning
2. Detailed reasoning steps (5-8 steps, each 1-2 sentences) that break down the problem thoroughly. Explore the question from multiple angles, consider why wrong answers might seem plausible, and build toward the conclusion logically.
3. A concise final answer
4. 3 plausible but INCORRECT alternative final answers as "distractors"
5. CRITICAL — Length matching: The correct answer and ALL distractors must be similar length (1-2 sentences). Do NOT make the correct answer longer or more detailed.
6. Questions must be SELF-CONTAINED — answerable without the source text. Do NOT reference "the text", "the passage", "the article", or "above". Include enough context in the question itself.
7. CRITICAL — TEMPORAL CONSTRAINT: All questions, reasoning, and answers must be grounded ONLY in knowledge available during the {start_year}-{end_year} period. Do NOT introduce any facts, events, outcomes, or references from after {end_year}.
8. IMPORTANT — Each question must address a DIFFERENT aspect, argument, or theme from the text. Avoid overlapping questions.

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
7. For each question, provide step-by-step reasoning (3-5 sentences) explaining why the correct answer is right and how it is supported by the passage.

Return a JSON object:
{{"questions": [{{"passage": "Your clean 150-300 word passage here.", "question": "What does the passage suggest about...?", "choices": {{"A": "Choice 1", "B": "Choice 2", "C": "Choice 3", "D": "Choice 4"}}, "correct": "B", "reasoning": "The passage states X in the second paragraph, which directly supports choice B. Choice A contradicts the passage because..."}}]}}

IMPORTANT: The "passage" field must be IDENTICAL across all items — write it once and repeat it in each item.

Source text:
{text}"""


# ---------------------------------------------------------------------------
# Generator D: Quantitative (from text with numbers)
# Formats: MC-4, Open-ended, CoT
# ---------------------------------------------------------------------------

QUANTITATIVE_PROMPT = """You are given text from a historical document published between {start_year} and {end_year}. Create {num_items} multi-step math word problems inspired by the numerical data in this text.

Requirements:
1. Extract real numbers, dates, percentages, or quantities from the text
2. Create word problems that require MULTI-STEP mathematical reasoning — each problem must require 3-5 chained calculations to solve. Do NOT create single-step problems (like just computing one percentage). Build problems where each step depends on the previous one.
3. Example of a good multi-step problem: "A factory produced X widgets in 1925 and Y in 1930. If each widget costs Z dollars, and the factory kept 40% of revenue as profit, how much total profit did the factory earn from the increase in production over this period?" (requires: compute increase, multiply by price, compute 40%)
4. Each problem must have detailed step-by-step reasoning showing 3-5 computation steps, where each step clearly states the operation and result.
5. Problems should be grounded in the historical context of the text
6. Questions must be SELF-CONTAINED word problems — include all necessary numbers and context in the question itself. Do NOT reference "the text" or "the passage".
7. CRITICAL — TEMPORAL CONSTRAINT: All problems must use only facts and numbers from the {start_year}-{end_year} period. Do NOT reference events or data from after {end_year}.
8. IMPORTANT — Each problem must use DIFFERENT numbers or calculations from the text. Vary the math type: percentage change, ratio, difference, rates, unit conversion, profit/loss, averages, etc.
9. For each problem, provide 3 plausible but INCORRECT final answers as "distractors"
10. Distractors must be the same format as the correct answer (numbers for numbers, percentages for percentages). Make distractors plausible by using COMMON ARITHMETIC MISTAKES — e.g., forgetting to subtract, multiplying instead of dividing, off-by-one errors, using the wrong base, computing a partial step as the final answer. Distractors should be the kind of wrong answers a student would get by making a single calculation error.
11. CRITICAL — All answers (correct AND distractors) must be CLEAN INTEGERS or simple fractions. Do NOT use decimal numbers like 99.9984 or 141.4295. Round to the nearest whole number if needed.

Return a JSON object:
{{"problems": [{{"question": "A factory produced 500 widgets in 1925 and 800 in 1930. If each widget sold for $3 and the factory kept 40% of revenue as profit, how much profit came from the additional production?", "reasoning": "Step 1: Calculate the increase in production: 800 - 500 = 300 widgets\\nStep 2: Calculate revenue from additional widgets: 300 * $3 = $900\\nStep 3: Calculate profit at 40%: $900 * 0.40 = $360", "answer": "$360", "distractors": ["$900", "$240", "$540"]}}]}}

Text:
{text}"""


# ---------------------------------------------------------------------------
# Generator E: Historical Completion (MC)
# Formats: MC-4
# Already produces 4 choices — no prompt change needed
# ---------------------------------------------------------------------------

COMPLETION_PROMPT = """You are given text from a historical document published between {start_year} and {end_year}. Create {num_items} sentence completion questions in HellaSwag style.

The goal is to test SITUATIONAL COMPREHENSION — whether the reader can identify which completion fits the current situation, without needing any external knowledge.

Requirements:
1. Take a sentence or short passage from the text and truncate it at a natural point
2. Create 4 possible completions (A, B, C, D): one correct (from the text) and three wrong
3. The correct completion should continue the SAME narrative thread as the stem
4. CRITICAL — LENGTH: Each completion (correct AND wrong) must be 2-3 sentences long (40-80 words). Short single-phrase completions are NOT acceptable. Include enough narrative detail that the reader must carefully evaluate each option.
5. Wrong completions MUST be:
   - Grammatically correct and the SAME length as the correct completion (2-3 sentences each)
   - SITUATIONALLY OFF — they shift to a different topic, subject, or type of action that does not fit the current situation. Example: if the stem discusses a prisoner exchange, wrong completions should be about unrelated matters (tariff negotiations, troop deployments, treaty provisions) — NOT alternative prisoner outcomes.
   - The test must be answerable by comprehension alone: "which completion continues what the sentence is actually about?" A reader should NOT need historical knowledge to eliminate wrong answers.
   - NEVER contradict the stem directly (if stem says "not satisfied", do NOT say "expressed satisfaction")
   - NEVER be vague (no "faced many challenges" or "took a different approach")
6. The context must be SELF-CONTAINED — it should make sense on its own without the source text. Do NOT reference "the text" or "the passage".
7. CRITICAL — TEMPORAL CONSTRAINT: All content must be grounded ONLY in knowledge available during the {start_year}-{end_year} period. Do NOT introduce any references from after {end_year}.
8. IMPORTANT — Each completion must start from a DIFFERENT sentence or section of the text. Do NOT create multiple completions from the same or adjacent sentences.

For each completion, provide step-by-step reasoning (3-5 sentences) explaining why the correct completion fits and why the others don't.

Return a JSON object:
{{"completions": [{{"context": "The beginning of the sentence or passage...", "choices": {{"A": "completion 1 (2-3 sentences)", "B": "completion 2 (2-3 sentences)", "C": "completion 3 (2-3 sentences)", "D": "completion 4 (2-3 sentences)"}}, "correct": "C", "reasoning": "The context discusses X, so the correct completion continues this topic. Completion A shifts to an unrelated subject, while completion C maintains the narrative thread about X."}}]}}

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
