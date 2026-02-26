"""
Prompt templates for all 8 synthetic data generators (A-H).

Generators A+B: extracted from run_direct.py
Generators C-H: new, designed per 03_SYNTHETIC_DATA_GENERATORS.md

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
4. For each pair, also provide 3 plausible but INCORRECT alternative answers as "distractors"
5. Distractors should be concise (similar length to the correct answer) and sound plausible
6. Return a JSON object with key "qa_pairs" containing an array:

{{"qa_pairs": [{{"question": "Question 1?", "answer": "Answer 1.", "distractors": ["Wrong answer A.", "Wrong answer B.", "Wrong answer C."]}}]}}

Text:
{text}"""


# ---------------------------------------------------------------------------
# Generator B: Chain-of-Thought
# Formats: MC-4, Open-ended, CoT
# ---------------------------------------------------------------------------

COT_PROMPT = """Create {num_items} complex reasoning examples from this text that demonstrate chain-of-thought thinking.

Each example should have:
1. A challenging question that requires step-by-step reasoning
2. Detailed reasoning steps that break down the problem
3. A concise final answer
4. 3 plausible but INCORRECT alternative final answers as "distractors"

Return a JSON object with key "cot_examples" containing an array:

{{"cot_examples": [{{"question": "Complex question?", "reasoning": "Step 1: First, I need to consider...\\nStep 2: Then, I analyze...\\nStep 3: Finally, I can conclude...", "answer": "Final answer based on the reasoning.", "distractors": ["Plausible wrong answer 1.", "Plausible wrong answer 2.", "Plausible wrong answer 3."]}}]}}

Text:
{text}"""


# ---------------------------------------------------------------------------
# Generator C: Reading Comprehension (MC)
# Formats: MC-4, MC-4+Passage, MC-2+Passage
# Already produces 4 choices — no prompt change needed
# ---------------------------------------------------------------------------

COMPREHENSION_PROMPT = """Read the following passage and create {num_items} reading comprehension multiple-choice questions.

Requirements:
1. Each question should test understanding of the passage (main idea, inference, vocabulary in context, supporting detail)
2. Each question must have exactly 4 answer choices labeled A, B, C, D
3. Exactly one choice must be correct
4. Wrong choices should be plausible but clearly incorrect based on the passage
5. Include a mix of difficulty levels

Return a JSON object:
{{"questions": [{{"question": "What does the passage suggest about...?", "choices": {{"A": "Choice 1", "B": "Choice 2", "C": "Choice 3", "D": "Choice 4"}}, "correct": "B"}}]}}

Passage:
{text}"""


# ---------------------------------------------------------------------------
# Generator D: Temporal Reasoning (metadata-based, no corpus)
# Formats: MC-4, Open-ended
# Already produces 4 choices — no prompt change needed
# ---------------------------------------------------------------------------

TEMPORAL_PROMPT = """Generate {num_items} temporal reasoning questions about historical events from the period {start_year}-{end_year}.

Requirements:
1. Questions should test temporal reasoning: ordering events, understanding cause-effect over time, comparing periods
2. All referenced events must have occurred WITHIN or BEFORE {end_year} (never after)
3. Each question must have exactly 4 answer choices labeled A, B, C, D
4. Exactly one choice must be correct
5. Include the domain and approximate year(s) being tested
6. Vary across domains: politics, science, technology, culture, economics
7. This is batch {batch_num} -- cover diverse sub-topics, avoid repetition

Return a JSON object:
{{"questions": [{{"question": "Which event occurred first: X or Y?", "choices": {{"A": "X occurred first", "B": "Y occurred first", "C": "They occurred simultaneously", "D": "Neither occurred during this period"}}, "correct": "A", "domain": "politics", "year": 1945}}]}}"""


# ---------------------------------------------------------------------------
# Generator E: Quantitative (from text with numbers)
# Formats: Open-ended, CoT
# No distractors needed — only generative formats
# ---------------------------------------------------------------------------

QUANTITATIVE_PROMPT = """Create {num_items} math word problems inspired by the numerical data in this text.

Requirements:
1. Extract real numbers, dates, percentages, or quantities from the text
2. Create word problems that require mathematical reasoning (arithmetic, percentages, comparisons, rates)
3. Each problem should have step-by-step reasoning and a final numerical answer
4. Problems should be grounded in the historical context of the text

Return a JSON object:
{{"problems": [{{"question": "If production increased from X to Y between 1920 and 1930, what was the average annual increase?", "reasoning": "Step 1: Calculate total increase: Y - X = Z\\nStep 2: Divide by number of years: Z / 10 = W", "answer": "The average annual increase was W units."}}]}}

Text:
{text}"""


# ---------------------------------------------------------------------------
# Generator F: Sentence Completion (MC, HellaSwag-style)
# Formats: MC-4, MC-2
# Already produces 4 choices — no prompt change needed
# ---------------------------------------------------------------------------

COMPLETION_PROMPT = """Create {num_items} sentence completion questions from this text in HellaSwag style.

Requirements:
1. Take a sentence or short passage from the text and truncate it at a natural point
2. Create 4 possible completions (A, B, C, D): one correct (from the text) and three plausible but wrong
3. The correct completion should follow naturally from the text
4. Wrong completions should be grammatically correct but factually wrong or contextually inappropriate

Return a JSON object:
{{"completions": [{{"context": "The beginning of the sentence or passage...", "choices": {{"A": "completion 1", "B": "completion 2", "C": "completion 3", "D": "completion 4"}}, "correct": "C"}}]}}

Text:
{text}"""


# ---------------------------------------------------------------------------
# Generator G: Instruction Following
# Formats: Open-ended, MC-4+Passage
# ---------------------------------------------------------------------------

INSTRUCT_PROMPT = """Create {num_items} instruction-response pairs grounded in this text.

Requirements:
1. Instructions should be diverse: summarize, explain, compare, analyze, list, describe
2. Responses must be directly supported by the text content
3. Responses should be detailed and well-structured (2-4 paragraphs)
4. Do NOT include information beyond what the text provides
5. For each pair, also provide:
   - "short_answer": a concise 1-2 sentence summary of the response
   - "distractors": 3 plausible but INCORRECT short answer alternatives (1-2 sentences each)

Return a JSON object:
{{"tasks": [{{"instruction": "Summarize the key developments described in this passage.", "response": "The passage describes...", "short_answer": "The passage covers X and Y developments.", "distractors": ["The passage focuses on Z.", "The main topic is W.", "The text describes Q."]}}]}}

Text:
{text}"""


# ---------------------------------------------------------------------------
# Generator H: Anti-Hallucination (metadata-based, no corpus)
# Formats: MC-4, Open-ended
# ---------------------------------------------------------------------------

ANTIHALLUC_PROMPT = """Generate {num_items} examples where a historical AI assistant (with knowledge only up to {end_year}) correctly refuses to answer questions about events after {end_year}.

Requirements:
1. The user question must ask about a real event/technology/person that became notable AFTER {end_year}
2. The assistant must politely decline, stating its knowledge ends at {end_year}
3. Where possible, the assistant should offer relevant PRE-{end_year} context
4. Vary across domains: technology, politics, science, culture, sports, economics
5. This is batch {batch_num} -- cover diverse post-{end_year} topics
6. For each example, also provide 3 plausible but INCORRECT responses as "distractors" that hallucinate knowledge beyond {end_year} (these should sound authoritative but contain made-up facts)

Return a JSON object:
{{"examples": [{{"question": "What can you tell me about [post-period event]?", "response": "I don't have knowledge about events after {end_year}. However, based on what I know up to {end_year}, [relevant pre-period context].", "distractors": ["[Confident but fabricated answer 1]", "[Confident but fabricated answer 2]", "[Confident but fabricated answer 3]"], "domain": "technology", "event_year": 2005}}]}}"""
