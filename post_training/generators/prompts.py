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
4. IMPORTANT — ADAPT TO CONTENT: If the text discusses science, engineering, medicine, or natural phenomena, create questions that test understanding of the UNDERLYING PRINCIPLE — like a grade-school science exam. Do NOT ask about what the author said or argued. Instead, ask about the science itself.
   BAD (asks about the author): "What did Wilder argue about abnormal organisms?"
   GOOD (asks about the science): "What biological process can cause two embryos to fuse into conjoined twins?"
   BAD: "How did the report describe electricity generation?"
   GOOD: "What form of energy does a coal-fired power plant convert into electricity?"
   BAD: "What did the study conclude about soil erosion?"
   GOOD: "Which natural process causes topsoil to be carried away by water?"
   When the text is about politics, law, or history, ask analytical cause-effect reasoning questions as usual.
5. For each pair, provide 3 plausible but INCORRECT alternative answers as "distractors"
6. CRITICAL — Keep answers SHORT: The correct answer and ALL distractors must be concise phrases (2-8 words each, like "a lightning strike" or "increased tariff revenue"). Do NOT write full sentences. Do NOT make the correct answer longer or more detailed than the distractors.
7. Each question must include specific context so it is fully answerable on its own. BAD: "What happened during the battle?" GOOD: "What role did Commodore Tattnall play at the Battle of Taku Forts in 1859?"
8. Do NOT use phrases like "according to the text", "the passage states", "mentioned above", "during the period described". Include all necessary context in the question itself.
9. CRITICAL — TEMPORAL CONSTRAINT: All questions and answers must be grounded ONLY in knowledge available during the {start_year}-{end_year} period. Do NOT introduce any facts, events, outcomes, terminology, or references from after {end_year}.
10. IMPORTANT — Each question must focus on a DIFFERENT topic, fact, or aspect of the text. Do NOT ask overlapping or rephrased versions of the same question.
11. For each pair, provide detailed step-by-step reasoning (3-5 sentences) explaining WHY the answer is correct. Show the thought process: what the question asks, what evidence supports the answer, and why alternatives are wrong.
12. Return a JSON object with key "qa_pairs" containing an array:

{{"qa_pairs": [{{"question": "Question 1?", "answer": "Answer 1.", "reasoning": "First, the question asks about X. The text states Y, which directly supports this answer. Alternative interpretations such as Z are incorrect because...", "distractors": ["Wrong answer A.", "Wrong answer B.", "Wrong answer C."]}}]}}

Text:
{text}"""


# ---------------------------------------------------------------------------
# Generator B: Physical Commonsense (PIQA-style)
# Formats: MC-2, CoT
# ---------------------------------------------------------------------------

PIQA_PROMPT = """You are given text from a historical document published between {start_year} and {end_year}. Use this text ONLY as inspiration for the time period and setting. Your task is to create {num_items} physical commonsense questions in the PIQA style.

GOAL: Test whether a reader understands how physical tasks actually work — which tool, action, sequence, or context is physically sensible, and which is absurd or inverted.

FORMAT of each item — you MUST use a different goal style for each of the {num_items} items. Cycle through these four styles in order (item 1 = style i, item 2 = style ii, item 3 = style iii, item 4 = style iv, then repeat if more items):
  (i) BARE OBJECT LABEL (1-3 words only, no verb): "mop", "washcloth", "cast iron skillet", "wool blanket", "kerosene lamp", "clothes line", "butter churn". The choices are then physical properties or uses of that object, e.g. "has a long wooden handle" vs "has a sharp cutting blade". This style is MANDATORY — do not skip it.
  (ii) Imperative: "Prevent spiders from entering house." / "Clean a cast iron skillet."
  (iii) "How to" / "how do you" phrase: "How to sharpen a kitchen knife." / "how do you light an oil lamp?"
  (iv) Sentence stem to complete: "To keep milk from spoiling in summer," — choices complete the sentence.
- Exactly TWO candidate methods:
  * solution: a physically sensible method (or, for style (i), a correct property of the object)
  * distractor: structurally similar but physically WRONG
- Step-by-step reasoning (3-5 short steps) explaining why the solution works and why the distractor fails

CRITICAL — WITHIN-CALL DIVERSITY:
Each of the {num_items} items in this call MUST be about a DIFFERENT object, material, or physical task. Do NOT produce two items about the same object (e.g., do not use "cast iron skillet" or "butter churn" in more than one item per call). Do NOT produce two items about the same activity (e.g., do not have two items about "preserving fruit"). If the source text focuses on one topic, draw on related-but-distinct objects and tasks from the same era.

CRITICAL — HOW TO MAKE THE DISTRACTOR WRONG:
Use ONE of these subtle-error patterns. Rotate across items — do not use the same pattern twice in a row.
  (a) Wrong tool/item swap: replace the correct tool with one that resembles it in name but cannot do the job. Ex: "strong cord" → "phone cord", "salad spinner" → "fidget spinner", "knife" → "axe".
  (b) Inverted action: flip the key verb. Ex: "seal cracks" → "open cracks", "wet sponge" → "dry sponge", "bring to a boil" → "let cool".
  (c) Wrong location/container: swap where the action happens. Ex: "put in the freezer" → "put in the sun", "square baking dish" → "round colander", "around the bar" → "around the wheels".
  (d) Missing critical step: omit a time, temperature, or sequence that is physically required. Ex: "let dry 24 hours" → "let dry" (no time), "boil 20 minutes then drain" → "boil 20 minutes then enjoy".
  (e) Nonsensical material: replace the right material with one that cannot achieve the purpose. Ex: "hardware cloth over opening" → "silken cloth over opening", "paste or glue" → "water".

CRITICAL — LENGTH MATCHING:
The correct solution and the distractor MUST be the same length (similar word count, same number of sentences). Do NOT make the correct answer longer or more detailed. Often the right and wrong candidates should differ by only one or two substituted words.

CRITICAL — TEMPORAL CONSTRAINT:
Use only tools, materials, and methods available during {start_year}-{end_year}. Use period-appropriate items: hand tools, wood stoves, iceboxes, sewing needles, horse-drawn implements, oil lamps, coal, kerosene, etc. Do NOT use microwaves, plastic, electric refrigerators (unless after ~1930), televisions, computers, or anything post-{end_year}.

Topics (pick a DIFFERENT topic for each item):
- Cooking, baking, food preservation (canning, smoking, salting, drying)
- Cleaning, laundry, household repair
- Farming, gardening, animal care
- Woodworking, metalwork, building, tool use
- Sewing, knitting, leatherwork
- First aid, home remedies
- Lighting, heating, fuel handling
- Packaging, storage, transport

Return a JSON object with key "piqa_items". The FIRST item MUST use style (i) — a bare object label with no verb. Example output with 2 items showing style (i) then style (ii):

{{"piqa_items": [{{"goal": "wool blanket", "solution": "can keep a person warm on a cold night.", "distractor": "can cool a person down on a hot night.", "reasoning": "Step 1: Wool fibers trap air, which is a poor conductor of heat.\\nStep 2: Trapped warm air reduces heat loss from the body.\\nStep 3: A blanket cannot actively cool a person — it only insulates, so the distractor is backwards."}}, {{"goal": "Prevent spiders from entering the house.", "solution": "Seal up any wall cracks with putty.", "distractor": "Open up any wall cracks with a chisel.", "reasoning": "Step 1: Spiders enter through small openings in walls.\\nStep 2: Sealing cracks physically blocks the path they use.\\nStep 3: Opening cracks would create MORE entry points, so the distractor achieves the opposite of the goal."}}]}}

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

PASSAGE STYLE — choose ONE of these styles based on what the source text best supports:
  (i)   Formal essay / article (general default — explanatory prose about a topic)
  (ii)  Narrative or personal account (first-person or third-person story involving named people, places, and events — e.g., a memoir excerpt, a news human-interest story, or a court testimony narrative)
  (iii) Letter or correspondence (if the source contains a letter, preserve the letter format with greeting and sign-off, e.g., "Dear Mr. Smith, ... Sincerely, John")
  (iv)  Dialogue or exchange (if the source contains dialogue or a reported conversation, present it as back-and-forth speech between named speakers)

If the source text is formal exposition, use style (i). If it contains people doing specific things, use style (ii). If it contains letters, use style (iii). If it contains conversation, use style (iv). Do NOT force a style that does not fit the source.

- Write the passage as direct prose about the subject matter — do NOT begin with meta-references like "The text discusses", "The passage describes", "This document outlines", "This passage covers", etc. Start directly with the content.

PART 2 — Create {num_items} multiple-choice questions about your passage:

QUESTION TYPE MIX — MANDATORY RULES (you will be graded on compliance):

Types:
  (a) BEST TITLE / MAIN IDEA — "Which of the following is the best title for the passage?" or "What is the main idea of the passage?"
  (b) DETAIL RECALL — a specific fact from the passage: a name, number, date, place, order of events, object, quantity. Example: "Where did X happen?" "How many Y were there?" "Who did Z?"
  (c) CHARACTER MOTIVATION or FEELING (only when the passage has people in it) — "Why did X do Y?" "How does X feel about Y?" "What kind of person is X?" "What is X's attitude toward Y?" "What is the most likely reason X did Z?"
  (d) INFERENCE or IMPLICATION — something the passage implies but does not state directly. "What can be inferred about X?" "The author suggests that..." "Which statement is most likely true based on the passage?"
  (e) VOCABULARY IN CONTEXT (when the passage has a notable phrase) — "What does [phrase] most likely mean in the passage?" "The word '[X]' as used in the passage most nearly means..."

MANDATORY DISTRIBUTION for {num_items} questions per call:
- If {num_items} == 2: exactly ONE question must be type (a) OR (b) (easier/surface level), and exactly ONE question must be type (c), (d), OR (e) (harder/deeper level). Never produce two type-(a) questions, never two type-(b) questions.
- If {num_items} >= 3: at least one type (a) title/main idea, at least one type (b) detail recall, and the rest MUST be from (c)/(d)/(e).

When the passage has PEOPLE doing things, having feelings, or making decisions (styles ii, iii, iv above), you MUST include at least one type (c) character-motivation question. Do not default to detail recall when the passage has narrative/human content.

When the passage contains a notable phrase, idiom, or technical term, prefer a type (e) vocabulary-in-context question over another detail-recall question.

Rules:
1. Each question must have exactly 4 answer choices labeled A, B, C, D
2. Exactly one choice must be correct
3. Wrong choices should be plausible, clearly incorrect based on the passage, and SIMILAR IN LENGTH to the correct choice (critical — do not make the correct answer longer or more elaborate)
4. For detail-recall questions (type b), distractors should be PLAUSIBLE WRONG FACTS of the same kind: if the correct answer is a number, distractors are other numbers; if the answer is a place, distractors are other places
5. For title questions (type a), distractors should be titles that partially fit but miss the main theme
6. For each question, provide step-by-step reasoning (3-5 sentences) explaining why the correct answer is right and how it is supported by the passage.
7. For each question, include a `question_type` field indicating which type above (a/b/c/d/e) it covers.

Return a JSON object. Each item must include a `question_type` field (one of a, b, c, d, e per the types above):
{{"questions": [{{"passage": "Your clean 150-300 word passage here.", "question": "What does the passage suggest about...?", "choices": {{"A": "Choice 1", "B": "Choice 2", "C": "Choice 3", "D": "Choice 4"}}, "correct": "B", "question_type": "d", "reasoning": "The passage states X in the second paragraph, which directly supports choice B. Choice A contradicts the passage because..."}}]}}

IMPORTANT: The "passage" field must be IDENTICAL across all items — write it once and repeat it in each item.

Source text:
{text}"""


# ---------------------------------------------------------------------------
# Generator D: Quantitative (from text with numbers)
# Formats: MC-4, Open-ended, CoT
# ---------------------------------------------------------------------------

QUANTITATIVE_PROMPT = """You are given text from a historical document published between {start_year} and {end_year}. Create {num_items} math word problems inspired by the numerical data in this text.

Requirements:
1. Extract real numbers, dates, percentages, or quantities from the text
2. CRITICAL — CHAINED REASONING: Each problem must require 2-4 computation steps where each step USES THE RESULT of the previous step. Do NOT create single-step problems (like just computing one percentage). The final answer must depend on intermediate results that themselves had to be computed.
   GOOD example (3 steps, chained): "A merchant buys 20 crates of goods at $8 each. He sells 15 crates at $12 each and the remaining 5 crates at $6 each. How much profit did he make?" → Step 1: Total cost = 20 × $8 = $160. Step 2: Total revenue = (15 × $12) + (5 × $6) = $180 + $30 = $210. Step 3: Profit = $210 - $160 = $50. (Each step builds on previous values.)
   BAD example (not chained): "What is 30% of 200?" → Only one step, no chaining.
3. Each problem should have step-by-step reasoning showing 2-4 computation steps. Vary the number of steps across problems.
4. Problems should be grounded in the historical context of the text
5. Questions must be SELF-CONTAINED word problems — include all necessary numbers and context in the question itself. Do NOT reference "the text" or "the passage".
6. CRITICAL — TEMPORAL CONSTRAINT: All problems must use only facts and numbers from the {start_year}-{end_year} period. Do NOT reference events or data from after {end_year}.
7. IMPORTANT — Each problem must use DIFFERENT numbers or calculations from the text. Vary the math type: buy/sell profit, percentage then subtract, unit rate then scale then compare, discount then tax, total then split then remainder, etc.
8. For each problem, provide 3 plausible but INCORRECT final answers as "distractors"
9. Distractors must be the same format as the correct answer (numbers for numbers, percentages for percentages). Make distractors plausible by using COMMON ARITHMETIC MISTAKES — e.g., forgetting to subtract, multiplying instead of dividing, off-by-one errors, using the wrong base, computing a partial step as the final answer.
10. CRITICAL — All answers (correct AND distractors) must be CLEAN INTEGERS or simple fractions. Do NOT use decimal numbers like 99.9984 or 141.4295. Round to the nearest whole number if needed.

Return a JSON object. IMPORTANT: the "answer" and "distractors" must be BARE NUMBERS ONLY (e.g. "360", "25%", "$900"). Do NOT wrap them in sentences.

{{"problems": [{{"question": "A merchant in 1925 buys 20 crates of goods at $8 each. He sells 15 crates at $12 each and the remaining 5 crates at $6 each. How much total profit did he make?", "reasoning": "Step 1: Calculate total cost: 20 × $8 = $160\\nStep 2: Calculate total revenue: (15 × $12) + (5 × $6) = $180 + $30 = $210\\nStep 3: Calculate profit: $210 - $160 = $50", "answer": "50", "distractors": ["210", "160", "30"]}}]}}

Text:
{text}"""


# ---------------------------------------------------------------------------
# Generator E: Historical Completion (MC)
# Formats: MC-4
# Already produces 4 choices — no prompt change needed
# ---------------------------------------------------------------------------

COMPLETION_PROMPT = """You are given text from a historical document published between {start_year} and {end_year}. Use this text ONLY as inspiration for the time period and setting. Your task is to create {num_items} sentence completion questions about EVERYDAY PHYSICAL ACTIVITIES that people did during this era.

The goal is to test SITUATIONAL COMPREHENSION — whether the reader can predict what a person does next in a real-world scene, based on common sense about how physical activities work.

CRITICAL — You must create TWO types of questions (mix both types):

TYPE 1 — SCENE DESCRIPTIONS: Describe someone doing a physical activity, then ask what they do next.
Example: "A farmer hitches his horse to the plow and begins working the field. After two hours, the horse slows and the farmer stops to let it rest."
A: "He unhitches the horse and leads it to the water trough, then sits under a tree to eat his lunch." ← correct
B: "He decides to ride the horse into town to attend a political rally at the courthouse."
C: "He begins repainting the barn, carefully mixing the red paint in a large bucket."
D: "He takes out a notebook and starts writing a letter to the newspaper editor."

TYPE 2 — HOW-TO INSTRUCTIONS: Write step-by-step instructions for a practical task, then ask what the next step is.
Example: "How to preserve vegetables for winter. Step: Wash each canning jar thoroughly with hot water and soap. Inspect the rim for any cracks or chips that could prevent a proper seal."
A: "Place the jars in a pot of boiling water for ten minutes to sterilize them completely." ← correct
B: "Fill the jars with fresh flowers picked from the garden to decorate the kitchen."
C: "Stack the jars in the cellar without drying them and cover with a blanket."
D: "Use the jars to store buttons, coins, and other small household items."

Topics for both types:
- Cooking, baking, canning, food preservation
- Farming, planting, harvesting, animal care
- Building, repairing, woodworking, metalwork
- Cleaning, washing, sewing, household tasks
- Shopping, trading, selling at a market
- Traveling by train, horse, car, or ship
- Manufacturing, factory work, machine operation
- Sports, games, physical exercise
- Medical care, first aid, treating injuries

Requirements:
1. Create a SCENE showing someone doing a specific physical activity in 2-3 sentences. Then provide 4 possible continuations of what happens next.
2. One completion must be the NATURAL NEXT STEP in the activity. The other three must describe unrelated actions.
3. CRITICAL — LENGTH MATCHING: All four completions must be the SAME length (similar word count and sentence count). If the correct answer is 2 sentences, all wrong answers must also be 2 sentences. Do NOT make the correct answer longer or more detailed than the distractors.
4. Wrong completions MUST shift to a completely DIFFERENT activity (not a variation of the same one).
5. The scene must be SELF-CONTAINED — make sense on its own.
6. CRITICAL — TEMPORAL CONSTRAINT: Use only tools, methods, and technology available during {start_year}-{end_year}. No computers, no plastic, no television, no microwave ovens. Use period-appropriate items: wood stoves, hand tools, horse-drawn carts, typewriters, telegraph, radio (if after 1920s).
7. Each question must describe a DIFFERENT everyday activity.

For each completion, provide step-by-step reasoning explaining why the correct completion is the natural next step and why the others don't fit.

Return a JSON object:
{{"completions": [{{"context": "The beginning of the sentence or passage...", "choices": {{"A": "completion 1 (2-3 sentences)", "B": "completion 2 (2-3 sentences)", "C": "completion 3 (2-3 sentences)", "D": "completion 4 (2-3 sentences)"}}, "correct": "C", "reasoning": "The context discusses X, so the correct completion continues this topic. Completion A shifts to an unrelated subject, while completion C maintains the narrative thread about X."}}]}}

Text:
{text}"""


# ---------------------------------------------------------------------------
# Generator F: Pronoun Resolution (Winogrande-style)
# Formats: MC-2, CoT
# ---------------------------------------------------------------------------

WINOGRANDE_PROMPT = """You are given text from a historical document published between {start_year} and {end_year}. Use this text ONLY as inspiration for the time period, entities, and register. Your task is to create {num_items} pronoun-resolution questions in the Winogrande style.

GOAL: Test whether a reader can resolve an ambiguous pronoun or blank in a single sentence using COMMONSENSE reasoning about the two named entities — not grammatical agreement or surface cues.

FORMAT of each item:
- ONE sentence (10-25 words) that names exactly TWO distinct entities and contains exactly ONE blank marked with a single underscore "_"
- Both entities must be CONCRETE actors or objects. Do NOT use abstract nouns (the law, the court, the case, justice, duty, honor, the verdict, the truth, tax laws, civil liberties) as one of the two fillers.
- Two candidate fillers: the TWO entities that are named in the sentence.

HARD RULE #1 — FILLERS MUST APPEAR IN THE SENTENCE:
Both option_a and option_b MUST be phrases that appear verbatim in the sentence (with or without the leading article "the" / "a"). If you want to use a name like "Elmer", the name must appear in the sentence; do NOT invent entity names from the source document that are not present in the sentence you wrote. Before emitting an item, check: can I highlight option_a inside the sentence? Can I highlight option_b inside the sentence? If the answer to either is no, REJECT and regenerate.

  GOOD: "Leon Breaux invited Freddie Delaune into his auto because _ wanted to enjoy a ride."
        Fillers: "Leon Breaux", "Freddie Delaune" — both appear in the sentence. ✓
  BAD:  "The mother relied on her son because _ contributed to her expenses."
        Fillers: "Elmer", "Freddie" — NEITHER appears in the sentence. ✗

HARD RULE #2 — GENUINE AMBIGUITY (BOTH READINGS SURFACE-PLAUSIBLE):
Both option_a and option_b must be plausible candidates for the blank on pure surface reading. A reader who knows ONLY the words of the sentence (no background commonsense) should be UNABLE to decide which filler is correct. Only commonsense about roles, physical properties, motivations, causality, or ownership should break the tie.

  REJECT these patterns (one-way surface inference):
  - "The DA argued against X because _ was violating the law." — By definition, DAs prosecute violators; the distractor is impossible by role, not ambiguous.
  - "The lion chased the rabbit because _ was a predator." — "Predator" is a definitional property of a lion; distractor is impossible by category.
  - "The judge ruled against the defendant because _ had broken the law." — Judges rule ON behavior; defendants are accused of breaking laws. One-way role.
  - "The doctor treated the patient because _ was ill." — "Patient" means ill by definition.

  ACCEPT these patterns (genuinely ambiguous on surface, commonsense breaks the tie):
  - "The trophy did not fit in the suitcase because _ was too big." — Trophy OR suitcase could plausibly be "too big"; commonsense about containers tells us the trophy is the one exceeding the suitcase's size.
  - "The farmer sold his corn to the miller because _ needed to grind it into flour." — Either could "need to grind"; commonsense about the miller's trade tells us it's the miller.
  - "The iron anvil would not fit on the wooden workbench because _ was too heavy." — Either could be "too heavy"; commonsense about iron vs wood tells us it's the anvil.

MANDATORY VERIFICATION STEP — before finalizing each item, you MUST fill two fields:
  `fillers_in_sentence`: "option_a appears in sentence: YES/NO. option_b appears in sentence: YES/NO."
  `verification`: two lines
    "If option_a fills the blank: <one-sentence judgment: does this reading make surface sense? does it make commonsense sense?>"
    "If option_b fills the blank: <one-sentence judgment: does this reading make surface sense? does it make commonsense sense?>"
Then set `correct` to the letter whose reading is commonsense-correct.
REJECT and regenerate if:
  - either filler's "appears in sentence" answer is NO
  - one reading is already implausible on pure surface reading (violation of HARD RULE #2)
  - both readings seem equally commonsense-valid (no clear answer)

GOOD example (commonsense anchor = professional role):
  Sentence: "The farmer sold his corn to the miller because _ needed to grind it into flour."
  Option A: "the farmer"   Option B: "the miller"   Correct: B
  Reasoning: Grinding grain is the miller's trade; the farmer grows grain but does not operate a mill.

GOOD example (commonsense anchor = physical property):
  Sentence: "The iron anvil would not fit on the wooden workbench because _ was too heavy."
  Option A: "the anvil"   Option B: "the workbench"   Correct: A
  Reasoning: Iron is denser than wood; the anvil is the heavy object that exceeds the bench's capacity.

BAD examples (DO NOT generate these patterns):
  - Gendered pronoun that gives it away: "Mary asked John because she needed help."
  - Grammatically impossible alternative: one filler is plural/singular mismatch with the verb
  - Obvious by surface meaning alone: "The lion chased the rabbit because _ was a predator."

CRITICAL — COMMONSENSE ANCHOR:
Rotate across these anchor types — use a DIFFERENT one for each of the {num_items} items:
  (a) Professional role / expertise: which entity performs the named trade/action
  (b) Physical property (comparative): size, weight, hardness, speed, temperature
  (c) Motivation / intention: which entity has the reason to act
  (d) Causality: which entity produced the state or outcome
  (e) Ownership / possession: which entity owns or controls the object in question

CRITICAL — TEMPORAL CONSTRAINT:
Use only entities, roles, and objects from the {start_year}-{end_year} period. Period-appropriate entities: farmer, miller, blacksmith, merchant, sailor, judge, clerk, doctor, soldier, weaver, tailor, butcher, printer, carpenter, stonemason, apothecary, telegraph operator, newspaper editor, coachman, stable hand, factory worker, factory owner, clergyman, schoolmaster, fisherman, innkeeper. No post-{end_year} roles (no programmer, pilot before 1910, televisions, computers).

CRITICAL — DIVERSITY WITHIN CALL:
Each of the {num_items} items MUST use a DIFFERENT entity pair and a DIFFERENT commonsense anchor. Do NOT reuse "farmer/miller" or the same role pairing in more than one item. Vary whether the correct answer is A or B — do not put the correct answer in the same slot for every item.

CRITICAL — FORMATTING:
- The sentence must contain exactly one "_" (a single underscore).
- Each option_a and option_b must match a phrase that actually appears in the sentence (with or without the leading article "the").
- Do NOT use personal pronouns (he, she, it, they) as the blank — use "_".
- Keep the sentence natural, period-register prose (not stilted).

Return a JSON object with key "winogrande_items". The first item demonstrates anchor (a); cycle through (a)→(b)→(c)→(d)→(e) across items:

{{"winogrande_items": [{{"sentence": "The farmer sold his corn to the miller because _ needed to grind it into flour.", "option_a": "the farmer", "option_b": "the miller", "fillers_in_sentence": "option_a appears in sentence: YES. option_b appears in sentence: YES.", "verification": "If the farmer fills the blank: surface sense YES (farmer is an animate actor who could 'need' something); commonsense sense NO (farmers grow grain, they do not mill it). If the miller fills the blank: surface sense YES; commonsense sense YES (millers run mills that grind grain).", "correct": "B", "anchor": "professional_role", "reasoning": "Step 1: The sentence describes a transaction: farmer sells corn, miller receives it.\\nStep 2: The blank refers to whoever needs to grind corn into flour.\\nStep 3: Grinding grain into flour is the defining work of a miller.\\nStep 4: Farmers grow grain but rarely mill it themselves.\\nStep 5: Therefore the miller is the one who needed to grind the corn."}}, {{"sentence": "The iron anvil would not fit on the wooden workbench because _ was too heavy.", "option_a": "the anvil", "option_b": "the workbench", "fillers_in_sentence": "option_a appears in sentence: YES. option_b appears in sentence: YES.", "verification": "If the anvil fills the blank: surface sense YES (both are objects that can have weight); commonsense sense YES (iron anvils are dense and heavy). If the workbench fills the blank: surface sense YES; commonsense sense NO (a wooden workbench is lighter than an iron anvil).", "correct": "A", "anchor": "physical_property", "reasoning": "Step 1: The sentence compares an iron anvil and a wooden workbench.\\nStep 2: The blank refers to whichever object's weight causes the fit problem.\\nStep 3: Iron is far denser than wood; a blacksmith's anvil typically weighs over 100 pounds.\\nStep 4: A workbench is designed to support tools; its weight is not the limiting factor.\\nStep 5: Therefore the anvil is the heavy object."}}]}}

Text:
{text}"""


# ---------------------------------------------------------------------------
# Deprecated: Instruction Following prompt (Gen F prior to 2026-04-15)
# Retained here only for reference; no longer wired into any generator.
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
