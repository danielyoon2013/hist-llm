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
4. CRITICAL — TEST THE GENERAL PRINCIPLE, NOT THE SPECIFIC SOURCE:
   When the text discusses science, engineering, medicine, or natural phenomena, your questions MUST test understanding of the UNDERLYING SCIENTIFIC PRINCIPLE that any educated person could know — NOT the specific finding, person, location, or experimental detail mentioned in the source. Use the text only to identify the topic; then write a question about the general principle behind it. The reader of the question must NOT need to have read the source.

   ABSOLUTELY FORBIDDEN PATTERNS (REJECT these — they are NOT acceptable):
     - "What was the significance of [Person]'s work on X?" (asks about a researcher)
     - "What did [Person] discover/argue/conclude about X?" (asks about a researcher)
     - "According to [Source/Person/Report], how does X work?" (cites the source)
     - "What effect does X have on Y, as discussed in [Document]?" (cites the source)
     - "What conditions are necessary for X in [Specific Place]?" (over-specific to source)
     - "How does the design of [Specific Building/Harbor/Machine] affect Y?" (over-specific case study)
     - "What process is described in the text for X?" (refers to text)

   REQUIRED PATTERN — direct grade-school-style questions about the principle:
     SOURCE topic: Hofmeister's plant cell research → ASK: "What is the main function of the cell wall in plant cells?"
     SOURCE topic: harbour silting at Ostend → ASK: "What natural process causes sediment to build up where rivers meet the sea?"
     SOURCE topic: radium emanation experiments → ASK: "What type of particle does a radioactive element emit during decay?"
     SOURCE topic: gyroscope demonstration by Thomson → ASK: "What property of a spinning object resists changes to its axis of rotation?"
     SOURCE topic: alloying calcium → ASK: "What is the term for a mixture of two or more metals?"
     SOURCE topic: simple retina formation → ASK: "Which part of the eye contains light-sensitive cells?"
     SOURCE topic: littoral drift along sandy coasts → ASK: "What primarily moves sand along an ocean shoreline?"

   The questions should read like 5th-8th grade science exam questions. Test concepts a curious 12-year-old should know — gravity, photosynthesis, evaporation, magnetism, simple circuits, plant biology, weather, rocks/minerals, cell structure, simple chemistry, light and sound, physical states of matter.

   HARD REJECT — the question must test a SCIENTIFIC PRINCIPLE that exists in nature, not a historical, institutional, or methodological fact about science. REJECT and SKIP the question if it tests any of:
     - History of science (when a discovery was made, who founded a field)
     - Research methodology, publishing practices, peer review
     - Education or institutional changes (universities adding botany, funding increases, curriculum changes)
     - Engineering-specific case studies tied to a particular structure (this harbour, that dredging operation, this specific machine design) — unless it's testing a general physics principle that applies anywhere
     - Sociology of scientists (collaboration patterns, scientific community trends)

     BAD examples to REJECT:
       - "After the mid-19th century, botanical literature shifted... what is a likely consequence?" (history of publishing)
       - "A group of universities incorporates botany... what is the cause of this increase?" (education history)
       - "A mail-steamer enters Flushing Harbour... what design flaw affects navigation?" (specific engineering case)
       - "A dredging operation increases the depth of Portsmouth Harbour..." (specific engineering project)
       - "A botanist studies isolated research papers... what impact on the community?" (research methodology)

     GOOD examples to KEEP:
       - "A river flows into the ocean and sediment builds up where freshwater meets saltwater. What process causes this?" (general geology)
       - "A botanist observes plants exhibiting changes in characteristics over time. What process explains these variations?" (evolution principle)
       - "A chemist combines potassium permanganate with oxalic acid. What is the role of cerium salts?" (chemistry principle)
       - "A hot rock is dropped into cool water. Heat transfers by..." (physics principle)

   If the source text only supports non-science questions (e.g., the chunk is purely about institutional history or a specific engineering case), generate fewer items rather than producing bad ones.

   CRITICAL — USE SCENARIO FRAMING (not definition-style):
   ARC-style science questions describe a CONCRETE SITUATION and ask the reader to APPLY knowledge to predict, infer, or explain what happens. They do NOT ask "what is X" or "what is the role of X" in the abstract. Always set up a scenario first, then ask the question.

   GOOD scenario-based examples (this is what to produce):
     "A hot rock is dropped into a pail of cool water. Heat energy is transferred from the rock to the water by:" → conduction
     "A scientist categorized a rock as an extrusive igneous rock. Another scientist could accurately categorize the same rock as:" → volcanic
     "A particular peach tree produces peaches that are more resistant to disease than other peaches. What method would reproduce these exact peaches?" → asexual reproduction
     "A brand of fertilizer claims that it contains all the chemicals a plant needs for rapid growth. It may be inferred that the fertilizer includes all these ingredients except:" → carbohydrates
     "Scientists studying a group of young sage grouse discovered a large percentage were infected with blood parasites. How will these blood parasites most likely affect the young birds?" → by decreasing the birds' overall health
     "A student observes morning dew on grass but a dry sidewalk. Which process most likely caused the grass to be wet?" → condensation
     "A blacksmith heats an iron horseshoe in a furnace, then drops it into a barrel of water. The water bubbles violently. What state change is the water undergoing?" → liquid to gas
     "A farmer notices his crops grow taller on the south side of a hill than the north side. What environmental factor most likely accounts for this?" → more sunlight exposure

   BAD definition-style (REJECT these — too abstract, unlike ARC):
     "What is the main function of X?"
     "What is the role of Y in process Z?"
     "What process explains how plants do X?"
     "What is the primary mechanism behind X?"

   STRUCTURE: every question should have ONE of these scenario openers:
     - "A [person/object] [does/has/observes] X. [Question about what happens, why, or what to infer]"
     - "[An object/system] [is in some state]. [Question about cause, effect, or classification]"
     - "After [an event], [some observation]. [Question about explanation]"
     - "Given [a condition], which [property/process/outcome] is most likely?"

   When the source is about politics, law, history, or other non-science: ask normal analytical cause-effect questions as usual.
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
# Generator B: Physical Commonsense (sensible-method selection)
# Formats: MC-2, CoT
# ---------------------------------------------------------------------------

PHYSICAL_COMMONSENSE_PROMPT = """You are given text from a historical document published between {start_year} and {end_year}. Use this text ONLY as inspiration for the time period and setting. Your task is to create {num_items} physical commonsense questions.

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

CRITICAL — You must create TWO types of questions in roughly equal numbers (alternate between them):

TYPE 1 — SHORT SCENE WITH TRAILING PRONOUN: A 1-2 sentence caption-style scene that ends with a pronoun ("he", "she", "they") or noun mid-thought, expecting completion. The CONTEXT must end mid-clause to be completed.

Example structure (use the FORMAT, but pick a topic from the source text — DO NOT copy this topic):
  Context: "A young child sits on the parlor floor, building a tower of wooden blocks while her grandmother watches from the rocking chair. she"
    A: "carefully balances another block on top, her tongue between her teeth in concentration." ← correct
    B: "stands up and runs to the window to watch the streetcar pass."
    C: "asks her grandmother to read aloud from the latest issue of Harper's."
    D: "begins humming a song her mother taught her last week."

The context ENDS with a pronoun ("he", "she", "they") or noun ("the man", "the boy") with no period. The completions begin lowercase and continue the sentence. Keep the scene to 1-2 short sentences, NOT 3 elaborate ones. The TOPIC of the scene comes from the source text (not from this example).

TYPE 2 — WIKIHOW-STYLE INSTRUCTIONS: Use the literal wikiHow markup format. Begin with [header], then [title] for each step, then [step] with the body text, ending mid-step or with [substeps] markers expecting completion.

Example structure (use the FORMAT, but pick a topic from the source text — DO NOT copy this topic):
  Context: "[header] How to soothe a child with a fever [title] Cool the body. [step] Wet a clean cloth in cool (not cold) water and wring it out so it does not drip. Place it gently on the child's forehead."
    A: "[substeps] Replace the cloth every few minutes as it warms, and offer small sips of cool water if the child is thirsty enough to drink." ← correct
    B: "[substeps] Use the cloth to scrub the kitchen counters before they begin to stain from the morning meal."
    C: "[substeps] Hang the cloth on the line outdoors to dry in the sun before applying it again the next day."
    D: "[substeps] Wrap the child in heavy wool blankets and place them next to the wood stove for warmth."

The context MUST include `[header]`, `[title]`, `[step]` markers. The completions MUST continue the wikiHow format, often starting with `[substeps]` or another `[title]`. The TOPIC comes from the source text (not from this example).

SOURCE-DRIVEN TOPIC SELECTION (CRITICAL — do not hardcode topics):

Derive the topic of each question from the SOURCE TEXT you were given. Read what the source actually discusses — a person, an event, an object, a process, a place — and build the scene or how-to around that. The source is what makes the question period-grounded; do not invent generic scenes that ignore the source.

VARY ACROSS LIFE DOMAINS (do not cluster around a few stereotypes):

Across the {num_items} items in this call, AND across all calls, the questions should span DIFFERENT life domains. Examples of domains (use these as inspiration, NOT a topic checklist):
  - Body and health (illness, injury, hygiene, posture, physical sensation)
  - Relationships and social life (visiting, courting, conflict, helping, mourning)
  - Mental and emotional states (worry, hope, decision-making, learning)
  - Domestic life (any household activity — wide range)
  - Work and trade (any occupation present in the source)
  - Travel and movement (any mode of getting somewhere)
  - Leisure and play (games, music, reading, gathering)
  - Survival and care (food, warmth, shelter, child care)
  - Civic and public life (markets, courts, gatherings, ceremonies)
  - Skill and craft (any specialized practice)

ANTI-STEREOTYPE RULE — REJECT and REGENERATE if the question is about:
  - "A blacksmith" / "A farmer" / "A seamstress" / "A housewife baking" — these are OVER-USED stereotypes from prior generations. Pick something else.
  - Generic "kneading dough", "harvesting corn", "canning fruit", "preserving vegetables" — these dominate our existing data. AVOID them unless the source text specifically discusses them.
  - Default-to-common occupations of the period. If you find yourself writing "A farmer..." stop and ask: what else is in the source text? A schoolteacher? A telegraph clerk? A nurse? A child playing? A passenger on a streetcar? An audience member at a lecture? A mother soothing a sick infant? A clerk filing papers?

The goal is broad domain coverage: questions should feel as different from each other as "How to recover from an emotional affair", "How to prune apple trees", "How to apply cologne", and "A man bends over into a pond. he". Reach for that range.

Requirements:
1. Alternate between TYPE 1 (short scene with trailing pronoun) and TYPE 2 (wikiHow markup). Roughly half of items should use each type.
2. For TYPE 1: scene is 1-2 sentences max, ending mid-thought with a pronoun. Each completion is 1-2 sentences. NO elaborate 3-sentence scenes.
3. For TYPE 2: must include `[header]`, `[title]`, and `[step]` markers in the context. Completions must use `[substeps]` or additional `[title]` markers.

4. CRITICAL — DISTRACTORS MUST TEST NARRATIVE COHERENCE, NOT TOPIC RECOGNITION.

   This is the HARDEST requirement. All 4 completions must stay ON THE SAME TOPIC as the scene, but only ONE should be narratively consistent (correct cause-effect, correct physical sequence, correct timing). The other 3 must look plausible at first glance but BREAK narrative consistency in subtle ways:

   (a) WRONG SEQUENCE: do a step before its prerequisites
   (b) WRONG CAUSE-EFFECT: produce an illogical outcome from the action
   (c) WRONG TIMING: skip ahead or jump back inappropriately
   (d) CONTRADICTS SETUP: do something the scene already ruled out

   GOOD example of HARD distractors (all stew-topic, only ONE coherent):
   Scene: "A woman carefully stirs a large pot of bubbling stew over the wood stove. She"
     A: leans in to taste the stew, adjusting the seasoning as needed. ← CORRECT (natural sequence: stir → taste → adjust)
     B: pours the bubbling stew directly into the bread dough she has been kneading. ← wrong sequence (mixes incompatible processes)
     C: sets the pot in the cellar to cool before lighting the fire under it. ← wrong cause-effect (cools then heats)
     D: covers the pot with a wet cloth to put out the flame inside the broth. ← contradicts physics (broth doesn't burn)

   BAD example of EASY distractors (different activities — REJECT this style):
   Scene: "A woman stirs stew. She"
     A: tastes and adjusts seasoning. ← CORRECT
     B: kneads bread dough on the counter. ← off-topic (different activity)
     C: hangs laundry outside on the line. ← off-topic
     D: writes a letter to her sister. ← off-topic

   The DIFFERENCE: HARD distractors stay in the cooking domain but break logic. EASY distractors switch to other domains. WE NEED HARD DISTRACTORS.

5. CRITICAL — LENGTH MATCHING: All four completions must be the SAME length (similar word count and sentence count). Do NOT make the correct answer longer or more detailed than the distractors.

6. CRITICAL — TEMPORAL CONSTRAINT: Use only tools, methods, and technology available during {start_year}-{end_year}. No computers, no plastic, no television, no microwave ovens. Use period-appropriate items: wood stoves, hand tools, horse-drawn carts, typewriters, telegraph, radio (if after 1920s).

7. Each question must describe a DIFFERENT everyday activity.

For each completion, provide step-by-step reasoning explaining why the correct completion is the natural next step and why the others don't fit.

Return a JSON object:
{{"completions": [{{"context": "The beginning of the sentence or passage...", "choices": {{"A": "completion 1 (2-3 sentences)", "B": "completion 2 (2-3 sentences)", "C": "completion 3 (2-3 sentences)", "D": "completion 4 (2-3 sentences)"}}, "correct": "C", "reasoning": "The context discusses X, so the correct completion continues this topic. Completion A shifts to an unrelated subject, while completion C maintains the narrative thread about X."}}]}}

Text:
{text}"""


# ---------------------------------------------------------------------------
# Generator F: Commonsense Reference (which entity fills the blank?)
# Formats: MC-2, CoT
# ---------------------------------------------------------------------------

COMMONSENSE_REFERENCE_PROMPT = """You are given a passage from a historical document published between {start_year} and {end_year}. Use the passage as inspiration for entities, register, and setting. Create {num_items} pronoun-resolution items.

CORE PRINCIPLE
Each item is one prose sentence (or two short sentences) containing TWO concrete entities and ONE blank "_". The blank must be resolvable ONLY through real-world commonsense about those entities — not grammar, not surface cues, not statistics. Both entities must be plausible fillers on surface reading; the answer turns on knowledge of how the world actually works (roles, physical properties, motivations, causality, ownership).

HARD CONSTRAINTS
1. Both fillers must appear verbatim (with or without "the"/"a") in the sentence. Never invent names that aren't in the sentence.
2. Both entities must be CONCRETE (people, animals, tools, places, vehicles, materials). Reject abstract nouns like "the law", "duty", "honor".
3. Reject one-way inferences where one filler is impossible by definition (e.g., "doctor treated patient. The _ was ill" — patient is ill by definition).
4. Reject items where gender pronouns or grammatical agreement leak the answer.
5. Period-appropriate entities and vocabulary only. No post-{end_year} concepts.

SENTENCE-FORM VARIETY (CRITICAL)
DO NOT default to "because _". Vary the connective across items. Use any of:
- declarative two-sentence form: "<setup>. The _ <attribute>."
- comma continuation: "<setup>, _ <action/state>."
- "so _", "but _", "but the _", "as _", "since _", "while _", "because _"

The choice of pattern should fit the relationship being tested, not be uniform.

COMMONSENSE ANCHORS (vary across items)
Each item resolves on one of: professional role / physical property / motivation / causality / ownership / temporal sequence.

VERIFICATION (every item)
Before emitting, check each filler against the sentence: does the reading make surface sense? Does it make commonsense sense? Set `correct` only when one reading is commonsense-correct AND the other is commonsense-wrong but surface-plausible. If both readings are equally valid OR one is surface-impossible, regenerate.

DIVERSITY WITHIN A CALL
Different entity pair per item. Different commonsense anchor per item. Different sentence pattern per item. Roughly balanced A/B as the correct answer.

ONE FULL SHAPE EXAMPLE (illustrates structure — DO NOT copy the topic):
{{
  "sentence": "Maria had to bend down to walk through the cellar door but barely tilted her head to enter the attic doorway. The _ is shorter.",
  "option_a": "cellar door",
  "option_b": "attic doorway",
  "fillers_in_sentence": "option_a appears in sentence: YES. option_b appears in sentence: YES.",
  "verification": "If cellar door: surface sense YES, commonsense YES (bending = much shorter opening). If attic doorway: surface sense YES, commonsense NO (only a head tilt suggests a near-normal-height opening).",
  "correct": "A",
  "anchor": "physical_property",
  "reasoning": "Greater body adjustment (bending) implies a much shorter opening; minor adjustment (head tilt) implies a roughly head-height opening."
}}

Return JSON: {{"winogrande_items": [<item>, <item>, ...]}}

Source passage:
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
