"""
Prompt templates for synthetic data generators (A-F).

All six prompts follow a unified schema for readability and paper-appendix
presentation:

  GOAL            — one-two sentence description of what the generator tests
  FORMAT          — structural requirements (what an item looks like)
  CORE PRINCIPLES — what makes a good item
  CONSTRAINTS     — hard rules (temporal, length, diversity, etc.)
  REJECT IF       — anti-patterns with brief reasons
  SHAPE EXAMPLE   — one representative item illustrating structure
  OUTPUT          — JSON return format

The schema is identical across generators; only the content differs. This
lets reviewers compare designs directly without parsing six different layouts.
"""

# ---------------------------------------------------------------------------
# Generator A: Factual QA (scenario-based, general-principle questions)
# Formats: MC-4, Open-ended, CoT
# ---------------------------------------------------------------------------

QA_PROMPT = """You are given text from a historical document published between {start_year} and {end_year}. Use it ONLY for the underlying scientific principle being discussed — IGNORE the author's name, the publication context, the institutional/historical framing. Create {num_items} grade-school-level science multiple-choice items in the style of standardized science exams (3rd-9th grade level).

GOAL
Test application of a general scientific principle to a novel scenario — physical science, life science, earth/space science, or lab procedure. Each question must be self-contained; the reader should not need to have read the source. The principle being tested must be one that any grade-school science student would learn (heat transfer, phase change, circuits, photosynthesis, rock cycle, animal adaptation, atomic composition, gravity, etc.), not specialized historical/engineering knowledge.

FORMAT
Each item has:
- ONE question (one of the six STEM STYLES below)
- ONE correct answer
- THREE plausible-but-wrong distractors, all from the SAME CATEGORY as the correct answer
- Step-by-step reasoning (3-5 sentences) explaining the principle and why each distractor fails

QUESTION STEM STYLES (rotate roughly evenly across items; do not use the same style twice in a row)
(i)   SENTENCE-COMPLETION: the question is an incomplete sentence ending with a preposition, article, or particle (with a / by / called / except / as / through / into); each choice directly completes it. Do NOT use a "___" placeholder — the sentence simply trails off and the choice is the completion. Examples of the FORMAT: "Current moving through a circuit is stopped with a", "Heat energy transferred from the rock to the water by", "When an animal population moves a long distance...it is called"
(ii)  PROPERTY/CLASSIFICATION: "Which X has property Y?" or "Which of these is Z?" Examples: "Which form of energy can travel through a vacuum?", "Which of the following is made of atoms?", "Which of these substances are elements?"
(iii) SCENARIO + CAUSE: a 1-2 sentence concrete scenario, then a "What is the most likely cause?" / "How does this affect X?" / "What process produces this?" question (or a sentence-completion ending with "to", "by", "into"). Examples: "Over a long period of time, running water in a river erodes the riverbed. This erosion causes the river to", "Scientists studying a group of young birds discovered a percentage were infected with parasites. How will these parasites most likely affect the birds?"
(iv)  COMPARATIVE / "BEST" SELECTION: "Which X best explains Y?" / "What method would Z?" / "Which behavior most likely caused W?" Examples: "Which behavior of golden eagles best explains why this adaptation was favored?", "What method would reproduce these exact peaches?"
(v)   NEGATION / EXCEPTION: ends with "except" / "...includes all these except" / "Which is NOT a Y?" Examples: "All of these types of waves travel from the Sun to Earth except", "It may be inferred that the fertilizer includes all these ingredients except"
(vi)  PROCEDURAL: "What should be done first?" / "Which procedure correctly does X?" Examples: "A student drops a slide and it shatters. What procedure should the student follow first?", "What could be done to show that air takes up space?"

DISTRACTOR DESIGN PATTERNS (each item's THREE distractors should each use a DIFFERENT pattern when possible)
(a) Same-category siblings: distractors are members of the SAME category as the correct answer. Correct=conduction → distractors=[radiation, convection, evaporation] (other heat-transfer modes). Correct=switch → distractors=[wire, battery, bulb] (other circuit parts). Correct=migration → distractors=[hibernation, navigation, breeding] (other animal behaviors).
(b) Synonym/sibling confusion: a term commonly confused with the correct one. Correct=volcanic → distractor=intrusive igneous (sounds like extrusive). Correct=heating and pressure → distractor=compaction and cementation (the OTHER rock-cycle process).
(c) Surface-feature trap: a choice that matches the question's surface words but not the principle. For "what action releases heat" — distractor "Chemicals are released to indicate thirst" matches "released" but not heat.
(d) Plausible-wrong-mechanism: a real biological/physical process that would produce a different outcome. For "how do parasites affect birds" — distractor "by mutating genetic traits" is a real concept but wrong mechanism.
(e) Off-by-one phase / wrong direction: for phase-change or cause-effect, swap the direction. Correct=become a solid (water at -2°C) → distractor=condense (wrong phase change).

CORE PRINCIPLES
- Pivot from author to phenomenon: never ask what an author argued, observed, or described. Ask about the underlying physical/biological/chemical mechanism the author was studying.
- General principle, not source-specific: ask about the science that any grade-school student could learn (heat transfer, gravity, circuits, photosynthesis, food chains), not the source's specific harbor, researcher, or institution.
- Same-category distractors: all three distractors must be the same KIND of thing as the correct answer (all heat-transfer modes, all rock types, all phase changes, all animal behaviors).
- Length-matched choices: distractors and correct have similar word count. Correct is NEVER the longest or most-detailed choice.
- Style variety: rotate across the (i)-(vi) stem styles. Do NOT use "What X?" for every item.

PIVOT EXAMPLES (BAD asks about the author/source; GOOD asks about the underlying science):
  Source topic: abnormal organisms / conjoined twins
    BAD:  "What did Wilder argue about abnormal organisms?"
    GOOD: "What biological process can cause two embryos to fuse into conjoined twins?"
  Source topic: power plant report
    BAD:  "How did the report describe electricity generation?"
    GOOD: "What form of energy does a coal-fired power plant convert into electricity?"
  Source topic: soil erosion study
    BAD:  "What did the study conclude about soil erosion?"
    GOOD: "Which natural process causes topsoil to be carried away by water?"
  Source topic: electrical switch patent
    BAD:  "What mechanism did Spielman patent for his electric lighter?"
    GOOD: "Current moving through a circuit is stopped with a"  [stem style (i)]
  Source topic: petroleum refining / compound separation
    BAD:  "What method did Travis use to separate suspended solids from oil?"
    GOOD: "Oxygen, hydrogen, and water are substances. Which of these substances are elements?"  [stem style (ii)]

CONSTRAINTS
- Temporal: all questions, answers, and distractors must be grounded ONLY in knowledge available during {start_year}-{end_year}. Do NOT introduce post-{end_year} facts, events, outcomes, terminology, or references.
- Self-contained: include all necessary context in the question itself. No phrases like "according to the text", "the passage states", "mentioned above", "during the period described".
- Within-call diversity: each of {num_items} items tests a DIFFERENT scientific principle AND uses a DIFFERENT stem style.
- Short answers preferred: 1-5 words each ("conduction", "switch", "potassium", "become a solid"). For procedural style, full sentences are OK.

REJECT IF
- Question asks about an author, researcher, publication, or institution ("What did X argue?", "How did the report describe...?", "What was the significance of Y's work?").
- Question is about history of science (when a discovery was made, who founded a field), research methodology, publishing practices, or sociology of scientists.
- Question is a site-specific engineering case study ("A mail-steamer enters Flushing Harbour...", "Gravity measurements at Washington in 1900..."). The underlying physics principle must be generalized to a non-specific scenario.
- Distractors are not all in the same category as the correct answer (e.g., correct is a process and one distractor is an object).
- More than half of {num_items} items use the same stem style (e.g., all "What X?" questions).
- Question uses phrases like "according to the text", "the passage states", "described in the text", "mentioned above".
- Question reuses the TOPIC of any example in this prompt. The following topics are ALREADY illustrated and MUST NOT be your chosen topic: animal migration, solar/vacuum energy, river erosion, asexual plant reproduction, fertilizer/NPK, air-takes-up-space, electrical switch/circuit, oxygen-hydrogen-elements, heat transfer (conduction/convection/radiation). Pick a DIFFERENT scientific principle from the source. These examples only show STRUCTURE — do not copy their content.

SHAPE EXAMPLES (one for each stem style; pick your own scientific principle from the source):
  Style (i) — SENTENCE-COMPLETION
    Q: "When an animal population moves a long distance to another area in order to survive, it is called"
    Correct: "migration"  Distractors: ["hibernation", "navigation", "breeding"]

  Style (ii) — PROPERTY / CLASSIFICATION
    Q: "Which form of energy can travel through a vacuum?"
    Correct: "solar"  Distractors: ["thermal", "mechanical", "chemical"]

  Style (iii) — SCENARIO + CAUSE
    Q: "Over a long period, running water in a river erodes the riverbed. This erosion causes the river to"
    Correct: "become deeper and wider"  Distractors: ["stop flowing", "create waves", "move faster and cleaner"]

  Style (iv) — COMPARATIVE / "BEST" SELECTION
    Q: "A peach tree produces peaches more disease-resistant than other peaches. Which method would reproduce these exact peaches?"
    Correct: "ensure the tree reproduces asexually"  Distractors: ["cross-pollinate with different peach trees", "use bees to pollinate the flowers", "increase genetic diversity in the orchard"]

  Style (v) — NEGATION / EXCEPTION
    Q: "A fertilizer claims to contain all the chemicals a plant needs for rapid growth. It may be inferred that the fertilizer includes all these ingredients except"
    Correct: "carbohydrates"  Distractors: ["potassium", "phosphorus", "nitrogen"]

  Style (vi) — PROCEDURAL
    Q: "Air has no color and cannot be seen, yet it takes up space. What could be done to show that air takes up space?"
    Correct: "blow up a beach ball or balloon"  Distractors: ["observe clouds forming", "measure the air temperature", "weigh a glass before and after it is filled with water"]

If the source only supports questions about specific historical engineering or sociology-of-science (no underlying generalizable principle), generate fewer items rather than producing weak ones.

OUTPUT
Return a JSON object with key "qa_pairs". Each item should specify which stem style it uses:
{{"qa_pairs": [{{"question": "When an animal population moves a long distance to another area in order to survive, it is called", "answer": "migration", "stem_style": "i", "reasoning": "Step 1: Long-distance seasonal or survival-driven relocation is a defined biological behavior. Step 2: Migration specifically describes this pattern of travel to a different area. Step 3: Hibernation is a dormancy behavior (not relocation); navigation is a skill used during migration, not the behavior itself; breeding refers to reproduction.", "distractors": ["hibernation", "navigation", "breeding"]}}]}}

Text:
{text}"""


# ---------------------------------------------------------------------------
# Generator B: Physical Commonsense (sensible-method selection)
# Formats: MC-2, CoT
# ---------------------------------------------------------------------------

PHYSICAL_COMMONSENSE_PROMPT = """You are given text from a historical document published between {start_year} and {end_year}. Use it ONLY for the time period and the kinds of materials/tools available — IGNORE the source's topic if it is abstract (politics, philosophy, religion, military strategy, court cases). Create {num_items} physical-commonsense items in the style of everyday DIY hacks, household tasks, cooking, crafts, gardening, and home repair from that era.

GOAL
Test whether a reader understands how concrete physical tasks actually work — which tool, action, quantity, direction, sequence, or material is physically sensible, and which is subtly wrong. Items should feel like instructional how-to tips, not philosophical or political claims.

FORMAT
Each item has ONE goal statement and TWO candidate methods:
- solution: physically sensible method (or, for style (i), a correct property/use of the object)
- distractor: structurally similar but physically WRONG in one specific way
- reasoning: 3-5 short steps explaining why the solution works and why the distractor fails

GOAL STYLES (rotate roughly evenly across items; do not use the same style twice in a row)
(i)   BARE OBJECT LABEL (1-3 words, no verb): a single concrete period-appropriate household, kitchen, workshop, garden, or craft object. Choices describe a physical property or use of the object. Examples of the FORMAT (not topics to copy): "safety pin", "washcloth", "wool", "spoons", "a bucket", "knives", "toilet tissue".
(ii)  IMPERATIVE TASK: a one-sentence task to accomplish. Examples of the FORMAT: "Prevent feet from sweating when wearing flats.", "Remove dust residue from chalk boards.", "Make an SOS signal.", "Set up a magazine holder."
(iii) HOW-TO QUESTION: "How to ..." or "how do you ..." prompting an instruction. Examples: "How to clean a toilet.", "how do you eat a boiled egg?", "How to beautify a hanging basket."
(iv)  "TO ..." INFINITIVE PURPOSE STEM: starts with "To ...," and the two choices are full method instructions completing the purpose. Examples: "To add fragrance to glycerin for making soap.", "To keep edges of strapping from continuing to fray.", "To store old beer-bottle tops to use for crafts later."

DISTRACTOR ERROR PATTERNS (rotate across items — do not reuse the same pattern twice in a row)
(a) Wrong tool/material: replace the correct item with one resembling it in name or category but unable to do the job. Ex: "soapy water" → "milk"; "panythose over coat hanger" → "newspaper over coat hanger"; "knife" → "axe".
(b) Inverted action: flip the key verb. Ex: "seal cracks" → "open cracks"; "be nice to everyone" → "be mean to everyone"; "burn the edges" → "paint the edges".
(c) Wrong location/direction/spatial relation: swap inside/outside, against wall/on floor, in/on. Ex: "sprinkle inside of shoes" → "sprinkle outside of shoes"; "lay against the wall" → "lay on the floor"; "add cleaner into the toilet" → "add cleaner on the toilet".
(d) Quantity inversion: swap the proportions or amounts. Ex: "3/4 water and 1/4 salt" → "1/4 water and 3/4 salt"; "a few cups of water" → "1/2 cup of water"; "short rapid breaths" → "long rapid breaths".
(e) Object-of-action substitution (style (i) + (iv) common): change WHAT the object/method acts on. Ex: "scoop ice cubes" → "scoop books"; "carry a book" → "carry a watermelon"; "can clean a car" → "can clean mold".
(f) Missing critical step or wrong sequence: omit a required time/temperature/order, or swap step order. Ex: "let dry 24 hours" → "let dry"; "boil then drain" → "boil then enjoy".

CORE PRINCIPLES
- Concrete physical task, not abstract concept: every item must be about something a person could literally DO with their hands or with everyday objects. NEVER use abstract topics (governance, justice, rights, philosophy, religion, military doctrine, ceremony).
- Length matching within an item: solution and distractor have similar word count; they typically differ by only 1-2 substituted words or a single inverted phrase.
- Length VARIATION across items is OK: short single-clause choices for styles (i)/(ii); longer multi-sentence instructional choices for styles (iii)/(iv) are acceptable when realistic for a how-to.
- Subtle error: the distractor should look plausible at first glance; only real-world physical understanding should reveal it is wrong.

CONSTRAINTS
- Temporal: only tools, materials, and methods available during {start_year}-{end_year}. Period-appropriate items: hand tools, wood stoves, iceboxes, sewing needles, oil lamps, coal, kerosene, panythose (post-1940), etc. Do NOT use microwaves, plastic, electric refrigerators (unless after ~1930), televisions, computers, or anything post-{end_year}.
- Setting-grounded, NOT topic-grounded: borrow the era and material culture from the source; do NOT borrow the source's subject matter if it is abstract. If the source is about constitutional law, do not write items about "judiciary" or "rights" — write about period-appropriate household, craft, or DIY tasks instead.
- Within-call diversity: each of {num_items} items is about a DIFFERENT object, material, or task. Do NOT reuse the same object across items. Do NOT reuse the same activity across items.

Topic inspiration (these are PIQA-style domains; use the source ONLY for setting/era):
- Cooking, baking, food preservation (canning, smoking, salting, drying)
- Cleaning, laundry, stain removal, household repair
- Crafts, sewing, knitting, leatherwork, paper crafts
- Gardening, plant care, hanging baskets
- First aid, home remedies, hygiene hacks
- Lighting, heating, fuel handling
- Storage, packaging, repurposing old items (jars, bottle caps, crates)
- Tool use, sharpening, fastening, fixing

REJECT IF
- Goal is an abstract concept (judiciary, rights, governance, ceremony, philosophy, military strategy, court ruling). It must be a concrete physical task.
- Correct answer is longer or more detailed than the distractor (within the same item).
- The two choices describe the same activity with a trivial rewording — the correct/wrong distinction must hinge on a SPECIFIC physical fact (a tool, direction, quantity, material, or step).
- Uses post-{end_year} technology or materials.

SHAPE EXAMPLES (one for each style; pick your own topics from the PIQA-style domain list above):
  Style (i) — BARE OBJECT LABEL
    Goal: "safety pin"
    Solution: "can hold together a diaper."
    Distractor: "can hold together a thick stack of paper."
    Reasoning: "Step 1: A safety pin pierces fabric and clasps shut, securing two layers of cloth. Step 2: Diaper fabric is thin enough for a pin to penetrate cleanly. Step 3: A thick paper stack would resist the pin and tear, so the distractor fails."

  Style (ii) — IMPERATIVE TASK
    Goal: "Prevent feet from sweating when wearing flats."
    Solution: "Sprinkle inside of shoes with dry shampoo."
    Distractor: "Sprinkle outside of shoes with dry shampoo."
    Reasoning: "Step 1: Sweat originates from feet inside the shoe. Step 2: Dry shampoo absorbs moisture. Step 3: Powder applied outside the shoe never contacts the foot, so it cannot absorb sweat."

  Style (iii) — HOW-TO QUESTION
    Goal: "How to clean a toilet."
    Solution: "Add cleaner into the toilet bowl and use a toilet brush to scrub around the inside."
    Distractor: "Add cleaner on top of the toilet seat and use a toilet brush to scrub around the inside."
    Reasoning: "Step 1: Stains form on the inside surface of the bowl. Step 2: Cleaner must contact those stains to dissolve them. Step 3: Cleaner placed on the seat never reaches the stains, so it cannot clean them."

  Style (iv) — "TO ..." INFINITIVE PURPOSE STEM
    Goal: "To keep edges of strapping from fraying inconspicuously,"
    Solution: "burn the edges of the strapping to seal it off."
    Distractor: "paint the edges of the strapping to seal it off."
    Reasoning: "Step 1: Strapping frays because loose fibers separate at the cut edge. Step 2: Heat melts and fuses synthetic fibers, locking them together. Step 3: Paint coats the surface but does not bind the fibers, so fraying continues underneath."

OUTPUT
Return a JSON object with key "piqa_items". Distribute {num_items} items across the four styles (do not pile onto one style):
{{"piqa_items": [{{"goal": "safety pin", "solution": "can hold together a diaper.", "distractor": "can hold together a thick stack of paper.", "reasoning": "Step 1: ... Step 2: ... Step 3: ..."}}]}}

Text:
{text}"""


# ---------------------------------------------------------------------------
# Generator C: Reading Comprehension (passage + MC questions)
# Formats: MC-4+Passage, Open-ended, CoT
# ---------------------------------------------------------------------------

COMPREHENSION_PROMPT = """You are given raw source text from a historical document published between {start_year} and {end_year}. The text may be poorly formatted (broken words, column artifacts, mid-sentence starts).

GOAL
Produce a clean period-appropriate passage plus {num_items} multiple-choice questions testing comprehension (main idea, detail recall, character motivation, inference, vocabulary in context).

FORMAT
Your response has two parts:

PART 1 — Write a clean passage (150-300 words):
- Start and end at natural sentence boundaries
- Faithfully represent the source — do NOT invent facts or include information beyond what is stated or clearly implied
- Fix OCR artifacts (broken words, garbled text) but preserve the original meaning
- Write as direct prose about the subject matter — do NOT begin with meta-references like "The text discusses", "The passage describes", "This document outlines". Start directly with the content.

PART 2 — Create {num_items} MC questions about the passage:
Each question has 4 choices (A/B/C/D), exactly one correct, similar-length distractors, a question_type label, and step-by-step reasoning.

PASSAGE STYLES (PREFER narrative/dialogue/letter over formal essay whenever the source even partially supports it — RACE passages are overwhelmingly conversational)
(i)   Formal essay / article (use ONLY when source has no named people, no decisions, no dialogue — pure exposition)
(ii)  Narrative / personal account (DEFAULT when any named person is doing, feeling, or deciding something — memoir excerpt, human-interest story, court testimony, anecdote). First- or third-person story.
(iii) Letter or correspondence (if the source contains a letter or correspondence, preserve "Dear Mr. Smith, ... Sincerely, John")
(iv)  Dialogue / exchange (if the source contains quoted speech, present it as back-and-forth between named speakers)

QUESTION TYPES
(a) BEST TITLE / MAIN IDEA — "Which of the following is the best title for the passage?" / "What is the main idea?"
(b) DETAIL RECALL — specific fact (name, number, date, place, order, object, quantity). "Where did X happen?" "How many Y were there?" "What color was Y?"
(c) CHARACTER MOTIVATION / FEELING — "Why did X do Y?" "How does X feel about Y?" "What kind of person is X?" "What is X's attitude toward Y?"
(d) INFERENCE / IMPLICATION — something implied but not stated. "What can be inferred about X?" "The author suggests that..." "From this passage we learn that..."
(e) VOCABULARY IN CONTEXT (when there is a notable phrase) — "What does [phrase] most likely mean in the passage?" "'[X]' as used in the passage most nearly means..."

MANDATORY DISTRIBUTION for {num_items} questions per call:
- If {num_items} == 2: exactly ONE type (a) OR (b) (surface-level) AND exactly ONE type (c), (d), OR (e) (deeper). Never two type-(a) or two type-(b).
- If {num_items} >= 3: at least one type (a), at least one type (b), and the rest from (c)/(d)/(e).
- Whenever the passage contains ANY named person making a decision, expressing a feeling, or acting on a motive (i.e. passage style ii/iii/iv), at least one type (c) motivation question is MANDATORY. Motivation/why questions are heavily tested and must not be skipped.
- When the passage contains a notable phrase, idiom, or technical term, prefer a type (e) vocabulary question over another detail-recall question.

DISTRACTOR DESIGN PATTERNS (each question's THREE distractors should each use a DIFFERENT pattern from this list)
(p1) Partial truth: a fact that IS stated in the passage but does NOT answer the specific question asked (true but irrelevant).
(p2) Opposite / reversal: reverses the key proposition from the passage (e.g., passage says "it wasn't lovely" → distractor says "it looked beautiful").
(p3) Fabricated plausible detail: a specific-sounding claim (named number, named item) that the passage never makes. The fabrication should fit the passage's genre.
(p4) Generic instead of specific: a generic reason/description where the passage gives a SPECIFIC one (e.g., "she needed to be alone" when the passage specifically says "she didn't hear him call").
(p5) Same-kind wrong value: for detail recall of names/numbers/places/colors, use other valid members of the same category (other colors, other days, other people named in passage).
(p6) Partial-theme title: for type (a), a title that captures a tangential topic from the passage but misses the main theme.

CORE PRINCIPLES
- Wrong choices should be plausible given the passage, clearly incorrect on close reading, and use one of the six distractor patterns (p1)-(p6).
- Length pattern: for type (b) detail-recall, the correct answer is often SHORTER than distractors (a single word, color, number, or short phrase). For type (c)/(d), length-match across choices. NEVER make the correct answer the longest or most elaborate.
- Short answers are GOOD for detail recall: one-word ("Red", "Green", "Team", "Tuesday"), short phrase ("They play games.", "After a meal.") are all acceptable correct answers.
- Title distractors should partially fit but miss the specific theme (pattern p6).

CONSTRAINTS
- Temporal: passage must read as if written during {start_year}-{end_year}. Do NOT introduce any knowledge, events, terminology, or references from after {end_year}.
- The "passage" field must be IDENTICAL across all {num_items} items — write it once and repeat it in each item.

REJECT IF
- Passage introduces post-{end_year} facts or terminology.
- A type (c)/(d) question's correct answer is noticeably longer or more detailed than distractors.
- A type (b) detail-recall question has the correct answer as the longest choice (detail recall correct answers should be short).
- Two questions test the same aspect of the passage.
- Passage style is (i) formal essay when the source actually contains named people doing things (use style ii instead).

SHAPE EXAMPLES (illustrate two question types with different distractor patterns — pick your own topic from the source):

  Narrative passage: "About two years ago, my wife Cathy brought home a little dog with a face only a mother could love. We named her Gertie. Gertie is the kind of dog that has to grow on you. At first I wondered why Cathy had picked such an unlovely creature, but over months Gertie's quiet devotion changed my mind completely."

  Q1: "Which of the following is the best title for the passage?"  (type a)
  Choices: A: "Why I Changed My Mind About Gertie" [CORRECT]; B: "How to Pick a Dog" (p6: partial-theme); C: "Cathy's Love of Animals" (p6: tangential); D: "A Mother's Bond with Her Pet" (p6: off-theme)
  Reasoning: "The passage is about the narrator's shift in feeling toward Gertie. A captures this exactly. B is generic; no picking-advice is given. C narrows on Cathy, but the passage focuses on the narrator. D mentions a metaphor from line 1, not the theme."

  Q2: "How did Gertie look when she was first brought home?"  (type b, detail recall)
  Choices: A: "It looked very beautiful." (p2: opposite); B: "It wasn't very lovely." [CORRECT]; C: "It wasn't necessary to be trained." (p1: partial truth — not asked); D: "It could change his life." (p3: fabricated — not stated)
  Reasoning: "The passage says Gertie had 'a face only a mother could love' (i.e., unlovely). B captures this plainly. A reverses it. C is about training — never discussed. D paraphrases an outcome, not the look."

OUTPUT
Return a JSON object with key "questions". Each item includes a `question_type` field (a/b/c/d/e):
{{"questions": [{{"passage": "Your clean 150-300 word passage here.", "question": "What does the passage suggest about...?", "choices": {{"A": "Choice 1", "B": "Choice 2", "C": "Choice 3", "D": "Choice 4"}}, "correct": "B", "question_type": "d", "reasoning": "The passage states X, which supports choice B. Choice A contradicts the passage because..."}}]}}

Source text:
{text}"""


# ---------------------------------------------------------------------------
# Generator D: Quantitative (chained math word problems)
# Formats: MC-4, Open-ended, CoT
# ---------------------------------------------------------------------------

QUANTITATIVE_PROMPT = """You are given text from a historical document published between {start_year} and {end_year}. Create {num_items} math word problems inspired by the numerical data in this text.

GOAL
Produce chained multi-step math problems grounded in the historical context of the source. Problems must require 2-4 computation steps where each step uses the result of the previous step.

FORMAT
Each item has:
- ONE self-contained word problem (all numbers and context in the question; no reference to "the text")
- ONE correct answer (bare number only: "50", "25%", "$900" — NOT wrapped in a sentence)
- THREE distractors (bare numbers, same format as correct)
- Step-by-step reasoning showing all 2-4 computation steps

OPERATION TYPES (rotate across items in a call — do NOT use the same type twice in a row)
(a) Percentage chain: percent of a value, then percent of the remainder, then sum/difference. Ex: "60% of harvest sold at $5, rest kept; total revenue?"
(b) Unit rate × quantity: rate × time × count, with one secondary operation. Ex: "300 units/day × 5 days/week × 4 weeks; 25% defective; non-defective count?"
(c) Discount / markup / tax: original price → discounted price → final with tax/fee. Ex: "$400 with 80% off, plus 3 hours labor at $50/hr; total paid?"
(d) Time × distance / round-trip: travel time + intermediate stop or return leg. Ex: "70 miles at 35 mph one way, 3 hours visit, then return; total time?"
(e) Profit / loss with cost subtraction: revenue minus cost (sometimes negative). Ex: "Rent car 10×/month at $25/hr × 3 hours, minus $500 car payment; profit?"
(f) Average / mean of N values: sum a list, divide by count (often a non-round divisor). Ex: "Birds seen across 7 days: 50+0+120+20+90; average per day?"
(g) Ratio / proportion / split: total divided in given ratio, or one share computed from another. Ex: "$400 split so Madeline pays 60%; weekly cost for Keenan?"
(h) Division with remainder / capacity: how many trucks/boxes/trips needed to cover N items at K-per-container (must round UP). Ex: "80 flagstones at 75 lbs each, trucks carry 2000 lbs; how many trucks?"

DISTRACTOR ERROR PATTERNS (each item's THREE distractors should each use a DIFFERENT pattern from this list)
(i)    Forgot final subtraction: report gross instead of net (revenue without cost subtracted, total without tax/fee added).
(ii)   Sign flip: report a negative when the answer is positive (or vice versa) — typically from subtracting in the wrong order. Include this pattern in AT LEAST 20% of items where profit/cost/difference is asked, so the student sees and rejects negative answers.
(iii)  Wrong percentage base: applied % to the remainder when it should be % of the total, or vice versa.
(iv)   Reported intermediate value as final: stop one step early (return revenue when profit was asked, return total when per-unit was asked).
(v)    Wrong operation: divided where multiplication was needed, or added where subtraction was needed.
(vi)   Off-by-one in count problems: rounded down when round-up was required, or counted endpoints wrong.
(vii)  Used wrong subset: averaged only some of the values, or summed only the spending categories not all categories.
(viii) Wrong-arithmetic decimal: the result of an INCORRECT calculation path that happens to be non-integer — e.g., dividing by the wrong denominator (400/3 = 133.33 instead of 400/4 = 100; 450/7 = 64.2857142857143 instead of 280/7 = 40), computing percentage of wrong base, stopping at an intermediate ratio. Include this pattern in AT LEAST 20% of items WHEN a plausible wrong-arithmetic path gives a decimal — the decimal MUST be the actual result of a specific miscalculation you describe in reasoning, NOT a random nearby number.
       Write the decimal in one of TWO realistic forms (pick one per item):
       (a) Rounded to 2 decimal places: "133.33", "64.29", "37.14", "73.20"
       (b) Full unrounded float (Python repr of a/b when the division doesn't terminate): "64.2857142857143", "33.33333333333333", "99.9984"
       The (b) form is especially informative — it signals "someone did the division and forgot to round", which is exactly what a sloppy calculator would produce. Alternate between (a) and (b) across items so both forms are learned as wrong.
       This teaches the student to reject spurious-looking decimals (both tidy and untidy) by verifying the arithmetic.

CORE PRINCIPLES
- Chained reasoning: each step USES the previous step's result. Final answer depends on intermediate results that themselves had to be computed.
- ONE final question, multi-step setup: the question asks for ONE quantity. All chaining lives in the narrative leading up to that single question. Banned phrases: "First, find X. Then...", "How many X? If Y, how many Z?", "After this, what is...". The reader computes intermediate values internally; only the FINAL value is asked.
- Historical grounding: use numbers and context inspired by the source (dates, quantities, prices from the period).
- Correct answer is ALWAYS a clean integer (round if the arithmetic gives a decimal). The teaching goal is that the model learns "the correct answer is a whole number."
- Distractors MAY be decimals or negatives when they come from a REAL wrong-arithmetic path (pattern viii) or sign-flip (pattern ii). A decimal distractor is a STRONG teaching signal — the model learns "a decimal choice = someone did wrong division and forgot to round." Similarly, a negative distractor teaches "a negative choice = wrong subtraction order."
- Realistic distractors: each distractor should be the literal output of one of the eight error patterns above, applied to the question's actual numbers — NOT a random nearby number.

CONSTRAINTS
- Temporal: all problems use only facts and numbers from {start_year}-{end_year}. Do NOT reference events or data from after {end_year}.
- Self-contained: include all necessary numbers and context in the question. Do NOT reference "the text" or "the passage".
- Within-call diversity: each of {num_items} problems uses a DIFFERENT operation type from the (a)-(h) list above. No two items in a call share an operation type.
- Vary step count (2-4 steps) across items in a call.

REJECT IF
- Problem is single-step (no chaining).
- Correct answer is a decimal (correct answer must be a whole number; round if necessary).
- Problem references the source ("as stated in the text") rather than being self-contained.
- All {num_items} items use the same operation type.
- Problem contains a compound question ("How many X? Then how many Y?", "If Z, how many W?"). The problem MUST end with exactly ONE final question. Multi-step reasoning lives in the SETUP (the narrative leading up to the question), never in stacked sub-questions.
- Distractor is a random nearby number rather than the output of a specific error pattern (i)-(viii).
- None of {num_items} items in a call contain a sign-flip (ii) or wrong-arithmetic-decimal (viii) distractor when profit/cost/difference/average operations appear. At least one item per call that supports those patterns MUST use them.

SHAPE EXAMPLES (illustrate the structure — pick your own operation type and numbers from the source):
  Example 1 — with SIGN-FLIP distractor:
    Q: "A merchant in 1925 buys 20 crates of goods at $8 each. He sells 15 crates at $12 each and the remaining 5 crates at $6 each. How much total profit did he make?"  [type (e): profit/loss]
    Reasoning: "Step 1: Total cost: 20 × $8 = $160. Step 2: Total revenue: (15 × $12) + (5 × $6) = $180 + $30 = $210. Step 3: Profit: $210 - $160 = $50."
    Answer: "50"
    Distractors: ["210" (pattern i: forgot to subtract cost), "160" (pattern iv: reported cost as final), "-50" (pattern ii: sign flip — subtracted in wrong order)]

  Example 2 — with LONG UNROUNDED DECIMAL distractor (pattern viii form b):
    Q: "A shop in 1930 recorded bird sightings over a week: 50 on days 1-2 combined, 0 on day 3, 120 on days 4-5 combined, 20 on day 6, and 90 on day 7. What was the average number of birds sighted per day across the seven-day week?"  [type (g): average/mean]
    Reasoning: "Step 1: Sum of all sightings: 50 + 0 + 120 + 20 + 90 = 280. Step 2: Days in the week: 7. Step 3: Correct average = 280 / 7 = 40 (whole number). A common wrong path: divide by 6, excluding the zero-bird day, which gives 280/6 = 46.666... written as full float 46.666666666666664. Another wrong path: forget day 3 entirely and divide the day-3-excluded sum 280 by some other wrong count."
    Answer: "40"
    Distractors: ["46.666666666666664" (pattern viii-b: wrong-arithmetic decimal, full unrounded float from dividing by 6 instead of 7), "280" (pattern iv: reported total as final), "-40" (pattern ii: sign flip)]

BAD (NOT chained) — would be REJECTED: "What is 30% of 200?" → only one step, no chaining.

COMPOUND-QUESTION EXAMPLES (BAD vs GOOD):
  BAD:  "A factory produces 1200 units/month for 8 months. How many total units? If 25% are defective, how many non-defective?"
  GOOD: "A factory produces 1200 units/month for 8 months, and 25% of all units produced are defective. How many non-defective units does the factory produce?"

  BAD:  "Each delegate owned 10 units of scrip. What was the profit per unit? What was the total profit for all 40 delegates?"
  GOOD: "Each of the 40 delegates owned 10 units of scrip, bought at 9 cents and sold at 100 cents per unit. What was the total profit for all delegates combined?"

OUTPUT
Return a JSON object with key "problems". The "answer" and "distractors" fields must be BARE NUMBERS ONLY (e.g. "360", "25%", "$900") — do NOT wrap in sentences:
{{"problems": [{{"question": "A merchant in 1925 buys 20 crates of goods at $8 each. He sells 15 crates at $12 each and the remaining 5 crates at $6 each. How much total profit did he make?", "reasoning": "Step 1: Calculate total cost: 20 × $8 = $160\\nStep 2: Calculate total revenue: (15 × $12) + (5 × $6) = $180 + $30 = $210\\nStep 3: Calculate profit: $210 - $160 = $50", "answer": "50", "distractors": ["210", "160", "30"]}}]}}

Text:
{text}"""


# ---------------------------------------------------------------------------
# Generator E: Situational Completion (scene / how-to continuation)
# Formats: MC-4
# ---------------------------------------------------------------------------

COMPLETION_PROMPT = """You are given text from a historical document published between {start_year} and {end_year}. Use it as inspiration for the time period and setting. Create {num_items} sentence-completion items.

GOAL
Test SITUATIONAL COMPREHENSION — whether the reader can predict what happens next in a real-world scene or how-to instruction, based on common sense about how activities actually work. All four completions stay on-topic; only ONE is narratively consistent.

FORMAT
Two item types in roughly equal numbers (alternate):

TYPE 1 — SHORT SCENE WITH TRAILING PRONOUN: a 1-2 sentence caption-style scene ending mid-clause with a pronoun ("he", "she", "they") or noun ("the man", "the boy") with NO period. Four completions (1-2 sentences each) begin lowercase and continue the sentence.

TYPE 2 — WIKIHOW-STYLE INSTRUCTIONS: uses the literal markup `[header]`, `[title]`, `[step]`, and `[substeps]` tags. Begin with `[header]`, then `[title]` for each step, then `[step]` with the body text. Four completions continue the wikiHow format, often starting with `[substeps]` or another `[title]`.

CORE PRINCIPLES
- Narrative consistency over topic recognition: all four completions must stay ON THE SAME TOPIC as the scene. Only ONE is narratively consistent; the other three break logic in subtle ways.
- Source-driven topic: derive each scene or how-to from what the SOURCE TEXT actually discusses — a person, event, object, process, or place. Do NOT invent generic scenes ignoring the source.
- Length matching: all four completions must be the SAME length (similar word count, same sentence count). The correct answer is NOT longer or more detailed.

DISTRACTOR ERROR PATTERNS (narrative-consistency violations — all on-topic)
(a) WRONG SEQUENCE: do a step before its prerequisites.
(b) WRONG CAUSE-EFFECT: produce an illogical outcome from the action.
(c) WRONG TIMING: skip ahead or jump back inappropriately.
(d) CONTRADICTS SETUP: do something the scene already ruled out.

GOOD example of HARD distractors (all stew-topic, only ONE coherent):
  Scene: "A woman carefully stirs a large pot of bubbling stew over the wood stove. She"
    A: "leans in to taste the stew, adjusting the seasoning as needed." ← CORRECT (natural sequence: stir → taste → adjust)
    B: "pours the bubbling stew directly into the bread dough she has been kneading." ← wrong sequence
    C: "sets the pot in the cellar to cool before lighting the fire under it." ← wrong cause-effect
    D: "covers the pot with a wet cloth to put out the flame inside the broth." ← contradicts physics

BAD example of EASY distractors (different activities — REJECT this style):
  Scene: "A woman stirs stew. She"
    A: "tastes and adjusts seasoning." ← CORRECT
    B: "kneads bread dough on the counter." ← off-topic
    C: "hangs laundry outside on the line." ← off-topic
    D: "writes a letter to her sister." ← off-topic

The DIFFERENCE: HARD distractors stay in the cooking domain but break logic. EASY distractors switch to other domains. WE NEED HARD DISTRACTORS.

TOPIC VARIETY (source-driven; no hardcoded list)
Derive the topic of each item from the source text. Across the {num_items} items, questions should span DIFFERENT life domains. Examples of domains (inspiration, NOT a checklist):
- Body and health (illness, injury, hygiene, posture, physical sensation)
- Relationships and social life (visiting, courting, conflict, helping, mourning)
- Mental and emotional states (worry, hope, decision-making, learning)
- Domestic life (any household activity)
- Work and trade (any occupation present in the source)
- Travel and movement (any mode of getting somewhere)
- Leisure and play (games, music, reading, gathering)
- Survival and care (food, warmth, shelter, child care)
- Civic and public life (markets, courts, gatherings, ceremonies)
- Skill and craft (any specialized practice)

ANTI-STEREOTYPE RULE
REJECT and REGENERATE if the item is about "A blacksmith", "A farmer", "A seamstress", "A housewife baking", or generic "kneading dough", "harvesting corn", "canning fruit", "preserving vegetables". These are over-used from prior generations. Pick something else from the source text — a schoolteacher, a telegraph clerk, a nurse, a child playing, a passenger on a streetcar, a mother soothing a sick infant, a clerk filing papers.

CONSTRAINTS
- Temporal: only tools, methods, and technology available during {start_year}-{end_year}. No computers, plastic, television, microwave ovens. Use period-appropriate items (wood stoves, hand tools, horse-drawn carts, typewriters, telegraph, radio if after 1920s).
- Within-call diversity: each item describes a DIFFERENT everyday activity.
- The goal is broad domain coverage: questions should feel as different from each other as "How to recover from an emotional affair", "How to prune apple trees", "How to apply cologne", and "A man bends over into a pond. he".

REJECT IF
- All four distractors switch to different activities (topic-shift instead of narrative-break).
- Correct answer is noticeably longer or more detailed than distractors.
- Scene reuses any of the forbidden stereotype subjects.
- Item uses post-{end_year} technology.

SHAPE EXAMPLES (illustrate structure — pick your own topic from the source)

TYPE 1:
  Context: "A young child sits on the parlor floor, building a tower of wooden blocks while her grandmother watches from the rocking chair. she"
    A: "carefully balances another block on top, her tongue between her teeth in concentration." ← CORRECT
    B: "sweeps the block tower into a dustpan and wipes the floor clean."  ← wrong cause-effect
    C: "places the next block six feet from the tower and waits for it to jump on its own." ← contradicts physics
    D: "unstacks the entire tower and puts each block away before adding any more." ← contradicts setup

TYPE 2:
  Context: "[header] How to soothe a child with a fever [title] Cool the body. [step] Wet a clean cloth in cool (not cold) water and wring it out so it does not drip. Place it gently on the child's forehead."
    A: "[substeps] Replace the cloth every few minutes as it warms, and offer small sips of cool water if the child is thirsty enough to drink." ← CORRECT
    B: "[substeps] Wrap the child in heavy wool blankets and place them next to the wood stove for warmth." ← contradicts goal (heating a fevered child)
    C: "[substeps] Leave the cloth on indefinitely, replacing it only when it freezes solid." ← wrong timing
    D: "[substeps] Apply the cloth, then immediately bathe the child in hot water before the forehead has cooled." ← wrong cause-effect

OUTPUT
Return a JSON object with key "completions":
{{"completions": [{{"context": "Scene or wikiHow-style context ending mid-clause...", "choices": {{"A": "completion 1", "B": "completion 2", "C": "completion 3", "D": "completion 4"}}, "correct": "A", "reasoning": "Step-by-step explanation of why the correct completion is narratively consistent and why each distractor breaks logic."}}]}}

Text:
{text}"""


# ---------------------------------------------------------------------------
# Generator F: Commonsense Reference (which entity fills the blank?)
# Formats: MC-2, CoT
# ---------------------------------------------------------------------------

COMMONSENSE_REFERENCE_PROMPT = """You are given a passage from a historical document published between {start_year} and {end_year}. Use the passage as inspiration for entities, register, and setting. Create {num_items} commonsense-reference items.

GOAL
Test whether a reader can resolve an ambiguous blank in a short sentence using commonsense about the two named entities — not grammar, not surface cues, not statistics. Both entities must be plausible fillers on surface reading; the answer turns on knowledge of how the world actually works (roles, physical properties, motivations, causality, ownership).

FORMAT
Build each item in THIS ORDER (do not reorder):
  1. Write option_a and option_b FIRST — the two entity names you will test.
  2. Write the sentence using the EXACT strings of option_a and option_b as-is, plus ONE blank "_".
  3. Copy option_a and option_b into the JSON fields unchanged — same capitalization and words.

Each item has:
- ONE sentence (10-25 words, or two short sentences) containing exactly TWO concrete entities and ONE blank marked with a single underscore "_"
- TWO candidate fillers (option_a, option_b) — each MUST be an EXACT substring of the sentence (case-insensitive). Only "the ", "a ", or "an " at the START of the option may be absent from the sentence.
- A `fillers_in_sentence` self-check field (see VERBATIM CHECK below)
- A `verification` self-check field evaluating each reading
- The correct letter (A or B)
- A `commonsense_anchor` label (professional_role / physical_property / motivation / causality / ownership / temporal_sequence)
- Step-by-step reasoning

VERBATIM CHECK (do this for EVERY item BEFORE emitting; if it fails, regenerate the item):
  - Take option_a. Read through the sentence. Is option_a an EXACT substring (ignoring only a leading "the "/"a "/"an ")? If NO → regenerate.
  - Take option_b. Same check. If NO → regenerate.
  - Copy the result into `fillers_in_sentence` as:
    "option_a='<x>' found in sentence at position YES/NO; option_b='<y>' found in sentence at position YES/NO"

GOOD vs BAD option writing (focus on VERBATIM MATCHING):
  Sentence: "The doctor tended to the feverish patient with care, but the _ showed signs of worsening symptoms."
  GOOD: option_a="doctor", option_b="patient"  ← both appear verbatim in sentence
  BAD:  option_a="physician", option_b="patient"  ← "physician" NOT in sentence (synonym mismatch)
  BAD:  option_a="the kindly doctor", option_b="the patient"  ← adds "kindly", NOT in sentence
  BAD:  option_a="Dr. Smith", option_b="patient"  ← invented name not in sentence

  Sentence: "A judge and a carpenter both entered the courtroom, but only the _ held a gavel."
  GOOD: option_a="judge", option_b="carpenter"
  BAD:  option_a="the wise judge", option_b="carpenter"  ← "wise" not in sentence
  BAD:  option_a="judge", option_b="woodworker"  ← synonym mismatch

  Sentence: "The cellar door and the attic doorway differed in height. The _ required Maria to bend down."
  GOOD: option_a="cellar door", option_b="attic doorway"  ← exact multi-word substrings
  BAD:  option_a="cellar", option_b="attic"  ← partial, not full entity

CORE PRINCIPLES
- Genuine ambiguity: a reader with NO real-world knowledge could not decide which filler is correct. Only commonsense breaks the tie.
- Concrete entities only: people, animals, tools, places, vehicles, materials. NEVER abstract nouns like "the law", "duty", "honor".
- Verbatim match is mandatory: never use a synonym, partial name, or modified version of the entity. The option string, stripped of leading "the "/"a "/"an ", must be findable as-is inside the sentence.

SENTENCE-FORM VARIETY (CRITICAL — do NOT default to "because _")
Vary the connective across items. Roughly: ~25% declarative two-sentence form "<setup>. The _ <attribute>."; ~15% comma continuation "<setup>, _ <action/state>."; and the rest spread across "so _", "but _", "but the _", "as _", "since _", "while _", "because _". The choice of pattern should fit the relationship being tested, not be uniform.

COMMONSENSE ANCHORS (vary across items; use a DIFFERENT anchor per item in a call)
- Professional role / expertise (which entity performs the named trade/action)
- Physical property (size, weight, hardness, speed, temperature — usually comparative)
- Motivation / intention (which entity has the reason to act)
- Causality (which entity produced the state or outcome)
- Ownership / possession (which entity owns or controls the object in question)
- Temporal sequence (which entity did something first)

VERIFICATION (every item, before emitting)
Check each filler: does the reading make surface sense? Does it make commonsense sense? Set `correct` only when one reading is commonsense-correct AND the other is commonsense-wrong but surface-plausible. If both readings are equally valid OR one is surface-impossible, regenerate.

CONSTRAINTS
- Temporal: only entities, roles, and objects from {start_year}-{end_year}. No post-{end_year} concepts.
- Within-call diversity: different entity pair per item, different commonsense anchor per item, different sentence pattern per item. Roughly balanced A/B as the correct answer.
- No personal pronouns (he/she/it/they) as the blank — use "_".
- Natural period-register prose (not stilted).

REJECT IF
- Either option (stripped of a leading "the "/"a "/"an ") is NOT an exact substring of the sentence.
- One reading is impossible by definition (e.g., "doctor treated the patient. The _ was ill" — patient is ill by definition).
- Gender pronoun or grammatical agreement leaks the answer.
- Both readings are equally commonsense-valid (no clear answer).
- Sentence uses "because _" when the call already has several "because" items (overuse).

SHAPE EXAMPLE (illustrates structure — DO NOT copy the topic)
{{
  "sentence": "Maria had to bend down to walk through the cellar door but barely tilted her head to enter the attic doorway. The _ is shorter.",
  "option_a": "cellar door",
  "option_b": "attic doorway",
  "fillers_in_sentence": "option_a appears in sentence: YES. option_b appears in sentence: YES.",
  "verification": "If cellar door: surface sense YES, commonsense YES (bending = much shorter opening). If attic doorway: surface sense YES, commonsense NO (only a head tilt suggests a near-normal-height opening).",
  "correct": "A",
  "commonsense_anchor": "physical_property",
  "reasoning": "Greater body adjustment (bending) implies a much shorter opening; minor adjustment (head tilt) implies a roughly head-height opening."
}}

OUTPUT
Return a JSON object with key "winogrande_items" (internal identifier — not a reference to any external benchmark):
{{"winogrande_items": [<item>, <item>, ...]}}

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
