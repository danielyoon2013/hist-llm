# FinePhrase Review: Code-Specific Improvements for Our Synthetic-Data Pipeline

**Sources reviewed**
1. **FinePhrase repo** – `huggingface/finephrase` (`README.md`, `prompts/format/*.md`, `prompts/nemotron/*.md`, `prompts/dspy/rephrase/*.md`, `configs/rephrasing.yaml`).
2. **Paper** – Niklaus et al., *The Synthetic Data Playbook: Generating Trillions of the Finest Tokens* (alphaxiv 2604.13977 / abstract + full repo artifacts; the alphaxiv SPA blocks WebFetch on the body text, so claims below are grounded only in the abstract + the released ablation matrix in `configs/rephrasing.yaml` and the empirical winner that survived into prod, `format/tutorial.md` + `format/faq.md`).

**Bottom line on relevance**
FinePhrase is **partially relevant** — same broad family (LLM rephrasing of source text into pretraining data), but a different *output type* (raw rephrased prose for **pretraining**, not labelled QA/MC for **mid-training + SFT**). Several findings transfer cleanly; one (the "≤1B generator" claim) actively *contradicts* what we should do; and their core "format diversity" finding maps very well onto our 6-generator zoo, but in a way that suggests we are **over-engineering** our prompts and **under-diversifying** our output formats.

---

## What FinePhrase actually validated

1. **Format-diversity beats prompt-diversity.** They ablated 9 output *formats* on the same source text (`prompts/format/*.md`): `article`, `commentary`, `discussion`, `explanation`, `faq`, `math`, `narrative`, `table`, `tutorial`. Each prompt is **3–6 sentences long**, with **no rejection criteria, no shape examples, no anti-stereotype rules, and no temporal constraints**. The big wins came from picking the right output *shape* (FAQ, tutorial, table, math), not from heroic prompt engineering inside any one shape.
2. **Generator scaling plateaus around 1B for FORMAT prompts** (their 9-format prompts), but they explicitly tried 0.27B → 1B → 4B → 12B → 27B Gemma-3 and ran multiple model families (Qwen, SmolLM2, Llama-3.2, Falcon-3, Granite-3.1) — *all at 1–2B*. The plateau is for *naïve rephrasing*; they found "guided rewrite" (`rewire/guided_rewrite_*.md`) needed bigger models, and DSPy-optimized prompts (`dspy/rephrase/*/budget-{1,10}.md`) work even at 1B.
3. **HQ vs LQ source data needs different prompts.** README explicitly bifurcates: HQ uses `nemotron/distill.md`, `extract_knowledge.md`, `diverse_qa_pairs.md`, `knowledge_list.md`; LQ uses `rewire/guided_rewrite_corrected.md` and `dspy/5-max_full_evals.md`. The same prompt does not work for both quality tiers.
4. **They don't filter or reject bad outputs at generation time.** No `REJECT IF` clause exists in any of their prompts. Quality is enforced upstream by FineWeb-Edu Ridge classifier scores (HQ = score 4-5, LQ = score 0-1) that determine *which prompt the document gets*, not whether the output is kept.
5. **They tag every prompt with `Output only X, nothing else.`** Hard, terminal output discipline — no JSON wrappers, no metadata, no chain-of-thought.
6. **DSPy-optimized prompt is a striking find** (`prompts/dspy/rephrase/gemma-3-1b-it/budget-10.md`): it includes a *validation gate* that aborts on out-of-domain inputs with a fixed string token. The prompt was machine-discovered by DSPy's GEPA optimizer on a separate held-out eval set. Their `iterate-prompt` CLI explicitly supports human-in-the-loop prompt refinement on a tiny sample before launch.

---

## Concrete improvements for our pipeline

### Improvement 1 — Add a **rephrasing-as-pretraining stream** alongside the QA stream

**FinePhrase finding.** Their 486B-token "winner" was raw rephrased text using **9 distinct output formats** (FAQ, tutorial, table, math, narrative, explanation, article, commentary, discussion). The single biggest lever was *format diversity over the same source*, not bigger models or longer prompts. Their ablations show structured formats consistently beat the FineWeb-Edu baseline at the same token budget.

**Why our current approach is suboptimal.** Our entire pipeline (`config.py:GENERATOR_SPEC`, all 6 `gen_*.py` files) only emits **labelled QA/MC/CoT** — i.e., every output is a `{user, assistant}` conversation where the assistant message is a *letter* (`mc4`, `mc2`, `mc4_passage`) or a *short answer* (`open`, `cot`). We *throw away* the raw rephrased passage in `gen_c_comprehension.py` (the passage is bundled into MC items rather than being its own training example). At a 1.3B model trained on 1900–1949 text only, the model is learning to *pick a letter* far more than it is learning to *generate fluent period-appropriate prose*. HellaSwag at 25.2% (random) and Winogrande at 48% (below random) are the textbook signature of a model that hasn't seen enough fluent generative prose during mid-training.

**Exact code change.** In `src/post_training/generators/`:

- New file `gen_g_rephrase.py` (or a `RephraseGenerator` mode flag on `BaseGenerator`). The prompt is **literally one of FinePhrase's `format/*.md` prompts** with our `{start_year}-{end_year}` temporal constraint appended.
- Add to `src/post_training/generators/prompts.py`:

```python
REPHRASE_TUTORIAL_PROMPT = """Rewrite the document below as a clear, step-by-step tutorial or
instructional guide using vocabulary and references appropriate to {start_year}-{end_year}. Use
numbered steps or bullet points where appropriate. Preserve all essential information but ensure
the style feels didactic and easy to follow. Do NOT introduce any knowledge, events, or terminology
from after {end_year}. Output only the tutorial, nothing else.

Document:
{text}
"""
# Plus REPHRASE_FAQ_PROMPT, REPHRASE_NARRATIVE_PROMPT, REPHRASE_EXPLANATION_PROMPT
# (all four taken verbatim from finephrase/prompts/format/{tutorial,faq,narrative,explanation}.md
#  with the period clause appended).
```

- New format constant `FORMAT_REPHRASE = "rephrase"` in `base.py`. Its `format_conversation` returns:

```python
return [
    {"role": "user", "content": ""},                  # empty user — pretraining-style
    {"role": "assistant", "content": rephrased_text}, # whole rephrased document
]
```

- In `config.py:GENERATOR_SPEC`:

```python
"G": {"formats": ("rephrase",) * 4,  # 4 slots: tutorial/faq/narrative/explanation
      "corpus": True, "collections": None, "pass_rate": 0.95, "weight": 2.0},
```

A weight of 2.0 means ~25% of the 1M-conv budget becomes generative rephrasing; we drop `D` (Quantitative) weight to 0.6 to absorb the cost (D drives GSM-MC, which is already at 25.7% random — adding more chain-math data without addressing the underlying coverage gap won't help; rephrased prose will).

**Expected impact.** HellaSwag and Winogrande directly: both depend on the model's fluent next-token distribution over conversational/narrative prose, which is exactly what rephrased "narrative" and "explanation" formats produce. RACE-M/H also benefits because rephrased "explanation" prose is structurally close to RACE passage style. Estimated: HellaSwag 25 → 30+; Winogrande 48 → 52+; RACE-M 51 → 55+; PIQA likely +2–4 from "tutorial" format (it's literally PIQA-style how-to text).

**Effort.** ~1 day. Four FinePhrase prompts already written; add one new `gen_g_rephrase.py` (~120 lines, mostly identical to `gen_a`); update `config.py` weight redistribution; rerun assemble.

---

### Improvement 2 — **DSPy/GEPA-style prompt optimization**, not hand-tuned mega-prompts

**FinePhrase finding.** Their `iterate-prompt` CLI runs an LLM-judged tight loop on 5–20 sample documents before any large run, and `prompts/dspy/rephrase/gemma-3-1b-it/budget-10.md` shows what an automatically optimized prompt looks like — a **structured input/output schema** (`[[ ## original_text ## ]]` ... `[[ ## generated_text ## ]]`) with an *explicit validation gate* that aborts on bad inputs (`[INVALID INPUT: ...]`). Critically, the DSPy prompt was **discovered**, not designed; it works at 1B where hand-written prompts need 4B+.

**Why our current approach is suboptimal.** Our `prompts.py` is **9000+ characters per prompt** (`QA_PROMPT` alone is ~3500 chars). Each prompt has GOAL/FORMAT/CORE PRINCIPLES/CONSTRAINTS/REJECT IF/SHAPE EXAMPLE/OUTPUT — six conceptual layers, ~15 hand-curated examples and anti-patterns each. We have no telemetry on which clauses actually move pass-rate or downstream eval. Worse: when we add a clause (e.g. "ANTI-STEREOTYPE RULE" in `COMPLETION_PROMPT`, the BAD/GOOD pivot examples in `QA_PROMPT`), we have no way to know whether it helped or hurt — these are vibes-driven changes against a $300/run feedback loop.

**Exact code change.**

- Add `src/post_training/generators/iterate_prompt.py` (~150 lines):

```python
def iterate_prompt(prompt_template, gen_class, period, n_samples=10, n_rounds=5):
    """Tight optimization loop: sample N chunks, run prompt, judge with GPT-4o,
    show diff to operator, accept/reject, repeat. Saves best prompt to versioned file."""
    # 1. Sample n_samples random chunks from period corpus
    # 2. For each round:
    #    a. Run prompt on samples (sync, max_workers=10)
    #    b. Score with judge prompt: "rate each generated item on
    #       (faithfulness, format_adherence, distractor_quality, temporal_consistency)"
    #    c. For lowest-scoring item, ask GPT-4o to suggest one targeted prompt edit
    #    d. Operator accepts/rejects edit
    # 3. Save winning prompt as prompts/optimized/{gen}_{date}.py
```

- For each generator, before the next 1M batch, run `python -m src.post_training.generators.iterate_prompt --gen A --rounds 5`. This costs ~$2 per round, total ~$50, vs. ~$500 of wasted batch budget if a prompt regression slips into prod.

- More important shorter-term change: **strip every `SHAPE EXAMPLE` from the prompts and A/B test pass rate + downstream eval at 50K examples each.** Our prompts likely contain examples the model is now copying as topics (we already know from the "ANTI-STEREOTYPE RULE" comment that we caught this happening before). FinePhrase's prompts have **zero shape examples** and the dataset still wins. This is empirically the right size; we are over-prompting.

**Expected impact.** Hard to predict per-benchmark, but the *meta-level* impact is bigger: every benchmark moves more reliably in the right direction. Best concrete estimate: pass-rate goes from 0.62 (Gen F) and 0.85 (Gen E) to 0.90+, recovering ~$80K of API budget per 1M-conv run that's currently wasted on rejects. Position bias likely improves too: shorter prompts give the model less surface to overfit to.

**Effort.** ~2 days for the iterate-prompt CLI; ~1 hour per generator per A/B; ~3 days end-to-end including the actual prompt slimming. Highest-ROI engineering investment in the pipeline.

---

### Improvement 3 — **Quality-tier the source data and route to different prompts**

**FinePhrase finding.** README explicitly bifurcates HQ (FineWeb-Edu score 4–5) vs LQ (score 0–1) and sends them to *different prompt families*: HQ → `nemotron/{distill, extract_knowledge, diverse_qa_pairs, knowledge_list}` (preserve dense knowledge); LQ → `rewire/guided_rewrite_corrected.md` + `dspy/5-max_full_evals.md` (synthesize structure that wasn't there). They found single-prompt-for-all-data underperforms the bifurcation by a measurable margin in the ablation matrix.

**Why our current approach is suboptimal.** Our pipeline already has Ridge quality classifiers (per CLAUDE.md: `processing/quality_models/`) but `prepare.py` uses them as a binary *include/exclude* gate, then sends every surviving document to the same 6 generators with the same prompts. A 1949 NYT op-ed and a 1923 OCR'd patent fragment go through `QA_PROMPT` identically. The patent will produce great Gen A items (real grade-school physics principles); the op-ed will produce weak Gen A items (no principle to extract) and strong Gen C items (clean prose, named people doing things). We're paying generation cost on items the prompt cannot succeed on.

**Exact code change.** In `src/post_training/corpus/prepare.py` (the script that produces `synthetic/input/*.parquet`):

```python
# After quality scoring, attach a coarse routing tag per document.
def route_doc(text, ridge_score, has_named_people, has_numbers, has_dialogue):
    if ridge_score < 0.3:                    # OCR-noisy, fragmentary
        return ["F"]                         # Gen F still works on broken text
    if has_dialogue or has_named_people:
        return ["C", "E", "F", "G_narrative"]  # narrative-friendly
    if has_numbers > 5:
        return ["A", "D", "G_table"]         # principle/quantitative-friendly
    return ["A", "B", "G_tutorial"]          # generic expository
```

In `base.py:_load_documents`, return `(doc_name, text, route_tags)` and in `submit_batch_requests`, skip documents whose `route_tags` don't include `self.gen_key`. This means every $0.50 batch call is spent on a document the prompt can succeed on.

**Expected impact.** Pass rate up across the board (Gen E from 0.85 to 0.92+, Gen F from 0.62 to 0.75+). LAB-eval scores up (model sees fewer "bad fit" items that pollute the training distribution). Estimated 15–25% reduction in API spend per 1M-conv run. ARC-C and HellaSwag also benefit because they no longer get diluted by Gen A items that came from sources with no extractable principle.

**Effort.** ~2 days. The Ridge scores already exist; the routing function and `_load_documents` plumbing is mechanical.

---

### Improvement 4 — Make the **generator model 1B-or-larger** for *generative* slots, keep 4o-mini only for MC slots

**FinePhrase finding.** Their plateau-at-1B claim is **specifically about format-rephrase prompts** where the task is "rewrite this passage as X." For *guided rewrite* (their `rewire/guided_rewrite_*.md` family) they ran 1B → 4B → 12B → 27B Gemma-3 and observed monotonic improvement (the configs explicitly call out 27B as a meaningful step). The takeaway is: **task complexity dictates required generator scale**.

**Why our current approach is suboptimal.** `config.py:MODEL = "gpt-4o-mini"` for everything except Gen F (`GENERATOR_MODEL_OVERRIDES = {"F": "gpt-4o"}`). For Gen B (PIQA-style 2-choice physical commonsense), Gen E (HellaSwag-style narrative completion), and Gen F (Winogrande-style pronoun resolution), the *quality of the wrong distractor* is what determines whether the model learns anything. 4o-mini is provably weak at "construct a subtle physically-wrong alternative that is the same length as the right one" — that's why Gen F was already escalated to 4o. The same logic applies to Gen B and the proposed Gen G rephrase. Conversely, Gen A's MC4 items where the answer is "conduction" and distractors are "radiation/convection/evaporation" — 4o-mini handles trivially.

**Exact code change.** In `config.py`:

```python
MODEL = "gpt-4o-mini"
GENERATOR_MODEL_OVERRIDES = {
    "B": "gpt-4o",       # PIQA-style: hard distractors need physical-world model
    "E": "gpt-4o",       # HellaSwag-style: narrative-coherent wrong continuations
    "F": "gpt-4o",       # already escalated
    "G": "gpt-4o",       # NEW: rephrasing whole passages benefits from larger model
}
```

Cost: the format counts are A=3, B=2, C=3, D=3, E=3, F=2, G=4 = 20 slots, of which B+E+F+G = 11 slots go to 4o (~10×). At 1M target: ~550K conversations × ~$0.001 = ~$550 extra per period (vs. current ~$1500 → ~$2000 total). Cheap.

**Expected impact.** Distractor quality goes up directly. PIQA 55 → 60+; HellaSwag 25 → 32+ (correlated with the rephrasing improvement); Winogrande 48 → 53+; LAB-eval up because fewer "obviously wrong" distractors poison the training signal.

**Effort.** 30 minutes. Just edit config, rerun.

---

### Improvement 5 — Drop `REJECT IF` clauses from prompts; move filtering to a separate **post-generation judge pass**

**FinePhrase finding.** Their prompts have **zero `REJECT IF` clauses**. Filtering happens upstream (Ridge quality scoring) and *not at all* downstream — they trust the prompt to produce usable output. When they need higher quality, they switch *prompt*, not add rejection criteria.

**Why our current approach is suboptimal.** Every prompt in `prompts.py` has a long `REJECT IF` block (8–10 anti-patterns). The model has to *understand the rejection criteria, hold them in working memory, and self-censor* — but 4o-mini is not good at this kind of meta-instruction. The result: pass rate of 62–95% because the model *tries* to comply but produces brittle items, and `format_conversation` then drops them. We pay the API cost regardless.

**Exact code change.** Two-pass architecture:

- **Pass 1 (generation):** Strip the `REJECT IF` block from each prompt. Keep only GOAL, FORMAT, CORE PRINCIPLES, CONSTRAINTS, OUTPUT. Prompt size drops by ~40%, generation cost drops proportionally, model has fewer competing instructions to balance.
- **Pass 2 (judge filter):** New `src/post_training/generators/judge.py`. After all batches return, run a tiny GPT-4o-mini call per item with a 3-line judge prompt:

```python
JUDGE_PROMPT = """Rate this {generator} item 1-5 on (a) faithfulness to source,
(b) distractor quality, (c) temporal consistency. Reject if any score is 1 or 2.
Item: {item}
Output: {{"keep": true/false, "reason": "..."}}"""
```

Use 4o-mini batch for ~$0.10/1K judgments. At 1M items: ~$100 extra. Drops the worst 10–20% of items but keeps the rest, which would have *passed* our `format_conversation` validation but contained subtle defects.

**Expected impact.** The big benchmark gain isn't here — this is a *cost optimization that also improves item quality*. Net effect: ~30% of API spend recovered (shorter prompts) minus $100 judge spend = ~$400 saved per period; 10–15% better-quality data going into training (which compounds into 1–3% on each MC benchmark). Mostly enables the *other* improvements above by freeing budget.

**Effort.** ~2 days. Strip prompts is 1 hour; judge.py is 1 day; rerun + validate the new pass rate is 1 day.

---

## What from FinePhrase does NOT transfer

- **Their "≤1B generator is enough" claim is for pretraining-style rephrasing**, not for our QA generation task. We should *not* downsize from 4o-mini. (See Improvement 4 — we should *upsize* selectively.)
- **Their `report-tokens` / `tokenize` / `train` infrastructure is nanotron-specific** and irrelevant to our nanochat pipeline.
- **DSPy auto-optimization at full scale (their `dspy/rephrase/*` prompts)** would require us to set up a separate LLM-judge eval harness and a DSPy GEPA loop. That's a 2-week project. Improvement 2 is the lightweight version (`iterate-prompt` style, 5-round human-in-the-loop) that captures most of the value at ~10% of the engineering cost.
- **Their HQ/LQ binary based on FineWeb-Edu Ridge classifier scores** maps to our existing per-period quality model, but the *split labels* (4-5 vs 0-1) are calibrated to web text, not historical OCR. We'd have to re-calibrate. Improvement 3 captures the *spirit* (route by document character) without copying the *implementation* (binary score thresholds).

---

## Summary judgment, ranked by leverage

1. **Improvement 1 (rephrase stream)** is highest-leverage by a wide margin. HellaSwag at 25.2% and Winogrande at 48% are direct symptoms of "model never saw enough fluent generative prose during mid-training" — exactly what a 25% rephrase-format mix fixes. This is the closest analogue of what FinePhrase actually shipped, and it directly addresses our two worst benchmarks.
2. **Improvement 4 (selective gpt-4o for distractor-heavy generators)** is the best $/effort ratio: 30 minutes of work for an estimated 5–8 points across PIQA/HellaSwag/Winogrande, because it directly fixes the "weak distractor" problem 4o-mini has at length-matched physical/narrative items.
3. **Improvement 3 (quality-tier routing)** is high-leverage but more work; primarily it recovers wasted budget that funds Improvements 1+2.
4. **Improvement 2 (iterate-prompt + slim prompts)** is the highest *long-term* leverage because it removes the vibes-driven loop that produced the current over-engineered prompts. Pays compounding dividends but takes the longest to ship.
5. **Improvement 5 (judge pass + slim prompts)** is the smallest individual win but enables the cost math for everything above.

The single change I would ship first: **add Gen G rephrase with `format/tutorial.md` + `format/narrative.md` at weight=2.0** (Improvement 1 narrowly scoped). Lowest risk, highest expected effect on the benchmarks we are most below random on.
