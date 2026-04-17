# FinePhrase Appendix B vs. Our `REPHRASE_PROMPT` — Review

**Source:** `https://github.com/huggingface/finephrase/tree/main/prompts/format/`
(Repo prompts are the canonical templates referenced in Appendix B "Prompt Templates", subsection B.4 "Structured Prompts" of arXiv 2604.13977 — *"How Can We Synthesize High-Quality Pretraining Data? A Systematic Study of Prompt Design, Generator Model, and Source Data."*)

The repo also contains a top-level `prompts/rephrase.md` used as the "generic" rephrase baseline. Inspection of `configs/rephrasing.yaml`, the `rephrase` CLI, and the README (`iterate-prompt --prompt format/faq.md` / `rephrase --prompt …`) confirms each `format/*.md` file is invoked **independently as a separate generation call**.

---

## 1. Verbatim Text of Each Format Prompt

### `format/tutorial.md`
```
Rewrite the document as a clear, step-by-step tutorial or instructional guide. Use numbered steps or bullet points where appropriate to enhance clarity. Preserve all essential information while ensuring the style feels didactic and easy to follow. Output only the tutorial, nothing else.

Document:
[TEXT]
```

### `format/faq.md`
```
Rewrite the document as a comprehensive FAQ (Frequently Asked Questions). Extract or infer the key questions a reader would have about this topic, then provide clear, direct answers. Order questions logically—from foundational to advanced, or by topic area. Each answer should be self-contained and understandable without reference to other answers. Ensure the FAQ works as a standalone document. Output only the FAQ, nothing else.

Document:
[TEXT]
```

### `format/narrative.md`
```
Rewrite the document as a clear narrative that emphasizes the temporal sequence and causal relationships between events or steps. Reorganize the content to show how actions, events, or situations naturally flow from one to the next, making cause-and-effect relationships explicit. If describing a process or activity, show the logical progression of steps and explain why each step follows from the previous one. Output only the narrative, nothing else.

Document:
[TEXT]
```

### `format/explanation.md`
```
Rewrite the document to provide clear scientific or logical explanations for concepts, phenomena, or processes mentioned in the text. Make implicit reasoning explicit by explaining why things work the way they do, what principles or mechanisms are at play, and how different factors relate to each other. Focus on building understanding through causal explanations rather than just describing facts. Output only the explanatory text, nothing else.

Document:
[TEXT]
```

### `format/math.md` (bonus, for context)
```
Rewrite the document to create a mathematical word problem based on the numerical data or relationships in the text. Provide a step-by-step solution that shows the calculation process clearly. Create a problem that requires multi-step reasoning and basic arithmetic operations. It should include the question followed by a detailed solution showing each calculation step. Output only the problem and solution, nothing else.

Document:
[TEXT]
```

### `prompts/rephrase.md` (top-level baseline — referenced by Section 2.2)
```
Rephrase the low-quality web document below into higher quality, more educational content. Make sure the rephrased document can stand on its own; do NOT reference the input text. Output only the rephrased text, nothing else.

Document:
[TEXT]

Here's a rephrased version of the provided text, aiming for a higher quality and more educational content, without referencing the original input:
```

The other format files (`article.md`, `commentary.md`, `discussion.md`, `table.md`) follow the same one-instruction-per-file pattern.

---

## 2. Structural Pattern

| Property                              | FinePhrase Appendix B                                                                         | Our `REPHRASE_PROMPT`                                          |
| ------------------------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Source-text + instruction?            | Yes — instruction first, then `Document: [TEXT]`                                              | Yes — instructions first, then `Source passage: {text}` last   |
| Topic extraction first?               | **No** — never. No topic field anywhere in repo.                                              | **Yes** — emits a `"topic"` field                              |
| Output format                         | **Plain text** (markdown allowed inside; e.g. `table.md` uses markdown table syntax)          | **JSON** with 5 fields                                         |
| One call per format vs. combined call | **Separate call per format** (one prompt file → one rewrite per call)                         | **One combined call** producing all 4 rewrites                 |
| Length targets                        | **None.** No word/token counts. Paper notes "wider length variance (4 to 4,000 tokens)" obs. | 200–400 words per format                                       |
| Forbids verbatim copying              | Soft — "stand on its own; do NOT reference the input text" (only in top-level `rephrase.md`); format prompts say only "Output only the X, nothing else" | Yes — "Do NOT include source verbatim"                         |
| Period / temporal constraints         | None (FinePhrase rephrases modern web data)                                                   | "Period-appropriate vocabulary", "No knowledge from after end_year" |
| Meta-reference guard                  | Implicit ("stand on its own", only in baseline)                                               | Explicit — 'no "the passage states"'                           |
| Preserve facts                        | Tutorial: "Preserve all essential information"; others implicit                               | Explicit — "Preserve essential factual information"            |
| Style guidance                        | Per-format style cue (e.g. narrative emphasizes temporal/causal; explanation emphasizes mechanisms) | Generic — just names the format                                |

---

## 3. One Combined Prompt vs. Separate Prompts?

**FinePhrase: separate prompts, separate calls.** Each format is a standalone `.md` file invoked with `--prompt format/<X>.md`. The CLI (`iterate-prompt`, `rephrase`) takes exactly one prompt at a time. The paper's experimental matrix (Section 3) sweeps prompts as the independent variable.

**We use:** one combined call producing a JSON object with all four rewrites at once.

---

## 4. Material Differences and Recommendations

### Diff 1 — Combined-JSON vs. separate calls **(MOST IMPORTANT)**
- **What:** We bundle all 4 formats into one JSON; FinePhrase issues 4 independent calls.
- **Matters?** **Yes, significantly.** Bundling forces the model to ration its 200-400 word budget across 4 outputs in one context, encouraging shallow rewrites and cross-contamination of style (the "tutorial" half-borrows phrasing from the "narrative" sibling because they share a context window). It also creates one large JSON-validation failure mode: if any field is malformed the whole record is dropped. FinePhrase's per-format calls let each rewrite use the full attention budget on a single goal — exactly the design the paper benchmarks.
- **Recommendation:** **Switch to per-format calls** (or accept that we are deliberately deviating to save 4× API cost). If cost is the driver, document it. Otherwise this is the change with the largest expected quality delta.

### Diff 2 — Topic extraction
- **What:** We ask for a `"topic"` (5-10 word noun phrase); FinePhrase never extracts a topic.
- **Matters?** Mild. A topic field doesn't appear in any FinePhrase prompt or output schema. It's a useful-feeling addition but it's *our* invention, not theirs.
- **Recommendation:** **Drop `topic`** if the goal is faithful FinePhrase reproduction. Keep it only if downstream code consumes it.

### Diff 3 — JSON output vs. plain text
- **What:** We require strict JSON; FinePhrase says "Output only the X, nothing else" → plain text.
- **Matters?** Yes for parser robustness — JSON validation fails harder than text trimming. But JSON gives clean field separation.
- **Recommendation:** If keeping the combined call, keep JSON. If switching to per-format calls, switch to plain text (matches paper).

### Diff 4 — Word-count targets (200-400)
- **What:** We pin 200-400 words per format; FinePhrase has no length target.
- **Matters?** Yes. Paper explicitly observes wide length variance and treats it as fine. Our 200-400 cap may truncate dense source passages or pad sparse ones, both reducing fidelity.
- **Recommendation:** **Drop the word-count caps.** Or relax to a single soft hint (e.g. "match the level of detail of the source"). Strict caps are not in Appendix B.

### Diff 5 — Period / temporal vocabulary constraint
- **What:** We add "Period-appropriate vocabulary" and "No knowledge from after {end_year}".
- **Matters?** Yes — but in our favor. This is our domain (historical corpora) and the FinePhrase corpus is not historical, so they have no reason to include it. The repo's other generators (A, B, D, E) already standardized on a TEMPORAL CONSTRAINT (per CLAUDE.md). Keeping this is consistent with the rest of the pipeline.
- **Recommendation:** **Keep** the temporal constraint. Document it as a hist-LLM-specific addendum, not part of the FinePhrase paper's prompt.

### Diff 6 — Verbatim / meta-reference guards
- **What:** We forbid "source verbatim" and meta-phrases. FinePhrase only says "stand on its own; do NOT reference the input text" and only in the top-level `rephrase.md`, not in the format prompts.
- **Matters?** Slightly. The format prompts trust the model. Our guards are stricter; probably fine.
- **Recommendation:** **Keep**, but understand they are our additions, not Appendix B text.

### Diff 7 — Per-format style cues
- **What:** FinePhrase's `narrative.md` specifically demands "temporal sequence and causal relationships"; `explanation.md` demands "scientific or logical explanations… mechanisms"; `faq.md` demands "self-contained" answers ordered foundational→advanced. Our prompt just says "rewrite as flowing prose narrative" / "multi-paragraph expository analysis" / "Q&A pairs (3-5 exchanges)".
- **Matters?** Yes. The FinePhrase wording is more directive and produces more differentiated outputs. Our wording is generic and risks all four rewrites converging to similar prose.
- **Recommendation:** **Adopt FinePhrase's per-format directive language verbatim** for tutorial / faq / narrative / explanation, even if we keep the combined-call structure.

---

## 5. Verdict

**We are *not* using the FinePhrase prompts correctly.** We adapted the *idea* (4 structured pedagogical formats) but rewrote the prompts ourselves with three significant deviations: (a) one combined call instead of four separate calls, (b) generic format names instead of the paper's directive style cues, and (c) hard 200-400 word caps the paper does not impose. Optional/minor: a fabricated `topic` field and JSON output instead of plain text.

### Specific text changes recommended

**Option A — Maximum fidelity (recommended for a paper-quality reproduction):**
Replace `REPHRASE_PROMPT` with four separate prompts that wrap each FinePhrase format file plus our temporal preamble. E.g. for tutorial:

```
You are given a passage from a historical document published between {start_year} and {end_year}. Use only period-appropriate vocabulary and no knowledge from after {end_year}.

Rewrite the document as a clear, step-by-step tutorial or instructional guide. Use numbered steps or bullet points where appropriate to enhance clarity. Preserve all essential information while ensuring the style feels didactic and easy to follow. Output only the tutorial, nothing else.

Document:
{text}
```

…and analogously for `faq`, `narrative`, `explanation`. Issue 4 separate API calls per chunk. Plain-text output. Drop `topic`. Drop word-count caps.

**Option B — Keep combined call, raise fidelity:**
Keep the JSON wrapper but replace each field's instruction with the verbatim FinePhrase directive sentences (the body of each format file, minus the `Document: [TEXT]` footer). Drop `topic`. Drop word-count caps or downgrade to "match the source's level of detail." Keep the temporal preamble.

Either option is materially closer to Appendix B than the current draft. Option A is what the paper actually evaluates; Option B is a defensible cost-saving compromise.
