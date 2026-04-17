# FinePhrase vs. Our Rephrase Outputs — Comparison

## 1. Source Confirmation

- **Dataset**: `HuggingFaceFW/finephrase`
- **URL**: https://huggingface.co/datasets/HuggingFaceFW/finephrase
- **Revision**: `78cf4a5ed0099214979c094c963e699c19163838` (lastModified 2026-03-31)
- **Generator model**: `HuggingFaceTB/SmolLM2-1.7B-Instruct` (temperature=1.0, max_tokens=2048)
- **Source corpus**: `HuggingFaceFW/fineweb-edu` (sample-350BT)
- **Published configs**: **only 4** — `faq`, `math`, `table`, `tutorial` (≈1.35B samples total)
- **Citation**: Niklaus et al. 2026, "The Synthetic Data Playbook: Generating Trillions of the Finest Tokens"

### CRITICAL FINDING: Format mismatch

The published FinePhrase dataset does **NOT** include `narrative` or `explanation` configs. FinePhrase only released `faq / math / table / tutorial`. Our `narrative` and `explanation` formats have **no direct FinePhrase analog** to compare against — they appear to be local inventions (possibly inspired by the paper's prompt families that were discussed but not shipped).

Published per-format mean completion tokens:
- `tutorial`: 436.49 tokens
- `faq`: 436.96 tokens
- `math`: 290.51 tokens
- `table`: 272.94 tokens

Our files are 3–5k chars each (≈500–700 words) per sample, i.e. **~600–900 tokens**, roughly 1.5–2× the FinePhrase mean.

## 2. Prompt comparison

### FinePhrase published prompts (verbatim, from README)

**tutorial**: "Rewrite the document as a clear, step-by-step tutorial or instructional guide. Use numbered steps or bullet points where appropriate to enhance clarity. Preserve all essential information while ensuring the style feels didactic and easy to follow. Output only the tutorial, nothing else."

**faq**: "Rewrite the document as a comprehensive FAQ (Frequently Asked Questions). Extract or infer the key questions a reader would have about this topic, then provide clear, direct answers. Order questions logically, from foundational to advanced, or by topic area. Each answer should be self-contained and understandable without reference to other answers. Ensure the FAQ works as a standalone document. Output only the FAQ, nothing else."

**math**: "Rewrite the document to create a mathematical word problem based on the numerical data or relationships in the text. Provide a step-by-step solution…"

**table**: "Rewrite the document as a structured table… generate one question-answer pair based on the table."

## 3. Side-by-side examples

### TUTORIAL

**FinePhrase (row 0, from "apology guide" source)**:
```
Step 1: Stop and reflect on the situation.
Step 2: Recognize the impact of your words and actions on others.
Step 3: Choose sincere apology as a first line of action.
Step 4: Express remorse for past actions using:
 - "What I said the other day was really insensitive of me..."
Step 5: Acknowledge your intention to change and future actions.
...
Step 9: Move forward with working on your good intentions and mend the situation.
```
(≈155 words; plain "Step N:" headers; no markdown bold; short declarative; source-faithful; no meta-framing.)

**Ours (tutorial sample 0, ≈517 words, 3561 chars)**:
```
1. **Understand the Nature of Government Power**: Recognize that in any
government, power that operates independently is also absolute...
2. **Trust in the People**: Acknowledge that true independence resides
with the populace...
...
11. **Uphold the Independence of Government Powers**: The sustainability
of government on its foundational principles...
```
(≈517 words; markdown bold headers; ornate didactic register; 11 numbered steps.)

**Verdict**: **Broadly similar but diverges in length and register.** Both use numbered steps. Ours are ~3.3× longer, use markdown bold section headers (FinePhrase uses plain "Step N:"), and have a more ornate academic register vs. FinePhrase's plain instructional voice. Structural fidelity matches.

### FAQ

**FinePhrase (row_idx 28–32, sampled)**:
```
Q: How do maps of food deserts help public health?
A: Maps of food deserts can help public health by providing an accurate,
visual representation of the amount and accessibility of fresh produce
in a given area...
```
Or (row 28): a single flowing-prose "The New World refers to the Western Hemisphere…" — the SmolLM2 model often produced **just one or two Q&A pairs** rather than a comprehensive FAQ.

Average published length ≈440 tokens (~330 words), but first-rows sampled outputs averaged only ~300–560 chars (~50–90 words) — suggesting the real distribution is highly bimodal.

**Ours (faq sample 0, 550 words, 3742 chars)**:
```
**Frequently Asked Questions (FAQ) on the Constitution and the Bill of Rights**

**1. What were the main proceedings of the Constitutional Convention?**
The Constitutional Convention was primarily concerned with establishing
a government that served the interests of property owners...

**2. Who were the prominent figures at the Constitutional Convention?**
The Convention comprised 55 delegates, predominantly lawyers...
[7+ numbered Q&A pairs with bold markdown]
```

**Verdict**: **Broadly similar in structure but diverges in formatting and title.** Both use Q&A. Ours ALWAYS opens with a bold title ("**Frequently Asked Questions (FAQ) on…**") which FinePhrase does NOT do (FinePhrase typically starts directly with a Q or with the answer). Ours uses bold markdown headers and numbered questions; FinePhrase uses plain "Q:/A:" or just prose. Ours is ~50% longer on average and much more consistently comprehensive (FinePhrase often produces only 1 Q&A pair).

### NARRATIVE

**FinePhrase**: **No published narrative config exists** on HuggingFaceFW/finephrase. Cannot be compared to a paper baseline.

**Ours (narrative sample 0, 732 words)**:
```
In the realm of governance, it is crucial to acknowledge the inherent
independence of the people... The essence of true independence resides
not within any single institution but among the populace collectively.
This principle is grounded in my interpretation of the Constitution...
[flowing first-person prose, literary register]
```

**Verdict**: **No-op — no FinePhrase baseline.** Style and length are internally consistent (literary/first-person) but there is no published FinePhrase "narrative" output to compare against. If the intent is to emulate a FinePhrase-like format, this format is fabricated and should be renamed or dropped.

### EXPLANATION

**FinePhrase**: **No published explanation config exists**. Cannot be compared.

**Ours (explanation sample 0, 559 words)**:
```
The narrative presents a profound exploration of the themes of love,
loss, and vengeance... Fire serves as a powerful metaphor throughout
the tale...
```
(Essayistic analysis register; third-person; themes/interpretation focus.)

**Verdict**: **No-op — no FinePhrase baseline.** The closest FinePhrase analog would be the `tutorial` prompt's "preserve all essential information while ensuring the style feels didactic", but there is no direct "explanation" family. Also a fabricated format.

## 4. Key stylistic divergences (faq + tutorial — where comparison is valid)

| Dimension | FinePhrase | Ours |
|-----------|-----------|------|
| Length | ~290–440 completion tokens (mean) | ~600–900 tokens |
| Opening | Starts immediately with Step 1 or Q1 | Starts with bold markdown title/header |
| Structural markers | Plain "Step 1:", "Q:/A:" | Bold markdown: `**1. Question?**`, `**Step N**:` |
| Register | Plain didactic, contemporary, direct | Ornate academic / literary, historical voice |
| Self-containedness | Occasionally refers to "the document" / "the text" | Occasionally says "in the case at hand" / "in this case" — minor meta-leakage |
| Question diversity | FinePhrase FAQ often has 1–3 pairs; tutorial 5–10 steps | Ours consistently 7–11 numbered items |
| Meta-references | The prompt forbids them; in practice some slip (row 1 echoes the source "News story originally written on October 15, 2000"). | We mostly avoid "the document" but have occasional "in the case at hand" / "in this specific case" in tutorial 3. |

**Meta-leakage check** (explicit violations of "standalone document" rule):
- Ours tutorial sample 3: "In this specific case, the court must determine…" — soft violation (context-dependent phrasing).
- Ours faq sample 3: "What is the context of this case?" — makes the FAQ about "this case" rather than about the general topic; significant violation of standalone-document principle.

## 5. Recommended prompt changes (top 2)

### Change 1: Remove the markdown-heavy title/bold-header bias
Our outputs uniformly open with `**Frequently Asked Questions (FAQ) on X**` or `**Guide to Understanding Y**` and use `**bold**` on every numbered item. FinePhrase outputs do neither — they use plain text markers. Our prompt is almost certainly instructing or few-shot-demonstrating bold headers.

**Fix**: Strip any "use a title", "use markdown bold headers", or bold-formatted examples from our rephrase prompts. Replace with FinePhrase's exact phrasing: *"Output only the FAQ, nothing else"* / *"Use numbered steps or bullet points where appropriate"*. Do NOT show bold examples.

### Change 2: Enforce topic-level framing, not document-level framing
Ours frequently says "this case", "in the case at hand", "the defendant", making the output read as an analysis of one specific source document rather than a standalone educational text. FinePhrase's faq prompt explicitly says *"Each answer should be self-contained and understandable without reference to other answers. Ensure the FAQ works as a standalone document."*

**Fix**: Add/strengthen in our prompt: *"Do not refer to 'this case', 'this document', 'this text', or any specific source. The output must read as if it were written independently about the general topic, not as an analysis of a particular document."* This is particularly important for `faq` where standalone-ness matters most.

## 6. Format-level recommendations

- `tutorial`: keep; reduce length target, drop bold headers. **Broadly similar → matches closely** with minor prompt fixes.
- `faq`: keep; drop the required bold title and topic-framing of "on X"; discourage "this case" framing. **Broadly similar → diverges in formatting/meta-references**.
- `narrative`: **FinePhrase has no such format.** If the goal is FinePhrase-alignment, this should be dropped or clearly marked as an extension. It is genuinely useful as a literary register but is not a FinePhrase format.
- `explanation`: **FinePhrase has no such format.** Same — either drop, rename, or document as an internal extension.

Additional missing FinePhrase formats we do NOT currently generate: `math` (word problem + solution) and `table` (markdown table + Q/A). Adding these would give us the full FinePhrase coverage at the cost of two new generators.

## Appendix: raw samples pulled

- Fetched via `https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW%2Ffinephrase&config={config}&split=train`
- Tutorial: 5 real outputs pulled at offsets 0 and 50 (row_idx 0–4, 50–54)
- FAQ: 34 rows from first-rows endpoint (subset with len > 400 analyzed: row_idx 27, 28, 31, 32)
- Our samples: 4 random from each `gen_g_rephrase_rephrase_{tutorial,faq,narrative,explanation}.jsonl` at `D:\hist_LLM\periods\1900_1949\posttraining_data\synthetic\by_generator\` (seed=42)
