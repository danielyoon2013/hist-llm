# Evaluation Protocol Comparison: Is Our Benchmark Table Apples-to-Apples?

**Date**: 2026-04-16
**Scope**: ARC-Challenge, HellaSwag, Winogrande, PIQA, GSM for LLaMA-1, Pythia, TinyLlama, OPT, BLOOM, GPT-2 XL, nanochat d20, and ours.

---

## TL;DR Verdict

**The comparison as currently framed is apples-to-oranges for 4 of 5 benchmarks.** Our pipeline uses forced-letter MC (all options shown; argmax over letter-token logits only). The baselines almost uniformly use lm-evaluation-harness's `multiple_choice` output_type, which is **loglikelihood-ranking over full candidate answer text with each candidate scored in isolation** (the "separation" setup). These produce materially different numbers on the same questions, by as much as 20+ points on ARC (per Balepur et al. 2024, arXiv:2412.17758).

The comparison is **apples-to-apples only against nanochat d20** (which uses the same code we use) — and even there, the exact 28.1% figure is not confirmed from Karpathy's own reports (confirmed d20 numbers from the miniseries discussions are ~0.124 base / ~0.306 after SFT).

---

## 1. Per-Source Evaluation Protocol Table

| Model / Paper | ARC-Challenge | HellaSwag | Winogrande | PIQA | GSM |
|---|---|---|---|---|---|
| **Ours (nanochat/tasks/arc.py etc.)** | 0-shot forced-letter MC; argmax over letter-token logits at answer position | same | same | same (2 letters A/B) | **GSM-MC** 0-shot forced-letter MC (4-way, Zhang et al. 2024 distractors) |
| **LLaMA-1** (Touvron 2023) | 0-shot loglikelihood **separation** (each candidate scored alone, options NOT shown together) | 0-shot LL rank | 0-shot LL rank | 0-shot LL rank | **GSM8K** generative, maj1@1 or 8-shot |
| **Pythia** (Biderman 2023, App. G) | lm-eval-harness `multiple_choice`; 0-shot; reports **both acc and acc_norm** (usually acc_norm) | harness MC, acc_norm | harness MC, acc | harness MC, acc_norm | N/A in core table |
| **TinyLlama v1.1** (Zhang 2024) | lm-eval-harness, 0-shot, GPT4All suite, **acc_norm by default** (per EVAL.md) | acc_norm | acc | acc_norm | Generative (InstructEval suite) |
| **OPT / BLOOM / GPT-2 XL** (as quoted by Pythia and TinyLlama) | harness multiple_choice, acc_norm (consistent with the table they appear in) | same | same | same | N/A |
| **nanochat d20** (Karpathy) | Same code as ours — forced-letter MC argmax over letter logits | same | same | same | GSM8K generative in nanochat |
| **lm-eval-harness** `arc_challenge`/`hellaswag`/`piqa`/`winogrande` | `output_type: multiple_choice` — computes loglikelihood of each `doc_to_choice` string as continuation of `doc_to_text` independently; argmax → acc; argmax of (LL / byte-length) → acc_norm | same | same (special preprocess) | same | `gsm8k` is `generate_until`, 5-shot, `exact_match` on regex-parsed numeric answer |

### Why this distinction matters (verified)

- **EleutherAI's own blog** (`blog.eleuther.ai/multiple-choice-normalization`) confirms: the harness "multiple_choice" type scores each candidate independently as a continuation. `acc` = argmax ΣlogP; `acc_norm` = argmax ΣlogP / byte-length.
- **Balepur et al. 2024** ("ARC Challenge Is Not That Challenging", arXiv:2412.17758) Table 1: LLaMA-2 70B scores **57.4% in "separation" vs 79.6% in "options"** — a 22-point artifact purely from evaluation format.
- **ARC.py / hellaswag.py / piqa.py / winogrande.py / gsm_mc.py** in our repo all render prompts with `render_mc()` showing all options + "Respond only with the letter of the correct answer", and `scripts/chat_eval.py` line 223 does `focus_logits.argmax(dim=-1)` over **only the letter-token IDs** at the final answer position. This is forced-letter MC.

---

## 2. Per-Benchmark Verdict

| Benchmark | Ours | Baseline protocol | Verdict |
|---|---|---|---|
| ARC-C 38.2% | forced-letter MC | LLaMA, Pythia, TinyLlama all use LL-separation / harness MC (acc_norm) | **Invalid.** Different tasks effectively. Format swing can be 20+ pts. |
| HellaSwag 25.2% | forced-letter MC | harness MC acc_norm (HellaSwag is famously acc_norm-preferred; lengths vary wildly) | **Invalid.** Our 25.2% is near chance on our format; the 76.1% LLaMA number is acc_norm on an entirely different scoring procedure. |
| Winogrande 48.1% | forced-letter MC | harness MC acc (Winogrande uses cloze, options are single words; fill-in-the-blank) | **Invalid.** Our number is below random-ish for 2-way; baselines compute LL of option1 vs option2 filling the `_`. |
| PIQA 55.0% | forced-letter MC | harness MC acc_norm | **Invalid.** Baselines rank sol1 vs sol2 by LL. |
| GSM 25.7% (GSM-MC) | forced-letter MC, 4-way with model-sourced distractors | LLaMA GSM8K 11% is **generative maj1@1** | **Invalid** — fundamentally different tasks. GSM-MC random baseline is 25%, so our 25.7% is essentially chance. LLaMA's 11% is on open-ended generation of the exact numeric answer. |

The only partially-valid comparison is **nanochat d20** (same eval code): but the 28.1% value isn't in any nanochat page/discussion I could verify — the nanochat miniseries v1 discussion (#420) reports d20 ARC-C ~0.1240 base, ~0.3063 post-SFT. **Flag: 28.1% cannot be corroborated from Karpathy's own publicly posted eval outputs.**

---

## 3. Correction Recommendations

### Option A — Re-run our eval under the standard protocol (recommended)
Install `lm-evaluation-harness` and evaluate our model on `arc_challenge` (acc_norm), `hellaswag` (acc_norm), `piqa` (acc_norm), `winogrande` (acc), `gsm8k` (exact_match, 5-shot). Report those numbers side-by-side with the published LLaMA/Pythia/TinyLlama numbers. This is the only way to claim apples-to-apples.

### Option B — Keep forced-letter MC but re-label and remove incompatible rows
Keep our numbers but:
- Title the column **"Forced-letter MC (all options shown)"**.
- **Remove** the LLaMA/Pythia/TinyLlama/OPT/BLOOM rows for ARC/HellaSwag/Winogrande/PIQA (they didn't run this protocol; cross-row comparison is meaningless).
- Keep only **nanochat d20** (verified identical protocol) as a baseline.
- For GSM: drop the LLaMA GSM8K row entirely, or move to a separate table labeled "generative GSM8K, provided for context — not comparable to GSM-MC."

### Option C — Caveat-heavy hybrid
Keep the table but add a bold note: "Our column uses forced-letter MC (categorical argmax over letter-token logits with all options shown). Baselines use lm-evaluation-harness loglikelihood-ranking (acc_norm for ARC-C/HellaSwag/PIQA, acc for Winogrande). Per Balepur et al. 2024, these can differ by 20+ points on the same model. Our numbers should be interpreted as a lower bound; direct numeric comparison is not meaningful." This is academically defensible but reviewers will likely flag it as insufficient.

### Recommendation
**Do Option A.** The rerun is cheap (<2 hrs on one GPU for 7B-scale) and the comparison table is the #1 reviewer target.

---

## 4. Biggest Single Correction Needed

**GSM-MC 25.7% vs LLaMA GSM8K 11%** is the most indefensible cell. They are not the same benchmark. GSM-MC random baseline is 25% (4-way); our 25.7% means our model cannot do the task. LLaMA-7B at 11% on open-ended generation is a stronger result than our 25.7% on MC. Reporting them as if ours wins is actively misleading — this cell must be removed or reframed.

The second biggest: **HellaSwag 25.2% vs LLaMA 76.1%**. 25.2% is ~chance for 4-way; LLaMA 76.1% is harness acc_norm. Keeping this in the paper without protocol disclosure will get it rejected.

---

## 5. nanochat d20 Verification Status

- **Eval code**: verified identical to ours (same repo).
- **28.1% ARC-C figure**: **NOT verified** from primary sources. Karpathy's miniseries v1 discussion reports d20 ~0.124 base / ~0.306 post-SFT. The 28.1% is plausible as a mid-training or specific-config checkpoint, but cite the exact source (commit, discussion post, or run log) or drop the number. If the user sourced 28.1% from a run log, include that provenance in the paper.

---

## Sources

- `C:\Users\danielyoon\Dropbox\hist_LLM\nanochat\tasks\arc.py`, `hellaswag.py`, `piqa.py`, `winogrande.py`, `gsm_mc.py`, `common.py` (render_mc)
- `C:\Users\danielyoon\Dropbox\hist_LLM\nanochat\scripts\chat_eval.py` (lines 95-228, `run_categorical_eval`, `_get_focus_logits`)
- lm-evaluation-harness tasks: `arc/arc_easy.yaml`, `hellaswag/hellaswag.yaml`, `piqa/piqa.yaml`, `winogrande/default.yaml`, `gsm8k/gsm8k.yaml` on GitHub main
- EleutherAI blog: https://blog.eleuther.ai/multiple-choice-normalization/
- Balepur et al. 2024, "ARC Challenge Is Not That Challenging", arXiv:2412.17758
- TinyLlama EVAL.md: https://github.com/jzhang38/TinyLlama/blob/main/EVAL.md
- Pythia paper Appendix G (Biderman et al. 2023, arXiv:2304.01373) — confirmed uses lm-eval-harness
- nanochat miniseries v1 discussion (GitHub Discussions #420)
