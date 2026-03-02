# Eval System Implementation Plan

## Goal

Unified evaluation across all training stages (base / mid / SFT) using the same benchmarks at regular step intervals. Simple, well-organized, no over-engineering.

---

## 1. Eval Data Directory Structure

All eval data lives in a **single consolidated folder** (`eval_data/`). This folder is the only thing that needs to be transferred when deploying to a remote server (e.g., Lambda AI). One CLI arg (`--eval-data-dir`) controls everything — all paths are derived internally.

### Canonical layout (local development)

```
D:\hist_LLM\
├── eval_data/                              ← CONSOLIDATED eval folder (all benchmarks)
│   ├── arc_challenge/                      ← ARC-Challenge (HuggingFace, cached here)
│   ├── hellaswag/                          ← HellaSwag (HuggingFace, cached here)
│   ├── race/                               ← RACE Middle+High (HuggingFace, cached here)
│   ├── winogrande/                         ← Winogrande (HuggingFace, cached here)
│   ├── mathqa/                             ← MathQA (HuggingFace, cached here)
│   ├── gsm_mc.jsonl                        ← GSM-MC (downloaded from GitHub)
│   ├── internal_mc.jsonl                   ← Internal MC test split (copied per period)
│   └── lab_eval.jsonl                      ← LAB eval questions (copied per period)
│
└── periods/{period}/
    ├── posttraining_data/
    │   └── final/
    │       └── test/
    │           └── hist_synthetic_test.jsonl    ← source for internal_mc.jsonl
    │
    └── model/
        ├── eval/
        │   ├── eval_log.jsonl                  ← ALL eval results (append-only, tagged by stage/step)
        │   └── lab_eval.jsonl                  ← source for eval_data/lab_eval.jsonl
        │
        ├── base_checkpoints/d{N}/              ← saved ONCE at end of base train
        ├── mid_checkpoints/d{N}/               ← saved ONCE at end of mid train
        └── chatsft_checkpoints/d{N}/           ← saved ONCE at end of SFT
```

### Where each eval dataset lives

| Dataset | Filename in `eval_data/` | Source | How it gets there |
|---------|--------------------------|--------|-------------------|
| **ARC-Challenge** | `arc_challenge/` | HuggingFace | `load_dataset("allenai/ai2_arc", cache_dir=eval_data_dir/"arc_challenge")` |
| **HellaSwag** | `hellaswag/` | HuggingFace | `load_dataset("Rowan/hellaswag", cache_dir=eval_data_dir/"hellaswag")` |
| **RACE-Middle/High** | `race/` | HuggingFace | `load_dataset("ehovy/race", cache_dir=eval_data_dir/"race")` |
| **Winogrande** | `winogrande/` | HuggingFace | `load_dataset("allenai/winogrande", cache_dir=eval_data_dir/"winogrande")` |
| **MathQA** | `mathqa/` | HuggingFace | `load_dataset("allenai/math_qa", cache_dir=eval_data_dir/"mathqa")` |
| **GSM-MC** | `gsm_mc.jsonl` | GitHub | Download once via `scripts/download_gsm_mc.py` |
| **Internal MC** | `internal_mc.jsonl` | Local | Copied from `periods/{period}/.../hist_synthetic_test.jsonl` before each run |
| **LAB Eval** | `lab_eval.jsonl` | Local | Copied from `periods/{period}/model/eval/lab_eval.jsonl` before each run |

### Key decisions

- **One folder, one arg:** `--eval-data-dir` is the only path needed. All filenames are fixed conventions — no per-dataset CLI args.
- All HuggingFace datasets use explicit `cache_dir` pointing to `eval_data/` subdirectories (not scattered in `~/.cache/`).
- Period-specific files (Internal MC, LAB) are **copied into** `eval_data/` before each period's training run.
- If a file doesn't exist in `eval_data/`, that benchmark is skipped gracefully (no error).
- `eval_log.jsonl` is a single append-only file per period — one line per eval run, tagged by stage/step.
- Checkpoints saved ONLY at end of each stage (not at eval intervals).

### Lambda AI deployment

When deploying to a remote server (Lambda AI), transfer the consolidated `eval_data/` folder:

```bash
# 1. On local machine: prepare eval_data/ for the target period
cp periods/1950_1999/.../hist_synthetic_test.jsonl  eval_data/internal_mc.jsonl
cp periods/1950_1999/model/eval/lab_eval.jsonl      eval_data/lab_eval.jsonl

# 2. Transfer to Lambda (one folder)
rsync -avz eval_data/ lambda:/path/to/hist_LLM/eval_data/

# 3. On Lambda: just point to it
EVAL_DATA_DIR="/path/to/hist_LLM/eval_data"
```

**TODO:** Build a `prepare_eval_data.py` script that automates step 1 — takes a period name, copies the period-specific files into `eval_data/`, and verifies all expected files are present. This script should also be usable as a pre-flight check before transferring to Lambda.

---

## 2. Eval Task Suite (same for all stages)

```
EVAL_TASKS = {
    # External benchmarks (from HuggingFace)
    "ARC-Challenge":  categorical, MC-4,  1172 problems,
    "HellaSwag":      categorical, MC-4, 10042 problems,
    "RACE-Middle":    categorical, MC-4,  1436 problems,
    "RACE-High":      categorical, MC-4,  3451 problems,
    "Winogrande":     categorical, MC-2,  1267 problems,
    "MathQA":         categorical, MC-5,  2985 problems,   ← NEW
    "GSM-MC":         categorical, MC-4,  1319 problems,   ← NEW

    # Internal benchmarks (period-specific local files)
    "InternalMC":     categorical, MC-4, ~50K problems,    ← NEW (from hist_synthetic_test.jsonl)
    "LAB":            categorical, MC-4,  5000 problems,   ← existing
}
```

**At each eval checkpoint:** run ALL tasks, log all results to `eval_log.jsonl`.

**Max problems per task during training:** Cap at `--eval-max-problems` (default 1024) for speed. Full eval only at stage boundaries and standalone.

---

## 3. New nanochat Task Files Needed

### 3a. `tasks/mathqa.py` — MathQA (MC-5)

```
Source: allenai/math_qa (HuggingFace)
Pattern: Follow arc.py
Format: 5-choice (a,b,c,d,e), need to parse raw option string
         "a ) 24 , b ) 120 , c ) 625 , d ) 720 , e ) 1024"
Letters: ["A", "B", "C", "D", "E"]
Split: test (2,985 problems)
eval_type: categorical
```

### 3b. `tasks/gsm_mc.py` — GSM-MC (MC-4)

```
Source: Local JSONL (downloaded from github.com/Geralt-Targaryen/MC-Evaluation)
Pattern: Follow lab_eval.py (loads local JSONL)
Format: 4-choice, candidates[0] is always correct → shuffle at load time
Letters: ["A", "B", "C", "D"]
Split: 1,319 problems
eval_type: categorical
```

### 3c. `tasks/internal_mc.py` — Internal MC Test Split (MC-4)

```
Source: Local JSONL (hist_synthetic_test.jsonl)
Pattern: Follow lab_eval.py
Format: Already has {"messages": [...], "letters": ["A","B","C","D"]}
        Exactly matches LABEval format — near-identical implementation
Letters: ["A", "B", "C", "D"]
Split: ~50K problems per period
eval_type: categorical
```

---

## 4. Changes to `chat_eval.py`

### 4a. Expand `task_module_map`

Add the 3 new tasks to the task registry. All paths derived from `eval_data_dir`:

```python
def build_task_map(eval_data_dir=None):
    """Build the task registry. If eval_data_dir is set, all paths are derived from it."""
    task_module_map = {
        # Existing (add cache_dir for consolidated eval folder)
        'HellaSwag':     partial(HellaSwag, split="validation", cache_dir=...),
        'Winogrande':    partial(Winogrande, split="validation", cache_dir=...),
        'PIQA':          partial(PIQA, split="validation", cache_dir=...),
        'RACE-Middle':   partial(RACE, subset="middle", split="test", cache_dir=...),
        'RACE-High':     partial(RACE, subset="high", split="test", cache_dir=...),
        'SpellingBee':   partial(SpellingBee, size=256, split="test"),
        'DyckLanguage':  partial(DyckLanguage),

        # New (HF-based)
        'ARC-Challenge': partial(ARC, subset="ARC-Challenge", split="test", cache_dir=...),
        'MathQA':        partial(MathQA, split="test", cache_dir=...),
    }
    # Local-file tasks: only add if the file exists in eval_data_dir
    if eval_data_dir:
        gsm_mc_file = os.path.join(eval_data_dir, "gsm_mc.jsonl")
        internal_mc_file = os.path.join(eval_data_dir, "internal_mc.jsonl")
        lab_file = os.path.join(eval_data_dir, "lab_eval.jsonl")
        if os.path.exists(gsm_mc_file):
            task_module_map['GSM-MC'] = partial(GSMMC, data_file=gsm_mc_file)
        if os.path.exists(internal_mc_file):
            task_module_map['InternalMC'] = partial(InternalMC, data_file=internal_mc_file)
        if os.path.exists(lab_file):
            task_module_map['LAB'] = partial(LABEval, data_file=lab_file)
    return task_module_map
```

**Design:** One arg (`--eval-data-dir`) drives everything. HF tasks get `cache_dir` subdirectories. Local-file tasks are auto-discovered — if the file exists, the task is available; if not, it's skipped. No per-dataset CLI args.

### 4b. Add `run_full_eval()` convenience function

```python
FULL_EVAL_TASKS = [
    'ARC-Challenge', 'HellaSwag', 'RACE-Middle', 'RACE-High',
    'Winogrande', 'MathQA', 'GSM-MC', 'InternalMC', 'LAB',
]

def run_full_eval(model, tokenizer, engine, eval_data_dir,
                  batch_size=64, max_problems=1024):
    """Run the full eval suite. Returns dict of {task_name: {accuracy, passed, total}}."""
    task_map = build_task_map(eval_data_dir)
    results = {}
    for task_name in FULL_EVAL_TASKS:
        if task_name not in task_map:
            continue  # file not present, skip gracefully
        acc, passed, total = run_chat_eval(
            task_name, model, tokenizer, engine,
            batch_size=batch_size, max_problems=max_problems,
            eval_data_dir=eval_data_dir,
        )
        results[task_name] = {"accuracy": acc, "passed": passed, "total": total}
    return results
```

### 4c. Eval log format

Each line in `eval_log.jsonl`:
```json
{
    "timestamp": "2026-03-01T23:45:00",
    "stage": "mid",
    "step": 1500,
    "label": "mid_step_1500",
    "results": {
        "ARC-Challenge": {"accuracy": 0.35, "passed": 410, "total": 1172},
        "HellaSwag": {"accuracy": 0.28, "passed": 287, "total": 1024},
        ...
    }
}
```

---

## 5. Changes to Training Scripts

### Reference: Previous 1950-1999 Run Stats
- Base: 26,760 steps
- Mid: 525 steps
- SFT: 14,517 steps

### 5a. `base_train.py` — Replace CORE metric with unified eval

**Current behavior:**
- Val BPB every 250 steps
- CORE metric every 2000 steps (uses `base_eval.py`, separate eval system)
- Checkpoint at end only

**Changes:**
1. **Replace `--core-metric-every` with `--chat-eval-every`** (default: 2000)
   - Remove old CORE metric code (`evaluate_model()` calls from base_eval.py)
   - Replace with `run_full_eval()` from chat_eval.py
   - At 26K steps / 2000 = ~13 eval points during base training
2. Add `--eval-data-dir` and `--eval-log` flags
3. **Keep val BPB as-is** (every 250 steps — monitors training stability)
4. **No checkpoint changes** — save only at end (model + optimizer for resume)

**Note:** Categorical eval works on base models — compares logits for letter tokens. No chat formatting needed.

### 5b. `mid_train.py` — TaskMixture + step-based eval

**Current behavior:**
- TaskSequence (sequential datasets)
- Boundary eval at dataset transitions
- Boundary checkpoints at each transition
- Val BPB every 150 steps

**Changes:**
1. **Replace TaskSequence with TaskMixture** for training data
   - `train_ds = TaskMixture([CustomJSON(f) for f in data_files])`
   - This shuffles all datasets together — no more sequential curriculum
   - Remove all boundary detection logic (`consumed` counter, `cumulative_lengths`, `boundary_crossed` sync)
   - This simplifies the code significantly

2. **Add `--chat-eval-every N` flag** (default: 250)
   - At 525 steps / 250 = ~2 eval points + start/end
   - Calls `run_full_eval()`, appends to `eval_log.jsonl`
   - Log to wandb with `eval/` prefix

3. **Delete all boundary-related code:**
   - Boundary detection (`consumed`, `cumulative_lengths`, `boundary_crossed`, `pending_boundary_indices`)
   - DDP sync for boundary crossing
   - Boundary checkpoints
   - Boundary eval calls

4. **Delete replay buffer** (`--replay-file`, `--replay-ratio`) — not needed when data is mixed

5. **Checkpoint only at end of mid-training**
   - Save model + optimizer (for resume to SFT if needed)
   - Single directory: `mid_checkpoints/d{N}/`

6. **Keep val BPB as-is** (every 150 steps, lightweight)

### 5c. `chat_sft.py` — TaskMixture + step-based eval

**Current behavior:**
- TaskSequence (sequential datasets) with replay buffer
- 3 eval mechanisms: val_loss (100 steps), metrics (200 steps), boundary (dataset transitions)
- Boundary checkpoints + boundary eval

**Changes:**
1. **Replace TaskSequence with TaskMixture** for training data
   - `train_ds = TaskMixture([CustomJSON(f) for f in data_files])`
   - Remove boundary detection logic
   - Remove replay buffer (not needed when data is mixed — replay was for preventing forgetting during sequential training)

2. **Consolidate 3 eval mechanisms into 2:**
   - **Val loss** (`--eval-every`, default 100): keep as-is (lightweight, fast)
   - **Full eval** (`--chat-eval-every`, default 200): replace the separate `--eval-metrics-every` (which only ran HellaSwag+Winogrande) with `run_full_eval()` running ALL 9 benchmarks

3. **Remove boundary eval code entirely** (no more boundaries)

4. **Checkpoint only at end of SFT**
   - Save model only (no optimizer — SFT is the final stage)
   - Single directory: `chatsft_checkpoints/d{N}/`

---

## 6. Summary of Changes Per File

### nanochat/tasks/ (NEW files)
| File | Lines | Description |
|------|-------|-------------|
| `mathqa.py` | ~60 | MathQA MC-5 task, parse option strings from HuggingFace |
| `gsm_mc.py` | ~50 | GSM-MC MC-4 task, load local JSONL, shuffle candidates |
| `internal_mc.py` | ~40 | Internal MC test split, identical to LABEval pattern |

### nanochat/tasks/ (MODIFIED — add `cache_dir` support)
| File | Change |
|------|--------|
| `arc.py` | Add optional `cache_dir` param to `__init__`, pass to `load_dataset()` |
| `hellaswag.py` | Same |
| `winogrande.py` | Same |
| `race.py` | Same |
| `piqa.py` | Same |

### nanochat/scripts/ (MODIFIED files)
| File | Change | Complexity |
|------|--------|------------|
| `chat_eval.py` | Add 3 tasks to registry, `--eval-data-dir` arg, `run_full_eval()` | Small |
| `base_train.py` | Add `--chat-eval-every` flag, call `run_full_eval()` | Small |
| `mid_train.py` | TaskSequence→TaskMixture, remove boundary logic, add `--chat-eval-every` | Medium |
| `chat_sft.py` | TaskSequence→TaskMixture, remove boundary+replay, consolidate eval | Medium |

### src/post_training/eval/ (NEW file)
| File | Description |
|------|-------------|
| `download_gsm_mc.py` | Downloads GSM-MC data from GitHub, writes to `eval_data/gsm_mc.jsonl` |

---

## 7. Changes to `speedrun_hist_llm.sh`

### Add variables at top:
```bash
EVAL_DATA_DIR="$DATA_ROOT/eval_data"   # single path — all eval data lives here
```

### Pre-run setup (copy period-specific files into eval_data/):
```bash
cp "$PERIOD_DIR/posttraining_data/final/test/hist_synthetic_test.jsonl" "$EVAL_DATA_DIR/internal_mc.jsonl"
cp "$PERIOD_DIR/model/eval/lab_eval.jsonl" "$EVAL_DATA_DIR/lab_eval.jsonl"
```

### Mid-training section:
**Remove:** `--replay-file`, `--replay-ratio`
**Add:** `--chat-eval-every 250`, `--eval-data-dir $EVAL_DATA_DIR`

### SFT section:
**Remove:** `--replay-file`, `--replay-ratio`
**Add:** `--chat-eval-every 500`, `--eval-data-dir $EVAL_DATA_DIR`

### Post-stage standalone evals — update task list:
```bash
torchrun ... scripts.chat_eval -- -i mid -g mid_final \
    -a "ARC-Challenge|HellaSwag|RACE-Middle|RACE-High|Winogrande|MathQA|GSM-MC|InternalMC|LAB" \
    --eval-data-dir $EVAL_DATA_DIR \
    --eval-log $EVAL_LOG --eval-label "mid_final"
```

---

## 8. What NOT to Change

- **Val BPB / val loss**: keep as-is (lightweight, different purpose — monitors training stability)
- **Tokenizer, model architecture, data generation pipeline**: untouched
- **Standalone `chat_eval.py` CLI**: keep working as-is, just add more tasks + flags
- **wandb integration**: keep, add `eval/` prefix for new metrics
- **DDP sync logic for BPB eval**: keep as-is
- **Generative eval tasks** (SpellingBee, DyckLanguage): keep in registry for standalone use, skip during training (too slow)

---

## 9. Eval Timing Estimate

All benchmarks are categorical (MC) — batched logit comparison, no sampling.

Per eval run (at 1024 max_problems, batch_size=64):
- 9 tasks × ~16 batches each = ~144 forward passes
- At ~0.1s per forward pass = ~15 seconds total per eval
- Negligible compared to training step time

Running every 500 steps adds < 1% overhead. Safe to run frequently.

---

## 10. Implementation Order

### Phase 1: Add new task files + consolidated eval folder (no training changes)
1. Create `tasks/mathqa.py`, `tasks/gsm_mc.py`, `tasks/internal_mc.py`
2. Add `cache_dir` support to existing HF tasks (`arc.py`, `hellaswag.py`, `winogrande.py`, `race.py`, `piqa.py`)
3. Create `src/post_training/eval/download_gsm_mc.py` — downloads GSM-MC data to `eval_data/gsm_mc.jsonl`
4. Update `chat_eval.py`: `--eval-data-dir` arg, expanded registry with `build_task_map()`, `run_full_eval()`
5. Test with standalone `python scripts/chat_eval.py -i sft --eval-data-dir D:/hist_LLM/eval_data -a "MathQA|GSM-MC|InternalMC"`

### Phase 2: Update training scripts
1. Add `--chat-eval-every` to `base_train.py`
2. Refactor `mid_train.py` (TaskMixture + step eval)
3. Refactor `chat_sft.py` (TaskMixture + consolidated eval)
4. Test each script independently

### Phase 3: Full pipeline test
1. Run base → mid → SFT for one period
2. Verify `eval_log.jsonl` has consistent entries across all stages
3. Verify checkpoints saved only at stage boundaries
4. Plot eval metrics over training steps (should see smooth curves)

---

## 11. Prompt for Next Claude Session

```
I'm implementing a unified eval system for my nanochat training framework.

Key files to read first:
1. Implementation plan: C:\Users\danielyoon\Dropbox\hist_LLM\src\post_training\eval_implementation.md
2. Post-training README (eval strategy): C:\Users\danielyoon\Dropbox\hist_LLM\src\post_training\README.md
3. Existing eval tasks for pattern reference: C:\Users\danielyoon\Dropbox\hist_LLM\nanochat\tasks\arc.py
4. Current eval dispatcher: C:\Users\danielyoon\Dropbox\hist_LLM\nanochat\scripts\chat_eval.py
5. Training scripts to modify: nanochat/scripts/base_train.py, mid_train.py, chat_sft.py
6. Launch script: C:\Users\danielyoon\Dropbox\hist_LLM\nanochat\runs\speedrun_hist_llm.sh

IMPORTANT: nanochat has its OWN git repo (separate from hist_LLM/src/).

Please read the implementation plan first, then proceed with Phase 1:
creating the 3 new task files and updating chat_eval.py.
```
