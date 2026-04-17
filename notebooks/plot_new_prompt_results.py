"""Generate three full-pipeline charts (External, Internal MC, LAB) for the
1900-1949 period showing all 3 phases: Base Pretraining (from old run) ->
Mid-Training (new prompts) -> SFT (new prompts). PIQA is included in the
External chart and will only appear where data exists."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPORT_OLD_DIR = Path(r"D:\hist_LLM\periods\1900_1949\model\report_old")  # base eval log
REPORT_NEW_DIR = Path(r"D:\hist_LLM\periods\1900_1949\report_new_prompt")  # new mid/sft eval log
REPORT_OUT_DIR = REPORT_NEW_DIR


def load_eval_log(path: Path) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def main() -> None:
    df_old = load_eval_log(REPORT_OLD_DIR / "eval_log.jsonl")
    df_new = load_eval_log(REPORT_NEW_DIR / "eval_log.jsonl")

    df_base = df_old[df_old.get("stage") == "base"].copy()
    df_mid = df_new[df_new.get("stage") == "mid"].copy()
    df_sft = df_new[df_new.get("stage") == "sft"].copy()

    # Build sequential eval index across base -> mid -> sft
    parts = []
    idx = 0
    for part_df in (df_base, df_mid, df_sft):
        part = part_df.copy()
        part["eval_idx"] = range(idx, idx + len(part))
        parts.append(part)
        idx += len(part)
    df_all = pd.concat(parts, ignore_index=True)

    print(f"Total eval points: {len(df_all)}")
    print(f"  Base: {len(df_base)} (steps {df_base['step'].min()}-{df_base['step'].max()})")
    print(f"  Mid:  {len(df_mid)} (steps {df_mid['step'].min()}-{df_mid['step'].max()})")
    print(f"  SFT:  {len(df_sft)} (steps {df_sft['step'].min()}-{df_sft['step'].max()})")

    base_end = len(df_base) - 0.5
    mid_end = len(df_base) + len(df_mid) - 0.5
    x_lo, x_hi = -0.5, len(df_all) - 0.5

    def draw_phase_overlays(ax, label_va="bottom", y_offset=0.01):
        """Draw phase boundary lines, colored bands, and phase labels."""
        ax.axvline(x=base_end, color="black", linestyle="-", alpha=0.5, linewidth=1.5)
        ax.axvline(x=mid_end, color="black", linestyle="-", alpha=0.5, linewidth=1.5)
        ax.axvspan(x_lo, base_end, alpha=0.04, color="blue")
        ax.axvspan(base_end, mid_end, alpha=0.04, color="orange")
        ax.axvspan(mid_end, x_hi, alpha=0.04, color="green")

        y_top = ax.get_ylim()[1]
        if label_va == "bottom":
            y_label = y_top + y_offset
        else:
            y_label = y_top - y_offset
        ax.text(len(df_base) / 2, y_label, "Base Pretraining",
                ha="center", va=label_va, fontsize=12, fontweight="bold",
                color="#333", alpha=0.7)
        ax.text(base_end + len(df_mid) / 2 + 0.5, y_label, "Mid-Training",
                ha="center", va=label_va, fontsize=12, fontweight="bold",
                color="#333", alpha=0.7)
        ax.text(mid_end + len(df_sft) / 2 + 0.5, y_label, "SFT",
                ha="center", va=label_va, fontsize=12, fontweight="bold",
                color="#333", alpha=0.7)

    # --------------------------------------------------------------
    # Plot 1: External Benchmarks (includes PIQA)
    # --------------------------------------------------------------
    EXTERNAL = ["ARC-Challenge", "HellaSwag", "RACE-Middle", "RACE-High",
                "Winogrande", "GSM-MC", "PIQA"]
    EXTERNAL = [b for b in EXTERNAL if b in df_all.columns]

    ext_colors = {
        "ARC-Challenge": "#e74c3c",
        "HellaSwag": "#3498db",
        "RACE-Middle": "#2ecc71",
        "RACE-High": "#27ae60",
        "Winogrande": "#9b59b6",
        "GSM-MC": "#e67e22",
        "PIQA": "#1abc9c",
    }
    ext_baselines = {
        "ARC-Challenge": 0.25, "HellaSwag": 0.25,
        "RACE-Middle": 0.25, "RACE-High": 0.25,
        "Winogrande": 0.50, "GSM-MC": 0.25,
        "PIQA": 0.50,
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    for bm in EXTERNAL:
        ax.plot(df_all["eval_idx"], df_all[bm], label=bm,
                color=ext_colors[bm], linewidth=1.8, alpha=0.85)
        ax.axhline(y=ext_baselines[bm], color=ext_colors[bm],
                   linestyle="--", alpha=0.15, linewidth=0.8)

    draw_phase_overlays(ax, label_va="bottom", y_offset=0.01)

    ax.set_xlabel("Evaluation Checkpoint", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("External Benchmark Performance Across Training Phases (1900-1949, new prompts)",
                 fontsize=14, pad=20)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.set_xlim(x_lo, x_hi)
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out1 = REPORT_OUT_DIR / "full_pipeline_external.png"
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out1}")

    # --------------------------------------------------------------
    # Plot 2: Per-Generator InternalMC
    # --------------------------------------------------------------
    INTERNAL = [f"InternalMC_{L}" for L in "ABCDEF"]
    INTERNAL = [b for b in INTERNAL if b in df_all.columns]

    gen_colors = {
        "InternalMC_A": "#e74c3c",
        "InternalMC_B": "#3498db",
        "InternalMC_C": "#2ecc71",
        "InternalMC_D": "#9b59b6",
        "InternalMC_E": "#e67e22",
        "InternalMC_F": "#1abc9c",
    }
    gen_labels = {
        "InternalMC_A": "Gen A (Factual QA)",
        "InternalMC_B": "Gen B (Chain-of-Thought)",
        "InternalMC_C": "Gen C (Comprehension)",
        "InternalMC_D": "Gen D (Quantitative)",
        "InternalMC_E": "Gen E (Completion)",
        "InternalMC_F": "Gen F (Instruction)",
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    for bm in INTERNAL:
        ax.plot(df_all["eval_idx"], df_all[bm], label=gen_labels.get(bm, bm),
                color=gen_colors[bm], linewidth=1.8, alpha=0.85)

    ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.4, linewidth=1,
               label="Random baseline (25%)")

    draw_phase_overlays(ax, label_va="bottom", y_offset=0.01)

    ax.set_xlabel("Evaluation Checkpoint", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Internal MC Performance by Generator Across Training Phases (1900-1949, new prompts)",
                 fontsize=14, pad=20)
    ax.legend(loc="center left", fontsize=10, framealpha=0.9)
    ax.set_xlim(x_lo, x_hi)
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out2 = REPORT_OUT_DIR / "full_pipeline_internal_mc.png"
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out2}")

    # --------------------------------------------------------------
    # Plot 3: LAB-Strict
    # --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(df_all["eval_idx"], df_all["LAB"], color="#e67e22", linewidth=2,
            label="LAB-Strict accuracy", zorder=3)
    ax.axhline(y=0.25, color="green", linestyle="--", alpha=0.6, linewidth=1.5,
               label="Random baseline (25%)")
    ax.fill_between(df_all["eval_idx"], 0.20, 0.30, alpha=0.08, color="green")
    ax.text(0.5, 0.205, "Random ± 2σ", fontsize=9, color="green", alpha=0.7)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0.15, 0.45)

    draw_phase_overlays(ax, label_va="top", y_offset=0.01)

    ax.set_xlabel("Evaluation Checkpoint", fontsize=13)
    ax.set_ylabel("LAB-Strict Accuracy", fontsize=13)
    ax.set_title("Temporal Isolation: LAB-Strict Accuracy Across Training Phases (1900-1949, new prompts)",
                 fontsize=14)
    ax.tick_params(labelsize=11)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out3 = REPORT_OUT_DIR / "full_pipeline_lab.png"
    plt.savefig(out3, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out3}")


if __name__ == "__main__":
    main()
