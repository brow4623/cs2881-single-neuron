"""Job 2: draw our Figure 2 from whatever judgment files exist, next to the paper's numbers.

One panel per judge. Bars are our attack success rate on JailbreakBench (no intervention vs. neuron
pinned); the black tick on each bar is the paper's value. Also writes a CSV with every number.
ASR = successes / responses judged; judge failures (success=null) count as not successful.

Usage: python scripts/plot_fig2.py
Outputs: results/figures/fig2.png, results/fig2_numbers.csv
"""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from single_neuron import RESULTS_DIR  # noqa: E402
from single_neuron.config import in_scope_models  # noqa: E402
from single_neuron.records import CONDITIONS, judgments_path, read_jsonl  # noqa: E402

JUDGE_TITLES = {"llamaguard": "Llama-Guard-3-8B", "llm_judge": "LLM judge (Claude Sonnet)"}
CONDITION_LABELS = {"baseline": "No intervention", "constant": "Neuron pinned (constant)"}
COLORS = {"baseline": "#2a78d6", "constant": "#eb6834"}  # categorical slots 1-2, validated
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def asr(rows: list[dict]) -> float | None:
    return 100 * sum(r["success"] is True for r in rows) / len(rows) if rows else None


def collect() -> list[dict]:
    table = []
    for spec in in_scope_models():
        for judge in JUDGE_TITLES:
            for condition in CONDITIONS:
                rows = read_jsonl(judgments_path(judge, spec.key, condition))
                table.append({
                    "model": spec.key, "judge": judge, "condition": condition,
                    "ours_asr": asr(rows), "n": len(rows), "n_judge_failed": sum(r["success"] is None for r in rows),
                    "paper_asr": spec.paper_fig2[condition][judge],
                })
    return table


def plot(table: list[dict]) -> None:
    models = [s.key for s in in_scope_models()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True, facecolor=SURFACE)
    width, gap = 0.36, 0.02
    for ax, judge in zip(axes, JUDGE_TITLES):
        ax.set_facecolor(SURFACE)
        for k, condition in enumerate(CONDITIONS):
            for i, model in enumerate(models):
                row = next(r for r in table if r["model"] == model and r["judge"] == judge and r["condition"] == condition)
                x = i + (k - 0.5) * (width + gap)
                if row["ours_asr"] is None:
                    # keep the label clear of the paper tick: above low ticks, below high ones
                    y = row["paper_asr"] + 3 if row["paper_asr"] < 50 else 2
                    ax.text(x, y, "not run", rotation=90, ha="center", va="bottom", fontsize=7, color=INK_2)
                else:
                    ax.bar(x, row["ours_asr"], width, color=COLORS[condition], zorder=2)
                    label_y = max(row["ours_asr"], row["paper_asr"]) + 2  # clear of both bar top and paper tick
                    ax.text(x, label_y, f"{row['ours_asr']:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
                ax.hlines(row["paper_asr"], x - width / 2, x + width / 2, colors=INK, linewidth=2, zorder=3)
        ax.set_title(JUDGE_TITLES[judge], fontsize=10, color=INK, loc="left")
        ax.set_xticks(range(len(models)), [m.replace("-", "-​") for m in models], fontsize=8, color=INK_2)
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"], fontsize=8, color=INK_2)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Attack success rate (JBB, n=100)", fontsize=9, color=INK_2)
    handles = [Patch(color=COLORS[c], label=CONDITION_LABELS[c]) for c in CONDITIONS]
    handles.append(Line2D([], [], color=INK, linewidth=2, label="Paper (Fig. 2)"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = RESULTS_DIR / "figures" / "fig2.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"saved {out.relative_to(RESULTS_DIR.parent)}")


def main() -> None:
    table = collect()
    out = RESULTS_DIR / "fig2_numbers.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"saved {out.relative_to(RESULTS_DIR.parent)}")
    for r in table:
        ours = "  -" if r["ours_asr"] is None else f"{r['ours_asr']:3.0f}"
        print(f"  {r['model']:<13} {r['judge']:<11} {r['condition']:<9} ours {ours}%  paper {r['paper_asr']:3d}%  (n={r['n']})")
    plot(table)


if __name__ == "__main__":
    main()
