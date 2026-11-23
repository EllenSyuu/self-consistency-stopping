"""Figures for the README from results/summary.json, in a light and a dark theme.

    python scripts/make_figures.py --summary results/summary.json --out docs/img
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WIDTH = 10.0


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink_2: str
    ink_3: str
    grid: str
    series: tuple[str, str, str, str]


LIGHT = Theme("light", "#ffffff", "#16202e", "#4b5768", "#7d8896", "#e6eaef",
              ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"))
DARK = Theme("dark", "#0d1117", "#e8edf3", "#a3afbd", "#6f7c8c", "#222b36",
             ("#3987e5", "#d95926", "#199e70", "#c98500"))

# rule name in summary.json -> (legend label, series slot, marker)
RULES = {
    "Adaptive-Consistency": ("Adaptive-Consistency", 0, "o"),
    "ESC": ("ESC", 1, "s"),
    "ConSol-SPRT": ("ConSol", 2, "D"),
    "MMC (ε=0.1)": ("MMC, ε = 0.1", 3, "^"),
}
PRETTY = {
    "GSM8K_Llama-3-70B-Instruct": "GSM8K · Llama-3-70B-Instruct",
    "GSM8K_Llama-3-8B-Instruct": "GSM8K · Llama-3-8B-Instruct",
    "MATH_Llama-3-70B-Instruct": "MATH · Llama-3-70B-Instruct",
    "MATH_Llama-3-8B-Instruct": "MATH · Llama-3-8B-Instruct",
    "MATH_Gemma-7B": "MATH · Gemma-7B",
    "MATH_Llama-3-8B": "MATH · Llama-3-8B",
    "MATH_Gemma-2B": "MATH · Gemma-2B",
}


def short(key: str, setting: dict) -> str:
    task, model = key.split("_", 1)
    model = model.replace("-Instruct", "\nInstruct")
    return f"{task}\n{model}\n{setting['single_sample_accuracy']:.0%}"


def style(t: Theme) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Inter", "DejaVu Sans"], "font.size": 12,
        "figure.facecolor": t.surface, "axes.facecolor": t.surface, "savefig.facecolor": t.surface,
        "savefig.dpi": 200, "axes.edgecolor": t.grid, "axes.labelcolor": t.ink_2, "axes.labelsize": 12,
        "axes.titlecolor": t.ink, "axes.titlesize": 13.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False, "axes.grid": True, "axes.grid.axis": "y",
        "axes.axisbelow": True, "grid.color": t.grid, "xtick.color": t.ink_2, "ytick.color": t.ink_2,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "ytick.major.size": 0, "text.color": t.ink,
        "legend.frameon": False, "legend.fontsize": 11, "legend.labelcolor": t.ink_2,
    })


def rows(setting: dict) -> dict[str, dict]:
    return {r["rule"]: r for r in setting["rules"]}


def frontier(ax, t: Theme, setting: dict, title: str, label_sc: bool) -> None:
    r = rows(setting)
    sc = sorted((v["mean_samples"], v["accuracy"]) for k, v in r.items() if k.startswith("SC@"))
    xs, ys = zip(*sc)
    ax.plot(xs, ys, color=t.ink_3, lw=2, zorder=2, label="Fixed-N self-consistency")
    ax.scatter(xs, ys, s=18, color=t.ink_3, zorder=2, edgecolor=t.surface, linewidth=1)
    if label_sc:
        ax.annotate("fixed N", (xs[-1], ys[-1]), xytext=(-4, 8), textcoords="offset points",
                    ha="right", color=t.ink_2, fontsize=10.5)
    for name, (label, slot, marker) in RULES.items():
        v = r[name]
        ax.scatter(v["mean_samples"], v["accuracy"], s=90, marker=marker, color=t.series[slot],
                   edgecolor=t.surface, linewidth=2, zorder=4, label=label)
    ax.set_xscale("log", base=2)
    ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    ax.set_xticks(ticks, [str(k) for k in ticks])
    ax.set_xlim(0.8, 320)
    lo, hi = min(ys), max(max(ys), max(r[n]["accuracy"] for n in RULES))
    pad = (hi - lo) * 0.12 or 0.01
    ax.set_ylim(lo - pad, hi + pad)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title(title)
    ax.set_ylabel("Accuracy")


def fig_frontiers(summary: dict, t: Theme, out: Path) -> None:
    picks = ["GSM8K_Llama-3-70B-Instruct", "MATH_Llama-3-70B-Instruct", "MATH_Gemma-2B"]
    fig, axes = plt.subplots(len(picks), 1, figsize=(WIDTH, 10.5))
    for k, (ax, key) in enumerate(zip(axes, picks)):
        s = summary["settings"][key]
        frontier(ax, t, s, f"{PRETTY[key]}  ·  one sample right {s['single_sample_accuracy']:.0%} of the time",
                 label_sc=k == 0)
    axes[-1].set_xlabel("Samples per question (mean, log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0), handletextpad=0.4,
               columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.2)
    fig.savefig(out / f"frontiers-{t.name}.png")
    plt.close(fig)


def fig_cost_by_strength(summary: dict, t: Theme, out: Path) -> None:
    settings = sorted(summary["settings"].items(), key=lambda kv: kv[1]["single_sample_accuracy"])
    x = np.arange(len(settings))
    fig, ax = plt.subplots(figsize=(WIDTH, 5.8))
    for name, (label, slot, marker) in RULES.items():
        y = [rows(s)[name]["mean_samples"] for _, s in settings]
        ax.plot(x, y, color=t.series[slot], lw=2, marker=marker, ms=8, mec=t.surface, mew=2, label=label, zorder=3)
        ax.annotate(label, (x[-1], y[-1]), xytext=(10, 0), textcoords="offset points", va="center",
                    color=t.ink_2, fontsize=10.5)
    ax.set_yscale("log", base=2)
    yt = [2, 4, 8, 16, 32, 64]
    ax.set_yticks(yt, [str(v) for v in yt])
    ax.set_ylim(2.5, 110)
    ax.set_xticks(x, [short(k, s) for k, s in settings], fontsize=9.5)
    ax.set_xlim(-0.4, len(settings) + 0.9)
    ax.set_xlabel("Task · model · how often a single sample is right")
    ax.set_ylabel("Samples per question (mean)")
    ax.set_title("The less often one sample is right, the fewer samples any rule saves")
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(out / f"cost_by_strength-{t.name}.png")
    plt.close(fig)


def fig_guarantee_audit(summary: dict, t: Theme, out: Path) -> None:
    settings = sorted(summary["settings"].items(), key=lambda kv: kv[1]["single_sample_accuracy"])
    audit = [("ConSol-SPRT", "ConSol (configured α = 0.05)", 2, "D", 0.05),
             ("Adaptive-Consistency", "Adaptive-Consistency (stops at 95% posterior)", 0, "o", 0.05),
             ("MMC (ε=0.1)", "MMC (certified ε = 0.1)", 3, "^", 0.10)]
    fig, ax = plt.subplots(figsize=(WIDTH, 5.8))
    xs = np.arange(len(settings))
    for j, (name, label, slot, marker, _nominal) in enumerate(audit):
        y = [rows(s)[name]["mode_error_when_stopped"] or 0.0 for _, s in settings]
        ax.scatter(xs + (j - 1) * 0.18, y, s=80, marker=marker, color=t.series[slot], edgecolor=t.surface,
                   linewidth=2, zorder=3, label=label)
    ax.axhline(0.05, color=t.ink_3, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate("5%", (len(settings) - 0.45, 0.05), xytext=(4, 0), textcoords="offset points", va="center",
                color=t.ink_2, fontsize=10.5, annotation_clip=False)
    ax.axhline(0.10, color=t.ink_3, lw=1.2, ls=(0, (1, 3)), zorder=1)
    ax.annotate("10%", (len(settings) - 0.45, 0.10), xytext=(4, 0), textcoords="offset points", va="center",
                color=t.ink_2, fontsize=10.5, annotation_clip=False)
    ax.set_xticks(xs, [short(k, s) for k, s in settings], fontsize=9.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Wrong mode when the rule stopped")
    ax.set_title("When a rule says it is sure, how often is its answer not the model's most likely one?")
    ax.legend(loc="upper right", ncol=1)
    ax.set_xlim(-0.6, len(settings) - 0.1)
    ax.set_ylim(-0.01, 0.30)
    fig.tight_layout()
    fig.savefig(out / f"guarantee_audit-{t.name}.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, default=Path("results/summary.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/img"))
    args = ap.parse_args()
    summary = json.loads(args.summary.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for t in (LIGHT, DARK):
        style(t)
        fig_frontiers(summary, t, args.out)
        fig_cost_by_strength(summary, t, args.out)
        fig_guarantee_audit(summary, t, args.out)
    print(f"wrote figures to {args.out}")


if __name__ == "__main__":
    main()
