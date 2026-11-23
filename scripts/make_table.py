"""Print the README results table (markdown) from results/summary.json.

    python scripts/make_table.py --summary results/summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SHOW = ["Adaptive-Consistency", "ESC", "ConSol-SPRT", "MMC (ε=0.1)"]
LABEL = {"ConSol-SPRT": "ConSol", "MMC (ε=0.1)": "MMC (ε = 0.1)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=Path("results/summary.json"))
    s = json.loads(ap.parse_args().summary.read_text())
    settings = sorted(s["settings"].items(), key=lambda kv: -kv[1]["single_sample_accuracy"])
    head = "| Setting | One sample | Fixed N = 40 | " + " | ".join(LABEL.get(r, r) for r in SHOW) + " |"
    print(head)
    print("|" + "---|" * (3 + len(SHOW)))
    for key, v in settings:
        rows = {r["rule"]: r for r in v["rules"]}
        task, model = key.split("_", 1)
        task = {"AIME25": "AIME 2025", "GPQA": "GPQA-Diamond"}.get(task, task)
        cells = [f"{task} · {model}", f"{v['single_sample_accuracy']:.1%}",
                 f"{rows['SC@40']['accuracy']:.1%} · 40"]
        for r in SHOW:
            x = rows[r]
            cells.append(f"{x['accuracy']:.1%} · {x['mean_samples']:.1f}")
        print("| " + " | ".join(cells) + " |")
    print()
    print(f"Each cell: accuracy · mean samples per question; {s['replicates']} random sample orders per problem.")


if __name__ == "__main__":
    main()
