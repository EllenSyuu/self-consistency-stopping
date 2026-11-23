"""Run every stopping rule on every prepared setting and write results/summary.json.

    python scripts/run_benchmark.py --data data --out results --replicates 200
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from scstop import (AdaptiveConsistencyBeta, ConSol, EarlyStoppingSC, FixedN,
                    MartingaleMajorityCertificate, replay, summarise)

MAX_LEN = 256   # longest budget of any rule (ConSol's default)


def rules_for(task: str) -> list:
    """Every rule at the settings its authors used for this kind of task.

    GSM8K uses the papers' settings for easier tasks (ESC w = 5, L = 40); MATH, AIME and
    GPQA use their settings for MATH (w = 8, L = 64).
    """
    big = 40 if task == "GSM8K" else 64
    sc = [FixedN(n) for n in (1, 2, 3, 5, 8, 10, 16, 20, 32, 40, 64, 96, 128, 256)]
    w = 5 if task == "GSM8K" else 8
    return sc + [
        AdaptiveConsistencyBeta(threshold=0.95, budget=40),
        EarlyStoppingSC(window=w, budget=big, name=f"ESC"),
        ConSol("sprt"),
        ConSol("msprt"),
        MartingaleMajorityCertificate(epsilon=0.1, budget=big, name="MMC (ε=0.1)"),
        MartingaleMajorityCertificate(epsilon=0.05, budget=big, name="MMC (ε=0.05)"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--replicates", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"replicates": args.replicates, "seed": args.seed, "settings": {}}
    for path in sorted(args.data.glob("*.npz")):
        config = path.stem
        task, model = config.split("_", 1)
        z = np.load(path)
        labels = z["labels"]
        tokens = z["tokens"] if "tokens" in z.files else None
        if task.startswith("AIME"):
            task = "AIME"
        t0 = time.perf_counter()
        results = replay(labels, rules_for(task), args.replicates, MAX_LEN, seed=args.seed, tokens=tokens)
        rows = [summarise(r) for r in results]
        ceiling = float(np.mean([np.mean(labels[q] == 0) > 0 and
                                 np.bincount(labels[q][labels[q] >= 0]).argmax() == 0
                                 for q in range(labels.shape[0])]))
        summary["settings"][config] = {
            "task": task, "model": model, "problems": int(labels.shape[0]), "pool": int(labels.shape[1]),
            "single_sample_accuracy": float(np.mean(labels == 0)),
            "mode_accuracy": ceiling,
            "rules": rows,
        }
        print(f"{config}: {time.perf_counter() - t0:.0f}s")
        for r in rows:
            print(f"  {r['rule']:<16} acc {r['accuracy']:.3f}  samples {r['mean_samples']:6.2f}  "
                  f"mode {r['matches_mode']:.3f}  stopped {r['stopped_share']:.2f}  "
                  f"mode-error|stopped {r['mode_error_when_stopped'] if r['mode_error_when_stopped'] is None else round(r['mode_error_when_stopped'], 3)}")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
