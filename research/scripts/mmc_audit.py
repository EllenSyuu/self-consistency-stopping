"""Monte Carlo search for violations of P(MMC stops with a non-modal answer) <= epsilon.

The paper's Theorem 3.1 controls sup_n e_n under the per-round null "the current leader is
not more likely than the current runner-up". The event a user cares about is "the rule
stopped and returned an answer that is not the mode". This script estimates the latter on
synthetic distributions chosen to stress leader changes. Synthetic only.

    python research/scripts/mmc_audit.py --reps 2000 --budget 300
"""
from __future__ import annotations

import argparse

import numpy as np

from scstop import MartingaleMajorityCertificate

CASES = {
    "two answers, tie 0.50/0.50": [0.5, 0.5],
    "two answers 0.52/0.48": [0.52, 0.48],
    "two answers 0.55/0.45": [0.55, 0.45],
    "two answers 0.60/0.40": [0.60, 0.40],
    "three answers 0.40/0.35/0.25": [0.40, 0.35, 0.25],
    "three answers 0.36/0.34/0.30": [0.36, 0.34, 0.30],
    "three answers, tie 0.45/0.45/0.10": [0.45, 0.45, 0.10],
    "mode 0.50 + runner-up 0.48 + noise 0.02": [0.50, 0.48, 0.02],
    "five answers 0.30/0.28/0.14/0.14/0.14": [0.30, 0.28, 0.14, 0.14, 0.14],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--epsilon", type=float, default=0.1)
    args = ap.parse_args()
    rng = np.random.default_rng(20261123)
    rule = MartingaleMajorityCertificate(epsilon=args.epsilon, budget=args.budget)
    print(f"epsilon = {args.epsilon}, budget = {args.budget}, {args.reps} runs per case")
    for name, p in CASES.items():
        p = np.asarray(p)
        modes = set(np.flatnonzero(p == p.max()) + 1)
        stopped = wrong = 0
        for _ in range(args.reps):
            d = rule.run(rng.choice(np.arange(1, len(p) + 1), size=args.budget, p=p))
            if d.status == "stopped":
                stopped += 1
                wrong += d.answer not in modes or len(modes) > 1
        se = np.sqrt(max(wrong, 1) / args.reps ** 2)
        print(f"  {name:<42} stopped {stopped / args.reps:.3f}   stopped on a non-unique-mode answer "
              f"{wrong / args.reps:.4f} (se {se:.4f})")
