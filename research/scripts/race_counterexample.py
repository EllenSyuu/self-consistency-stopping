"""Synthetic counterexample: completion-order consumption breaks an anytime-valid certificate.

Answer A (probability 0.3) takes 1,000 tokens; answer B (probability 0.7, the mode) takes
`ratio` times longer. Serial sampling and launch-order consumption see i.i.d. answers.
Completion-order consumption with K workers sees the short answer first.

    python research/scripts/race_counterexample.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from scstop import AdaptiveConsistencyBeta, MartingaleMajorityCertificate

spec = importlib.util.spec_from_file_location("race", Path(__file__).with_name("parallel_race.py"))
race_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(race_mod)

if __name__ == "__main__":
    rng = np.random.default_rng(5)
    reps, n = 2000, 400
    for ratio in (1.0, 2.0, 10.0):
        for workers in (4, 16):
            for rule in (MartingaleMajorityCertificate(epsilon=0.1, budget=n), AdaptiveConsistencyBeta(budget=n)):
                out = {}
                for mode in ("serial", "completion", "launch-order"):
                    bad = stop = 0
                    r = np.random.default_rng(99)
                    for _ in range(reps):
                        lab = np.where(r.random(n) < 0.3, 1, 2).astype(np.int16)
                        tok = np.where(lab == 1, 1000.0, 1000.0 * ratio)
                        d, _, _ = race_mod.race(lab, tok, np.arange(n), workers, rule, mode)
                        s = d.status == "stopped"
                        stop += s
                        bad += s and d.answer == 1
                    out[mode] = bad / reps
                print(f"length ratio {ratio:>4}, {workers:>2} workers, {rule.name:<20} "
                      f"P(stop on the minority answer): serial {out['serial']:.4f}  "
                      f"completion {out['completion']:.4f}  launch-order {out['launch-order']:.4f}")
