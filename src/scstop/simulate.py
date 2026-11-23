"""Replay stopping rules on recorded samples.

Each problem has a pool of recorded answer labels (10,000 in Large Language Monkeys). One
replicate draws a random order of the pool without replacement and feeds it to every rule,
so all rules see the same streams. With a pool this large relative to the budgets, this
is close to sampling the model afresh.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rules import NO_ANSWER, Rule

CORRECT = 0


def pool_mode(pool: np.ndarray) -> int | None:
    """The most frequent valid label in the whole pool (the model's answer mode, estimated)."""
    valid = pool[pool != NO_ANSWER]
    if valid.size == 0:
        return None
    labels, counts = np.unique(valid, return_counts=True)
    return int(labels[np.argmax(counts)])


@dataclass
class Result:
    rule: str
    correct: np.ndarray      # [problems, replicates] bool
    samples: np.ndarray      # [problems, replicates] int
    matches_mode: np.ndarray # [problems, replicates] bool
    status: np.ndarray       # [problems, replicates] str


def replay(labels: np.ndarray, rules: list[Rule], replicates: int, max_len: int, seed: int = 0) -> list[Result]:
    n_problems, pool = labels.shape
    rng = np.random.default_rng(seed)
    modes = [pool_mode(labels[q]) for q in range(n_problems)]
    shape = (n_problems, replicates)
    out = {r.name: Result(r.name, np.zeros(shape, bool), np.zeros(shape, np.int32),
                          np.zeros(shape, bool), np.empty(shape, dtype=object)) for r in rules}
    for q in range(n_problems):
        for k in range(replicates):
            stream = labels[q, rng.choice(pool, size=max_len, replace=False)]
            for rule in rules:
                d = rule.run(stream)
                res = out[rule.name]
                res.correct[q, k] = d.answer == CORRECT
                res.samples[q, k] = d.samples
                res.matches_mode[q, k] = d.answer is not None and d.answer == modes[q]
                res.status[q, k] = d.status
    return [out[r.name] for r in rules]


def summarise(res: Result) -> dict:
    stopped = res.status == "stopped"
    return {
        "rule": res.rule,
        "accuracy": float(res.correct.mean()),
        "mean_samples": float(res.samples.mean()),
        "median_samples": float(np.median(res.samples)),
        "matches_mode": float(res.matches_mode.mean()),
        "stopped_share": float(stopped.mean()),
        "mode_error_when_stopped": float((~res.matches_mode[stopped]).mean()) if stopped.any() else None,
    }
