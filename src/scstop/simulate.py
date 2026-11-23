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
    tokens: np.ndarray | None = None  # [problems, replicates] output tokens spent, if known


def replay(labels: np.ndarray, rules: list[Rule], replicates: int, max_len: int, seed: int = 0,
           tokens: np.ndarray | None = None, replace: bool = False,
           correct: np.ndarray | None = None) -> list[Result]:
    """Feed every rule the same random orders of each problem's pool.

    `tokens`, if given, holds each recorded sample's output length; a rule's token cost is
    the total length of the samples it drew.
    `replace=True` draws with replacement: i.i.d. samples from the pool's empirical
    distribution, whose mode is then exactly the reference mode.
    `correct`, if given, is a bool array [problems, classes] saying which answer classes are
    right; without it only class 0 is right (labels built with the grader's help).
    """
    n_problems, pool = labels.shape
    max_len = max_len if replace else min(max_len, pool)
    rng = np.random.default_rng(seed)
    modes = [pool_mode(labels[q]) for q in range(n_problems)]
    shape = (n_problems, replicates)
    out = {r.name: Result(r.name, np.zeros(shape, bool), np.zeros(shape, np.int32),
                          np.zeros(shape, bool), np.empty(shape, dtype=object),
                          np.zeros(shape, np.int64) if tokens is not None else None) for r in rules}
    for q in range(n_problems):
        for k in range(replicates):
            order = rng.choice(pool, size=max_len, replace=replace)
            stream = labels[q, order]
            cum = np.concatenate(([0], np.cumsum(tokens[q, order]))) if tokens is not None else None
            for rule in rules:
                d = rule.run(stream)
                res = out[rule.name]
                if correct is None:
                    res.correct[q, k] = d.answer == CORRECT
                else:
                    res.correct[q, k] = d.answer is not None and bool(correct[q, d.answer])
                res.samples[q, k] = d.samples
                res.matches_mode[q, k] = d.answer is not None and d.answer == modes[q]
                res.status[q, k] = d.status
                if cum is not None:
                    res.tokens[q, k] = cum[d.samples]
    return [out[r.name] for r in rules]


def _bootstrap(per_problem: np.ndarray, weights: np.ndarray | None, draws: np.ndarray) -> list[float]:
    """95% interval of a (weighted) mean over problems, resampling problems."""
    if weights is None:
        stats = per_problem[draws].mean(axis=1)
    else:
        w = weights[draws]
        tot = w.sum(axis=1)
        stats = np.where(tot > 0, (per_problem[draws] * w).sum(axis=1) / np.maximum(tot, 1), np.nan)
        stats = stats[~np.isnan(stats)]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return [float(lo), float(hi)]


def summarise(res: Result, n_boot: int = 2000, seed: int = 0) -> dict:
    """Means over problems and replicates, with 95% intervals from resampling problems.

    Problems are resampled with the same indices for every rule (same seed), so the
    intervals of two rules on one setting are paired.
    """
    stopped = res.status == "stopped"
    n = res.correct.shape[0]
    draws = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    stop_count = stopped.sum(axis=1)
    wrong_when_stopped = (~res.matches_mode & stopped).sum(axis=1) / np.maximum(stop_count, 1)
    out = {
        "rule": res.rule,
        "accuracy": float(res.correct.mean()),
        "accuracy_ci": _bootstrap(res.correct.mean(axis=1), None, draws),
        "mean_samples": float(res.samples.mean()),
        "mean_samples_ci": _bootstrap(res.samples.mean(axis=1), None, draws),
        "median_samples": float(np.median(res.samples)),
        "matches_mode": float(res.matches_mode.mean()),
        "stopped_share": float(stopped.mean()),
        "mode_error_when_stopped": float((~res.matches_mode[stopped]).mean()) if stopped.any() else None,
        "mode_error_when_stopped_ci": (_bootstrap(wrong_when_stopped, stop_count.astype(float), draws)
                                       if stopped.any() else None),
    }
    if res.tokens is not None:
        out["mean_tokens"] = float(res.tokens.mean())
        out["mean_tokens_ci"] = _bootstrap(res.tokens.mean(axis=1), None, draws)
    return out
