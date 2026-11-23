"""Does completion-order observation break stopping rules? A race on recorded samples.

K workers generate in parallel; a sample's duration is its output-token count. A serial
system sees answers in launch order (i.i.d.). A parallel system that reacts to whichever
generation finishes first sees them in completion order, which over-represents short
generations whenever it stops. If answer and length are dependent, the answers a rule has
seen when it stops are not an i.i.d. sample of the model's answer distribution.

Protocols compared, each feeding the same rule:
  serial        one generation at a time, launch order (the setting every paper analyses)
  completion    K workers, answers consumed as they finish; on stop, in-flight work is cancelled
  launch-order  K workers, answer i is consumed only once answers 1..i have all finished

Cost = output tokens generated, including the partial tokens of cancelled generations.
Needs data/AIME25_QwQ-32B.npz (labels and tokens); prints only.

    python research/scripts/parallel_race.py --workers 8 --replicates 200
"""
from __future__ import annotations

import argparse
import heapq

import numpy as np

from scstop import AdaptiveConsistencyBeta, ConSol, MartingaleMajorityCertificate, pool_mode


def race(labels, tokens, order, workers, rule, mode: str):
    """Return (decision, tokens spent, wall time). `order` is the launch order of pool indices."""
    if mode == "serial":
        d = rule.run(labels[order])
        spent = int(tokens[order[: d.samples]].sum())
        return d, spent, spent
    # event simulation: (finish time, launch index)
    heap, launched, start = [], 0, {}
    for _ in range(workers):
        start[launched] = 0.0
        heapq.heappush(heap, (float(tokens[order[launched]]), launched))
        launched += 1
    seen, done, consumed_upto = [], set(), 0
    budget = len(order)
    while heap:
        t, i = heapq.heappop(heap)
        done.add(i)
        if mode == "completion":
            seen.append(labels[order[i]])
        else:
            while consumed_upto in done:
                seen.append(labels[order[consumed_upto]])
                consumed_upto += 1
        d = rule.run(np.array(seen, dtype=labels.dtype)) if seen else None
        if d is not None and d.status != "need_more":
            spent = sum(tokens[order[j]] for j in done) + sum(t - start[j] for _, j in heap)
            return d, int(spent), t
        if launched < budget:
            start[launched] = t
            heapq.heappush(heap, (t + float(tokens[order[launched]]), launched))
            launched += 1
    d = rule.run(np.array(seen, dtype=labels.dtype))
    return d, int(tokens[order].sum()), t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/AIME25_QwQ-32B.npz")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--replicates", type=int, default=200)
    args = ap.parse_args()
    z = np.load(args.data)
    labels, tokens = z["labels"], z["tokens"]
    right, wrong, none = tokens[labels == 0], tokens[labels > 0], tokens[labels == -1]
    print(f"mean tokens: correct {right.mean():.0f}, wrong {wrong.mean():.0f}, no answer {none.mean():.0f}")
    within = [np.corrcoef(tokens[q], labels[q] == 0)[0, 1] for q in range(len(labels))
              if 0.02 < np.mean(labels[q] == 0) < 0.98]
    print(f"within-problem correlation(length, correct): mean {np.mean(within):.3f} over {len(within)} mixed problems")
    modes = [pool_mode(labels[q]) for q in range(len(labels))]
    rules = [AdaptiveConsistencyBeta(), ConSol("sprt"), MartingaleMajorityCertificate(epsilon=0.1, budget=64)]
    rng = np.random.default_rng(2026)
    orders = [[rng.choice(labels.shape[1], size=256, replace=False) for _ in range(args.replicates)]
              for _ in range(len(labels))]
    print(f"{'rule':<22}{'protocol':<14}{'accuracy':>9}{'tokens':>10}{'wall':>9}{'stopped':>9}{'wrong mode | stopped':>22}")
    for rule in rules:
        for mode in ("serial", "completion", "launch-order"):
            acc = tok = wall = stop = bad = 0
            for q in range(len(labels)):
                for order in orders[q]:
                    d, spent, t = race(labels[q], tokens[q], order, args.workers, rule, mode)
                    acc += d.answer == 0
                    tok += spent
                    wall += t
                    s = d.status == "stopped"
                    stop += s
                    bad += s and d.answer != modes[q]
            n = len(labels) * args.replicates
            print(f"{rule.name:<22}{mode:<14}{acc / n:>9.3f}{tok / n / 1e3:>9.0f}K{wall / n / 1e3:>8.0f}K"
                  f"{stop / n:>9.2f}{bad / max(stop, 1):>22.4f}")
