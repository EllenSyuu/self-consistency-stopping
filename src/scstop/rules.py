"""Early-stopping rules for self-consistency, each as described by its authors.

Every rule consumes a stream of answer labels (one per sampled solution) and returns a
`Decision`: the answer it outputs, how many samples it used, and why it stopped.

Label conventions (see scripts/prepare_monkey_business.py): an int per sample, -1 meaning
"no final answer could be extracted". Such a sample costs one call and casts no vote.

Shared conventions, applied identically to every rule:
* Majority vote over valid labels; ties go to the label that reached the tied count first.
* A rule may stop only after a sample; it never peeks ahead in the stream.
* If the stream ends before the rule stops or reaches its budget, the status is
  "need_more": draw another sample and call `run` again on the longer stream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np
from scipy.special import betainc, betaln

NO_ANSWER = -1


@dataclass(frozen=True)
class Decision:
    answer: int | None      # None if no valid answer was ever seen
    samples: int            # calls spent, including samples with no answer
    status: str             # "stopped", "budget", "abstained", or "need_more" if the stream
                            # ended before the rule stopped or spent its budget (online use)


class Rule(Protocol):
    name: str

    def run(self, stream: np.ndarray) -> Decision: ...


class _Tally:
    """Vote counts with first-to-reach tie-breaking."""

    def __init__(self) -> None:
        self.counts: dict[int, int] = {}
        self.reached: dict[tuple[int, int], int] = {}   # (label, count) -> time it reached count
        self.t = 0

    def add(self, label: int) -> None:
        self.t += 1
        if label == NO_ANSWER:
            return
        c = self.counts.get(label, 0) + 1
        self.counts[label] = c
        self.reached[(label, c)] = self.t

    def ranked(self) -> list[tuple[int, int]]:
        """(label, count), most votes first; ties broken by who reached that count first."""
        return sorted(self.counts.items(), key=lambda kv: (-kv[1], self.reached[(kv[0], kv[1])]))

    def top_two(self) -> tuple[int, int]:
        r = self.ranked()
        return (r[0][1] if r else 0), (r[1][1] if len(r) > 1 else 0)

    def leader(self) -> int | None:
        r = self.ranked()
        return r[0][0] if r else None


def majority(labels: Iterable[int]) -> int | None:
    t = _Tally()
    for x in labels:
        t.add(int(x))
    return t.leader()


@dataclass
class FixedN:
    """Self-consistency (Wang et al., 2023): sample N times, return the majority."""

    n: int

    @property
    def name(self) -> str:
        return f"SC@{self.n}"

    def run(self, stream: np.ndarray) -> Decision:
        if len(stream) < self.n:
            return Decision(majority(stream), len(stream), "need_more")
        return Decision(majority(stream[: self.n]), self.n, "budget")


@dataclass
class AdaptiveConsistencyBeta:
    """Adaptive-Consistency, Beta criterion (Aggarwal et al., EMNLP 2023).

    After each sample, with v1 >= v2 the two largest vote counts, stop when
    P(p1 > p2 | v) = 1 - I_{1/2}(v1 + 1, v2 + 1) > threshold. Paper: threshold 0.95,
    budget 40 for math tasks, one sample at a time.
    """

    threshold: float = 0.95
    budget: int = 40
    name: str = "Adaptive-Consistency"

    def run(self, stream: np.ndarray) -> Decision:
        t = _Tally()
        for i in range(min(self.budget, len(stream))):
            t.add(int(stream[i]))
            v1, v2 = t.top_two()
            if v1 > 0 and 1.0 - betainc(v1 + 1, v2 + 1, 0.5) > self.threshold:
                return Decision(t.leader(), i + 1, "stopped")
        if len(stream) < self.budget:
            return Decision(t.leader(), len(stream), "need_more")
        return Decision(t.leader(), self.budget, "budget")


@dataclass
class EarlyStoppingSC:
    """Early-Stopping Self-Consistency (Li et al., ICLR 2024).

    Sample in windows of w; stop after the first window whose w answers are all the same
    (zero entropy) and return that window's answer; otherwise return the majority of all
    L samples. Paper: w = 5, L = 40 for GSM8K; w = 8, L = 64 for MATH. Here a window that
    contains a sample with no answer is not unanimous.
    """

    window: int = 5
    budget: int = 40
    name: str = "ESC"

    def run(self, stream: np.ndarray) -> Decision:
        used = 0
        for start in range(0, min(self.budget, len(stream)) - self.window + 1, self.window):
            win = stream[start:start + self.window]
            used = start + self.window
            if win[0] != NO_ANSWER and np.all(win == win[0]):
                return Decision(int(win[0]), used, "stopped")
        if len(stream) < self.budget:
            return Decision(majority(stream[:used]), used, "need_more")
        return Decision(majority(stream[:used]), used, "budget")


@dataclass
class ConSol:
    """ConSol (Lee et al., 2025), using the authors' `consol` package (v0.3.0) defaults.

    Tests the top two counts (first, second) with Wald's SPRT of p = 1/2 against p = p1,
    or with a mixture SPRT under a Beta(a, b) prior truncated to (1/2, 1). It draws
    samples in batches: each batch is the smallest number of further samples that would
    make the test stop if they all went to the leader. It stops when the log-likelihood
    ratio crosses either boundary. Defaults from the package: max_trials 256, alpha 0.05,
    SPRT p1 = 0.5001 and beta = 1 - 0.05 - 0.000024; mSPRT a = b = 1e6 and
    beta = 1 - 0.05 - 0.00006.
    """

    kind: str = "sprt"
    budget: int = 256
    alpha: float = 0.05
    beta: float | None = None
    p1: float = 0.5001
    prior: float = 1e6

    @property
    def name(self) -> str:
        return "ConSol-SPRT" if self.kind == "sprt" else "ConSol-mSPRT"

    def __post_init__(self) -> None:
        if self.beta is None:
            self.beta = 1 - 0.05 - (0.000024 if self.kind == "sprt" else 0.00006)
        self._logA = np.log((1 - self.beta) / self.alpha)
        self._logB = np.log(self.beta / (1 - self.alpha))

    def _llr(self, first: int, second: int) -> float:
        if self.kind == "sprt":
            return first * np.log(self.p1 / 0.5) + second * np.log((1 - self.p1) / 0.5)
        a = b = self.prior
        norm = 1 - betainc(a, b, 0.5)
        tail = 1 - betainc(first + a, second + b, 0.5)
        if tail <= 0:
            return np.inf
        h1 = betaln(first + a, second + b) + np.log(tail) - betaln(a, b) - np.log(norm)
        return h1 - np.log(0.5) * (first + second)

    def test(self, first: int, second: int) -> bool:
        llr = self._llr(first, second)
        return llr >= self._logA or llr <= self._logB

    def _trials(self, first: int, second: int, used: int) -> int:
        trials = 0
        for trials in range(0, self.budget + 1):
            if first + trials == 0:
                continue
            if self.test(first + trials, second):
                break
        return min(trials, self.budget - used)

    def run(self, stream: np.ndarray) -> Decision:
        t = _Tally()
        used = 0
        cap = min(self.budget, len(stream))
        while True:
            first, second = t.top_two()
            k = min(self._trials(first, second, used), cap - used)
            if k <= 0:
                break
            for x in stream[used:used + k]:
                t.add(int(x))
            used += k
        first, second = t.top_two()
        if first + second > 0 and self.test(first, second):
            return Decision(t.leader(), used, "stopped")
        return Decision(t.leader(), used, "budget" if used >= self.budget else "need_more")


def _log_beta_upper_half(a: float, b: float) -> float:
    """log of int_{1/2}^1 t^(a-1) (1-t)^(b-1) dt, stable when the integral is tiny.

    Uses 1 - I_{1/2}(a, b) = I_{1/2}(b, a), which avoids cancellation, and falls back to
    the exact series int_0^x u^(p-1) (1-u)^(q-1) du = x^p (1-x)^q / p * 2F1(p+q, 1; p+1; x),
    with u = 1 - t, p = b, q = a, x = 1/2, in arbitrary precision when even that underflows.
    """
    upper = betainc(b, a, 0.5)
    if upper > 1e-280:
        return betaln(a, b) + np.log(upper)
    import mpmath
    with mpmath.workdps(40):
        return float((a + b) * mpmath.log(0.5) - mpmath.log(b) + mpmath.log(mpmath.hyp2f1(a + b, 1, b + 1, 0.5)))


@dataclass
class MartingaleMajorityCertificate:
    """Martingale Majority Certificate (Cordero-Encinar and Duncan, 2025), Algorithm 1.

    Two e-processes, leader vs runner-up and leader vs all others, each a mixture over a
    Beta(a, b) prior truncated to (1/2, 1] (the paper's option A), updated one sample at a
    time with predictable per-round factors. Stop when both reach 1/epsilon; abstain (and
    return the majority) at the budget. The leader and runner-up are re-read from the
    counts before every draw, ties broken as in `_Tally`. Before the first valid answer
    there is no leader, so those draws leave both e-processes unchanged.
    """

    epsilon: float = 0.1
    budget: int = 40
    a: float = 0.5
    b: float = 0.5
    name: str = "MMC"

    def _factor(self, hits: int, misses: int, hit: bool) -> float:
        """Predictive factor 2 E[theta] (hit) or 2 E[1 - theta] (miss), theta ~ truncated posterior."""
        base = _log_beta_upper_half(self.a + hits, self.b + misses)
        if hit:
            return 2.0 * np.exp(_log_beta_upper_half(self.a + hits + 1, self.b + misses) - base)
        return 2.0 * np.exp(_log_beta_upper_half(self.a + hits, self.b + misses + 1) - base)

    def run(self, stream: np.ndarray) -> Decision:
        t = _Tally()
        s = f = o = 0
        log_run = log_oth = 0.0
        threshold = np.log(1.0 / self.epsilon)
        cap = min(self.budget, len(stream))
        for i in range(cap):
            ranked = t.ranked()
            A = ranked[0][0] if ranked else None
            B = ranked[1][0] if len(ranked) > 1 else None
            x = int(stream[i])
            if A is not None and x != NO_ANSWER:
                if x == A:
                    log_run += np.log(self._factor(s, f, True))
                    log_oth += np.log(self._factor(s, o, True))
                    s += 1
                elif x == B:
                    log_run += np.log(self._factor(s, f, False))
                    f += 1
                else:
                    log_oth += np.log(self._factor(s, o, False))
                    o += 1
            t.add(x)
            if log_run >= threshold and log_oth >= threshold:
                return Decision(t.leader(), i + 1, "stopped")
        if len(stream) < self.budget:
            return Decision(t.leader(), cap, "need_more")
        return Decision(t.leader(), cap, "abstained")
