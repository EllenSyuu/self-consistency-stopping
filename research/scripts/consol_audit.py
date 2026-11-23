"""ConSol's stopping test, analysed exactly. Synthetic distributions only; no benchmark data.

    python research/scripts/consol_audit.py [--authors-file path/to/consol/confidence_models.py]

Prints: (1) the log-likelihood-ratio boundaries implied by the package defaults; (2) the
stopping region as a function of (first, second), ours and optionally the authors' own
class; (3) exact error of the resulting vote-margin rule against Monte Carlo of the full
batched procedure; (4) the type-I error of the textbook fixed-pair SPRT at these settings.
"""
from __future__ import annotations

import argparse
import importlib.util

import numpy as np

from scstop import ConSol

P0, P1, ALPHA = 0.5, 0.5001, 0.05
BETA = 1 - 0.05 - 0.000024


def boundaries() -> None:
    up, down = np.log(P1 / P0), np.log((1 - P1) / (1 - P0))
    log_a, log_b = np.log((1 - BETA) / ALPHA), np.log(BETA / (1 - ALPHA))
    print("1. Boundaries")
    print(f"   step per leader vote {up:+.6e}, per runner-up vote {down:+.6e}")
    print(f"   log A = {log_a:+.6e}  ->  stop when lead >= {log_a / up:.3f}, i.e. lead >= {int(np.ceil(log_a / up))}")
    print(f"   log B = {log_b:+.6e}  ->  with lead 0 and k votes each, LLR = k*{up + down:.3e}; "
          f"crossed at k >= {int(np.ceil(log_b / (up + down)))} (budget is 256 samples)")
    print(f"   Wald's inequality bounds the type-I error by alpha/(1-beta) = {ALPHA / (1 - BETA):.4f}")


def stopping_region(authors_file: str | None) -> None:
    ours = {kind: ConSol(kind) for kind in ("sprt", "msprt")}
    theirs = {}
    if authors_file:
        spec = importlib.util.spec_from_file_location("cm", authors_file)
        cm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cm)
        theirs = {"sprt": cm.SprtConfidenceModel(), "msprt": cm.MsprtConfidenceModel()}
    print("2. Stopping region over all (first >= second) with first + second <= 256")
    for kind, rule in ours.items():
        leads, disagree, lower = set(), 0, 0
        for second in range(0, 129):
            for first in range(second, 257 - second):
                stop = rule.test(first, second)
                if kind in theirs and bool(theirs[kind].test(first, second)) != stop:
                    disagree += 1
                if stop and first - second < 3:
                    lower += 1
                if stop != (first - second >= 3):
                    leads.add((first, second))
        print(f"   {kind}: states where 'stop' differs from 'lead >= 3': {len(leads)}"
              f" (all are lower-boundary stops at near-ties: {lower == len(leads)}; smallest first+second "
              f"{min((f + s for f, s in leads), default=None)})"
              + (f"; disagreements with the authors' class: {disagree}" if kind in theirs else ""))


def ruin_error(p: float, lead: int = 3) -> float:
    """P(the less likely of two answers gets `lead` ahead first), mode share p among the two."""
    r = p / (1 - p)
    return 1.0 / (1.0 + r ** lead)


def margin_rule(rng: np.random.Generator) -> None:
    print("3. Two answers, mode share p: exact error of 'first to lead by 3' vs Monte Carlo of ConSol-SPRT")
    rule = ConSol("sprt")
    for p in (0.5, 0.52, 0.55, 0.6, 0.7, 0.8, 0.9):
        reps, wrong, used, stopped = 4000, 0, 0, 0
        for _ in range(reps):
            d = rule.run(np.where(rng.random(256) < p, 1, 2))
            wrong += d.answer != 1
            used += d.samples
            stopped += d.status == "stopped"
        print(f"   p = {p:.2f}: exact {ruin_error(p):.4f}   simulated {wrong / reps:.4f}"
              f"   stopped {stopped / reps:.3f}   mean samples {used / reps:.1f}")
    print("   Three answers (0.40, 0.35, 0.25), Monte Carlo only:")
    reps, wrong = 4000, 0
    for _ in range(reps):
        wrong += rule.run(rng.choice([1, 2, 3], size=256, p=[0.40, 0.35, 0.25])).answer != 1
    print(f"   returned a non-modal answer in {wrong / reps:.4f} of runs")


def fixed_pair_type_one() -> None:
    """Textbook SPRT on a fixed labelled pair under H0: p' = 1/2.

    The walk starts at 0, rejects at lead +3, accepts at lead -1 (the first state below
    log B). For a symmetric walk P(hit +3 before -1) = 1/4.
    """
    print("4. Fixed-pair SPRT under the null p' = 1/2 (no leader re-selection): "
          "P(reject) = P(walk hits +3 before -1) = 1/4 = 0.25, against nominal alpha = 0.05")
    print("   With the leader re-selected every round, as in Algorithm 1, a tie ends in 'reject' "
          "as soon as either answer leads by 3: probability ~1 within the budget.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors-file")
    args = ap.parse_args()
    boundaries()
    stopping_region(args.authors_file)
    margin_rule(np.random.default_rng(20261123))
    fixed_pair_type_one()
