"""Each rule against its published definition, plus the shared conventions."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import betainc

from scstop import (AdaptiveConsistencyBeta, ConSol, EarlyStoppingSC, FixedN,
                    MartingaleMajorityCertificate, majority, pool_mode, replay, summarise)
from scstop.rules import _log_beta_upper_half

A, B, C, NA = 1, 2, 3, -1


def test_majority_breaks_ties_by_first_to_reach():
    assert majority([A, B, B, A]) == B          # B reached 2 votes first
    assert majority([A, B]) == A
    assert majority([NA, NA]) is None
    assert majority([NA, B, NA]) == B


def test_fixed_n_uses_exactly_n_samples():
    d = FixedN(3).run(np.array([A, B, B, A, A]))
    assert (d.answer, d.samples, d.status) == (B, 3, "budget")


def test_adaptive_consistency_matches_beta_formula():
    # unanimous stream: P = 1 - I_{1/2}(n + 1, 1) = 1 - 2^-(n+1) > 0.95 first at n = 4
    d = AdaptiveConsistencyBeta().run(np.full(40, A))
    assert (d.answer, d.samples, d.status) == (A, 4, "stopped")
    rng = np.random.default_rng(0)
    stream = rng.choice([A, B, C], size=40, p=[.5, .3, .2])
    d = AdaptiveConsistencyBeta().run(stream)
    counts = {}
    for i, x in enumerate(stream[: d.samples]):
        counts[x] = counts.get(x, 0) + 1
    v = sorted(counts.values(), reverse=True) + [0]
    if d.status == "stopped":
        assert 1 - betainc(v[0] + 1, v[1] + 1, 0.5) > 0.95


def test_adaptive_consistency_respects_budget():
    d = AdaptiveConsistencyBeta(budget=6).run(np.array([A, B] * 20))
    assert (d.samples, d.status) == (6, "budget")


def test_esc_stops_on_first_unanimous_window_and_returns_its_answer():
    stream = np.array([A, B, A, A, A] + [B] * 5 + [A] * 30)
    d = EarlyStoppingSC(window=5, budget=40).run(stream)
    assert (d.answer, d.samples, d.status) == (B, 10, "stopped")


def test_esc_window_with_missing_answer_is_not_unanimous():
    stream = np.array([A, A, NA, A, A] + [B, A] * 20)
    d = EarlyStoppingSC(window=5, budget=10).run(stream)
    assert d.status == "budget" and d.samples == 10


def test_consol_sprt_stops_at_a_lead_of_three():
    # log LR = lead * log(0.5001 / 0.5) must reach log((1 - beta) / alpha)
    d = ConSol("sprt").run(np.full(256, A))
    assert (d.answer, d.samples, d.status) == (A, 3, "stopped")
    d = ConSol("sprt").run(np.array([A, B, A, A, A] + [A] * 50))
    assert d.samples == 5 and d.status == "stopped"


def test_consol_batches_like_the_package():
    # batches are the fewest further samples that would stop the test if all went to the
    # leader: 3 (A B A), then 2 (B B: B leads 3-2), then 2 (B B: B leads 5-2, stop)
    rule = ConSol("sprt")
    d = rule.run(np.array([A, B, A, B, B, B, B] + [A] * 50))
    assert (d.answer, d.samples, d.status) == (B, 7, "stopped")
    assert rule._trials(1, 1, 2) == 3


def test_mmc_closed_form_matches_recursion_for_a_constant_leader():
    rule = MartingaleMajorityCertificate(epsilon=1e-12, budget=12)
    a, b = rule.a, rule.b
    log_e = 0.0
    for n in range(10):
        log_e += np.log(rule._factor(n, 0, True))
    closed = 10 * np.log(2) + _log_beta_upper_half(a + 10, b) - _log_beta_upper_half(a, b)
    assert log_e == pytest.approx(closed, rel=1e-10)


def test_mmc_stops_on_a_clear_majority_and_certifies_it():
    d = MartingaleMajorityCertificate(epsilon=0.1, budget=200).run(np.full(200, A))
    assert d.answer == A and d.status == "stopped" and d.samples <= 10


def test_mmc_type_one_error_is_controlled_when_the_top_two_tie():
    """Ville: under p_A = p_B the rule should certify (stop) with probability <= epsilon."""
    rng = np.random.default_rng(7)
    eps, reps = 0.1, 400
    rule = MartingaleMajorityCertificate(epsilon=eps, budget=400)
    stops = sum(rule.run(rng.choice([A, B], size=400)).status == "stopped" for _ in range(reps))
    assert stops / reps <= eps + 3 * np.sqrt(eps * (1 - eps) / reps)


def test_replay_feeds_every_rule_the_same_streams_and_summarises():
    rng = np.random.default_rng(1)
    labels = rng.choice([0, 1, 2, -1], size=(5, 1000), p=[.6, .2, .15, .05]).astype(np.int16)
    res = replay(labels, [FixedN(40), FixedN(41)], replicates=7, max_len=64, seed=3)
    assert res[0].correct.shape == (5, 7)
    # identical streams: the first 40 votes agree, so answers differ only by the 41st sample
    assert (res[0].correct == res[1].correct).mean() > 0.9
    s = summarise(res[0])
    assert 0 <= s["accuracy"] <= 1 and s["mean_samples"] == 40
    assert pool_mode(np.array([NA, NA, A, B, B])) == B


@pytest.mark.parametrize("a,b,ref", [
    (10.5, 3.5, -7.434775305918952),
    (1.5, 900.5, -631.3280056782984),        # references: 50-digit quadrature with fine knots
    (3.5, 2000.5, -1395.9737058713667),
    (0.5, 5000.5, -3474.2532959328264),
])
def test_upper_half_beta_integral_is_accurate_when_tiny(a, b, ref):
    assert _log_beta_upper_half(a, b) == pytest.approx(ref, abs=1e-8)


@pytest.mark.parametrize("rule", [FixedN(10), AdaptiveConsistencyBeta(), EarlyStoppingSC(5, 40), ConSol("sprt"),
                                  MartingaleMajorityCertificate(budget=40)])
def test_a_short_undecided_stream_asks_for_more_samples(rule):
    d = rule.run(np.array([A, B]))
    assert d.status == "need_more" and d.samples <= 2


def test_online_use_reaches_the_same_decision_as_a_full_stream():
    rng = np.random.default_rng(5)
    stream = rng.choice([A, B, C], size=64, p=[.6, .3, .1])
    rule = MartingaleMajorityCertificate(epsilon=0.1, budget=64)
    for t in range(1, 65):
        d = rule.run(stream[:t])
        if d.status != "need_more":
            break
    assert d == rule.run(stream)


def test_token_cost_is_the_length_of_the_samples_drawn():
    labels = np.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=np.int16)
    tokens = np.array([[10, 20, 30, 40, 50, 60, 70, 80]])
    res = replay(labels, [FixedN(3), FixedN(8)], replicates=50, max_len=8, seed=0, tokens=tokens)
    assert np.all(res[1].tokens == tokens.sum())
    assert 60 <= res[0].tokens.min() and res[0].tokens.max() <= 210


def test_summary_intervals_contain_the_estimate_and_pair_across_rules():
    rng = np.random.default_rng(2)
    labels = rng.choice([0, 1, 2], size=(40, 500), p=[.5, .3, .2]).astype(np.int16)
    res = replay(labels, [FixedN(5), AdaptiveConsistencyBeta()], replicates=5, max_len=64, seed=1)
    for r in res:
        s = summarise(r)
        lo, hi = s["accuracy_ci"]
        assert lo <= s["accuracy"] <= hi
        lo, hi = s["mean_samples_ci"]
        assert lo <= s["mean_samples"] <= hi
