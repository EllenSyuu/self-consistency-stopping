# Statistical guarantee audit

Branch `research-v0.3`, 23 November 2026. Labels: **VERIFIED** (shown here by derivation,
exact computation or the cited source), **CONDITIONAL** (true under stated assumptions that
the benchmark does not check), **UNVERIFIED**, **FALSE**.

Numbers in this file are analytic or come from synthetic distributions
(`research/scripts/`). Numbers measured on the recorded model samples are not committed;
each claim names the script that reproduces them.

## 1. ConSol

Sources read: Lee et al., *ConSol: Sequential Probability Ratio Testing to Find Consistent
LLM Reasoning Paths Efficiently*, arXiv:2503.17587v1, Sections 2–3, Appendices A–B and the
limitations section; the authors' package `consol` 0.3.0 (`confidence_models.py`,
`confident_solver.py`).

### 1.1 What the paper claims

| Claim | Status |
|---|---|
| The paper states a theorem bounding the probability of returning a non-modal answer. | **FALSE.** No theorem or proof of error control appears. The paper says SPRT allows "rigorously controlling Type I and Type II errors" and that the method "carefully controls Type I error", without a bound for its own procedure. |
| The null is H0: p' = 1/2 against H1: p' = 0.5001 (SPRT) or p' > 1/2 with a Beta(1e6, 1e6) prior truncated to (1/2, 1] (mSPRT), where p' is the share of the leading answer among the two most frequent. | **VERIFIED** (Sec. 2.3; package defaults). |
| Thresholds are A = (1 − β)/α and B = β/(1 − α), with α = 0.05 and β = 0.949976 (SPRT), 0.94994 (mSPRT). | **VERIFIED** (Appendix A; package defaults). The paper chooses "β ≈ 1 − α" on purpose, to stop early. |
| The top two answers are re-selected at every step. | **VERIFIED** (Algorithm 1). The paper does not analyse the effect on error rates. |

### 1.2 What α controls with these parameters

Let the log-likelihood ratio after `first` leader votes and `second` runner-up votes be
`L = first·log(p1/p0) + second·log((1−p1)/(1−p0))` with p0 = 1/2, p1 = 0.5001.

* Step sizes: +1.9998e-4 per leader vote, −2.0002e-4 per runner-up vote.
* Upper boundary: log A = log(0.050024/0.05) = 4.7988e-4. It is reached when the lead is at
  least 2.4, that is, **at a lead of 3 votes**.
* Lower boundary: log B = −2.5263e-5. With the leader ahead or level it is reached only at
  an exact tie after 632 votes each, beyond the 256-sample budget.

So, for every reachable state, **ConSol-SPRT with its default parameters stops exactly when
the leading answer is 3 votes ahead of the runner-up.** VERIFIED by enumerating all states
with first + second ≤ 256 against both our implementation and the authors' own class
(0 disagreements; `consol_audit.py`). The mSPRT differs from the same rule in 2 of those
states, both lower-boundary stops at near-ties with 254 or more votes.

Consequences, all VERIFIED:

1. **Wald's inequality is vacuous here.** For a fixed pair the SPRT guarantees
   P(reject H0 | H0) ≤ α/(1 − β) = 0.05/0.050024 = 0.9995. The nominal α = 0.05 is a
   bound on the type-I error only through Wald's no-overshoot approximation, which fails
   because one vote moves the statistic by eight times the distance to the lower boundary.
2. **Fixed-pair type-I error is 1/4.** With the pair fixed in advance and p' = 1/2, the walk
   rejects at +3 and accepts at −1, so P(reject) = 1/4.
3. **With leader re-selection, a tie always ends in "reject".** Either answer reaching a
   3-vote lead stops the procedure, which happens with probability ≈ 1 within the budget.
4. **Exact error for two answers.** If the mode has share p among the two, the probability
   that the other answer leads by 3 first is 1/(1 + (p/(1−p))³): 0.354 at p = 0.55, 0.229 at
   0.60, 0.073 at 0.70, 0.015 at 0.80. Monte Carlo of the full batched procedure agrees
   within simulation error (`consol_audit.py`, part 3).
5. **The same formula explains the benchmark.** Applying it to each problem's two most
   frequent recorded answers predicts ConSol's measured rate of non-modal answers in all 12
   settings to within 3.5 points (reproduce: the snippet in `implementation_audit.md` §8).

### 1.3 Classification of the v0.2 finding

The directive's options, for "ConSol at α = 0.05 returns a non-modal answer in up to 26.9%
of stops":

| Option | Verdict |
|---|---|
| A. Violation of a stated theorem | **No.** There is no such theorem. |
| B. Our misreading of its guarantee | **Partly yes.** v0.2 placed ConSol's rate beside lines at 5% and 10% under the heading '"Guaranteed" does not always mean what it says', which reads as a broken guarantee. ConSol never guaranteed mode identification. |
| C. A valid guarantee for a different event | **No valid nominal-level guarantee exists for any event**: even the fixed-pair type-I error is 0.25, and the only rigorous bound is 0.9995. |
| D. Implementation difference | **No.** Our stopping test agrees with the authors' class in every reachable state. Two documented protocol differences remain (unparsable outputs count as spent calls; shared tie-break) and do not affect the stopping test. |
| E. Leader re-selection / sequential testing | **Contributes**, but the main cause is the parameter choice (β ≈ 1 − α, p1 = 0.5001), which collapses the test to a 3-vote margin. |
| F. Answer grouping or finite pools | **No.** The rate is reproduced with replacement and predicted from pool frequencies; it is concentrated on problems whose two leading answers are close. |

**What survives:** the measured behaviour, and the statement "ConSol with default parameters
is the rule 'stop at a 3-vote lead'; its α is not an error rate for mode identification".
**What must be withdrawn:** any wording that ConSol breaks a guarantee it made. The
correction to the README is listed in `implementation_audit.md` §1 and awaits approval for
`main`. Nothing here should be communicated to the authors as a theoretical error; it is a
characterisation of a parameter choice the paper itself describes as trading evidence for
early stopping (Sec. 2.4.2).

## 2. Martingale Majority Certificate

Sources read: Cordero-Encinar & Duncan, arXiv:2510.17472, Section 3, Algorithm 1 and
Appendix B (Lemma B.1, Theorem B.2 = Theorem 3.1). Corollary 3.2 and Appendix B.5 were not
in the retrieved text. UNVERIFIED for those parts.

| Claim | Status |
|---|---|
| Each per-round factor is a valid e-value for the null "the current leader is not more likely than the current runner-up" (resp. "than all other answers pooled"), with the leader and runner-up chosen from past data. | **VERIFIED.** For the truncated-Beta mixture the factors are 2E[θ] and 2E[1−θ] under the current posterior; under θ ≤ 1/2 their mean is at most 1, with equality at 1/2. Predictable choice of the pair preserves this by conditioning. |
| Theorem 3.1: sup-probability that either process reaches 1/ε is at most ε **under the null that the per-round inequality holds at every round**. | **VERIFIED** as stated (Ville's inequality). |
| At the stopping time the returned answer is the mode with probability at least 1 − ε. | **CONDITIONAL / not proved in the text read.** The event "stopped on a non-modal answer" does not imply the per-round null at every round: in rounds where the true mode leads, the factors have mean above 1 and the processes grow legitimately; a later change of leader inherits that growth. The paper's stated aim P(ĉ ≠ c*) ≤ ε does not follow from Theorem 3.1 without an extra argument. |
| Empirically the certificate holds on the stress cases tried. | **VERIFIED on 9 synthetic distributions** (`mmc_audit.py`, ε = 0.1, budget 300, 2,000 runs each, v0.2 code): P(stop on a non-unique-mode answer) between 0.001 and 0.044 when the mode is unique. This is evidence, not a proof; no counterexample was found. |
| At an exact two-way tie the stop probability is at most ε. | **FALSE for the v0.2 implementation, VERIFIED after the fix.** See below. |

### 2.1 A defect in our v0.2 implementation (fixed)

While only one answer had been seen, v0.2 had no runner-up, and the first vote for a
different answer was scored as an "other" vote. The leader-versus-runner-up process then
collected its gains on leader votes without ever paying for the vote that creates the
runner-up. With two equally likely answers the process is not a martingale: expected factor
on that round is ½·2E[θ] + ½ > 1.

* Before the fix: 1,514 stops in 14,000 runs at an exact tie = 0.108 (standard error
  0.0026), above ε = 0.1.
* After the fix (the first different answer is scored as a runner-up vote): 823 in 12,000 =
  0.069.

The paper's Algorithm 1 does not say what the runner-up is before a second answer exists, so
this is our defect, not the paper's. Regression tests:
`test_mmc_first_runner_up_vote_is_charged_to_the_leader_vs_runner_up_process`,
`test_mmc_stop_probability_at_an_exact_two_way_tie_is_within_epsilon`. Effect on the
benchmark: MMC's accuracy changes by at most 0.1 point and its mean samples by at most 1.3
in any setting (rerun, not committed).

## 3. Adaptive-Consistency and ESC

| Claim | Status |
|---|---|
| Adaptive-Consistency's 0.95 threshold is a frequentist error bound. | **FALSE**, and not claimed by its authors. It is a posterior probability under a uniform prior on the top-two share, evaluated at a data-dependent time. |
| ESC carries an error guarantee. | **FALSE**, and not claimed. It is a unanimity heuristic; the paper gives a bound relating window size to the probability that the window answer equals the full-sample answer (not re-derived here: UNVERIFIED). |

## 4. Definitions used by the benchmark

| Quantity | Definition | Status |
|---|---|---|
| Reference mode | Most frequent valid label among all recorded samples of the problem. | **CONDITIONAL.** It is the exact mode of the finite pool. As an estimate of the model's mode it is uncertain: in 0 to 44 problems per setting the two leading answers are not separable at the 1% level (`implementation_audit.md` §6). |
| "Mode error when stopped" | Share of stopped runs whose answer is not the reference mode, pooled over problems and orders. | **VERIFIED** as computed; it is a mixture over problems, not a per-problem error probability, and it is conditional on stopping. The unconditional rate (stopped *and* non-modal) is lower and should be reported beside it. |
| Any α or ε compared with these rates | Per-problem, unconditional probabilities in the papers. | The v0.2 figure compared pooled conditional rates with per-problem unconditional levels. For MMC the conclusion is unaffected (its rates are far below ε on both definitions); for ConSol see §1.3. |
