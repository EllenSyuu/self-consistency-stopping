# Experimental protocol (to be frozen before the next real-data run)

Nothing in this file has been executed as a confirmatory study. The numbers obtained so far
on real samples are exploratory.

## Data

| Set | Use | Licence | Status |
|---|---|---|---|
| Large Language Monkeys (7 settings, 10,000 samples per problem) | sample-count comparisons | MIT | in use; grader-free labels still to be generated |
| sc-genrm-scaling: QwQ-32B AIME 2025 (512 × 30, token counts), GPQA-Diamond, MATH128 × 3 models | tokens; parallel race | Apache-2.0 | in use |
| Candidates not yet inspected for licence and structure: jonathanyin AIME 1983–2023 traces (DeepSeek-R1, QwQ-32B, R1 distills, with token counts; no licence stated), vectorzhou AIME 2024/2025 DeepSeek-R1-0528 (Apache-2.0, 16 responses per problem), CoffeeGitta difficulty-AIME generations (MIT, cost fields, k small) | more reasoning models | mixed | **licence check required**; sets without a licence are not used |

Leakage: AIME 2025 post-dates QwQ-32B's training cut-off by the dataset authors' account
(UNVERIFIED by us); MATH and GSM8K test items are widely contaminated, which affects
absolute accuracy, not comparisons between stopping rules on the same samples.

## Answer classes

Primary analysis: **grader-free classes** (normalised answer string). Secondary: grader-merged
classes, reported as an oracle-canonicaliser upper bound. Robustness: a stricter and a looser
normaliser.

## Comparisons

1. Author-recommended configuration of each rule (what v0.2 reports).
2. Equal maximum budget (all rules at 40, and all at 64).
3. Equal expected output tokens, where token counts exist.
4. Pareto frontiers: each rule's own parameter swept (threshold, window, ε, vote margin).
5. Tuned on a held-out half of the problems, evaluated on the other half.

## Metrics

Accuracy; output tokens; samples; unconditional P(stopped and non-modal); the same
conditional on stopping; both on all problems and on problems with a separable mode; share
of runs stopped; mean, 90th and 99th percentile of tokens per question; cost split by
problem difficulty tercile. For the parallel study: wall time in token-units, tokens
including cancelled partial generations, for serial / completion-order / launch-order /
windowed launch-order, K ∈ {2, 4, 8, 16, 32}.

## Uncertainty

* Resample problems (never orders). Report intervals for **differences** between rules on
  the same resamples.
* Equivalence claims use a pre-stated non-inferiority margin: 0.5 accuracy point against
  fixed N = 40. "No significant difference" is never reported as equivalence.
* With 30 AIME problems, per-rule accuracy intervals are about ±14 points; only paired
  differences are interpretable there.

## Synthetic grid (pre-registered before any confirmatory run)

Answer laws: two-way and three-way near-ties (gaps 0.02, 0.05, 0.1, 0.2), a dominant mode
with a long tail, uniform noise. Length laws: independent of the answer; mode shorter;
mode longer, with ratios 1.5, 2, 5, 10. Workers 1–64. Held-out scenarios, drawn after the
rules and policies are frozen: three laws fitted to QwQ-32B problems not used in tuning.

## Stop conditions

The parallel-validity study is dropped as the lead direction if (a) a full read of MARS,
short-m@k, First Finish Search or Glynn–Heidelberger shows the validity result is known in
this form, or (b) on at least three reasoning-model datasets completion order never moves
the certified error by more than the Monte Carlo error. In case (b) it is reported as a
negative finding inside the benchmark paper.
