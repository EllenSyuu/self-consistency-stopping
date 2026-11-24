# Implementation and benchmark audit

Branch `research-v0.3`. Same labels as `statistical_guarantee_audit.md`. Real-data numbers
are described qualitatively; exact values were reported to the author and are reproducible
with the commands given.

## 1. Confirmed defects and their status

| # | Defect | Severity | Status |
|---|---|---|---|
| 1 | **Answer classes were built with the grader's help.** Every sample the grader accepts is merged into one class; wrong answers are grouped by string only. A deployed system cannot do this. | High: affects every rule's accuracy and cost on MATH. | Quantified on two settings (§4). `prepare_sc_genrm.py` now also writes grader-free `online_labels`; `replay(correct=…)` scores them. `prepare_monkey_business.py` not yet converted. Published numbers not yet regenerated. |
| 2 | **ESC ran with window 5 on AIME and GPQA** while the README documented window 8. | Medium: two table cells and one token figure are wrong in v0.2. | Fixed (`w = 5 if task == "GSM8K" else 8`), regression test added. On AIME the corrected ESC uses about 40% more samples than published and spends more tokens than fixed N = 40 rather than fewer. |
| 3 | **MMC charged the first runner-up vote to the wrong process** (see the statistical audit §2.1). | Medium for validity, small for results. | Fixed, two regression tests. |
| 4 | **README finding 4 presents ConSol as breaking a guarantee.** | Wording, but it is a public claim about other authors' work. | Correct text drafted below; not applied to `main` without approval. |

Proposed replacement for README finding 4: "MMC's certificate held in every setting […].
ConSol states no mode-identification guarantee; with its default parameters its test is
exactly 'stop when the leader is three votes ahead' (derivation in
`research/statistical_guarantee_audit.md`), so on problems whose two leading answers are
close it often returns the runner-up."

## 2. Rule configurations against their sources

| Rule | Setting used | Source | Status |
|---|---|---|---|
| Adaptive-Consistency | Beta criterion, threshold 0.95, budget 40, one sample at a time | Aggarwal et al. 2023, Eq. 2 and experiments | **VERIFIED** against the paper text retrieved. No minimum sample count is stated there. |
| ESC | w = 5, L = 40 (GSM8K); w = 8, L = 64 (MATH) | Li et al. 2024 | **VERIFIED** for GSM8K and MATH. For AIME and GPQA the paper has no setting; using the MATH setting is **our choice** and must be labelled so. |
| ConSol | package 0.3.0 defaults, budget 256 | authors' code | **VERIFIED** (state-by-state agreement). |
| MMC | truncated Beta(½, ½), ε ∈ {0.1, 0.05}, budget 40/64 | Cordero-Encinar & Duncan 2025, Alg. 1, App. B.6 (N = 64, ε = 0.1) | **VERIFIED** for the factors; the runner-up-before-second-answer convention is ours. |
| Budgets 40 / 64 / 256 | each paper's own | — | **CONDITIONAL.** Rules are compared at unequal budgets. This is the "author-recommended configuration" comparison only; equal-budget and equal-token comparisons are not yet run. A rule with budget 256 can exceed any accuracy reachable at budget 40. |

## 3. Sampling protocol

| Item | Finding | Status |
|---|---|---|
| Draws without replacement from each problem's pool | Finite-population sampling; negatively dependent draws. Rerunning three settings with replacement changed stopped-mode-error rates by at most about one point. | **VERIFIED** small for pools ≥ 1,024. `replay(replace=True)` added. |
| Pools of 256 with budgets up to 256 | Fixed N = 256 then returns the pool mode by construction, so "matches mode" is 1.000 there and the top of the fixed-N curve is an artefact. | **VERIFIED** defect of presentation; those points should be dropped or drawn with replacement. |
| 200 orders per problem | Orders are replicates of the same problems, not independent tasks. | **VERIFIED** handled: intervals resample problems only; per-problem means are taken over orders first. |
| Paired comparisons | Every rule sees identical orders; the bootstrap uses the same problem resamples for every rule. Intervals of **differences** between rules are not yet reported. | **CONDITIONAL**: the machinery is paired, the reported intervals are marginal. "Matches fixed N = 40 to within 0.1 point" is a point estimate, not a non-inferiority result. |
| Information unavailable online | Besides §1 item 1: none found. Rules see only the prefix; ties are broken without look-ahead (tested: online and offline decisions coincide). | **VERIFIED**. |

## 4. Grader-informed answer classes (defect 1)

On MATH with Qwen-2.5-7B and Llama-3.3-70B, the grader accepts on average 2.0 and 1.7
distinct answer strings per problem (up to 16). Replaying the same orders with classes
formed from the normalised string alone:

* every rule loses accuracy, by roughly half a point to a point and a half;
* every adaptive rule needs more samples, by a few percent up to about a quarter;
* single-sample accuracy is unchanged, and the ordering of the rules is unchanged.

Reproduce: `python scripts/prepare_sc_genrm.py --out /tmp/d --only MATH_Qwen-2.5-7B-Instruct`,
then `replay(z["online_labels"], rules, …, correct=z["online_correct"])`.

Status: the v0.2 tables are **CONDITIONAL** on an oracle canonicaliser. GSM8K (numeric
answers), AIME (integers) and GPQA (letters) are expected to be nearly unaffected:
**UNVERIFIED** until rerun.

## 5. Invalid answers, ties, normalisation, grading

| Item | Status |
|---|---|
| Samples with no extractable answer cost a call and cast no vote, for every rule. | **VERIFIED** (tests). ConSol's package instead drops them before counting; ours is the stricter accounting. |
| Ties go to the answer that first reached the tied count. | **VERIFIED** (tests). |
| Wrong answers that are mathematically equal but written differently vote separately. | **VERIFIED** limitation; it pushes against early stopping on wrong answers and so flatters accuracy slightly. Not quantified. |
| MATH grading by math-verify; 13 comparisons timed out and count as wrong. | **VERIFIED** count; a spot check of 8 random problems found no misgraded class among the five most frequent answers. Not a full audit: **CONDITIONAL**. |
| Large Language Monkeys correctness flags | Taken from the dataset as released: **UNVERIFIED** by us. |

## 6. Reference mode under a finite pool

Counting problems whose two most frequent answers are not separable by a two-sided binomial
test at 1%: none to a handful on GSM8K and AIME, but a quarter or more of the problems in
the weakest MATH settings and in MATH/Llama-3.1-8B (pool 1,024). On those problems
"non-modal" largely means "the other of two nearly tied answers". Restricting to separable
problems lowers ConSol's stopped-mode-error by roughly a third to a half and leaves MMC's
at essentially zero.

Recommendation: report mode error on separable problems, or against an ε-mode set (all
answers within a margin of the maximum), in addition to the pooled rate.

## 7. Bootstrap

Problems are resampled; orders are not treated as independent (VERIFIED). The conditional
rate is a ratio estimator, bootstrapped with stop counts as weights (VERIFIED in code).
Not done: intervals for between-rule differences; stratification by task; any correction
for selecting the "best" rule after seeing results.

## 8. Reproducing the real-data statements

```bash
python scripts/prepare_monkey_business.py --out data && python scripts/prepare_sc_genrm.py --out data
python scripts/run_benchmark.py --data data --out /tmp/results_v03 --replicates 200
python research/scripts/parallel_race.py --workers 8 --replicates 40      # QwQ-32B race
```

ConSol prediction from pool frequencies, per setting: for each problem take the two largest
label counts c1 ≥ c2, p = c1/(c1 + c2), predicted error 1/(1 + (p/(1 − p))³), average over
problems; compare with `mode_error_when_stopped` of `ConSol-SPRT`.
