# Research directions

## 0. First question: do heterogeneous costs change optimal stopping?

Model: generations t = 1, 2, … produce i.i.d. pairs (Y_t, C_t), answer and cost. The pair
may be dependent (long generations may favour some answers). A procedure is a stopping time
τ for the filtration F_t = σ((Y_s, C_s), s ≤ t) and a decision â_τ; it is δ-correct if
P(â_τ ≠ a*) ≤ δ for every law with a unique mode.

**Proposition 0 (reduction; PROVED).** For every stopping time τ with E[τ] < ∞ and
E[C] < ∞, E[Σ_{t≤τ} C_t] = E[C]·E[τ].
*Proof.* Σ_{t≤τ} C_t = Σ_t C_t·1{τ ≥ t}. The event {τ ≥ t} = {τ ≤ t−1}ᶜ is in F_{t−1} and
(Y_t, C_t) is independent of F_{t−1}, so E[C_t 1{τ ≥ t}] = E[C]·P(τ ≥ t). Sum over t
(Wald's identity). ∎

So when the cost is realised only after generation and the procedure cannot act on it,
**minimising expected cost is minimising the expected number of samples**, whatever the
dependence between cost and answer. Two remarks complete the picture:

* *Costs as side information.* If the conditional law of C given Y is a nuisance unrelated
  to p, observing C adds no information about the mode (the likelihood factorises), so the
  lower bound T_bb·log(1/δ) on E[τ] stands and the cost lower bound is E[C]·T_bb·log(1/δ).
* *Costs known or predicted before the request*, with no control over which request to
  make, change nothing either: the request must be made to obtain the next sample.

**Conclusion.** "Cost-aware anytime-valid stopping" in the sequential answer-only model is
the classical problem with a relabelled axis. A cost-aware theorem there would be
manufactured. Tokens still belong in the *evaluation*; they do not create a new stopping
principle. (Direction A below is therefore closed.)

Cost creates a genuinely new problem only where the procedure's **actions interact with
generation length**:

1. cancelling generations in flight (parallel sampling) — the answers seen so far are then
   selected by length;
2. truncating generations at a token limit — a censoring choice that changes both cost and
   what is observed;
3. choosing among sampling configurations with different answer laws and cost laws.

## Directions compared

| | Direction | Technically defensible? | New relative to the matrix? | Data available free? | Verdict |
|---|---|---|---|---|---|
| A | Cost-aware stopping with predictable heterogeneity | Reduces to the classical problem (Prop. 0) | No | — | **Closed.** |
| B | Adaptive choice among sampling configurations | Yes, but configurations have different modes; the target must be redefined (mode of a reference configuration) and cheap configurations are then biased proxies. Becomes multi-fidelity best-arm identification. | Partly (multi-fidelity BAI is established) | No: needs the same problems sampled under several configurations | Defer. |
| C | Certified stop, else abstain or escalate to a stronger model | Yes, but close to BARS (refusal), SAFER (abstention), cascade literature | Low | Partly | Engineering feature, not the paper. |
| **D** | **Valid stopping under parallel generation, where consuming answers in completion order selects by length** | Yes: elementary counterexample, exact fix, and a quantifiable price | **Appears open** (pending a full read of MARS, short-m@k, FFS, Glynn–Heidelberger) | **Yes**: QwQ-32B AIME 2025, 512 samples per problem with token counts | **Lead direction.** |
| E | Risk-controlled allocation across a workload | Yes | Low: PETS, SAFER, PAC-Bayes certificates | Yes | Crowded. |

## Direction D in one paragraph

Every guarantee in the matrix assumes the answers reach the rule as an i.i.d. sequence.
Production systems generate K traces in parallel and want to cancel the rest as soon as the
vote is settled. If the rule consumes answers as they finish, then at any stopping moment
the finished answers over-represent short generations. Answer and length are dependent in
reasoning models (documented by short-m@k and First Finish Search; in our QwQ-32B data
wrong answers are on average almost twice as long as correct ones). The i.i.d. assumption
fails precisely because of the stopping.

Evidence so far:

* **Synthetic counterexample (VERIFIED by simulation).** Minority answer with probability
  0.3 and length 1; mode with probability 0.7 and length 2. With 16 workers, MMC at ε = 0.1
  certifies the minority answer in about 32% of runs; with a length ratio of 10, about 64%.
  Serial sampling: 0.05%. (`race_counterexample.py`)
* **Exact fix (PROVED, `main_theorem.md` Thm 1).** Consume answers in launch order.
  Validity of any i.i.d.-valid rule is restored exactly, for any dependence between answer
  and length. The counterexample's error returns to 0.05%.
* **Real data (measured, small sample).** On QwQ-32B/AIME 2025 with 8 workers, completion
  order raises the stopped-wrong-mode rate only slightly for Adaptive-Consistency and
  ConSol and not at all for MMC, because there the short answer is usually the mode. The
  launch-order fix costs roughly 5–15% more tokens than completion order and keeps most of
  the wall-clock gain over serial sampling. So on this dataset the hazard is latent, not
  realised: the honest framing is "a certificate that is void in a regime practitioners
  are moving into", not "current systems are badly wrong".

The research question is then quantitative: **what is the cheapest valid way to use K
workers?** Launch order wastes the finished answers stuck behind a slow one. Candidates and
theory are in `main_theorem.md`.

## The benchmark on its own

A rigorous evaluation with the ConSol characterisation, grader-free labels and token
costs is useful in its own right; the experimental protocol for it is in
`experimental_protocol.md`.
