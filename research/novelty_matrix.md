# Novelty matrix (literature through November 2026)

**Reading status** is stated per row. "Read" means the method and theory sections were
retrieved and read in this audit; "abstract" means only the abstract or a search excerpt;
"memory" means cited from prior knowledge and not re-read. Nothing below the "Read" level
may be cited in a paper before it is read in full.

| Work | Target | Assumptions | Guarantee | Stopping rule | Cost | Optimality | Coverage | Code | Status |
|---|---|---|---|---|---|---|---|---|---|
| Wang et al., Self-Consistency (ICLR 2023) | accuracy of majority vote | — | none | fixed N | samples | — | arithmetic, commonsense | yes | memory |
| Aggarwal et al., Adaptive-Consistency (EMNLP 2023), arXiv:2305.11860 | agree with fixed-N vote | i.i.d. answers | none (posterior threshold) | Beta/Dirichlet posterior on top-two > 0.95 | samples | none | 17 datasets, 3 models | yes | read (method, experiments) |
| Li et al., ESC (ICLR 2024), arXiv:2401.10480 | agree with fixed-N vote | i.i.d. | none (heuristic; a bound on window-vs-full agreement, not re-derived) | unanimous window of w | samples | none | 6 benchmarks | yes | read (algorithm, settings) |
| Lee et al., ConSol (2025), arXiv:2503.17587 | "consistency" with the mode | i.i.d.; top two re-selected | **none stated**; α, β nominal; Wald bound vacuous at defaults (our derivation) | SPRT/mSPRT on top two; equals a 3-vote lead | samples, tokens | cites Wald–Wolfowitz for SPRT | AIME24, GPQA, reasoning models | yes | read (Secs 2–3, App. A–B) |
| Cordero-Encinar & Duncan, Certified Self-Consistency / MMC (2025), arXiv:2510.17472 | majority vote = mode | i.i.d.; answers in discrete bins | Thm 3.1: Ville bound under a per-round null; error at stopping asserted | two e-processes ≥ 1/ε | samples | expected-stop approximation (Eq. 5), no lower bound | several | project page | read (Sec. 3, App. B; not Cor. 3.2, B.5) |
| Ota et al., CITE (2026), arXiv:2605.05873 | a **prespecified** answer is the unique mode | i.i.d.; unknown label set | Thm 4.1: false certification ≤ ε under any stopping | intersection–union of e-processes | samples | Thm 4.3 stopping-time bound, category-count-free | — | — | read (algorithm, Thm 4.1/4.3 statements) |
| BARS (Nov 2026), arXiv:2611.14171 | majority identification with budget and refusal | i.i.d., unknown distribution | certified mode covered by theorems; practical mode empirical | top-two gap confidence width, then run-length rule; refuse if infeasible | samples within budget E* | none found | Qwen2.5-7B, 16 decodes, GSM8K/MATH | GPL-3.0 | read (Sec. 3, defaults) |
| Huang et al., ASC-D (Sep 2026), arXiv:2609.38931 | mode, δ-PAC | i.i.d.; **known K**; per-trajectory answer distribution observed | Thm 3.1 lower bound, Thm 4.1 matching upper bound for grey-box feedback | betting wealth vs every challenger ≥ (K−1)/δ | **one call per trajectory** | asymptotically optimal; black-box rate T_bb attributed to Shah et al. 2020, Jain et al. 2022 | small models, multiple choice | anonymised | read (summary of all sections) |
| Kim et al., ReASC (2026), arXiv:2601.02970 | efficient sampling using response-level confidence | — | none found | evidence-sufficiency with confidence weights | samples/cost | none | Gemma-3-4B and others, GSM8K | — | abstract |
| Li et al., Adaptive Thinking / Sonata (ICLR 2026) | allocate thinking budget per query | calibration set | none (learned adapter) | predicted self-consistency sets the budget before thinking | thinking tokens | none | Qwen3, GPT-OSS; AIME25, GSM8K, MATH500, GPQA | — | abstract |
| PETS (2026), arXiv:2602.16745 | agreement with infinite-budget vote across a workload | — | "theoretical guarantees" (not read) | allocation across questions | samples | — | GPQA and others | yes | abstract |
| Bayesian ASC with informative prior (2026), arXiv:2602.05395 | Bayes-optimal stopping | prior on answer distribution | Bayesian | sequential test, combinatorial posterior | samples | Bayes-optimal rule | — | — | abstract |
| SAFER (ICLR 2026) | risk-controlled sampling budget with abstention | exchangeable calibration data | Learn-then-Test risk control | calibrated budget | samples | none | QA | — | abstract |
| Anytime-Valid PAC-Bayes Certificates for Adaptive Test-Time Scaling (Nov 2026), arXiv:2611.21241 | risk of adaptive budget policies | calibration data | anytime-valid PAC-Bayes | — | tokens reported | none | 192 GSM8K items | — | abstract |
| MARS (2026), arXiv:2606.12935 | early-stopped vote = full-budget vote, **parallel** traces | probing of partial traces; switch probabilities | high-probability match given true switch probabilities | margin-adversarial bound | tokens of parallel traces | none | reasoning models | — | abstract |
| Parallel-Probe (2026), arXiv:2602.03845 | efficient parallel thinking | probing | none | consensus stability + pruning | tokens | none | — | — | abstract |
| Hassid et al., short-m@k (2025), arXiv:2505.17813; First Finish Search (2025), arXiv:2505.18149; Serving LLM Reasoning Efficiently and Accurately (2025), arXiv:2505.13326 | accuracy/latency | — | none | stop after the first m finished traces | tokens, latency | none | reasoning models | — | abstract. **They establish that shorter traces are more often correct and deliberately exploit completion order.** |
| Self-Consistency Is Losing Its Edge (2025), arXiv:2511.00751 | diminishing returns of SC | — | — | — | — | — | modern models | — | abstract |
| Shah, Choudhury, Karamchandani, Gopalan (2020), sequential mode estimation; Jain et al. (AISTATS 2022), PPR martingale confidence sequences for PAC mode estimation | mode, δ-PAC | i.i.d. categorical | δ-PAC with matching instance lower bound ~ p₁/(p₁−p₂)² · log(1/δ) | one-versus-one PPR confidence sequences | samples | near-optimal | — | — | memory; formula confirmed via ASC-D Eq. 2 |
| Wald (1945); Wald & Wolfowitz (1948) | simple-vs-simple test | i.i.d. | α* ≤ α/(1−β), β* ≤ β/(1−α) | SPRT | samples | optimal ASN | — | — | memory |
| Garivier & Kaufmann (2016); Kaufmann, Cappé, Garivier (2016) | best-arm identification | — | δ-PAC lower bounds by change of measure | track-and-stop | samples | asymptotically optimal | — | — | memory |
| Howard, Ramdas, McAuliffe, Sekhon (2021); Waudby-Smith & Ramdas (2024); Ramdas et al. (2023) | confidence sequences, e-processes, betting | — | time-uniform | — | — | — | — | — | memory |
| Glynn & Heidelberger (1990, 1991) | bias of time-budgeted parallel simulation | i.i.d. replications with random durations | completed-replication estimators are biased; corrections | — | — | — | — | — | memory; **must be read before claiming the parallel result** |

## What is already settled

1. **Anytime-valid mode identification from i.i.d. answers, counted in samples.** Valid
   procedures (PPR, MMC, CITE, ASC-D with labels) and the asymptotic lower bound
   T_bb = [p₁ log(2p₁/(p₁+p₂)) + p₂ log(2p₂/(p₁+p₂))]⁻¹ ≈ 4p̄/Δ² are known. A new
   sample-count stopping rule in this model is not a contribution.
2. **Richer per-trajectory feedback** (ASC-D, ReASC, CITE's confidence variant).
3. **Calibration-based risk control of budgets** (SAFER, Learn-then-Test, PAC-Bayes).
4. **Budget and refusal** (BARS).
5. **Parallel traces with early cancellation as a heuristic** (short-m@k, FFS, Parallel-Probe)
   and with a probing-based guarantee (MARS, not yet read).

## What appears unresolved (to be confirmed by reading the "abstract" rows)

* **Validity of sequential certificates when answers are consumed in completion order**
  under parallel generation with length-dependent answers. No row above states the
  problem; short-m@k and FFS exploit the dependence, MARS addresses a different event
  (agreement with the full-budget vote using probes). Closest classical result: Glynn &
  Heidelberger. See `research_directions.md`, direction D, and `main_theorem.md`.
* **A like-for-like comparison of the published rules** with grader-free answer classes,
  equal budgets, token costs and paired intervals. No row provides it.
* **An exact characterisation of ConSol's default test** (ours; elementary).
