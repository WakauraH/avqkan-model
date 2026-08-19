# Pre-registered confirmatory study: quantum vs. classical at d = 12, 14, 16

**Status: registered BEFORE any confirmatory run was launched.**
The git commit timestamp of this file precedes the creation of every file in
`results_confirm*/`. Exploratory motivation: on the standardized family
$\widetilde{\rm sep}_d$ with seeds 0–9, natural-gradient VQKAN beat the
per-seed best of four classical baselines at d = 12 (9.99 vs 13.61, p = 0.0039,
n = 10, not pre-specified). This study tests whether that effect is real and
whether it extends to larger d.

## Hypotheses (primary family, Holm correction, m = 3, α = 0.05)

- **H12**: at d = 12, natural-gradient VQKAN < oracle-best classical (test error)
- **H14**: at d = 14, natural-gradient VQKAN < oracle-best classical
- **H16**: at d = 16, natural-gradient VQKAN < oracle-best classical

One-sided direction is stated for clarity; tests are two-sided Wilcoxon
signed-rank on paired seeds, and a pass requires Holm-adjusted p < 0.05 AND
median paired difference < 0.

## Design (fixed before launch)

- **Seeds: 10–29** (n = 20), disjoint from the exploratory seeds 0–9.
  Each seed fixes the 10 training and 50 test points.
- **Targets**: $\widetilde{\rm sep}_d$, d ∈ {12, 14, 16}, standardization
  constants (μ_d, σ_d) fixed in `avqkan.py` (Monte-Carlo, seed 20260819,
  2×10⁵ samples), k = 2.5, computed and committed before launch.
- **Quantum arm**: natural gradient (`--method ite`), n_q = d, readout
  `all_pairs`, 40 steps, δτ = 0.1, growth every 10 steps, initial ansatz X₁,
  exact simulation — identical protocol to the exploratory runs.
- **Classical oracle arm**: per-seed minimum of {classical KAN (ridge,
  8 control points), MLP 8×8, MLP 16 (tanh, L-BFGS, seed-matched), linear
  regression}, all trained on the same 10 points, outputs clipped to [−1, 1].
  This oracle favors the classical side (selection with test knowledge of
  which model won is not available in practice).
- **Metric**: test error (sum of absolute deviations over the 50 test points)
  evaluated ONCE at the step of minimum training cost C (quantum arm);
  classical models have no trajectory (single fit).
- **Validity gates** (checked per size; failure of a gate at size d voids the
  test at that d rather than counting as pass or fail):
  1. Quantum arm beats the constant predictor (training median) in mean.
  2. Oracle-classical arm beats the constant predictor in mean.
- **Secondary (BH q-values, exploratory label)**: equal-budget COBYLA
  (maxiter 30) vs oracle-best classical at each d; natural gradient vs COBYLA
  at each d.
- **No other comparisons** on these runs count as confirmatory.

## Analysis script

`analyze_confirm.py`, committed together with this file. It reads only
`results_confirm/` (ite), `results_confirm_m30/` (cobyla), and computes the
classical arm deterministically from seeds. No analysis parameters may change
after launch; any deviation must be reported as such.

## Disclosure

d = 16 was not part of the exploratory analysis; d = 14 was exploratory only
on the DEGENERATE (unstandardized) family and on seeds 0–9 of the standardized
family it was not run. The exploratory d = 12 result used seeds 0–9; those
seeds are excluded here.
