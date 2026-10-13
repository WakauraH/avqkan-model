# Pre-registered confirmatory test: linear-Z Adaptive VQKAN vs cross-validated kernels

**Status: registered at commit 3faa5c5 (2026-10-14), BEFORE any run in `results_linz_confirm/` was launched.**

## Exploratory motivation (seeds 0–9, NOT confirmatory)
Across 7 model variants × 4 targets × N ∈ {10, 40, 100} (`results_model/variants_summary.md`), exactly one cell
showed the quantum model nominally below BOTH the RBF kernel ridge and the sorted-input (permutation-invariant) kernel
ridge, each with hyperparameters chosen by closed-form leave-one-out CV on the training set:
sepn_12, N = 100, variant `linz_m200_g5` (linear-Z ridge readout, COBYLA maxiter 200, growth every 5 steps):
2.6 ± 0.6 vs 3.2 ± 0.5 (p = 0.13) and 3.3 ± 0.5 (p = 0.08), n = 10. This is one hit among 48 cells and must be
treated as a hypothesis, not a result.

## Hypotheses (Holm correction, m = 2, α = 0.05; a pass requires adjusted p < 0.05 AND median paired difference < 0)
- **H-RBF**: on sepn_12 with N = 100 training points, the test error of `linz_m200_g5` is lower than that of
  RBF kernel ridge (LOO-CV over γ ∈ logspace(−2, 1, 7), λ ∈ logspace(−4, 1, 6)).
- **H-SORT**: same, against sorted-input RBF kernel ridge (same grids).
The claim "beats cross-validated kernel methods" is made only if BOTH pass.

## Design (fixed before launch)
- Seeds 30–49 (n = 20), disjoint from the exploratory seeds 0–9. Each seed fixes 100 training and 50 test points.
- Target `sepn_12` (`make_family("sepn", 12)`, standardisation constants from `_FAMILY_STATS`, seed 20260819).
- Quantum arm: `run("cobyla", "sepn_12", seed, 40, 0.1, 5, ..., maxiter=200, nq=12, n_train=100,
  readout_mode="linear_z")`, i.e. 40 steps, growth every 5 steps from `X1`, paper operator pool, cost-based selection,
  spline shared / arccos map / grids 8, exact simulation. Metric: test error at the step of minimum training cost
  (the training cost is the LOO error of the ridge readout; the test set is never used for selection).
- Classical arms computed deterministically from the same seeds by `classical_kernels.fit_predict`.
- Two-sided paired Wilcoxon signed-rank tests. Validity gate: all arms beat the constant predictor in mean.
- Secondary (exploratory, BH): the same quantum arm vs `krr_sym` and `krr_add`; and `linz_m200` (growth every 10).
- No other comparison on these runs is confirmatory. Analysis script: `analyze_linz_confirm.py`, committed with this file.
