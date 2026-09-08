# Pre-registered confirmatory study (model paper, arXiv:2503.21336 revision)

**Status: DRAFT — not yet committed. Commit this file BEFORE launching any run in `results_model_confirm*/`.**

Exploratory motivation (already public in the v2 companion, arXiv:2506.22801, `REVISION_LEDGER.md`):
on the standardized family sepn_d with seeds 10–29, COBYLA-trained Adaptive VQKAN beat the per-seed
oracle-best of four classical baselines at d = 12, 14, 16 (secondary endpoint there, p ≤ 1e-4), and
natural gradient and COBYLA were indistinguishable (p ≥ 0.25). This study (i) replicates that model
effect on fresh seeds with COBYLA as the *primary* training rule, (ii) extends it to d = 18, and
(iii) tests whether it survives finite measurement shots.

## Hypotheses (primary family, Holm correction, m = 4, α = 0.05)

- **H12**: at d = 12 (exact simulation), AVQKAN-COBYLA test error < oracle-best classical
- **H16**: at d = 16 (exact simulation), same
- **H18**: at d = 18 (exact simulation), same
- **H12s**: at d = 12 with 256 shots per circuit evaluation (training only; metric evaluated exactly), same

Two-sided paired Wilcoxon signed-rank tests on seed-paired errors; a pass requires Holm-adjusted p < 0.05
AND median paired difference < 0.

## Design (fixed before launch)

- **Seeds: 30–49** (n = 20), disjoint from all seeds used so far (0–29). Each seed fixes the 10 training
  and 50 test points.
- **Targets**: sepn_d, d ∈ {12, 16, 18}; standardization constants in `avqkan.py` `_SEPN_STATS`
  (d = 18 computed 2026-09-08 with the original protocol: seed 20260819, 2×10⁵ uniform samples, k = 2.5).
- **Quantum arm**: `--method cobyla --maxiter 30`, n_q = d, readout `all_pairs`, 40 steps,
  growth every 10 steps (way 2, paper pool), initial ansatz X1, spline evaluated directly.
  H12s adds `--shots 256` (Z-basis multinomial sampling of every ⟨Z_j⟩ and ⟨H⟩ seen by the optimizer).
- **Classical oracle arm**: per-seed minimum over {classical KAN (ridge, 8 control points), MLP 8×8,
  MLP 16 (tanh, L-BFGS, seed-matched), linear regression}, same 10 points, outputs clipped to [−1, 1].
  This selection uses test knowledge and therefore favors the classical side.
- **Metric**: test error (sum of |error| over the 50 test points) evaluated once at the step of minimum
  training cost (quantum arm); classical models are single fits.
- **Validity gates** (per hypothesis; a failed gate voids that test, it does not count as pass or fail):
  1. quantum arm beats the constant predictor (training median) in mean;
  2. oracle-classical arm beats the constant predictor in mean.
- **Secondary (BH q-values, exploratory label)**: AVQKAN vs each classical baseline separately;
  AVQKAN vs tuned QNN (L-BFGS, L ∈ {1,2,3,4} chosen by training cost) at d = 12; d = 20 (seeds 30–39
  only, compute-limited) reported descriptively.
- **No other comparisons** on these runs count as confirmatory.

## Analysis script

`analyze_model_confirm.py`, to be committed together with this file. It reads only
`results_model_confirm/` (exact) and `results_model_confirm_s256/` (shots) and computes the classical
arm deterministically from seeds. No analysis parameter may change after launch.

## Disclosure

d = 18 and the 256-shot condition have never been run on this model. d = 12 and 16 were run on seeds
0–29 in the companion study; those seeds are excluded here.
