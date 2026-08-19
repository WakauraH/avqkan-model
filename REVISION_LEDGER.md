# Panel-review revision ledger (2026-08-19)

Adversarial two-reviewer panel (GPT-family Codex + Claude Opus, 3 rounds, both
reproduction-based). What survived, what changed, what was retracted.

## Survived (strengthened)
| Claim | Old evidence | New evidence |
|---|---|---|
| Natural gradient ≫ fidelity objective | 10/10 vs 5-factor-confounded v1 arm, leaky metric | Clean objective-only arm (`results_b2`): eq6 23.41 vs 8.50, log 11.94 vs 2.75, p=0.002 both |
| Metric (vs plain GD) helps | 9/10, leaky metric | Early-stop + BH: 3/5 at q≤0.005, same direction 5/5 |
| NG > budget-matched COBYLA (eq6, single readout) | leaky best-test | Pre-specified P1/P2, train-cost early stop, Holm m=2: adj p=0.027 both |
| Estimator (not noise) breaks measured NG | confounded (reg+grad+metric changed together) | De-confounding control (`results_b1`): exact metric + clip reg + FD grad = 8.59 ≈ all-exact 8.30 (p=0.43); FD metric twin = 21.44 (p=0.002) → metric estimator isolated |

## Changed (mechanism / framing)
| Item | Before | After |
|---|---|---|
| Framing | "VarQITE / imaginary time" | Quantum natural gradient on data-averaged cost; ρ⊗2 correspondence (factor 2) single-sample only; sample-averaged metric is no pure-state FS metric |
| Readout dependence mechanism | multi-term smoothing vs curvature | Output range / expressivity headroom: 0.5·Z0Z1 (single term, half range) kills the advantage (13.30 vs 13.35, p=0.85) (`results_b3`) |
| Primary metric | best over test trajectory (leak) | test error at argmin of training cost; final secondary; best exploratory only |
| Statistics | uncorrected, ddof=0 | Holm (primary), BH q (exploratory), ddof=1 |

## Retracted
| Claim | Reason |
|---|---|
| "Advantage grows with system size" (old sep_d) | Old family degenerate: both methods below constant predictor at nq≥10. Standardized family (sepn, `results_c*`): no significant NG-vs-COBYLA difference at any size after Holm (nominal p=0.027 at d=12 → adj 0.137) |
| "COBYLA fails even with fivefold budget" (as tested claim) | p=0.065–0.105 at high budgets; wording now: stays 10.4–11.8 across sweep, not every pairwise test significant |
| expo / log as benchmarks | Degenerate under trivial baselines (constant beats every model on expo; beats quantum on log) |
| "64× shots increase" | Fig uses 1024–16384 = 16×; 256-shot data is from the log-target study |
| %2π-discontinuity divergence mechanism (panel speculation) | Diagnosed: grad-norm spikes not followed by test degradation |

## New findings
- Linear-regression baseline beats all quantum methods at d=4 on eq6/frac/radius.
- On standardized sepn family, quantum beats constant predictor at all sizes and
  linear regression for d≥8; at d=12 NG beats per-seed oracle-best of 4 classical
  baselines (9.99 vs 13.61, p=0.0039, exploratory).
- Paper-text encoding (arccos(2x−1), all qubits) degrades both methods and washes
  out the NG–COBYLA difference (p=0.28): update-rule rankings are encoding-specific.

## Reproduce
`analyze2.py` (confirmatory analysis), `results_b1..b4/`, `results_c*/`,
`gen_paper_figs.py`. Manuscript: `arXiv-2506.22801v1 2/mainwakaura2_v2.tex`
(pre-panel version preserved as `mainwakaura2_v2_prepanel.bak.tex`).
