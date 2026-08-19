# Natural-gradient imaginary-time optimization of Adaptive VQKAN

Reference implementation and full data for the paper

> H. Wakaura, R. Mulyawan, A. B. Suksmono,
> *Natural-gradient imaginary-time optimization of Adaptive Variational Quantum
> Kolmogorov-Arnold Networks* (v2 of arXiv:2506.22801).

Imaginary-time training of the Adaptive Variational Quantum Kolmogorov-Arnold
Network (VQKAN) is formulated as a McLachlan natural-gradient flow of the
regression cost, `A(θ) θ̇ = −∇C(θ)`, where `A` is the Fubini-Study metric of
the variational family. The repository contains the training rule, all
baselines compared in the paper (the fidelity-objective VarQITE of arXiv
v1, COBYLA, plain gradient descent, classical KAN/MLP), the finite-shot and
metric-estimator studies, and every CSV underlying the figures and tables.

## Install

```bash
pip install -r requirements.txt
```

Python ≥ 3.9. Simulation uses the blueqat SDK (exact statevector); no quantum
hardware is required.

## Verify

```bash
python test_avqkan.py
```

Five checks (T1–T5): qubit-ordering conventions, circuit-level equivalence of
the swap test used by the v1 code to `|⟨ψ₁|ψ₂⟩|²`, positive semidefiniteness
of the Fubini-Study metric, analytic-vs-numerical gradient agreement, and the
staircase artifact of the v1 grid-quantized spline evaluation.

## Quick start

```bash
python avqkan.py --method ite    --target eq6 --seed 0 --steps 60   # natural-gradient ITE
python avqkan.py --method cobyla --target eq6 --seed 0 --steps 60   # COBYLA baseline
python avqkan.py --method legacy --target eq6 --seed 0 --steps 60   # fidelity-objective v1
```

`--method {ite, gd, cobyla, legacy, ite_spsa}`,
`--target {eq6, log, frac, radius, expo, sep4 … sep14}`,
`--shots N` (0 = exact), `--nq`, `--readout-mode {single, all_pairs}`.
Every run is seed-deterministic and writes a per-step CSV + JSON metadata.

## Reproducing the paper

All result CSVs are already included (`results*/`), so the analysis and
figures run immediately:

```bash
python analyze.py            # Tables 1–2 + Wilcoxon signed-rank tests
python gen_paper_figs.py     # Figures 1–4 → figs/
```

To regenerate the raw data from scratch:

| Paper item | Command |
|---|---|
| Table 1, Fig. 1 (5 targets × 4 methods, 10 seeds) | `python run_compare.py <target> 10` for each target |
| Fig. 2 (equal-budget COBYLA sweep) | `python run_budget.py eq6 10` / `python run_budget.py log 10` |
| Fig. 3a (readout dependence) | `python run_readout.py` |
| Fig. 3b (qubit scaling, equal budget) | `python run_scaling_budget.py <nq>` for nq = 4…10; `python run_one.py {ite,cobyla} {12,14} <seed> [30]` |
| Fig. 4a (finite shots) | `python run_shots2.py <shots> <method>` |
| Fig. 4b (metric estimators, noise-free controls) | `python run_control_one.py <seed>`, `python run_eta.py <eta> <seed>`, `python tune_spsa.py` |
| Table 2 (classical baselines) | `python classical_baseline.py --seeds 10` |

A full regeneration takes a few hours on a laptop; the individual `run_*.py`
scripts are embarrassingly parallel over seeds.

## Layout

| File | Content |
|---|---|
| `avqkan.py` | Model (forward pass faithful to the v1 notebooks), targets, the four training rules, finite-shot estimators, CLI |
| `classical_baseline.py` | Classical KAN (closed-form ridge) and MLP baselines |
| `test_avqkan.py` | Validation tests T1–T5 |
| `analyze.py` | Statistics: means, paired Wilcoxon tests, classical comparison |
| `run_*.py`, `tune_spsa.py` | Experiment drivers (one per paper study) |
| `gen_paper_figs.py`, `plot_*.py` | Figures |
| `results*/` | All raw per-step CSVs and run metadata (~10 MB) |

## Notes on the v1 baseline

`--method legacy` reproduces the arXiv v1 formulation faithfully, including
its random restart kicks, mid-run spline-grid change, and grid-quantized
spline evaluation. Its objective couples to the regression target only
through fixed sample weights, which is the subject of Sec. 2.3 and Fig. 1b
of the paper.

## License

MIT — see `LICENSE`.
