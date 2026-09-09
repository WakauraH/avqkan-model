# Few-sample regression with an adaptively grown variational quantum Kolmogorov-Arnold network

Reference implementation, pre-registration, and full data for

> H. Wakaura, R. Mulyawan, A. B. Suksmono, *Few-sample regression with an adaptively grown
> variational quantum Kolmogorov-Arnold network* (revision of arXiv:2503.21336).

The repository also contains the training-rule study of the companion paper (arXiv:2506.22801);
see the section "Companion study" below.

## Reproducing the paper

```bash
pip install -r requirements.txt
./reproduce.sh                             # everything below in one go (a few minutes)
```

or step by step:

```bash
python test_avqkan.py                      # validation tests T1-T6
python qiskit_bridge.py --test             # blueqat -> qiskit translation check
python analyze_model.py                    # Fig. 2 table, ablation (Table 1)  -> results_model/table_fig2.md
python analyze_model_confirm.py            # pre-registered study (Table 3)      -> stdout
python classical_regularized.py sepn12 12 10 30-49 results_model_confirm/cobyla_sepn12_seedSEED_nq12_m30.csv
python analyze_nsweep.py                   # sample-size sweep (Fig. 5)
python analyze_cls.py                      # classification (Appendix A)
python analyze_noise.py                    # gate noise + hardware (Sec. 3.3, Table 1) -> results_model/noise_summary.md
python gen_model_figs.py                   # Figs. 2-5 and resource table       -> figs_model/
```

All result CSVs are included, so the analyses run in minutes. To regenerate the raw data:

| Paper item | Command |
|---|---|
| Fig. 2 quantum arm (d=4, 5 targets, seeds 0-9) | `python run_compare.py <target> 10` (cobyla arm) |
| Fig. 2 QNN arms | `python run_qnn_sweep.py` |
| Table 1 ablation | `python run_ablation.py` |
| Fig. 3 / Table 3 pre-registered study (seeds 30-49) | `python run_model_confirm.py` (after reading `PREREGISTRATION_model.md`) |
| Fig. 4 finite shots | `python run_shots_eq6.py`; the d=12 arm is part of `run_model_confirm.py` |
| Fig. 5 sample-size sweep | `python run_nsweep.py` |
| Appendix A classification | `python run_cls.py` |
| Saved model parameters (`models/`) | `python run_save_models.py` (deterministic re-training) |
| Gate-noise inference / noisy training | `python noise_study.py --models "models/d4/*.json"`, `--models "models/d12/*.json"`, `--backend-noise ibm_marrakesh`, `--train --seed S` |
| Hardware inference | `python hardware_inference.py --backend ibm_marrakesh --models "models/hw_batch/*.json" --submit`, then `--collect <job_id>` (needs a saved IBM Quantum account; the paper's jobs are in `results_hardware/`) |

`PREREGISTRATION_model.md` was committed (74a528d) before any run in `results_model_confirm*/`
was launched; `analyze_model_confirm.py` is the frozen analysis. The kernel-ridge comparison
(`classical_regularized.py`) was added after the development seeds had been analysed and is
labelled post hoc in the paper.

## Companion study (training rules)

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
