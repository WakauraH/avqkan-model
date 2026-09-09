# Depolarising noise, noisy training, and hardware inference

## d=4 (eq. 1, seeds 0-9): n=10, exact (statevector) 11.48 ± 4.23, CZ per circuit 4.0

| p2 (2q depolarising) | test error | vs exact p |
|---|---|---|
| 0 | 11.52 ± 4.21 | 0.432 |
| 0.001 | 11.51 ± 4.20 | 0.492 |
| 0.003 | 11.48 ± 4.18 | 0.846 |
| 0.01 | 11.39 ± 4.10 | 0.492 |
| 0.03 | 11.29 ± 3.88 | 0.432 |
| backend noise model | 11.35 ± 4.07 | 0.375 |

## d=12 (sepn12, seeds 30-49): n=20, exact (statevector) 11.14 ± 1.95, CZ per circuit 6.0

| p2 (2q depolarising) | test error | vs exact p |
|---|---|---|
| 0 | 11.12 ± 1.98 | 0.261 |
| 0.001 | 11.14 ± 1.98 | 0.430 |
| 0.003 | 11.14 ± 1.99 | 0.330 |
| 0.01 | 11.16 ± 2.00 | 0.123 |
| 0.03 | 11.22 ± 2.06 | 0.105 |
| backend noise model | 11.18 ± 1.97 | 0.076 |

## Noisy training, d=4 eq. 1: p2=0.01, 1024 shots, n=10

test error at min training cost: 13.07 ± 4.43 (noise-free training, same seeds: 11.48 ± 4.23; paired Wilcoxon p = 0.037); params 21

## Hardware inference

| job | backend | model | CZ | depth | exact | hardware |
|---|---|---|---|---|---|---|
| dagepnb9 | ibm_marrakesh | cobyla_eq6_seed0.json | 2 | 15 | 12.65 | 12.61 |
| dageqs39 | ibm_marrakesh | cobyla_eq6_seed1.json | 4 | 21 | 11.76 | 11.50 |
| dageqs39 | ibm_marrakesh | cobyla_eq6_seed2.json | 4 | 13 | 15.60 | 15.47 |
| dageqs39 | ibm_marrakesh | cobyla_eq6_seed3.json | 2 | 14 | 7.66 | 7.49 |
| dageqs39 | ibm_marrakesh | cobyla_eq6_seed4.json | 4 | 13 | 16.31 | 16.03 |
| dageqs39 | ibm_marrakesh | cobyla_eq6_seed5.json | 4 | 13 | 6.05 | 6.03 |
| dageqs39 | ibm_marrakesh | cobyla_sepn12_seed30_nq12_m30.json | 4 | 13 | 11.26 | 11.20 |
| dageqs39 | ibm_marrakesh | cobyla_sepn12_seed31_nq12_m30.json | 4 | 14 | 11.48 | 11.39 |
| dageqs39 | ibm_marrakesh | cobyla_sepn12_seed32_nq12_m30.json | 4 | 13 | 11.53 | 11.57 |
| dageqs39 | ibm_marrakesh | cobyla_sepn12_seed33_nq12_m30.json | 4 | 14 | 8.17 | 8.28 |
| dageqs39 | ibm_marrakesh | cobyla_sepn12_seed34_nq12_m30.json | 6 | 20 | 13.03 | 12.81 |

11 models: hardware − exact = -0.099 ± 0.124 (max |diff| 0.282), paired Wilcoxon p = 0.032; QPU usage 337 s

