# d=4, 5 targets, 10 paired seeds — test error at train-cost early stop (mean ± sd, ddof=1)

| target | AVQKAN (COBYLA) | QNN paper (L=3) | QNN tuned | cKAN | MLP 8x8 | MLP 16 | linear | const |
|---|---|---|---|---|---|---|---|---|
| eq6 | 11.48 ± 4.23 | 10.72 ± 1.69 | 11.18 ± 2.35 | 10.74 ± 1.89 | 7.92 ± 1.63 | 8.95 ± 2.71 | 7.74 ± 1.80 | 17.45 ± 1.97 |
| log | 2.29 ± 0.63 | 1.59 ± 0.89 | 2.33 ± 1.72 | 0.59 ± 0.33 | 0.84 ± 0.37 | 0.92 ± 0.37 | 1.06 ± 0.23 | 1.82 ± 0.27 |
| frac | 11.52 ± 1.76 | 5.49 ± 4.36 | 5.38 ± 2.08 | 9.75 ± 3.01 | 2.10 ± 2.25 | 1.95 ± 1.48 | 6.87 ± 1.52 | 23.65 ± 2.67 |
| radius | 5.32 ± 1.42 | 6.14 ± 2.34 | 5.44 ± 1.39 | 5.75 ± 0.76 | 2.69 ± 1.21 | 2.62 ± 1.04 | 3.65 ± 0.41 | 13.25 ± 1.39 |
| expo | 13.15 ± 4.89 | 0.71 ± 0.87 | 0.26 ± 0.17 | 1.64 ± 0.57 | 0.51 ± 0.39 | 0.36 ± 0.20 | 0.30 ± 0.20 | 0.24 ± 0.13 |

Params (mean): eq6: avqkan=24, qnn_paper=24, qnn_tuned=31, kan=37, mlp8x8=121, mlp16=97, linear=5, const=1; log: avqkan=22, qnn_paper=24, qnn_tuned=29, kan=19, mlp8x8=105, mlp16=65, linear=3, const=1; frac: avqkan=22, qnn_paper=24, qnn_tuned=32, kan=19, mlp8x8=105, mlp16=65, linear=3, const=1; radius: avqkan=24, qnn_paper=24, qnn_tuned=30, kan=28, mlp8x8=113, mlp16=81, linear=4, const=1; expo: avqkan=23, qnn_paper=24, qnn_tuned=30, kan=28, mlp8x8=113, mlp16=81, linear=4, const=1

## Paired Wilcoxon, AVQKAN vs each baseline (median diff < 0 = AVQKAN better; BH q)

| target | baseline | median diff | p | q |
|---|---|---|---|---|
| eq6 | qnn_paper | -0.80 | 0.7695 | 0.8162 |
| eq6 | qnn_tuned | -0.22 | 0.9219 | 0.9219 |
| eq6 | kan | +0.73 | 0.6250 | 0.6836 |
| eq6 | mlp8x8 | +4.22 | 0.0195 | 0.0273 ** |
| eq6 | mlp16 | +3.30 | 0.0840 | 0.1089 |
| eq6 | linear | +4.09 | 0.0137 | 0.0199 ** |
| eq6 | const | -6.85 | 0.0098 | 0.0149 ** |
| log | qnn_paper | +0.78 | 0.0645 | 0.0868 |
| log | qnn_tuned | +0.54 | 0.6250 | 0.6836 |
| log | kan | +1.68 | 0.0020 | 0.0036 ** |
| log | mlp8x8 | +1.50 | 0.0020 | 0.0036 ** |
| log | mlp16 | +1.46 | 0.0020 | 0.0036 ** |
| log | linear | +1.07 | 0.0020 | 0.0036 ** |
| log | const | +0.25 | 0.0020 | 0.0036 ** |
| frac | qnn_paper | +6.93 | 0.0059 | 0.0093 ** |
| frac | qnn_tuned | +5.13 | 0.0020 | 0.0036 ** |
| frac | kan | +1.54 | 0.1309 | 0.1636 |
| frac | mlp8x8 | +9.12 | 0.0020 | 0.0036 ** |
| frac | mlp16 | +9.11 | 0.0020 | 0.0036 ** |
| frac | linear | +4.22 | 0.0020 | 0.0036 ** |
| frac | const | -12.27 | 0.0020 | 0.0036 ** |
| radius | qnn_paper | -0.45 | 0.2754 | 0.3324 |
| radius | qnn_tuned | -0.39 | 0.8457 | 0.8706 |
| radius | kan | +0.06 | 0.4922 | 0.5742 |
| radius | mlp8x8 | +2.12 | 0.0020 | 0.0036 ** |
| radius | mlp16 | +2.43 | 0.0039 | 0.0065 ** |
| radius | linear | +1.33 | 0.0039 | 0.0065 ** |
| radius | const | -8.06 | 0.0020 | 0.0036 ** |
| expo | qnn_paper | +12.95 | 0.0020 | 0.0036 ** |
| expo | qnn_tuned | +12.93 | 0.0020 | 0.0036 ** |
| expo | kan | +11.47 | 0.0020 | 0.0036 ** |
| expo | mlp8x8 | +12.90 | 0.0020 | 0.0036 ** |
| expo | mlp16 | +12.95 | 0.0020 | 0.0036 ** |
| expo | linear | +12.89 | 0.0020 | 0.0036 ** |
| expo | const | +12.98 | 0.0020 | 0.0036 ** |

## Ablation on eq6 (COBYLA, 10 seeds), vs base = way 2 / paper pool / init X1

| config | test | params | p vs base | q |
|---|---|---|---|---|
| base | 11.48 ± 4.23 | 24 | - | - |
| way1 | 9.32 ± 2.73 | 24 | 0.0371 | 0.1113 |
| pool_full | 11.48 ± 4.23 | 24 | 0.7500 | 0.7695 |
| way1_full | 9.20 ± 2.71 | 24 | 0.0371 | 0.1113 |
| init_Y1 | 14.56 ± 4.89 | 23 | 0.1309 | 0.1963 |
| init_Z1 | 12.20 ± 3.38 | 24 | 0.7695 | 0.7695 |
| init_2layer | 15.55 ± 3.61 | 32 | 0.0840 | 0.1680 |
