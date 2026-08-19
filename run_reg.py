"""ショットノイズ下の ite 劣化が「計量推定の雑音」由来かを切り分ける。
計量の正則化 reg を強くする（＝計量を単位行列に近づける）と gd に漸近するはず。
差分幅 eta_m も振る（信号 ~ eta_m^2 なので大きいほど雑音に強い）。"""
import json, sys
import numpy as np
from avqkan import run
SHOTS = 4096
SEEDS = range(5)
print(f"{'reg':>8s} {'eta_m':>6s} {'final mean±std':>20s} {'best mean±std':>20s}")
out = {}
for reg, eta_m in [(1e-3, 0.3), (1e-2, 0.3), (1e-1, 0.3), (1.0, 0.3),
                   (1e-3, 0.6), (1e-1, 0.6)]:
    fin, best = [], []
    for s in SEEDS:
        rows, _ = run("ite", "log", s, 30, 0.1, 10, "results_reg", maxiter=200,
                      verbose=False, shots=SHOTS, ite_reg=reg, eta_m=eta_m,
                      tag_suffix=f"_reg{reg}_eta{eta_m}")
        te = [r["test_absdist"] for r in rows]
        fin.append(te[-1]); best.append(min(te))
    out[f"reg{reg}_eta{eta_m}"] = dict(final_mean=float(np.mean(fin)),
                                       final_std=float(np.std(fin)),
                                       best_mean=float(np.mean(best)),
                                       best_std=float(np.std(best)))
    print(f"{reg:8.4g} {eta_m:6.2f} {np.mean(fin):11.3f} ±{np.std(fin):7.3f} "
          f"{np.mean(best):11.3f} ±{np.std(best):7.3f}", flush=True)
json.dump(out, open("results_reg/summary_reg.json", "w"), indent=2)
