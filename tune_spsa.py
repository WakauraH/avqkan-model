"""SPSA-QFI のハイパーパラメータ探索（shots=4096, 5 シード, 30 ステップ）。"""
import numpy as np
from avqkan import run
SEEDS = range(5)
print(f"{'reg':>7s} {'eps':>5s} {'k':>2s} {'final mean±std':>18s} {'best mean±std':>18s} {'calls':>8s}")
for reg, eps, k in [(1e-1,0.3,8), (1e-1,0.3,16), (3e-2,0.3,8), (1e-1,0.5,8), (3e-1,0.3,8), (1e-2,0.3,16)]:
    fin, best, calls = [], [], []
    for s in SEEDS:
        rows,_ = run("ite_spsa","log",s,30,0.1,10,"results_spsa",verbose=False,
                     shots=4096, ite_reg=reg, eta_m=eps, spsa_resamplings=k,
                     tag_suffix=f"_r{reg}_e{eps}_k{k}")
        te=[r["test_absdist"] for r in rows]
        fin.append(te[-1]); best.append(min(te)); calls.append(rows[-1]["opt_calls"])
    print(f"{reg:7.4g} {eps:5.2f} {k:2d} {np.mean(fin):11.3f} ±{np.std(fin):5.2f} "
          f"{np.mean(best):11.3f} ±{np.std(best):5.2f} {int(np.mean(calls)):8d}", flush=True)
