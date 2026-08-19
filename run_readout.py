"""(a) 読み出しハミルトニアン依存性の切り分け。

§1.5(eq6, 読み出し Z0Z1) と §2.2(sep4, 読み出し (Z0Z1+Z2Z3)/2) は
対象関数が同一（sep4 == eq6、差 0.0 を検算済み）にもかかわらず結論が違った。
ただしスケジュールも違った（60 ステップ/成長 20 vs 40 ステップ/成長 10）ので、
ここでは **スケジュールと予算を完全に揃え、readout_mode だけを変える**。
"""
import json, sys
import numpy as np
from scipy.stats import wilcoxon
from avqkan import run

SEEDS = list(range(10))
STEPS, GROW, NQ, TARGET = 40, 10, 4, "sep4"     # sep4 == eq6
OUT = "results_readout"
res = {}

def go(method, rmode, mi=None):
    fin, best, calls = [], [], []
    for s in SEEDS:
        rows, _ = run(method, TARGET, s, STEPS, 0.1, GROW, OUT,
                      maxiter=(mi or 1000), verbose=False, nq=NQ,
                      readout_mode=rmode, tag_suffix=f"_{rmode}_{mi}")
        te = [r["test_absdist"] for r in rows]
        fin.append(te[-1]); best.append(min(te)); calls.append(rows[-1]["opt_calls"])
    return np.array(fin), np.array(best), int(np.mean(calls))

print(f"{'readout':11s} {'method':13s} {'calls':>7s} {'final mean±std':>18s} {'best mean±std':>18s}")
store = {}
for rmode in ("single", "all_pairs"):
    for label, method, mi in (("ite", "ite", None),
                              ("cobyla max30", "cobyla", 30),
                              ("cobyla max200", "cobyla", 200)):
        f, b, c = go(method, rmode, mi)
        store[(rmode, label)] = b
        res[f"{rmode}|{label}"] = dict(final_mean=float(f.mean()), final_std=float(f.std()),
                                       best_mean=float(b.mean()), best_std=float(b.std()),
                                       opt_calls=c)
        print(f"{rmode:11s} {label:13s} {c:7d} {f.mean():11.3f} ±{f.std():5.2f} "
              f"{b.mean():11.3f} ±{b.std():5.2f}", flush=True)

print()
print("Wilcoxon（best, ite を基準, 負なら ite が良い）")
for rmode in ("single", "all_pairs"):
    for other in ("cobyla max30", "cobyla max200"):
        A, B = store[(rmode, "ite")], store[(rmode, other)]
        W, p = wilcoxon(A, B)
        print(f"  {rmode:11s} ite vs {other:14s} 中央値差={np.median(A-B):+8.3f} p={p:.4f}")
json.dump(res, open(f"{OUT}/summary.json", "w"), indent=2)
