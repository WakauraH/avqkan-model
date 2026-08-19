"""スケーリングの等予算版: COBYLA の maxiter を絞って ITE と予算を揃える。"""
import json, sys
import numpy as np
from avqkan import run
NQ = int(sys.argv[1]); SEEDS = range(10); STEPS, GROW = 40, 10
TARGET, OUT = f"sep{NQ}", "results_scaling"
res = {}
for label, mi in (("cobyla_max30", 30), ("cobyla_max20", 20)):
    fin, best, calls = [], [], []
    for s in SEEDS:
        rows, _ = run("cobyla", TARGET, s, STEPS, 0.1, GROW, OUT, maxiter=mi,
                      verbose=False, nq=NQ, readout_mode="all_pairs",
                      tag_suffix=f"_nq{NQ}_m{mi}")
        te = [r["test_absdist"] for r in rows]
        fin.append(te[-1]); best.append(min(te)); calls.append(rows[-1]["opt_calls"])
    res[label] = dict(final_mean=float(np.mean(fin)), final_std=float(np.std(fin)),
                      best_mean=float(np.mean(best)), best_std=float(np.std(best)),
                      opt_calls=int(np.mean(calls)))
    v = res[label]
    print(f"nq={NQ} {label:14s} calls={v['opt_calls']:7d} final={v['final_mean']:8.3f}"
          f" ±{v['final_std']:6.3f} best={v['best_mean']:8.3f} ±{v['best_std']:6.3f}",
          flush=True)
# ITE 側も opt_calls を取り直す
fin, best, calls = [], [], []
for s in SEEDS:
    rows, _ = run("ite", TARGET, s, STEPS, 0.1, GROW, OUT, verbose=False, nq=NQ,
                  readout_mode="all_pairs", tag_suffix=f"_nq{NQ}_b")
    te = [r["test_absdist"] for r in rows]
    fin.append(te[-1]); best.append(min(te)); calls.append(rows[-1]["opt_calls"])
res["ite"] = dict(final_mean=float(np.mean(fin)), final_std=float(np.std(fin)),
                  best_mean=float(np.mean(best)), best_std=float(np.std(best)),
                  opt_calls=int(np.mean(calls)))
v = res["ite"]
print(f"nq={NQ} {'ite':14s} calls={v['opt_calls']:7d} final={v['final_mean']:8.3f}"
      f" ±{v['final_std']:6.3f} best={v['best_mean']:8.3f} ±{v['best_std']:6.3f}")
json.dump(res, open(f"{OUT}/summary_budget_nq{NQ}.json", "w"), indent=2)
