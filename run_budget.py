"""回路評価回数を揃えたときの精度比較。

COBYLA 側の評価回数は maxiter が決めるので、maxiter を振って ITE と同じ
予算になる点を作り、そこで精度を比べる。ITE 側は 2P+1 の差分で回数が決まる
ため予算を直接は選べない（1 点だけ）。
指標の記録に使った評価は opt_calls から除外済み。
"""
import json, sys
import numpy as np
from avqkan import run

TARGET = sys.argv[1] if len(sys.argv) > 1 else "log"
SEEDS = list(range(int(sys.argv[2]) if len(sys.argv) > 2 else 10))
STEPS, GROW = 60, 20
out = {}

def summarize(label, rows_list):
    fin = [r[-1]["test_absdist"] for r in rows_list]
    bst = [min(x["test_absdist"] for x in r) for r in rows_list]
    cal = [r[-1]["opt_calls"] for r in rows_list]
    out[label] = dict(final_mean=float(np.mean(fin)), final_std=float(np.std(fin)),
                      best_mean=float(np.mean(bst)), best_std=float(np.std(bst)),
                      opt_calls=int(np.mean(cal)))
    v = out[label]
    print(f"{label:16s} {v['opt_calls']:9d} {v['final_mean']:9.3f} ±{v['final_std']:6.3f} "
          f"{v['best_mean']:9.3f} ±{v['best_std']:6.3f}", flush=True)

print(f"{'setting':16s} {'最適化評価':>9s} {'final mean±std':>17s} {'best mean±std':>17s}")
summarize("ite", [run("ite", TARGET, s, STEPS, 0.1, GROW, "results_budget",
                      verbose=False, tag_suffix="_bud")[0] for s in SEEDS])
for mi in (10, 20, 50, 100, 200):
    summarize(f"cobyla max{mi}",
              [run("cobyla", TARGET, s, STEPS, 0.1, GROW, "results_budget",
                   maxiter=mi, verbose=False, tag_suffix=f"_bud{mi}")[0] for s in SEEDS])
json.dump(out, open(f"results_budget/summary_{TARGET}.json", "w"), indent=2)
