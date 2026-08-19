"""量子ビット数スケーリング: nq = 4,6,8,10。

問題自体を nq とともに拡大する:
  対象関数  sep{nq} = eq.(6) の nq 変数への一般化
  読み出し  H = (2/nq) Σ_j Z_{2j} Z_{2j+1}   （⟨H⟩ ∈ [-1,1] に正規化）
これをやらないと、追加した量子ビットが問題に一切関与せず結果が nq に依存しない。
"""
import json, os, sys
import numpy as np
from avqkan import run

NQ = int(sys.argv[1])
SEEDS = list(range(int(sys.argv[2]) if len(sys.argv) > 2 else 5))
STEPS, GROW = 40, 10
TARGET = f"sep{NQ}"
OUT = "results_scaling"

res = {}
for method in ("cobyla", "ite"):
    fin, best, calls, params = [], [], [], []
    for s in SEEDS:
        rows, meta = run(method, TARGET, s, STEPS, 0.1, GROW, OUT,
                         maxiter=200, verbose=False, nq=NQ,
                         readout_mode="all_pairs", tag_suffix=f"_nq{NQ}")
        te = [r["test_absdist"] for r in rows]
        fin.append(te[-1]); best.append(min(te))
        calls.append(rows[-1]["forward_calls"]); params.append(rows[-1]["n_params"])
        print(f"nq={NQ} {method:7s} seed{s} final={te[-1]:8.3f} best={min(te):8.3f} "
              f"P={rows[-1]['n_params']} calls={rows[-1]['forward_calls']} "
              f"({meta['elapsed_sec']}s)", flush=True)
    res[method] = dict(final_mean=float(np.mean(fin)), final_std=float(np.std(fin)),
                       best_mean=float(np.mean(best)), best_std=float(np.std(best)),
                       forward_calls=int(np.mean(calls)), n_params=int(np.mean(params)))
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(f"{OUT}/summary_nq{NQ}.json", "w"), indent=2)
print(json.dumps(res, indent=2))
