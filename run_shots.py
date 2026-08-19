"""ショットノイズ下での比較。shots を引数で受ける（0 = 無限）。"""
import json, os, sys
import numpy as np
from avqkan import run

SHOTS = int(sys.argv[1])
SEEDS = list(range(int(sys.argv[2]) if len(sys.argv) > 2 else 5))
STEPS, GROW, TARGET = 30, 10, "log"
OUT = "results_shots"

res = {}
for method in ("cobyla", "gd", "ite"):
    fin, best = [], []
    for s in SEEDS:
        rows, meta = run(method, TARGET, s, STEPS, 0.1, GROW, OUT,
                         maxiter=200, verbose=False, shots=SHOTS,
                         tag_suffix=f"_shots{SHOTS}")
        te = [r["test_absdist"] for r in rows]
        fin.append(te[-1]); best.append(min(te))
        print(f"shots={SHOTS} {method:7s} seed{s} final={te[-1]:8.3f} "
              f"best={min(te):8.3f} ({meta['elapsed_sec']}s)", flush=True)
    res[method] = dict(final_mean=float(np.mean(fin)), final_std=float(np.std(fin)),
                       best_mean=float(np.mean(best)), best_std=float(np.std(best)))
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(f"{OUT}/summary_shots{SHOTS}.json", "w"), indent=2)
print(json.dumps(res, indent=2))
