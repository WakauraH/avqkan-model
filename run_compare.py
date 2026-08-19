"""4 手法 × 5 シードの比較を回し、集計を出す。"""
import numpy as np, json, os, sys
from avqkan import run

TARGET = sys.argv[1] if len(sys.argv) > 1 else "log"
SEEDS = list(range(int(sys.argv[2]) if len(sys.argv) > 2 else 5))
STEPS = 60
OUT = "results"

METHODS = [
    ("legacy", dict(kick=0.1, ng_growth=True,  spline_eval="grid")),   # 原実装の再現
    ("cobyla", dict(kick=0.0, ng_growth=False, spline_eval="direct")), # 前報方式
    ("gd",     dict(kick=0.0, ng_growth=False, spline_eval="direct")), # 計量なし
    ("ite",    dict(kick=0.0, ng_growth=False, spline_eval="direct")), # 修正版 ITE
]

summary = {}
for method, kw in METHODS:
    finals, bests = [], []
    for s in SEEDS:
        rows, meta = run(method, TARGET, s, STEPS, 0.1, 20, OUT,
                         maxiter=200, verbose=False, **kw)
        te = [r["test_absdist"] for r in rows]
        finals.append(te[-1]); bests.append(min(te))
        print(f"{method:7s} seed{s}  final={te[-1]:8.3f}  best={min(te):8.3f}  "
              f"({meta['elapsed_sec']}s)", flush=True)
    summary[method] = dict(
        final_mean=float(np.mean(finals)), final_std=float(np.std(finals)),
        best_mean=float(np.mean(bests)),   best_std=float(np.std(bests)))

print("\n=== テスト点 50 点の絶対距離の和（5 シード） target=%s ===" % TARGET)
print(f"{'method':10s} {'final mean±std':>22s} {'best mean±std':>22s}")
for m, v in summary.items():
    print(f"{m:10s} {v['final_mean']:10.3f} ± {v['final_std']:<9.3f} "
          f"{v['best_mean']:10.3f} ± {v['best_std']:<9.3f}")
json.dump(summary, open(os.path.join(OUT, f"summary_{TARGET}.json"), "w"), indent=2)
