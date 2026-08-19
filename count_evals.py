"""手法ごとの回路（状態ベクトル）評価回数を計測する。seed 0 のみ。"""
import csv
from avqkan import run
CFG = {"legacy": dict(kick=0.1, ng_growth=True,  spline_eval="grid"),
       "cobyla": dict(kick=0.0, ng_growth=False, spline_eval="direct"),
       "gd":     dict(kick=0.0, ng_growth=False, spline_eval="direct"),
       "ite":    dict(kick=0.0, ng_growth=False, spline_eval="direct")}
print(f"{'method':8s} {'回路評価回数':>14s} {'最終 test':>10s} {'最良 test':>10s} {'秒':>7s}")
for m, kw in CFG.items():
    rows, meta = run(m, "log", 0, 60, 0.1, 20, "results", maxiter=200,
                     verbose=False, **kw)
    te = [r["test_absdist"] for r in rows]
    print(f"{m:8s} {rows[-1]['forward_calls']:14d} {te[-1]:10.3f} {min(te):10.3f} "
          f"{meta['elapsed_sec']:7.1f}", flush=True)
