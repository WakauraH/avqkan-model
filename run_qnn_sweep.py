"""QNN ベースラインの掃引: 5 対象関数 × 10 シード × 層数 {1,2,3,4} × 最適化器。

  paper : 論文どおり（乱数初期化 1 回、COBYLA maxiter=1000）
  tuned : L-BFGS-B（数値勾配）を 5 回の乱数初期値から走らせ、訓練コスト最小を採用

出力は results_qnn/。VQKAN 側（results/cobyla_*）と同じシード＝同じ訓練/テスト点。
使い方: python run_qnn_sweep.py [n_workers]
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from avqkan import run_qnn

TARGETS = ["eq6", "log", "frac", "radius", "expo"]
SEEDS = range(10)
LAYERS = [1, 2, 3, 4]
OUT = "results_qnn"


def job(kw):
    rows, meta = run_qnn(verbose=False, out_dir=OUT, **kw)
    return kw, rows[-1]["test_absdist"], meta["elapsed_sec"]


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    jobs = []
    for t in TARGETS:
        for s in SEEDS:
            for L in LAYERS:
                jobs.append(dict(target=t, seed=s, layers=L, opt="cobyla", maxiter=1000, restarts=1))
                jobs.append(dict(target=t, seed=s, layers=L, opt="lbfgs", maxiter=200, restarts=5))
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for kw, te, sec in ex.map(job, jobs):
            print(f"{kw['target']:7s} s{kw['seed']} L{kw['layers']} {kw['opt']:6s} "
                  f"final test={te:8.3f}  ({sec}s)", flush=True)
