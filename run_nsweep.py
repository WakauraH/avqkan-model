"""訓練点数掃引（探索的, Fig. 5 用）: d=12, N ∈ {10, 20, 40, 100}, seeds 0–9, COBYLA maxiter 30。
古典側は analyze 時に同じ N で再計算する。
使い方: python run_nsweep.py [n_workers]
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from avqkan import run

NS = (10, 20, 40, 100)
SEEDS = range(10)
OUT = "results_nsweep"


def job(a):
    N, s = a
    rows, meta = run("cobyla", "sepn12", s, 40, 0.1, 10, OUT, maxiter=30, verbose=False,
                     nq=12, readout_mode="all_pairs", n_train=N, tag_suffix=f"_nq12_N{N}")
    co = [r["cost"] for r in rows]
    return N, s, rows[co.index(min(co))]["test_absdist"], meta["elapsed_sec"]


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    jobs = sorted([(N, s) for N in NS for s in SEEDS], key=lambda a: -a[0])
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for N, s, te, sec in ex.map(job, jobs):
            print(f"N={N:3d} seed{s}  earlystop test={te:8.3f}  ({sec}s)", flush=True)
