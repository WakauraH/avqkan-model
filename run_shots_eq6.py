"""有限ショット（Fig. 4a 用, 探索的）: d=4 eq6, COBYLA maxiter 200, 60 ステップ, shots ∈ {256,1024,4096}, seeds 0–9。
指標は常に厳密値。無限ショットの対照は results/cobyla_eq6_seed*.csv。"""
import sys
from concurrent.futures import ProcessPoolExecutor
from avqkan import run

OUT = "results_shots_eq6"


def job(a):
    shots, s = a
    rows, meta = run("cobyla", "eq6", s, 60, 0.1, 20, OUT, maxiter=200, verbose=False,
                     shots=shots, tag_suffix=f"_s{shots}")
    co = [r["cost"] for r in rows]
    return shots, s, rows[co.index(min(co))]["test_absdist"], meta["elapsed_sec"]


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    jobs = [(sh, s) for sh in (256, 1024, 4096) for s in range(10)]
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for sh, s, te, sec in ex.map(job, jobs):
            print(f"shots={sh:5d} seed{s}  earlystop test={te:8.3f}  ({sec}s)", flush=True)
