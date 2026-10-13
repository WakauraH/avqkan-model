"""変種探索の第 2 段: 最良変種 linz（線形 Z 読み出し）に最適化予算と成長頻度を足す。
  linz_m200     maxiter 200
  linz_m200_g5  maxiter 200 + 5 ステップごとに成長（最大 8 演算子）
targets sepn_12 / eq6, N ∈ {10, 40, 100}, seeds 0–9。出力は results_variants/ に同じ命名。"""
import sys, os
from concurrent.futures import ProcessPoolExecutor
from avqkan import run
from run_variants import TARGETS, OUT

V2 = {"linz_m200": dict(maxiter=200, grow=10), "linz_m200_g5": dict(maxiter=200, grow=5)}


def job(a):
    t, v, N, s = a
    d, ro = TARGETS[t]
    tag = f"_{v}_N{N}"
    if os.path.exists(f"{OUT}/cobyla_{t}_seed{s}{tag}.csv"):
        return a, None
    rows, meta = run("cobyla", t, s, 40, 0.1, V2[v]["grow"], OUT, verbose=False, nq=d, n_train=N,
                     tag_suffix=tag, maxiter=V2[v]["maxiter"], readout_mode="linear_z")
    co = [r["cost"] for r in rows]
    return a, (rows[co.index(min(co))]["test_absdist"], meta["elapsed_sec"])


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    jobs = [(t, v, N, s) for t in ("sepn_12", "eq6") for v in V2 for N in (10, 40, 100) for s in range(10)]
    jobs.sort(key=lambda a: -(TARGETS[a[0]][0] * a[2]))
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for a, r in ex.map(job, jobs):
            print(a, (f"test={r[0]:.3f} ({r[1]}s)" if r else "cached"), flush=True)
