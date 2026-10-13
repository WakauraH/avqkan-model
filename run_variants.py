"""モデル変種の探索（seeds 0–9, 事前登録外）。
targets: sepn_12, step_12, addsin_12 (d=12, all_pairs), eq6 (d=4, single)。N ∈ {10, 40}。
variants:
  base       現行（共有スプライン, arccos, 読み出し固定, COBYLA maxiter 30）
  base_m200  最適化予算のみ増加（maxiter 200）
  perin      入力ごとスプライン（grids 4）
  tanh       角度写像 tanh
  linz       線形 Z 読み出し（1 体 + 2 体 ⟨Z⟩ のリッジ, LOO 目的関数）
  perin_tanh 入力ごと + tanh
  all3       入力ごと + tanh + 線形 Z
出力: results_variants/cobyla_<target>_seed<s>_<variant>_N<N>.csv"""
import sys, os
from concurrent.futures import ProcessPoolExecutor
from avqkan import run

OUT = "results_variants"
TARGETS = {"sepn_12": (12, "all_pairs"), "step_12": (12, "all_pairs"), "addsin_12": (12, "all_pairs"), "eq6": (4, "single")}
NS = (10, 40)
V = {
    "base":       dict(),
    "base_m200":  dict(maxiter=200),
    "perin":      dict(spline_share="per_input", grids=4),
    "tanh":       dict(angle_map="tanh"),
    "linz":       dict(readout_mode="linear_z"),
    "perin_tanh": dict(spline_share="per_input", grids=4, angle_map="tanh"),
    "all3":       dict(spline_share="per_input", grids=4, angle_map="tanh", readout_mode="linear_z"),
}


def job(a):
    t, v, N, s = a
    d, ro = TARGETS[t]
    kw = dict(maxiter=30, grids=8, readout_mode=ro); kw.update(V[v])
    tag = f"_{v}_N{N}"
    if os.path.exists(f"{OUT}/cobyla_{t}_seed{s}{tag}.csv"):
        return a, None
    rows, meta = run("cobyla", t, s, 40, 0.1, 10, OUT, verbose=False, nq=d, n_train=N, tag_suffix=tag, **kw)
    co = [r["cost"] for r in rows]
    return a, (rows[co.index(min(co))]["test_absdist"], meta["elapsed_sec"])


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    jobs = [(t, v, N, s) for t in TARGETS for v in V for N in NS for s in range(10)]
    jobs.sort(key=lambda a: -(TARGETS[a[0]][0] * (7 if a[1] == "base_m200" else 1)))
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for a, r in ex.map(job, jobs):
            print(a, (f"test={r[0]:.3f} ({r[1]}s)" if r else "cached"), flush=True)
