"""探索（seeds 0–9, 事前登録外）: KAN 構造が有利な関数族と読み出しを探す。
family ∈ _FAMILIES, d ∈ {8, 12}, readout ∈ {all_pairs, sum_z}, N=10, COBYLA maxiter 30, 40 ステップ, 10 ごとに成長。
出力: results_explore/cobyla_<family>_<d>_seed<s>_<readout>.csv
使い方: python run_explore.py [n_workers]"""
import sys, os
from concurrent.futures import ProcessPoolExecutor
from avqkan import run, _FAMILIES

OUT = "results_explore"
FAMS = list(_FAMILIES)
DS = (8, 12)
READOUTS = ("all_pairs", "sum_z")
SEEDS = range(10)


def job(a):
    fam, d, ro, s = a
    tag = f"_{ro}"
    path = f"{OUT}/cobyla_{fam}_{d}_seed{s}{tag}.csv"
    if os.path.exists(path):
        return a, None
    rows, meta = run("cobyla", f"{fam}_{d}", s, 40, 0.1, 10, OUT, maxiter=30, verbose=False,
                     nq=d, readout_mode=ro, tag_suffix=tag)
    co = [r["cost"] for r in rows]
    return a, rows[co.index(min(co))]["test_absdist"]


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    jobs = sorted([(f, d, r, s) for f in FAMS for d in DS for r in READOUTS for s in SEEDS], key=lambda a: -a[1])
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for a, te in ex.map(job, jobs):
            print(a, f"earlystop test={te:.3f}" if te is not None else "cached", flush=True)
