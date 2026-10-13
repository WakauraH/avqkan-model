"""PREREGISTRATION_linz.md の量子アーム。seeds 30–49, sepn_12, N=100, linz_m200_g5（主）と linz_m200（副次）。"""
import sys, os
from concurrent.futures import ProcessPoolExecutor
from avqkan import run
OUT = "results_linz_confirm"


def job(a):
    v, s = a
    grow = 5 if v == "linz_m200_g5" else 10
    tag = f"_{v}_N100"
    if os.path.exists(f"{OUT}/cobyla_sepn_12_seed{s}{tag}.csv"):
        return a, None
    rows, meta = run("cobyla", "sepn_12", s, 40, 0.1, grow, OUT, verbose=False, nq=12, n_train=100,
                     tag_suffix=tag, maxiter=200, readout_mode="linear_z")
    co = [r["cost"] for r in rows]
    return a, (rows[co.index(min(co))]["test_absdist"], meta["elapsed_sec"])


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    jobs = [(v, s) for v in ("linz_m200_g5", "linz_m200") for s in range(30, 50)]
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for a, r in ex.map(job, jobs):
            print(a, (f"test={r[0]:.3f} ({r[1]}s)" if r else "cached"), flush=True)
