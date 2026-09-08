"""事前登録研究（PREREGISTRATION_model.md）の量子アーム。

  results_model_confirm/       d ∈ {12,16,18}, seeds 30–49, 厳密シミュレーション
  results_model_confirm_s256/  d = 12, seeds 30–49, 256 shots（訓練のみ雑音、指標は厳密）
  results_model_confirm/ (d=20, seeds 30–39) は計算資源の都合で記述的報告のみ

使い方: python run_model_confirm.py [n_workers] [d ...]
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from avqkan import run, run_qnn

SEEDS = range(30, 50)


def job(a):
    d, s, shots = a
    if shots == -1:   # 副次: d=12 の調整済み QNN（L はシードごとに訓練コストで選ぶ）
        for L in (1, 2, 3):
            run_qnn(f"sepn{d}", s, "results_model_confirm_qnn", layers=L, opt="lbfgs",
                    maxiter=100, restarts=2, nq=d, readout_mode="all_pairs", verbose=False,
                    tag_suffix=f"_nq{d}")
        return d, s, shots, float("nan"), 0
    out = "results_model_confirm_s256" if shots else "results_model_confirm"
    rows, meta = run("cobyla", f"sepn{d}", s, 40, 0.1, 10, out, maxiter=30, verbose=False,
                     nq=d, readout_mode="all_pairs", shots=shots, tag_suffix=f"_nq{d}_m30")
    te = [r["test_absdist"] for r in rows]
    co = [r["cost"] for r in rows]
    return d, s, shots, te[co.index(min(co))], meta["elapsed_sec"]


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    ds = [int(x) for x in sys.argv[2:]] or [12, 16, 18]
    jobs = []
    for d in ds:
        seeds = SEEDS if d < 20 else range(30, 40)
        jobs += [(d, s, 0) for s in seeds]
        if d == 12:
            jobs += [(12, s, 256) for s in SEEDS]
            jobs += [(12, s, -1) for s in SEEDS]
    # 重い d を先に投入して末尾の待ちを減らす
    jobs.sort(key=lambda a: -a[0])
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for d, s, shots, te, sec in ex.map(job, jobs):
            print(f"d={d:2d} seed{s} shots={shots:4d}  earlystop test={te:8.3f}  ({sec}s)", flush=True)
