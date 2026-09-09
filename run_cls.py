"""分類（SI 用）: cls2d, seeds 0–9。AVQKAN（COBYLA, BCE 損失, 60 ステップ, 20 ごとに成長）と
QNN（論文設定 3 層 COBYLA / 調整版 L-BFGS L∈{1,2,3}）。古典（ロジスティック, SVM-RBF, 多数決）は解析時。
使い方: python run_cls.py [n_workers]"""
import sys
from concurrent.futures import ProcessPoolExecutor
from avqkan import run, run_qnn

OUT = "results_cls"


def job(a):
    kind, s = a
    if kind == "avqkan":
        rows, meta = run("cobyla", "cls2d", s, 60, 0.1, 20, OUT, maxiter=200, verbose=False)
    elif kind == "avqkan_way1":
        rows, meta = run("cobyla", "cls2d", s, 60, 0.1, 20, OUT, maxiter=200, verbose=False,
                         select="grad", tag_suffix="_way1")
    elif kind == "qnn_paper":
        rows, meta = run_qnn("cls2d", s, OUT, layers=3, opt="cobyla", maxiter=1000, verbose=False)
    else:
        for L in (1, 2, 3):
            rows, meta = run_qnn("cls2d", s, OUT, layers=L, opt="lbfgs", maxiter=200, restarts=5, verbose=False)
    co = [r["cost"] for r in rows]
    return kind, s, rows[co.index(min(co))]["test_absdist"], rows[-1]["ansatz"]


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    jobs = [(k, s) for k in ("avqkan", "avqkan_way1", "qnn_paper", "qnn_tuned") for s in range(10)]
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for k, s, te, ans in ex.map(job, jobs):
            print(f"{k:12s} seed{s}  test errors/50 = {te:4.0f}  {ans}", flush=True)
