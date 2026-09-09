"""実機推論・雑音評価用に、論文のモデルを同じシードで再訓練しパラメータを保存する（決定論的なので同一軌跡）。
  models/d4/   eq6, seeds 0-9  (60 steps, grow 20, maxiter 200)
  models/d12/  sepn12, seeds 30-49 (40 steps, grow 10, maxiter 30, all_pairs)
保存後、CSV の test_absdist が元の結果と一致することを確認する。"""
import sys, csv, os
from concurrent.futures import ProcessPoolExecutor
from avqkan import run


def job(a):
    kind, s = a
    if kind == "d4":
        rows, meta = run("cobyla", "eq6", s, 60, 0.1, 20, "models/d4", maxiter=200, verbose=False)
        ref = f"results/cobyla_eq6_seed{s}.csv"
    else:
        rows, meta = run("cobyla", "sepn12", s, 40, 0.1, 10, "models/d12", maxiter=30, verbose=False,
                         nq=12, readout_mode="all_pairs", tag_suffix="_nq12_m30")
        ref = f"results_model_confirm/cobyla_sepn12_seed{s}_nq12_m30.csv"
    old = [float(r["test_absdist"]) for r in csv.DictReader(open(ref))]
    new = [r["test_absdist"] for r in rows]
    same = max(abs(a - b) for a, b in zip(old, new)) < 1e-6
    return kind, s, same, meta["best"]["test_absdist"], meta["best"]["ansatz"]


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    jobs = [("d12", s) for s in range(30, 50)] + [("d4", s) for s in range(10)]
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for kind, s, same, te, ans in ex.map(job, jobs):
            print(f"{kind} seed{s} reproduces={same} best_test={te:.3f} ansatz={ans}", flush=True)
