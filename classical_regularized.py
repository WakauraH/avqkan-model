"""正則化した古典ベースライン（探索的・重要）。

N=10 の少数データで量子モデルが古典 oracle に勝つ理由が「低容量モデルの暗黙の正則化」なら、
正則化を正しく入れた古典モデルも同じ利益を得るはず。ここで確認する:
  ridge      : リッジ回帰、λ を leave-one-out CV（訓練 10 点のみ）で選択
  krr_rbf    : RBF カーネルリッジ、(λ, γ) を LOO-CV グリッドで選択
  gp_rbf     : ガウス過程回帰（RBF + 白色雑音、周辺尤度最大化）
  kan_ridge  : 古典 KAN（8 制御点）、λ を LOO-CV で選択
いずれも訓練点だけで超パラメータを決める（テスト情報は使わない）。出力は [-1,1] にクリップ。
使い方: python classical_regularized.py <target> <nq> <n_train> <seeds> [quantum_csv_pattern_with_SEED]
"""
import sys, csv
import numpy as np
from scipy.stats import wilcoxon
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from avqkan import Config, Problem
from classical_baseline import kan_design


def loo_ridge(A, y, lams):
    best = None
    for lam in lams:
        err = 0.0
        for i in range(len(y)):
            m = np.ones(len(y), bool); m[i] = False
            w = np.linalg.solve(A[m].T @ A[m] + lam * np.eye(A.shape[1]), A[m].T @ y[m])
            err += (A[i] @ w - y[i]) ** 2
        if best is None or err < best[0]:
            best = (err, lam)
    lam = best[1]
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y), lam


def run(target, nq, n_train, seeds, qpat=None):
    lams = np.logspace(-4, 2, 13)
    res = {k: [] for k in ("ridge", "krr_rbf", "gp_rbf", "kan_ridge")}
    for s in seeds:
        prob = Problem(Config(nq=nq, seed=s, n_train=n_train), target)
        d = prob.cfg.dim
        X, Xf, y, yf = prob.X[:, :d], prob.Xf[:, :d], prob.f, prob.ff
        A = np.hstack([X, np.ones((len(X), 1))]); Af = np.hstack([Xf, np.ones((len(Xf), 1))])
        w, _ = loo_ridge(A, y, lams)
        res["ridge"].append(np.sum(np.abs(np.clip(Af @ w, -1, 1) - yf)))
        Ak, Akf = kan_design(X, d, 8), kan_design(Xf, d, 8)
        w, _ = loo_ridge(Ak, y, lams)
        res["kan_ridge"].append(np.sum(np.abs(np.clip(Akf @ w, -1, 1) - yf)))
        gs = GridSearchCV(KernelRidge(kernel="rbf"),
                          {"alpha": np.logspace(-4, 1, 6), "gamma": np.logspace(-2, 1, 7)},
                          cv=LeaveOneOut(), scoring="neg_mean_squared_error").fit(X, y)
        res["krr_rbf"].append(np.sum(np.abs(np.clip(gs.predict(Xf), -1, 1) - yf)))
        k = ConstantKernel(1.0) * RBF(length_scale=np.ones(d)) + WhiteKernel(1e-2)
        gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3, random_state=s).fit(X, y)
        res["gp_rbf"].append(np.sum(np.abs(np.clip(gp.predict(Xf), -1, 1) - yf)))
    res = {k: np.array(v) for k, v in res.items()}
    Q = None
    if qpat:
        Q = []
        for s in seeds:
            rows = list(csv.DictReader(open(qpat.replace("SEED", str(s)))))
            te = np.array([float(r["test_absdist"]) for r in rows]); co = np.array([float(r["cost"]) for r in rows])
            Q.append(te[int(np.argmin(co))])
        Q = np.array(Q)
        print(f"AVQKAN     {Q.mean():7.2f} ± {Q.std(ddof=1):5.2f}")
    for k, v in res.items():
        line = f"{k:10s} {v.mean():7.2f} ± {v.std(ddof=1):5.2f}"
        if Q is not None:
            line += f"   vs AVQKAN med diff {np.median(Q - v):+6.2f}  p={wilcoxon(Q, v)[1]:.4f}"
        print(line)
    return res, Q


if __name__ == "__main__":
    t, nq, n, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    seeds = range(int(ns.split("-")[0]), int(ns.split("-")[1]) + 1) if "-" in ns else range(int(ns))
    run(t, nq, n, seeds, sys.argv[5] if len(sys.argv) > 5 else None)
