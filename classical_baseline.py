"""古典ベースライン: KAN（B-spline エッジ）と MLP。

VQKAN と条件を揃える:
  - 同じ訓練 10 点 / テスト 50 点（`Problem` と同じシードから生成）
  - 同じ入力変数の数
  - 出力は [-1, 1] にクリップ（VQKAN の読み出し ⟨H⟩ が [-1,1] に限られるため）
  - 指標はテスト 50 点の絶対距離の和

KAN（1 層）は Liu et al. (2024) の定式化
    φ_i(x) = w_i · silu(x) + Σ_j c_ij B_j(x),   y = Σ_i φ_i(x_i)
を用いる。スプライン係数と基底重みについて **線形** なので、リッジ回帰の閉形式で解ける
（10 点しかない訓練集合に対し、反復最適化より公平で再現性が高い）。
"""

import argparse
import csv
import json
import os

import numpy as np
from scipy.interpolate import BSpline

from avqkan import Config, Problem, TARGETS


def bspline_basis(x, n_ctrl, k=3, lo=0.0, hi=1.0):
    """区間 [lo,hi] 上の n_ctrl 個の B-spline 基底を x（配列）で評価する。"""
    n_knots = n_ctrl + k + 1
    inner = np.linspace(lo, hi, n_knots - 2 * k)
    knots = np.concatenate([np.full(k, lo), inner, np.full(k, hi)])
    out = np.zeros((len(np.atleast_1d(x)), n_ctrl))
    for j in range(n_ctrl):
        c = np.zeros(n_ctrl); c[j] = 1.0
        out[:, j] = BSpline(knots, c, k, extrapolate=True)(np.atleast_1d(x))
    return out


def silu(x):
    return x / (1.0 + np.exp(-x))


def kan_design(X, dim, n_ctrl):
    """1 層 KAN の設計行列。列 = [各入力のスプライン基底..., 各入力の silu, 定数項]"""
    rows = []
    for i in range(dim):
        rows.append(bspline_basis(X[:, i], n_ctrl))
    D = np.hstack(rows)
    D = np.hstack([D, silu(X[:, :dim]), np.ones((X.shape[0], 1))])
    return D


def fit_kan(prob, n_ctrl=8, ridge=1e-6):
    dim = prob.cfg.dim
    A = kan_design(prob.X, dim, n_ctrl)
    y = prob.f
    P = A.shape[1]
    coef = np.linalg.solve(A.T @ A + ridge * np.eye(P), A.T @ y)
    pred = np.clip(kan_design(prob.Xf, dim, n_ctrl) @ coef, -1, 1)
    train = np.clip(A @ coef, -1, 1)
    return pred, train, P


def fit_mlp(prob, hidden=(8, 8), seed=0, max_iter=20000):
    from sklearn.neural_network import MLPRegressor
    dim = prob.cfg.dim
    m = MLPRegressor(hidden_layer_sizes=hidden, activation="tanh",
                     solver="lbfgs", max_iter=max_iter, random_state=seed,
                     alpha=1e-4)
    m.fit(prob.X[:, :dim], prob.f)
    pred = np.clip(m.predict(prob.Xf[:, :dim]), -1, 1)
    train = np.clip(m.predict(prob.X[:, :dim]), -1, 1)
    P = sum(c.size for c in m.coefs_) + sum(b.size for b in m.intercepts_)
    return pred, train, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=sorted(TARGETS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-ctrl", type=int, default=8)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    summary = {}
    print(f"{'target':8s} {'model':12s} {'params':>7s} "
          f"{'test mean±std':>20s} {'train mean±std':>20s}")
    for tgt in a.targets:
        summary[tgt] = {}
        for name in ("kan", "mlp8x8", "mlp16"):
            te, tr, Ps = [], [], []
            for s in range(a.seeds):
                cfg = Config(seed=s)
                prob = Problem(cfg, tgt)
                if name == "kan":
                    pred, trp, P = fit_kan(prob, a.n_ctrl)
                elif name == "mlp8x8":
                    pred, trp, P = fit_mlp(prob, (8, 8), s)
                else:
                    pred, trp, P = fit_mlp(prob, (16,), s)
                te.append(float(np.sum(np.abs(pred - prob.ff))))
                tr.append(float(np.sum(np.abs(trp - prob.f))))
                Ps.append(P)
            summary[tgt][name] = dict(
                params=int(np.mean(Ps)),
                test_mean=float(np.mean(te)), test_std=float(np.std(te)),
                train_mean=float(np.mean(tr)), train_std=float(np.std(tr)))
            v = summary[tgt][name]
            print(f"{tgt:8s} {name:12s} {v['params']:7d} "
                  f"{v['test_mean']:10.3f} ± {v['test_std']:<7.3f} "
                  f"{v['train_mean']:10.3f} ± {v['train_std']:<7.3f}")
    with open(os.path.join(a.out, "summary_classical.json"), "w") as fp:
        json.dump(summary, fp, indent=2)


if __name__ == "__main__":
    main()
