"""カーネルリッジ回帰の古典ベースライン（閉形式 LOO-CV で超パラメータ選択、訓練点のみ使用）。

  krr_rbf : RBF カーネル exp(-γ|x-x'|²)                      — 既定の古典比較
  krr_add : 加法 RBF カーネル Σ_i exp(-γ(x_i-x'_i)²)            — 加法構造を知っている古典
  krr_sym : 置換対称特徴量 s=(Σx, Σx², Σx³)/d 上の RBF カーネル  — 対称性を知っている古典
LOO 残差は e_i = (y_i − ŷ_i)/(1 − H_ii), H = K(K+λI)^{-1}（閉形式）。出力は [-1,1] にクリップ。"""
import numpy as np

LAMS = np.logspace(-4, 1, 6)


def _rbf(A, B, g):
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-g * np.maximum(d2, 0))


def _add(A, B, g):
    return sum(np.exp(-g * (A[:, i][:, None] - B[:, i][None, :]) ** 2) for i in range(A.shape[1])) / A.shape[1]


def _symfeat(X):
    return np.stack([X.sum(1), (X ** 2).sum(1), (X ** 3).sum(1)], 1) / X.shape[1]


KERNELS = {
    "krr_rbf": (lambda A, B, g: _rbf(A, B, g), np.logspace(-2, 1, 7), lambda X: X),
    "krr_add": (lambda A, B, g: _add(A, B, g), np.logspace(-1, 2, 7), lambda X: X),
    "krr_sym": (lambda A, B, g: _rbf(A, B, g), np.logspace(-1, 3, 9), _symfeat),
    # 入力をソートしてから RBF: 特徴量を仮定しない汎用の置換不変カーネル
    "krr_sort": (lambda A, B, g: _rbf(A, B, g), np.logspace(-2, 1, 7), lambda X: np.sort(X, 1)),
}


def fit_predict(name, X, y, Xf):
    kern, gammas, feat = KERNELS[name]
    F, Ff = feat(X), feat(Xf)
    n = len(y); best = (np.inf, None, None)
    for g in gammas:
        K = kern(F, F, g)
        for lam in LAMS:
            M = np.linalg.inv(K + lam * np.eye(n))
            H = K @ M
            yhat = H @ y
            e = (y - yhat) / np.clip(1 - np.diag(H), 1e-8, None)
            loo = float(np.mean(e ** 2))
            if loo < best[0]:
                best = (loo, g, lam)
    _, g, lam = best
    alpha = np.linalg.solve(kern(F, F, g) + lam * np.eye(n), y)
    return np.clip(kern(Ff, F, g) @ alpha, -1, 1), dict(gamma=float(g), lam=float(lam), loo=best[0])


if __name__ == "__main__":
    # 自己検証: 以前の GridSearchCV 版（classical_regularized.krr_rbf）と同じ問題で近い値になること
    from avqkan import Config, Problem
    for s in range(3):
        prob = Problem(Config(nq=12, seed=s), "sepn12")
        for k in KERNELS:
            pred, info = fit_predict(k, prob.X[:, :12], prob.f, prob.Xf[:, :12])
            print(f"seed{s} {k:8s} test={np.sum(np.abs(pred - prob.ff)):6.2f} gamma={info['gamma']:.3g} lam={info['lam']:.3g}")
