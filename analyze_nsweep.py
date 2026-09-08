"""訓練点数掃引の集計（探索的）: d=12, N ∈ {10,20,40,100}, seeds 0–9。
量子: 訓練コスト早期停止のテスト誤差。古典: 同じ N で再フィット（KAN/MLP/線形/定数）、oracle = per-seed 最小。"""
import csv, json, os
import numpy as np
from scipy.stats import wilcoxon
from avqkan import Config, Problem
from classical_baseline import fit_kan, fit_mlp

NS, SEEDS, D = (10, 20, 40, 100), range(10), 12
out = {}
print(f"{'N':>4s} {'AVQKAN':>14s} {'oracle cls':>14s} {'cKAN':>8s} {'MLP8x8':>8s} {'MLP16':>8s} {'linear':>8s} {'const':>7s} {'med diff':>9s} {'p':>7s}")
for N in NS:
    Q, cls = [], {k: [] for k in ("kan", "mlp8x8", "mlp16", "linear", "const")}
    for s in SEEDS:
        rows = list(csv.DictReader(open(f"results_nsweep/cobyla_sepn12_seed{s}_nq12_N{N}.csv")))
        te = np.array([float(r["test_absdist"]) for r in rows]); co = np.array([float(r["cost"]) for r in rows])
        Q.append(te[int(np.argmin(co))])
        prob = Problem(Config(nq=D, seed=s, n_train=N), f"sepn{D}")
        pred, _, _ = fit_kan(prob, 8); cls["kan"].append(np.sum(np.abs(pred - prob.ff)))
        pred, _, _ = fit_mlp(prob, (8, 8), s); cls["mlp8x8"].append(np.sum(np.abs(pred - prob.ff)))
        pred, _, _ = fit_mlp(prob, (16,), s); cls["mlp16"].append(np.sum(np.abs(pred - prob.ff)))
        Xa = np.hstack([prob.X[:, :D], np.ones((N, 1))]); Xfa = np.hstack([prob.Xf[:, :D], np.ones((50, 1))])
        w, *_ = np.linalg.lstsq(Xa, prob.f, rcond=None)
        cls["linear"].append(np.sum(np.abs(np.clip(Xfa @ w, -1, 1) - prob.ff)))
        cls["const"].append(np.sum(np.abs(prob.ff - np.median(prob.f))))
    Q = np.array(Q); cls = {k: np.array(v) for k, v in cls.items()}
    O = np.min(np.stack([cls[k] for k in ("kan", "mlp8x8", "mlp16", "linear")]), axis=0)
    p = wilcoxon(Q, O)[1]
    print(f"{N:4d} {Q.mean():7.2f}±{Q.std(ddof=1):5.2f} {O.mean():7.2f}±{O.std(ddof=1):5.2f} "
          + " ".join(f"{cls[k].mean():8.2f}" for k in ("kan", "mlp8x8", "mlp16", "linear")) + f" {cls['const'].mean():7.2f}"
          f" {np.median(Q-O):+9.2f} {p:7.4f}")
    out[N] = dict(avqkan=Q.tolist(), oracle=O.tolist(), p=float(p), **{k: v.tolist() for k, v in cls.items()})
os.makedirs("results_model", exist_ok=True)
json.dump(out, open("results_model/nsweep.json", "w"), indent=2)
