"""事前登録済み確証解析（PREREGISTRATION.md 参照）。
実行前にコミットされ、以後変更しないこと。"""
import csv
import numpy as np
from scipy.stats import wilcoxon
from avqkan import Config, Problem
from classical_baseline import fit_kan, fit_mlp

SEEDS = list(range(10, 30))
SIZES = (12, 14, 16)


def es(pat):
    out = []
    for s in SEEDS:
        rows = list(csv.DictReader(open(pat.replace("SEED", str(s)))))
        te = np.array([float(r["test_absdist"]) for r in rows])
        co = np.array([float(r["cost"]) for r in rows])
        out.append(te[int(np.argmin(co))])
    return np.array(out)


def classical_oracle(d):
    best, const = [], []
    for s in SEEDS:
        cfg = Config(nq=d, seed=s)
        prob = Problem(cfg, f"sepn{d}")
        vals = []
        pred, _, _ = fit_kan(prob, 8); vals.append(np.sum(np.abs(pred - prob.ff)))
        pred, _, _ = fit_mlp(prob, (8, 8), s); vals.append(np.sum(np.abs(pred - prob.ff)))
        pred, _, _ = fit_mlp(prob, (16,), s); vals.append(np.sum(np.abs(pred - prob.ff)))
        X, Xf = prob.X[:, :d], prob.Xf[:, :d]
        Xa = np.hstack([X, np.ones((len(X), 1))])
        Xfa = np.hstack([Xf, np.ones((len(Xf), 1))])
        w, *_ = np.linalg.lstsq(Xa, prob.f, rcond=None)
        vals.append(np.sum(np.abs(np.clip(Xfa @ w, -1, 1) - prob.ff)))
        best.append(min(vals))
        const.append(np.sum(np.abs(prob.ff - np.median(prob.f))))
    return np.array(best), np.array(const)


def holm(ps):
    order = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def main():
    print("=" * 88)
    print("事前登録済み確証解析（シード 10-29, n=20, Holm m=3）")
    print("=" * 88)
    ps, rows = [], []
    for d in SIZES:
        I = es(f"results_confirm/ite_sepn{d}_seedSEED.csv")
        O, K = classical_oracle(d)
        gate1, gate2 = I.mean() < K.mean(), O.mean() < K.mean()
        _, p = wilcoxon(I, O)
        ps.append(p)
        rows.append((d, I, O, K, gate1, gate2, p))
    adj = holm(np.array(ps))
    print(f"{'d':>3s} {'量子(NG)':>16s} {'オラクル古典':>16s} {'定数':>8s} "
          f"{'gate':>5s} {'中央値差':>9s} {'p':>8s} {'Holm p':>8s} 判定")
    for (d, I, O, K, g1, g2, p), ap in zip(rows, adj):
        gate = "OK" if (g1 and g2) else "VOID"
        if gate == "VOID":
            verdict = "無効（ゲート不成立）"
        elif ap < 0.05 and np.median(I - O) < 0:
            verdict = "PASS（量子優位を確証）"
        else:
            verdict = "非有意"
        print(f"{d:3d} {I.mean():8.3f}±{I.std(ddof=1):5.2f} "
              f"{O.mean():8.3f}±{O.std(ddof=1):5.2f} {K.mean():8.2f} {gate:>5s} "
              f"{np.median(I - O):+9.3f} {p:8.4f} {ap:8.4f} {verdict}")

    print()
    print("副次（探索的, BH なしの生 p）: COBYLA vs オラクル古典 / NG vs COBYLA")
    for d in SIZES:
        try:
            C = es(f"results_confirm_m30/cobyla_sepn{d}_seedSEED.csv")
        except FileNotFoundError:
            print(f"  d={d}: cobyla 未完"); continue
        I = es(f"results_confirm/ite_sepn{d}_seedSEED.csv")
        O, _ = classical_oracle(d)
        _, p1 = wilcoxon(C, O); _, p2 = wilcoxon(I, C)
        print(f"  d={d:2d}: cobyla {C.mean():7.3f}±{C.std(ddof=1):5.2f} "
              f"vs古典 p={p1:.4f} | NG vs cobyla 中央値差 {np.median(I-C):+7.3f} p={p2:.4f}")


if __name__ == "__main__":
    main()
