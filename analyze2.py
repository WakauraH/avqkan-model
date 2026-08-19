"""パネル査読後の主解析（analyze.py を置き換える確証的解析）。

事前指定:
  主指標   : 訓練コスト C の argmin ステップで停止し、その時点のテスト誤差を1回だけ評価
             （テスト集合はいかなる選択にも使わない）
  主要評価項目族（4検定, Holm 補正, α=0.05）:
    H1: eq6 等予算       ite vs cobyla max20   (results_budget)
    H2: 読み出し single   ite vs cobyla max30   (results_readout)
    H3: nq=10 等予算      ite vs cobyla max30   (results_scaling)
    H4: nq=12 等予算      ite vs cobyla max30   (results_scaling)
  副指標   : final（最終ステップ）
  探索的   : best（テスト軌跡最小 — 情報漏洩があるため参考表示のみ）
  その他の検定は探索的とみなし BH の q 値を併記
  ばらつきは標本標準偏差 (ddof=1)
"""
import csv, json, os
import numpy as np
from scipy.stats import wilcoxon

SEEDS = range(10)


def load_traj(path):
    rows = list(csv.DictReader(open(path)))
    te = np.array([float(r["test_absdist"]) for r in rows])
    co = np.array([float(r["cost"]) for r in rows])
    return te, co


def metric(path_pat, kind="earlystop", n=10):
    out = []
    for s in range(n):
        te, co = load_traj(path_pat.replace("SEED", str(s)))
        if kind == "earlystop":
            out.append(te[int(np.argmin(co))])
        elif kind == "final":
            out.append(te[-1])
        else:                       # best（探索的・漏洩あり）
            out.append(te.min())
    return np.array(out)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def bh(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = p[idx] * m / (rank + 1)
        prev = min(prev, val)
        q[idx] = prev
    return q


def fmt(a):
    return f"{a.mean():8.3f} ± {a.std(ddof=1):6.3f}"


PRIMARY = [
    ("H1 eq6 等予算 (vs max20)",
     "results_budget/ite_eq6_seedSEED_bud.csv",
     "results_budget/cobyla_eq6_seedSEED_bud20.csv"),
    ("H2 読み出し single (vs max30)",
     "results_readout/ite_sep4_seedSEED_single_None.csv",
     "results_readout/cobyla_sep4_seedSEED_single_30.csv"),
    ("H3 nq=10 等予算 (vs max30)",
     "results_scaling/ite_sep10_seedSEED_nq10_b.csv",
     "results_scaling/cobyla_sep10_seedSEED_nq10_m30.csv"),
    ("H4 nq=12 等予算 (vs max30)",
     "results_scaling/ite_sep12_seedSEED_nq12_ite.csv",
     "results_scaling/cobyla_sep12_seedSEED_nq12_m30.csv"),
]


def main():
    print("=" * 96)
    print("主要評価項目族（訓練コスト早期停止・Holm 補正・n=10 対応 Wilcoxon）")
    print("=" * 96)
    ps, rows = [], []
    for name, a_pat, b_pat in PRIMARY:
        A = metric(a_pat); B = metric(b_pat)
        _, p = wilcoxon(A, B)
        ps.append(p)
        rows.append((name, A, B, p))
    adj = holm(np.array(ps))
    print(f"{'検定':34s} {'ITE':>18s} {'COBYLA':>18s} {'中央値差':>9s} {'p':>8s} {'Holm p':>8s} 判定")
    for (name, A, B, p), ap in zip(rows, adj):
        verdict = "PASS (ite良)" if ap < 0.05 and np.median(A - B) < 0 else \
                  ("PASS (cobyla良)" if ap < 0.05 else "非有意")
        print(f"{name:34s} {fmt(A):>18s} {fmt(B):>18s} {np.median(A-B):+9.3f} "
              f"{p:8.4f} {ap:8.4f} {verdict}")

    print()
    print("=" * 96)
    print("副指標（final）での同じ4検定")
    print("=" * 96)
    for name, a_pat, b_pat in PRIMARY:
        A = metric(a_pat, "final"); B = metric(b_pat, "final")
        _, p = wilcoxon(A, B)
        print(f"{name:34s} {fmt(A):>18s} {fmt(B):>18s} {np.median(A-B):+9.3f} {p:8.4f}")

    # 探索的: 主比較 5 対象 × ite vs {legacy, gd, cobyla} を早期停止で、BH q 値つき
    print()
    print("=" * 96)
    print("探索的（早期停止, BH q 値）: 5 対象の ite vs 他手法")
    print("=" * 96)
    tests = []
    for t in ("eq6", "log", "frac", "radius", "expo"):
        I = metric(f"results/ite_{t}_seedSEED.csv")
        for m in ("legacy", "gd", "cobyla"):
            M = metric(f"results/{m}_{t}_seedSEED.csv")
            _, p = wilcoxon(I, M)
            tests.append((f"{t:7s} ite vs {m:7s}", I, M, p))
    qs = bh(np.array([x[3] for x in tests]))
    for (name, I, M, p), q in zip(tests, qs):
        print(f"{name:26s} {fmt(I):>18s} {fmt(M):>18s} {np.median(I-M):+9.3f} "
              f"p={p:7.4f} q={q:7.4f}")

    # 自明ベースライン
    print()
    print("=" * 96)
    print("自明ベースライン（テスト誤差, 訓練10点から構成: 平均 / 中央値 / 線形回帰）")
    print("=" * 96)
    import avqkan as A
    print(f"{'target':8s} {'定数(平均)':>10s} {'定数(中央値)':>11s} {'線形回帰':>9s} "
          f"{'ite(早期停止)':>14s} {'cobyla':>10s}")
    for t in ("eq6", "log", "frac", "radius", "expo"):
        fn, nvar = A.TARGETS[t]
        cm, cmed, lin = [], [], []
        for s in SEEDS:
            cfg = A.Config(seed=s); prob = A.Problem(cfg, t)
            mu, md = prob.f.mean(), np.median(prob.f)
            cm.append(np.sum(np.abs(prob.ff - mu)))
            cmed.append(np.sum(np.abs(prob.ff - md)))
            X, Xf = prob.X[:, :nvar], prob.Xf[:, :nvar]
            Xa = np.hstack([X, np.ones((len(X), 1))])
            Xfa = np.hstack([Xf, np.ones((len(Xf), 1))])
            w, *_ = np.linalg.lstsq(Xa, prob.f, rcond=None)
            lin.append(np.sum(np.abs(np.clip(Xfa @ w, -1, 1) - prob.ff)))
        I = metric(f"results/ite_{t}_seedSEED.csv")
        C = metric(f"results/cobyla_{t}_seedSEED.csv")
        print(f"{t:8s} {np.mean(cm):10.3f} {np.mean(cmed):11.3f} {np.mean(lin):9.3f} "
              f"{I.mean():14.3f} {C.mean():10.3f}")


if __name__ == "__main__":
    main()
