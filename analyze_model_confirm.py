"""事前登録済み確証解析（PREREGISTRATION_model.md 参照）。
コミット後は変更しないこと。

主要: H12, H16, H18（厳密）, H12s（256 shots）: AVQKAN-COBYLA vs オラクル古典, Holm m=4
副次（探索的, BH q）: 各古典ベースラインとの個別比較, d=12 の調整済み QNN, d=20 の記述
"""
import csv, os
import numpy as np
from scipy.stats import wilcoxon
from avqkan import Config, Problem
from classical_baseline import fit_kan, fit_mlp
from analyze2 import holm, bh

SEEDS = list(range(30, 50))
PRIMARY = [("H12", 12, "results_model_confirm/cobyla_sepn12_seedSEED_nq12_m30.csv"),
           ("H16", 16, "results_model_confirm/cobyla_sepn16_seedSEED_nq16_m30.csv"),
           ("H18", 18, "results_model_confirm/cobyla_sepn18_seedSEED_nq18_m30.csv"),
           ("H12s", 12, "results_model_confirm_s256/cobyla_sepn12_seedSEED_nq12_m30.csv")]


def es(pat, seeds=SEEDS):
    out = []
    for s in seeds:
        rows = list(csv.DictReader(open(pat.replace("SEED", str(s)))))
        te = np.array([float(r["test_absdist"]) for r in rows])
        co = np.array([float(r["cost"]) for r in rows])
        out.append(te[int(np.argmin(co))])
    return np.array(out)


def es_qnn_tuned(d, seeds=SEEDS, layers=(1, 2, 3)):
    out = []
    for s in seeds:
        best = None
        for L in layers:
            p = f"results_model_confirm_qnn/qnn{L}_lbfgs_sepn{d}_seed{s}_nq{d}.csv"
            rows = list(csv.DictReader(open(p)))
            te = np.array([float(r["test_absdist"]) for r in rows])
            co = np.array([float(r["cost"]) for r in rows])
            if best is None or co.min() < best[1]:
                best = (te[int(np.argmin(co))], co.min())
        out.append(best[0])
    return np.array(out)


def classical(d, seeds=SEEDS):
    arms = {k: [] for k in ("kan", "mlp8x8", "mlp16", "linear")}
    const = []
    for s in seeds:
        prob = Problem(Config(nq=d, seed=s), f"sepn{d}")
        pred, _, _ = fit_kan(prob, 8); arms["kan"].append(np.sum(np.abs(pred - prob.ff)))
        pred, _, _ = fit_mlp(prob, (8, 8), s); arms["mlp8x8"].append(np.sum(np.abs(pred - prob.ff)))
        pred, _, _ = fit_mlp(prob, (16,), s); arms["mlp16"].append(np.sum(np.abs(pred - prob.ff)))
        Xa = np.hstack([prob.X[:, :d], np.ones((len(prob.X), 1))])
        Xfa = np.hstack([prob.Xf[:, :d], np.ones((len(prob.Xf), 1))])
        w, *_ = np.linalg.lstsq(Xa, prob.f, rcond=None)
        arms["linear"].append(np.sum(np.abs(np.clip(Xfa @ w, -1, 1) - prob.ff)))
        const.append(np.sum(np.abs(prob.ff - np.median(prob.f))))
    arms = {k: np.array(v) for k, v in arms.items()}
    oracle = np.min(np.stack(list(arms.values())), axis=0)
    return oracle, np.array(const), arms


def main():
    print("=" * 92)
    print("事前登録済み確証解析（シード 30-49, n=20, Holm m=4）")
    print("=" * 92)
    rows, ps = [], []
    for name, d, pat in PRIMARY:
        if not os.path.exists(pat.replace("SEED", str(SEEDS[-1]))):
            print(f"{name}: 未完（{pat}）"); rows.append(None); ps.append(1.0); continue
        Q = es(pat); O, K, _ = classical(d)
        g1, g2 = Q.mean() < K.mean(), O.mean() < K.mean()
        _, p = wilcoxon(Q, O)
        rows.append((name, d, Q, O, K, g1, g2, p)); ps.append(p)
    adj = holm(np.array(ps))
    print(f"{'H':>5s} {'d':>3s} {'AVQKAN':>16s} {'oracle classical':>16s} {'const':>7s} {'gate':>5s} "
          f"{'med diff':>9s} {'p':>8s} {'Holm p':>8s} verdict")
    for r, ap in zip(rows, adj):
        if r is None:
            continue
        name, d, Q, O, K, g1, g2, p = r
        gate = "OK" if (g1 and g2) else "VOID"
        verdict = ("VOID (gate)" if gate == "VOID" else
                   "PASS" if ap < 0.05 and np.median(Q - O) < 0 else "not significant")
        print(f"{name:>5s} {d:3d} {Q.mean():8.3f}±{Q.std(ddof=1):5.2f} {O.mean():8.3f}±{O.std(ddof=1):5.2f} "
              f"{K.mean():7.2f} {gate:>5s} {np.median(Q - O):+9.3f} {p:8.4f} {ap:8.4f} {verdict}")

    print("\n副次（探索的, BH q）")
    sec = []
    for name, d, pat in PRIMARY:
        if not os.path.exists(pat.replace("SEED", str(SEEDS[-1]))):
            continue
        Q = es(pat); _, _, arms = classical(d)
        for k, v in arms.items():
            sec.append((f"{name} vs {k}", float(np.median(Q - v)), float(wilcoxon(Q, v)[1])))
    qp = "results_model_confirm_qnn/qnn1_lbfgs_sepn12_seed49_nq12.csv"
    if os.path.exists(qp) and os.path.exists(PRIMARY[0][2].replace("SEED", "49")):
        Q = es(PRIMARY[0][2]); N = es_qnn_tuned(12)
        sec.append(("H12 vs tuned QNN", float(np.median(Q - N)), float(wilcoxon(Q, N)[1])))
        print(f"  tuned QNN d=12: {N.mean():.3f}±{N.std(ddof=1):.2f}")
    if sec:
        for (lab, md, p), q in zip(sec, bh([p for *_, p in sec])):
            print(f"  {lab:22s} med diff {md:+8.3f}  p={p:.4f}  q={q:.4f}")
    p20 = "results_model_confirm/cobyla_sepn20_seedSEED_nq20_m30.csv"
    if os.path.exists(p20.replace("SEED", "39")):
        Q = es(p20, range(30, 40)); O, K, _ = classical(20, range(30, 40))
        print(f"\nd=20（記述のみ, n=10）: AVQKAN {Q.mean():.3f}±{Q.std(ddof=1):.2f}  "
              f"oracle {O.mean():.3f}±{O.std(ddof=1):.2f}  const {K.mean():.2f}")


if __name__ == "__main__":
    main()
