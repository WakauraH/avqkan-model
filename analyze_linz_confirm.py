"""事前登録済み確証解析（PREREGISTRATION_linz.md）。コミット後は変更しない。"""
import csv, os
import numpy as np
from scipy.stats import wilcoxon
from avqkan import Config, Problem
from classical_kernels import fit_predict
from analyze2 import holm, bh

SEEDS = list(range(30, 50)); OUT = "results_linz_confirm"


def es(v):
    out = []
    for s in SEEDS:
        p = f"{OUT}/cobyla_sepn_12_seed{s}_{v}_N100.csv"
        if not os.path.exists(p):
            return None
        rows = list(csv.DictReader(open(p)))
        te = np.array([float(r["test_absdist"]) for r in rows]); co = np.array([float(r["cost"]) for r in rows])
        out.append(float(te[int(np.argmin(co))]))
    return np.array(out)


def classical():
    res = {k: [] for k in ("krr_rbf", "krr_sort", "krr_sym", "krr_add", "const")}
    for s in SEEDS:
        prob = Problem(Config(nq=12, seed=s, n_train=100), "sepn_12")
        X, Xf = prob.X[:, :12], prob.Xf[:, :12]
        for k in ("krr_rbf", "krr_sort", "krr_sym", "krr_add"):
            pred, _ = fit_predict(k, X, prob.f, Xf); res[k].append(float(np.sum(np.abs(pred - prob.ff))))
        res["const"].append(float(np.sum(np.abs(prob.ff - np.median(prob.f)))))
    return {k: np.array(v) for k, v in res.items()}


def m(a): return f"{a.mean():.2f} ± {a.std(ddof=1):.2f}"


def main():
    Q = es("linz_m200_g5"); C = classical()
    if Q is None:
        print("quantum arm incomplete"); return
    print(f"quantum linz_m200_g5: {m(Q)}   const: {m(C['const'])}")
    gate = Q.mean() < C["const"].mean() and all(C[k].mean() < C["const"].mean() for k in ("krr_rbf", "krr_sort"))
    ps = [wilcoxon(Q, C[k])[1] for k in ("krr_rbf", "krr_sort")]
    adj = holm(np.array(ps))
    both = True
    for k, p, ap in zip(("krr_rbf", "krr_sort"), ps, adj):
        md = np.median(Q - C[k]); ok = gate and ap < 0.05 and md < 0; both &= ok
        print(f"  H vs {k:8s}: classical {m(C[k])}  median diff {md:+.2f}  p={p:.4f}  Holm p={ap:.4f}  -> {'PASS' if ok else 'not significant'}")
    print("VERDICT:", "quantum beats both cross-validated kernels" if both else "claim NOT supported")
    print("\nsecondary (exploratory, BH):")
    sec = [(k, np.median(Q - C[k]), wilcoxon(Q, C[k])[1]) for k in ("krr_sym", "krr_add")]
    Q2 = es("linz_m200")
    if Q2 is not None:
        sec.append(("linz_m200 (growth 10) vs g5", np.median(Q2 - Q), wilcoxon(Q2, Q)[1]))
        print(f"  linz_m200: {m(Q2)}")
    for (k, md, p), q in zip(sec, bh([p for *_, p in sec])):
        print(f"  {k:28s} median diff {md:+.2f}  p={p:.4f}  q={q:.4f}")


if __name__ == "__main__":
    main()
