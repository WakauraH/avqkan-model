"""変種探索の集計: (target, N) ごとに各変種の early-stop テスト誤差を、krr_rbf / krr_sort（LOO-CV）と対応あり比較。"""
import csv, os
import numpy as np
from scipy.stats import wilcoxon
from avqkan import Config, Problem
from classical_kernels import fit_predict
from run_variants import TARGETS, V, OUT
NS = (10, 40, 100)
V = dict(V, linz_m200={}, linz_m200_g5={})

SEEDS = range(10)


def classical(t, N):
    d, _ = TARGETS[t]
    res = {"krr_rbf": [], "krr_sort": [], "const": []}
    for s in SEEDS:
        prob = Problem(Config(nq=d, seed=s, n_train=N), t)
        dd = prob.cfg.dim
        for k in ("krr_rbf", "krr_sort"):
            pred, _ = fit_predict(k, prob.X[:, :dd], prob.f, prob.Xf[:, :dd])
            res[k].append(float(np.sum(np.abs(pred - prob.ff))))
        res["const"].append(float(np.sum(np.abs(prob.ff - np.median(prob.f)))))
    return {k: np.array(v) for k, v in res.items()}


def quantum(t, v, N):
    out, P = [], []
    for s in SEEDS:
        p = f"{OUT}/cobyla_{t}_seed{s}_{v}_N{N}.csv"
        if not os.path.exists(p):
            return None, None
        rows = list(csv.DictReader(open(p)))
        te = np.array([float(r["test_absdist"]) for r in rows]); co = np.array([float(r["cost"]) for r in rows])
        out.append(float(te[int(np.argmin(co))])); P.append(int(rows[-1]["n_params"]))
    return np.array(out), float(np.mean(P))


def m(a): return f"{a.mean():.1f}±{a.std(ddof=1):.1f}"


lines = ["# Model variants (seeds 0-9): early-stop test error vs LOO-CV kernels\n"]
for t in TARGETS:
    for N in NS:
        if not any(os.path.exists(f"{OUT}/cobyla_{t}_seed0_{v}_N{N}.csv") for v in V):
            continue
        C = classical(t, N)
        lines.append(f"\n## {t}, N={N}: krr_rbf {m(C['krr_rbf'])}, krr_sort {m(C['krr_sort'])}, const {m(C['const'])}\n")
        lines.append("| variant | params | test | vs rbf (med diff, p) | vs sort (med diff, p) |"); lines.append("|---|---|---|---|---|")
        for v in V:
            Q, P = quantum(t, v, N)
            if Q is None:
                continue
            r = {k: (np.median(Q - C[k]), wilcoxon(Q, C[k])[1]) for k in ("krr_rbf", "krr_sort")}
            flag = " **WIN**" if all(r[k][0] < 0 and r[k][1] < 0.05 for k in r) else ""
            lines.append(f"| {v} | {P:.0f} | {m(Q)} | {r['krr_rbf'][0]:+.1f}, {r['krr_rbf'][1]:.3f} | {r['krr_sort'][0]:+.1f}, {r['krr_sort'][1]:.3f}{flag} |")
text = "\n".join(lines); print(text)
os.makedirs("results_model", exist_ok=True); open("results_model/variants_summary.md", "w").write(text + "\n")
