"""探索の集計: family × d × readout ごとに AVQKAN(early stop) と古典カーネル 4 種（LOO-CV）、定数を比較。
候補 = AVQKAN が krr_rbf と krr_sort の両方に p<0.05 で勝つ条件。出力: results_model/explore_summary.md"""
import csv, glob, os, json
import numpy as np
from scipy.stats import wilcoxon
from avqkan import Config, Problem, _FAMILIES
from classical_kernels import fit_predict, KERNELS

OUT = "results_explore"; SEEDS = range(10)
cls_cache = {}


def classical(fam, d):
    key = (fam, d)
    if key in cls_cache:
        return cls_cache[key]
    res = {k: [] for k in list(KERNELS) + ["const"]}
    for s in SEEDS:
        prob = Problem(Config(nq=d, seed=s), f"{fam}_{d}")
        X, Xf = prob.X[:, :d], prob.Xf[:, :d]
        for k in KERNELS:
            pred, _ = fit_predict(k, X, prob.f, Xf)
            res[k].append(float(np.sum(np.abs(pred - prob.ff))))
        res["const"].append(float(np.sum(np.abs(prob.ff - np.median(prob.f)))))
    cls_cache[key] = {k: np.array(v) for k, v in res.items()}
    return cls_cache[key]


def quantum(fam, d, ro):
    out = []
    for s in SEEDS:
        p = f"{OUT}/cobyla_{fam}_{d}_seed{s}_{ro}.csv"
        if not os.path.exists(p):
            return None
        rows = list(csv.DictReader(open(p)))
        te = np.array([float(r["test_absdist"]) for r in rows]); co = np.array([float(r["cost"]) for r in rows])
        out.append(float(te[int(np.argmin(co))]))
    return np.array(out)


def m(a): return f"{a.mean():.1f}±{a.std(ddof=1):.1f}"


lines = ["# Exploratory search (seeds 0-9, N=10): AVQKAN vs LOO-CV kernel baselines\n",
         "| family | d | readout | AVQKAN | krr_rbf | p | krr_sort | p | krr_sym | p | krr_add | const | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
cands = []
for fam in _FAMILIES:
    for d in (8, 12):
        C = classical(fam, d)
        for ro in ("all_pairs", "sum_z"):
            Q = quantum(fam, d, ro)
            if Q is None:
                lines.append(f"| {fam} | {d} | {ro} | (pending) | {m(C['krr_rbf'])} | | {m(C['krr_sort'])} | | {m(C['krr_sym'])} | | {m(C['krr_add'])} | {m(C['const'])} | |"); continue
            ps = {k: wilcoxon(Q, C[k])[1] for k in ("krr_rbf", "krr_sort", "krr_sym")}
            wins = {k: (np.median(Q - C[k]) < 0 and ps[k] < 0.05) for k in ps}
            verdict = "CANDIDATE" if wins["krr_rbf"] and wins["krr_sort"] else ("beats rbf only" if wins["krr_rbf"] else "")
            if wins["krr_rbf"] and wins["krr_sort"] and wins["krr_sym"]:
                verdict = "STRONG (beats sym too)"
            if verdict:
                cands.append((fam, d, ro, verdict))
            lines.append(f"| {fam} | {d} | {ro} | {m(Q)} | {m(C['krr_rbf'])} | {ps['krr_rbf']:.3f} | {m(C['krr_sort'])} | {ps['krr_sort']:.3f} | "
                         f"{m(C['krr_sym'])} | {ps['krr_sym']:.3f} | {m(C['krr_add'])} | {m(C['const'])} | {verdict} |")
lines.append("\nCandidates: " + (", ".join(f"{f}_{d}/{r} [{v}]" for f, d, r, v in cands) or "none"))
text = "\n".join(lines); print(text)
os.makedirs("results_model", exist_ok=True); open("results_model/explore_summary.md", "w").write(text + "\n")
