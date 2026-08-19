"""論文改訂版の図を results*/ から生成して ../arXiv-2506.22801v1 2/img2/ に置く。"""
import csv, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False})
CL = {"legacy": "#999999", "cobyla": "#1f77b4", "gd": "#2ca02c",
      "ite": "#d62728", "ite_spsa": "#9467bd"}
LB = {"legacy": "fidelity-objective ITE (v1)", "cobyla": "COBYLA",
      "gd": "gradient descent", "ite": "natural-gradient ITE", "ite_spsa": "QN-SPSA ITE"}

def best(path_pat, n=10):
    return np.array([min(float(r["test_absdist"]) for r in
                         csv.DictReader(open(path_pat.replace("SEED", str(s)))))
                     for s in range(n)])

def curve(path_pat, n=10):
    Y = []
    for s in range(n):
        rows = list(csv.DictReader(open(path_pat.replace("SEED", str(s)))))
        Y.append([float(r["test_absdist"]) for r in rows])
    return np.array([float(r["tau"]) for r in rows]), np.array(Y)

# ---- Fig 1: main comparison -------------------------------------------------
TGT = ["eq6", "log", "frac", "radius", "expo"]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
w = 0.2
for i, m in enumerate(["legacy", "cobyla", "gd", "ite"]):
    mu = [best(f"results/{m}_{t}_seedSEED.csv").mean() for t in TGT]
    sd = [best(f"results/{m}_{t}_seedSEED.csv").std() for t in TGT]
    ax[0].bar(np.arange(5) + (i - 1.5) * w, mu, w, yerr=sd, capsize=2,
              color=CL[m], label=LB[m])
ax[0].set_xticks(range(5)); ax[0].set_xticklabels(["Eq.(7)", "log", "frac", "radius", "exp"])
ax[0].set_ylabel("Best sum of absolute distances"); ax[0].legend(fontsize=8)
ax[0].set_title("(a) Five targets, 10 seeds")
tau, _ = curve("results/ite_eq6_seedSEED.csv")
for m in ["legacy", "cobyla", "gd", "ite"]:
    t, Y = curve(f"results/{m}_eq6_seedSEED.csv")
    ax[1].plot(t, np.median(Y, 0), color=CL[m], label=LB[m])
    ax[1].fill_between(t, np.percentile(Y, 25, 0), np.percentile(Y, 75, 0),
                       color=CL[m], alpha=0.15)
ax[1].set_xlabel(r"Imaginary time $\tau$"); ax[1].set_ylabel("Sum of absolute distances")
ax[1].set_title("(b) Trajectories on Eq.(7), median & IQR")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_main.png", dpi=200); plt.close(fig)

# ---- Fig 2: equal budget ----------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for k, tgt in enumerate(["eq6", "log"]):
    xs, mu, sd = [], [], []
    for mi in (10, 20, 50, 100, 200):
        v = best(f"results_budget/cobyla_{tgt}_seedSEED_bud{mi}.csv")
        import json
        calls = np.mean([json.load(open(f"results_budget/cobyla_{tgt}_seed{s}_bud{mi}.json"))
                         is not None for s in range(1)])  # placeholder
        xs.append(mi); mu.append(v.mean()); sd.append(v.std())
    I = best(f"results_budget/ite_{tgt}_seedSEED_bud.csv")
    CALLS = {"eq6": {10: 11620, 20: 14060, 50: 30980, 100: 59899, 200: 105493},
             "log": {10: 11780, 20: 14060, 50: 30980, 100: 58466, 200: 95752}}
    ICALL = {"eq6": 20460, "log": 20780}
    ax[k].errorbar([CALLS[tgt][m_] for m_ in (10, 20, 50, 100, 200)], mu, yerr=sd,
                   marker="o", color=CL["cobyla"], label="COBYLA (maxiter sweep)", capsize=3)
    ax[k].errorbar([ICALL[tgt]], [I.mean()], yerr=[I.std()], marker="*", ms=16,
                   color=CL["ite"], label="natural-gradient ITE", capsize=3)
    ax[k].set_xscale("log"); ax[k].set_xlabel("Circuit evaluations (optimization only)")
    ax[k].set_title(f"({'ab'[k]}) {'Eq.(7)' if tgt=='eq6' else 'log'}")
ax[0].set_ylabel("Best sum of absolute distances"); ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_budget.png", dpi=200); plt.close(fig)

# ---- Fig 3: readout & scaling ----------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
groups = [("single", r"$Z_0Z_1$"), ("all_pairs", r"$(Z_0Z_1{+}Z_2Z_3)/2$")]
bars = [("ite", "None"), ("cobyla", "30"), ("cobyla", "200")]
blab = ["ITE", "COBYLA (equal budget)", r"COBYLA ($4\times$ budget)"]
bcol = [CL["ite"], CL["cobyla"], "#7fb3d5"]
for gi, (rm, glab) in enumerate(groups):
    for bi, (m, mi) in enumerate(bars):
        v = best(f"results_readout/{m}_sep4_seedSEED_{rm}_{mi}.csv")
        ax[0].bar(gi * 4 + bi, v.mean(), 0.85, yerr=v.std(), capsize=3,
                  color=bcol[bi], label=blab[bi] if gi == 0 else None)
ax[0].set_xticks([1, 5]); ax[0].set_xticklabels([g[1] for g in groups])
ax[0].set_ylabel("Best sum of absolute distances")
ax[0].set_title("(a) Readout dependence, Eq.(7)"); ax[0].legend(fontsize=8)
NQ = [4, 6, 8, 10, 12, 14]
for m, tag in (("ite", "b"), ("cobyla", "m30")):
    mu, sd = [], []
    for nq in NQ:
        suf = f"_nq{nq}_{'ite' if (nq in (12,14) and m=='ite') else tag}" \
            if not (nq in (12, 14) and m == "cobyla") else f"_nq{nq}_m30"
        if nq in (12, 14) and m == "ite":
            suf = f"_nq{nq}_ite"
        elif nq in (12, 14):
            suf = f"_nq{nq}_m30"
        else:
            suf = f"_nq{nq}_b" if m == "ite" else f"_nq{nq}_m30"
        v = best(f"results_scaling/{m}_sep{nq}_seedSEED{suf}.csv")
        mu.append(v.mean()); sd.append(v.std())
    ax[1].errorbar(NQ, mu, yerr=sd, marker="s", capsize=3, color=CL[m],
                   label="natural-gradient ITE" if m == "ite" else "COBYLA (equal budget)")
for nq, y, txt in ((10, 20.5, "*"), (12, 24.5, "**")):
    ax[1].text(nq, y, txt, ha="center", fontsize=14)
ax[1].set_xticks(NQ); ax[1].set_xlabel("Number of qubits $n_q$ (= input dimension)")
ax[1].set_title("(b) Scaling on sep$_d$, equal budget"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_readout_scaling.png", dpi=200); plt.close(fig)

# ---- Fig 4: shots & estimators ----------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
SH = [1024, 4096, 16384]
for m in ("cobyla", "gd", "ite", "ite_spsa"):
    mu = [best(f"results_shots2/{m}_sep4_seedSEED_sh{s}.csv").mean() for s in SH]
    sd = [best(f"results_shots2/{m}_sep4_seedSEED_sh{s}.csv").std() for s in SH]
    v0 = best(f"results_shots2/{m}_sep4_seedSEED_sh0.csv")
    ax[0].errorbar(SH, mu, yerr=sd, marker="o", capsize=3, color=CL[m], label=LB[m])
    ax[0].axhline(v0.mean(), color=CL[m], ls=":", lw=1)
ax[0].set_xscale("log"); ax[0].set_xlabel("Shots per circuit evaluation")
ax[0].set_ylabel("Best sum of absolute distances")
ax[0].set_title("(a) Finite shots, Eq.(7), $Z_0Z_1$ readout\n(dotted: exact-simulation value)")
ax[0].legend(fontsize=7)
labels = ["exact\nmetric", r"FD $\eta{=}0.3$", r"FD $\eta{=}0.1$", r"FD $\eta{=}0.03$",
          r"FD $\eta{=}0.01$", "QN-SPSA", "no metric\n(grad.desc.)"]
vals = [best("results_shots2/ite_sep4_seedSEED_sh0.csv"),
        best("results_control/ite_sep4_seedSEED_fdexact.csv"),
        best("results_eta/ite_sep4_seedSEED_eta0.1.csv"),
        best("results_eta/ite_sep4_seedSEED_eta0.03.csv"),
        best("results_eta/ite_sep4_seedSEED_eta0.01.csv"),
        best("results_shots2/ite_spsa_sep4_seedSEED_sh0.csv"),
        best("results_shots2/gd_sep4_seedSEED_sh0.csv")]
cols = [CL["ite"], "#e8a0a0", "#e8a0a0", "#e8a0a0", "#e8a0a0", CL["ite_spsa"], CL["gd"]]
ax[1].bar(range(7), [v.mean() for v in vals], yerr=[v.std() for v in vals],
          capsize=3, color=cols)
ax[1].set_xticks(range(7)); ax[1].set_xticklabels(labels, fontsize=7)
ax[1].set_title("(b) Metric estimators, exact simulation (no shot noise)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_shots.png", dpi=200); plt.close(fig)
print("wrote 4 figures")
