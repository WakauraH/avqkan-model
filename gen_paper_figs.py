"""改訂原稿（パネル査読後）の図を生成する。主指標は訓練コスト早期停止。"""
import csv, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

OUT = "figs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False})
CL = {"legacy": "#999999", "cobyla": "#1f77b4", "gd": "#2ca02c",
      "ite": "#d62728", "ite_spsa": "#9467bd", "fidonly": "#555555"}
LB = {"legacy": "fidelity obj. (v1 impl.)", "cobyla": "COBYLA",
      "gd": "gradient descent", "ite": "natural gradient (QNG)",
      "ite_spsa": "QN-SPSA"}

def es(pat, n=10):
    """早期停止（訓練コスト argmin）時点のテスト誤差"""
    out = []
    for s in range(n):
        rows = list(csv.DictReader(open(pat.replace("SEED", str(s)))))
        te = np.array([float(r["test_absdist"]) for r in rows])
        co = np.array([float(r["cost"]) for r in rows])
        out.append(te[int(np.argmin(co))])
    return np.array(out)

def curve(pat, n=10):
    Y = []
    for s in range(n):
        rows = list(csv.DictReader(open(pat.replace("SEED", str(s)))))
        Y.append([float(r["test_absdist"]) for r in rows])
    return np.array([float(r["tau"]) for r in rows]), np.array(Y)

sd = lambda a: a.std(ddof=1)

# ---- Fig 1: main comparison (early stopping) --------------------------------
TGT = ["eq6", "log", "frac", "radius", "expo"]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
w = 0.2
for i, m in enumerate(["legacy", "cobyla", "gd", "ite"]):
    mu = [es(f"results/{m}_{t}_seedSEED.csv").mean() for t in TGT]
    er = [sd(es(f"results/{m}_{t}_seedSEED.csv")) for t in TGT]
    ax[0].bar(np.arange(5) + (i - 1.5) * w, mu, w, yerr=er, capsize=2,
              color=CL[m], label=LB[m])
ax[0].set_xticks(range(5)); ax[0].set_xticklabels(["Eq.(8)", "log", "frac", "radius", "exp"])
ax[0].set_ylabel("Test error at training-cost early stop")
ax[0].legend(fontsize=8); ax[0].set_title("(a) Five targets, 10 seeds")
for m in ["legacy", "cobyla", "gd", "ite"]:
    t, Y = curve(f"results/{m}_eq6_seedSEED.csv")
    ax[1].plot(t, np.median(Y, 0), color=CL[m], label=LB[m])
    ax[1].fill_between(t, np.percentile(Y, 25, 0), np.percentile(Y, 75, 0),
                       color=CL[m], alpha=0.15)
ax[1].set_xlabel(r"Imaginary time $\tau$"); ax[1].set_ylabel("Test error")
ax[1].set_title("(b) Trajectories on Eq.(8), median & IQR")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_main.png", dpi=200); plt.close(fig)

# ---- Fig 2: budget sweep ----------------------------------------------------
CALLS = {"eq6": {10: 11620, 20: 14060, 50: 30980, 100: 59899, 200: 105493},
         "log": {10: 11780, 20: 14060, 50: 30980, 100: 58466, 200: 95752}}
ICALL = {"eq6": 20460, "log": 20780}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for k, tgt in enumerate(["eq6", "log"]):
    mu, er = [], []
    for mi in (10, 20, 50, 100, 200):
        v = es(f"results_budget/cobyla_{tgt}_seedSEED_bud{mi}.csv")
        mu.append(v.mean()); er.append(sd(v))
    I = es(f"results_budget/ite_{tgt}_seedSEED_bud.csv")
    ax[k].errorbar([CALLS[tgt][m_] for m_ in (10, 20, 50, 100, 200)], mu, yerr=er,
                   marker="o", color=CL["cobyla"], label="COBYLA (maxiter sweep)", capsize=3)
    ax[k].errorbar([ICALL[tgt]], [I.mean()], yerr=[sd(I)], marker="*", ms=16,
                   color=CL["ite"], label="natural gradient", capsize=3)
    ax[k].set_xscale("log"); ax[k].set_xlabel("Circuit evaluations (optimization only)")
    ax[k].set_title(f"({'ab'[k]}) {'Eq.(8)' if tgt == 'eq6' else 'log target'}")
ax[0].set_ylabel("Test error at early stop"); ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_budget.png", dpi=200); plt.close(fig)

# ---- Fig 3: readout (incl. scaled control) + normalized-family scaling ------
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
groups = [
    (r"$Z_0Z_1$", "results_readout/ite_sep4_seedSEED_single_None.csv",
     "results_readout/cobyla_sep4_seedSEED_single_30.csv"),
    (r"$0.5\,Z_0Z_1$", "results_b3/ite_sep4_seedSEED.csv",
     "results_b3_m30/cobyla_sep4_seedSEED.csv"),
    (r"$(Z_0Z_1{+}Z_2Z_3)/2$", "results_readout/ite_sep4_seedSEED_all_pairs_None.csv",
     "results_readout/cobyla_sep4_seedSEED_all_pairs_30.csv"),
]
for gi, (glab, ipat, cpat) in enumerate(groups):
    I, C = es(ipat), es(cpat)
    ax[0].bar(gi * 3, I.mean(), 0.85, yerr=sd(I), capsize=3, color=CL["ite"],
              label="natural gradient" if gi == 0 else None)
    ax[0].bar(gi * 3 + 1, C.mean(), 0.85, yerr=sd(C), capsize=3, color=CL["cobyla"],
              label="COBYLA (equal budget)" if gi == 0 else None)
ax[0].set_xticks([0.5, 3.5, 6.5]); ax[0].set_xticklabels([g[0] for g in groups])
ax[0].set_ylabel("Test error at early stop")
ax[0].set_title("(a) Readout dependence, Eq.(8)\n(middle: range-halved single string)")
ax[0].legend(fontsize=8)

NQ = [4, 6, 8, 10, 12]
import avqkan as A
triv, lin = [], []
for d in NQ:
    tv, lv = [], []
    for s in range(10):
        cfg = A.Config(nq=d, seed=s); prob = A.Problem(cfg, f"sepn{d}")
        tv.append(np.sum(np.abs(prob.ff - np.median(prob.f))))
        X, Xf = prob.X[:, :d], prob.Xf[:, :d]
        Xa = np.hstack([X, np.ones((10, 1))]); Xfa = np.hstack([Xf, np.ones((50, 1))])
        wgt, *_ = np.linalg.lstsq(Xa, prob.f, rcond=None)
        lv.append(np.sum(np.abs(np.clip(Xfa @ wgt, -1, 1) - prob.ff)))
    triv.append(np.mean(tv)); lin.append(np.mean(lv))
for m, pat, lab in (("ite", "results_c/ite_sepnNQ_seedSEED.csv", "natural gradient"),
                    ("cobyla", "results_c_m30/cobyla_sepnNQ_seedSEED.csv",
                     "COBYLA (equal budget)")):
    mu = [es(pat.replace("NQ", str(d))).mean() for d in NQ]
    er = [sd(es(pat.replace("NQ", str(d)))) for d in NQ]
    ax[1].errorbar(NQ, mu, yerr=er, marker="s", capsize=3, color=CL[m], label=lab)
ax[1].plot(NQ, triv, "k:", label="constant predictor (train median)")
ax[1].plot(NQ, lin, "k--", label="linear regression")
ax[1].set_xticks(NQ); ax[1].set_xlabel("Qubits $n_q$ (= input dimension $d$)")
ax[1].set_title("(b) Difficulty-controlled family $\\widetilde{\\rm sep}_d$")
ax[1].legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_readout_scaling.png", dpi=200); plt.close(fig)

# ---- Fig 4: shots + estimator ablation with B1 control ----------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
SH = [1024, 4096, 16384]
for m in ("cobyla", "gd", "ite", "ite_spsa"):
    mu = [es(f"results_shots2/{m}_sep4_seedSEED_sh{s}.csv").mean() for s in SH]
    er = [sd(es(f"results_shots2/{m}_sep4_seedSEED_sh{s}.csv")) for s in SH]
    v0 = es(f"results_shots2/{m}_sep4_seedSEED_sh0.csv")
    ax[0].errorbar(SH, mu, yerr=er, marker="o", capsize=3, color=CL[m], label=LB[m])
    ax[0].axhline(v0.mean(), color=CL[m], ls=":", lw=1)
ax[0].set_xscale("log"); ax[0].set_xlabel("Shots per circuit evaluation")
ax[0].set_ylabel("Test error at early stop")
ax[0].set_title("(a) Finite shots, Eq.(8), $Z_0Z_1$\n(dotted: exact simulation)")
ax[0].legend(fontsize=7)
labels = ["exact\n(all)", "exact metric\n+clip+FD grad", r"FD $\eta{=}0.3$",
          r"FD $\eta{=}0.01$", "QN-SPSA", "no metric\n(GD)"]
vals = [es("results_shots2/ite_sep4_seedSEED_sh0.csv"),
        es("results_b1/ite_ctl_sep4_seedSEED.csv"),
        es("results_control/ite_sep4_seedSEED_fdexact.csv"),
        es("results_eta/ite_sep4_seedSEED_eta0.01.csv"),
        es("results_shots2/ite_spsa_sep4_seedSEED_sh0.csv"),
        es("results_shots2/gd_sep4_seedSEED_sh0.csv")]
cols = [CL["ite"], "#f4a261", "#e8a0a0", "#e8a0a0", CL["ite_spsa"], CL["gd"]]
ax[1].bar(range(6), [v.mean() for v in vals], yerr=[sd(v) for v in vals],
          capsize=3, color=cols)
ax[1].set_xticks(range(6)); ax[1].set_xticklabels(labels, fontsize=7)
ax[1].set_title("(b) Metric estimators, zero shot noise\n(orange: de-confounding control)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_shots.png", dpi=200); plt.close(fig)
print("wrote 4 figures")
