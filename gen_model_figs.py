"""モデル論文の図 2–5 とリソース表を生成する（figs_model/）。
色は固定割当（系列→色）: AVQKAN 青, QNN 橙, 古典 oracle 青緑, KRR 紫, 定数 灰。
古典ベースラインは計算に数分かかるので results_model/figdata.json にキャッシュする。"""
import csv, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

from avqkan import Config, Problem
from classical_baseline import fit_kan, fit_mlp
from classical_regularized import run as reg_run

OUT = "figs_model"; os.makedirs(OUT, exist_ok=True)
CACHE = "results_model/figdata.json"
C = dict(avqkan="#2a78d6", qnn="#eb6834", cls="#1baf7a", krr="#4a3aa7", const="#7a7a76", shots="#eda100")
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 300})


def es(pat, seeds):
    out = []
    for s in seeds:
        rows = list(csv.DictReader(open(pat.replace("SEED", str(s)))))
        te = np.array([float(r["test_absdist"]) for r in rows]); co = np.array([float(r["cost"]) for r in rows])
        out.append(float(te[int(np.argmin(co))]))
    return out


def es_qnn_tuned(pat_L, seeds, layers):
    out = []
    for s in seeds:
        best = None
        for L in layers:
            rows = list(csv.DictReader(open(pat_L.replace("SEED", str(s)).replace("LAYER", str(L)))))
            te = np.array([float(r["test_absdist"]) for r in rows]); co = np.array([float(r["cost"]) for r in rows])
            if best is None or co.min() < best[1]:
                best = (float(te[int(np.argmin(co))]), co.min())
        out.append(best[0])
    return out


def classical_arms(target, d, seeds, n_train=10):
    arms = {k: [] for k in ("kan", "mlp8x8", "mlp16", "linear", "const")}
    for s in seeds:
        prob = Problem(Config(nq=d, seed=s, n_train=n_train), target)
        dd = prob.cfg.dim
        pred, _, _ = fit_kan(prob, 8); arms["kan"].append(float(np.sum(np.abs(pred - prob.ff))))
        pred, _, _ = fit_mlp(prob, (8, 8), s); arms["mlp8x8"].append(float(np.sum(np.abs(pred - prob.ff))))
        pred, _, _ = fit_mlp(prob, (16,), s); arms["mlp16"].append(float(np.sum(np.abs(pred - prob.ff))))
        Xa = np.hstack([prob.X[:, :dd], np.ones((len(prob.X), 1))]); Xfa = np.hstack([prob.Xf[:, :dd], np.ones((50, 1))])
        w, *_ = np.linalg.lstsq(Xa, prob.f, rcond=None)
        arms["linear"].append(float(np.sum(np.abs(np.clip(Xfa @ w, -1, 1) - prob.ff))))
        arms["const"].append(float(np.sum(np.abs(prob.ff - np.median(prob.f)))))
    arms["oracle"] = np.min(np.array([arms[k] for k in ("kan", "mlp8x8", "mlp16", "linear")]), axis=0).tolist()
    return arms


def build_data():
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    D = {}
    # Fig 3: scaling, seeds 30-49
    D["scaling"] = {}
    for d in (12, 16, 18):
        seeds = range(30, 50)
        e = dict(avqkan=es(f"results_model_confirm/cobyla_sepn{d}_seedSEED_nq{d}_m30.csv", seeds))
        e.update(classical_arms(f"sepn{d}", d, seeds))
        reg, _ = reg_run(f"sepn{d}", d, 10, seeds)
        e["krr"] = reg["krr_rbf"].tolist(); e["gp"] = reg["gp_rbf"].tolist(); e["ridge"] = reg["ridge"].tolist()
        if d == 12:
            e["avqkan_s256"] = es("results_model_confirm_s256/cobyla_sepn12_seedSEED_nq12_m30.csv", seeds)
            e["qnn_tuned"] = es_qnn_tuned("results_model_confirm_qnn/qnnLAYER_lbfgs_sepn12_seedSEED_nq12.csv", seeds, (1, 2, 3))
        D["scaling"][str(d)] = e
    # Fig 4: shots at d=4 eq6, seeds 0-9
    D["shots4"] = {"inf": es("results/cobyla_eq6_seedSEED.csv", range(10))}
    for sh in (4096, 1024, 256):
        D["shots4"][str(sh)] = es(f"results_shots_eq6/cobyla_eq6_seedSEED_s{sh}.csv", range(10))
    # Fig 5: N sweep at d=12 seeds 0-9 (+ KRR per N)
    ns = json.load(open("results_model/nsweep.json"))
    D["nsweep"] = {}
    for N, v in ns.items():
        reg, _ = reg_run("sepn12", 12, int(N), range(10))
        v["krr"] = reg["krr_rbf"].tolist()
        D["nsweep"][N] = v
    # ansatz composition in confirm runs
    comp = Counter(); npar = []
    for d in (12, 16, 18):
        for s in range(30, 50):
            rows = list(csv.DictReader(open(f"results_model_confirm/cobyla_sepn{d}_seed{s}_nq{d}_m30.csv")))
            ops = rows[-1]["ansatz"].split(";")[1:]
            for o in ops:
                comp[re.sub(r"\d+", "", o)] += 1
            npar.append(int(rows[-1]["n_params"]))
    D["ansatz"] = dict(composition=dict(comp), n_params=npar)
    json.dump(D, open(CACHE, "w"), indent=1)
    return D


def msd(a):
    a = np.asarray(a); return a.mean(), a.std(ddof=1)


def dot(ax, x, arr, color, label=None, marker="o", ms=4, dx=0.0):
    m, s = msd(arr)
    ax.errorbar(x + dx, m, yerr=s, fmt=marker, color=color, ms=ms, capsize=2, elinewidth=0.8,
                markeredgecolor="white", markeredgewidth=0.5, label=label, zorder=3)


def fig2():
    S = json.load(open("results_model/summary.json"))
    targets = ["eq6", "log", "frac", "radius", "expo"]
    names = {"eq6": "eq. (1)", "log": "log", "frac": "fractional", "radius": "radius", "expo": "exponential"}
    arms = [("avqkan", "Adaptive VQKAN", C["avqkan"]), ("qnn_paper", "QNN (3 layers)", C["qnn"]),
            ("qnn_tuned", "QNN (tuned)", C["qnn"]), ("kan", "classical KAN", C["cls"]),
            ("mlp8x8", "MLP 8×8", C["cls"]), ("mlp16", "MLP 16", C["cls"]),
            ("linear", "linear", C["cls"]), ("const", "constant", C["const"])]
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 2.2), sharey=False)
    for ax, t in zip(axes, targets):
        for i, (k, lab, col) in enumerate(arms):
            v = S[t][k]["per_seed"]
            m, s = msd(v)
            ax.errorbar(m, i, xerr=s, fmt="o", color=col, ms=3.5, capsize=1.5, elinewidth=0.7,
                        markeredgecolor="white", markeredgewidth=0.4, zorder=3)
        ax.set_yticks(range(len(arms))); ax.set_yticklabels([a[1] for a in arms] if t == "eq6" else [])
        ax.invert_yaxis(); ax.set_title(names[t], fontsize=8); ax.set_xlim(left=0)
        ax.grid(axis="x", color="#e6e6e3", lw=0.5, zorder=0)
        ax.set_xlabel("test error")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(f"{OUT}/fig2_d4.pdf"); fig.savefig(f"{OUT}/fig2_d4.png"); plt.close(fig)


def fig3(D):
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ds = [12, 16, 18]
    series = [("avqkan", "Adaptive VQKAN", C["avqkan"], "o", -0.25),
              ("krr", "kernel ridge (LOO-CV)", C["krr"], "D", -0.08),
              ("oracle", "best unregularised classical", C["cls"], "s", 0.08),
              ("const", "constant", C["const"], "_", 0.25)]
    for k, lab, col, mk, dx in series:
        for d in ds:
            dot(ax, d, D["scaling"][str(d)][k], col, lab if d == 12 else None, mk, dx=dx * 2)
    dot(ax, 12, D["scaling"]["12"]["qnn_tuned"], C["qnn"], "QNN (tuned)", "^", dx=-0.9)
    ax.set_xticks(ds); ax.set_xlabel("input dimension d = number of qubits"); ax.set_ylabel("test error (10 training points)")
    ax.set_ylim(0, 24); ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=6.5, columnspacing=0.8)
    ax.text(18, 0.6, "20 seeds per point", ha="right", fontsize=6, color=C["const"])
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_scaling.pdf"); fig.savefig(f"{OUT}/fig3_scaling.png"); plt.close(fig)


def fig4(D):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    xs = [256, 1024, 4096, 20000]
    for x, key in zip(xs, ["256", "1024", "4096", "inf"]):
        dot(ax, x, D["shots4"][key], C["avqkan"])
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(["256", "1024", "4096", "exact"])
    ax.set_xlabel("shots per circuit evaluation"); ax.set_ylabel("test error"); ax.set_ylim(0, 22)
    ax.set_title("d = 4, eq. (1), 10 seeds", fontsize=8)
    ax = axes[1]
    e = D["scaling"]["12"]
    dot(ax, 0, e["avqkan_s256"], C["avqkan"], "Adaptive VQKAN, 256 shots")
    dot(ax, 1, e["avqkan"], C["avqkan"], "Adaptive VQKAN, exact", marker="o")
    dot(ax, 2, e["krr"], C["krr"], "kernel ridge", marker="D")
    dot(ax, 3, e["oracle"], C["cls"], "best unregularised classical", marker="s")
    dot(ax, 4, e["const"], C["const"], "constant", marker="_")
    ax.set_xticks(range(5)); ax.set_xticklabels(["VQKAN\n256 shots", "VQKAN\nexact", "kernel\nridge", "best\nclassical", "constant"])
    ax.set_ylim(0, 22); ax.set_title("d = 12, 20 seeds", fontsize=8); ax.set_ylabel("test error")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_shots.pdf"); fig.savefig(f"{OUT}/fig4_shots.png"); plt.close(fig)


def fig5(D):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    Ns = sorted(int(n) for n in D["nsweep"])
    for k, lab, col, mk in [("avqkan", "Adaptive VQKAN", C["avqkan"], "o"), ("krr", "kernel ridge (LOO-CV)", C["krr"], "D"),
                            ("oracle", "best unregularised classical", C["cls"], "s"), ("const", "constant", C["const"], "_")]:
        ms_ = [msd(D["nsweep"][str(N)][k]) for N in Ns]
        ax.errorbar(Ns, [m for m, _ in ms_], yerr=[s for _, s in ms_], fmt=mk + "-", color=col, ms=4, lw=1.2,
                    capsize=2, elinewidth=0.8, markeredgecolor="white", markeredgewidth=0.5, label=lab)
    ax.set_xscale("log"); ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("number of training points N"); ax.set_ylabel("test error (d = 12)"); ax.set_ylim(0, 24)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=2, columnspacing=0.8)
    ax = axes[1]
    comp = D["ansatz"]["composition"]
    keys = sorted(comp, key=lambda k: -comp[k])
    ax.barh(range(len(keys)), [comp[k] for k in keys], color=C["avqkan"], height=0.7)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys); ax.invert_yaxis()
    ax.set_xlabel("operators adopted (d = 12–18, 60 runs)")
    ax.set_title(f"parameters at end: {np.mean(D['ansatz']['n_params']):.0f} (spline coefficients)", fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_nsweep.pdf"); fig.savefig(f"{OUT}/fig5_nsweep.png"); plt.close(fig)


def resource_table(D):
    """表 1: パラメータ数と最適化回路評価回数（d = 4 eq6 と d = 12）。"""
    rows = []
    for lab, pat, seeds in [("Adaptive VQKAN, d=4 (eq. 1)", "results/cobyla_eq6_seedSEED.csv", range(10)),
                            ("QNN 3 layers, d=4", "results_qnn/qnn3_cobyla_eq6_seedSEED.csv", range(10)),
                            ("Adaptive VQKAN, d=12", "results_model_confirm/cobyla_sepn12_seedSEED_nq12_m30.csv", range(30, 50)),
                            ("Adaptive VQKAN, d=16", "results_model_confirm/cobyla_sepn16_seedSEED_nq16_m30.csv", range(30, 50))]:
        P, calls, ops = [], [], []
        for s in seeds:
            r = list(csv.DictReader(open(pat.replace("SEED", str(s)))))[-1]
            P.append(int(r["n_params"])); calls.append(int(r.get("opt_calls", r["forward_calls"])))
            ops.append(len(r["ansatz"].split(";")) if not r["ansatz"].startswith("qnn") else float("nan"))
        rows.append((lab, np.mean(P), np.mean(ops), np.mean(calls)))
    with open(f"{OUT}/table1_resources.md", "w") as f:
        f.write("| model | trainable parameters | parametrised operators | optimiser circuit evaluations |\n|---|---|---|---|\n")
        for lab, P, o, c in rows:
            f.write(f"| {lab} | {P:.0f} | {'–' if np.isnan(o) else f'{o:.1f}'} | {c:,.0f} |\n")
    print(open(f"{OUT}/table1_resources.md").read())


if __name__ == "__main__":
    D = build_data()
    fig2(); fig3(D); fig4(D); fig5(D); resource_table(D)
    for d in ("12", "16", "18"):
        e = D["scaling"][d]
        print(d, {k: f"{msd(e[k])[0]:.2f}±{msd(e[k])[1]:.2f}" for k in ("avqkan", "krr", "oracle", "const")})
    print("figs ->", OUT)
