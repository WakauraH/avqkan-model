"""モデル論文（arXiv:2503.21336 改訂）用の d=4 主表: Adaptive VQKAN(COBYLA) vs 全ベースライン。

指標: テスト 50 点の絶対距離の和。量子手法・QNN は訓練コスト argmin ステップで 1 回だけ
テスト評価（テスト集合で選択しない）。古典は単一フィット。
比較はすべて同一シード（同一訓練/テスト分割）の対応あり Wilcoxon 符号順位検定、BH q 値。

入力:
  results/cobyla_{target}_seed{s}.csv              Adaptive VQKAN（way 2, init X1, 限定プール）
  results_qnn/qnn{L}_{opt}_{target}_seed{s}.csv    QNN（paper: L=3 COBYLA / tuned: 訓練コスト最小の L）
  results_ablation/cobyla_eq6_seed{s}_{name}.csv   アブレーション（eq6 のみ）
古典（KAN, MLP, 線形回帰, 定数）はここで再計算する（数秒）。
出力: results_model/table_fig2.md, results_model/summary.json
"""
import csv, json, os
import numpy as np
from scipy.stats import wilcoxon

from avqkan import Config, Problem
from classical_baseline import fit_kan, fit_mlp
from analyze2 import bh, holm

TARGETS = ["eq6", "log", "frac", "radius", "expo"]
SEEDS = range(10)
OUT = "results_model"
os.makedirs(OUT, exist_ok=True)


def earlystop(path):
    rows = list(csv.DictReader(open(path)))
    te = np.array([float(r["test_absdist"]) for r in rows])
    co = np.array([float(r["cost"]) for r in rows])
    return float(te[int(np.argmin(co))]), float(co.min()), int(rows[-1]["n_params"]), int(rows[-1].get("opt_calls", rows[-1]["forward_calls"]))


def quantum(tag_pat):
    """tag_pat に SEED を含む。欠損があれば None。"""
    te, P, calls = [], [], []
    for s in SEEDS:
        p = tag_pat.replace("SEED", str(s))
        if not os.path.exists(p):
            return None
        t, _, n, c = earlystop(p)
        te.append(t); P.append(n); calls.append(c)
    return dict(test=np.array(te), params=float(np.mean(P)), calls=float(np.mean(calls)))


def qnn_tuned(target):
    """層数 L∈{1,2,3,4} のうち訓練コスト最小のものをシードごとに選ぶ（テスト非依存）。"""
    te, P, calls, Ls = [], [], [], []
    for s in SEEDS:
        best = None
        for L in (1, 2, 3, 4):
            p = f"results_qnn/qnn{L}_lbfgs_{target}_seed{s}.csv"
            if not os.path.exists(p):
                return None
            t, c, n, k = earlystop(p)
            if best is None or c < best[1]:
                best = (t, c, n, k, L)
        te.append(best[0]); P.append(best[2]); calls.append(best[3]); Ls.append(best[4])
    return dict(test=np.array(te), params=float(np.mean(P)), calls=float(np.mean(calls)),
                layers=Ls)


def classical(target):
    out = {k: [] for k in ("kan", "mlp8x8", "mlp16", "linear", "const")}
    P = {k: [] for k in out}
    for s in SEEDS:
        prob = Problem(Config(seed=s), target)
        d = prob.cfg.dim
        for name, fn in (("kan", lambda: fit_kan(prob, 8)),
                         ("mlp8x8", lambda: fit_mlp(prob, (8, 8), s)),
                         ("mlp16", lambda: fit_mlp(prob, (16,), s))):
            pred, _, n = fn()
            out[name].append(float(np.sum(np.abs(pred - prob.ff)))); P[name].append(n)
        A = np.hstack([prob.X[:, :d], np.ones((len(prob.X), 1))])
        coef = np.linalg.lstsq(A, prob.f, rcond=None)[0]
        pred = np.clip(np.hstack([prob.Xf[:, :d], np.ones((len(prob.Xf), 1))]) @ coef, -1, 1)
        out["linear"].append(float(np.sum(np.abs(pred - prob.ff)))); P["linear"].append(d + 1)
        out["const"].append(float(np.sum(np.abs(np.median(prob.f) - prob.ff)))); P["const"].append(1)
    return {k: dict(test=np.array(v), params=float(np.mean(P[k])), calls=0.0) for k, v in out.items()}


def fmt(a):
    return f"{a.mean():.2f} ± {a.std(ddof=1):.2f}"


def main():
    summary, lines, tests = {}, [], []
    lines.append("# d=4, 5 targets, 10 paired seeds — test error at train-cost early stop (mean ± sd, ddof=1)\n")
    hdr = ["target", "AVQKAN (COBYLA)", "QNN paper (L=3)", "QNN tuned", "cKAN", "MLP 8x8", "MLP 16",
           "linear", "const"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    for t in TARGETS:
        arms = {"avqkan": quantum(f"results/cobyla_{t}_seedSEED.csv"),
                "qnn_paper": quantum(f"results_qnn/qnn3_cobyla_{t}_seedSEED.csv"),
                "qnn_tuned": qnn_tuned(t)}
        arms.update(classical(t))
        summary[t] = {k: (None if v is None else dict(mean=float(v["test"].mean()),
                                                     sd=float(v["test"].std(ddof=1)),
                                                     params=v["params"], calls=v["calls"],
                                                     per_seed=v["test"].tolist(),
                                                     **({"layers": v["layers"]} if "layers" in v else {})))
                      for k, v in arms.items()}
        row = [t] + [fmt(arms[k]["test"]) if arms[k] is not None else "-"
                     for k in ("avqkan", "qnn_paper", "qnn_tuned", "kan", "mlp8x8", "mlp16", "linear", "const")]
        lines.append("| " + " | ".join(row) + " |")
        if arms["avqkan"] is not None:
            for k, v in arms.items():
                if k == "avqkan" or v is None:
                    continue
                diff = arms["avqkan"]["test"] - v["test"]
                if np.allclose(diff, 0):
                    continue
                _, p = wilcoxon(arms["avqkan"]["test"], v["test"])
                tests.append((t, k, float(np.median(diff)), float(p)))
    lines.append("\nParams (mean): " + "; ".join(
        f"{t}: " + ", ".join(f"{k}={v['params']:.0f}" for k, v in summary[t].items() if v)
        for t in TARGETS))
    if tests:
        q = bh([p for *_, p in tests])
        lines.append("\n## Paired Wilcoxon, AVQKAN vs each baseline (median diff < 0 = AVQKAN better; BH q)\n")
        lines.append("| target | baseline | median diff | p | q |")
        lines.append("|---|---|---|---|---|")
        for (t, k, md, p), qq in zip(tests, q):
            flag = " **" if qq < 0.05 else ""
            lines.append(f"| {t} | {k} | {md:+.2f} | {p:.4f} | {qq:.4f}{flag} |")
        summary["tests"] = [dict(target=t, baseline=k, median_diff=md, p=p, q=float(qq))
                            for (t, k, md, p), qq in zip(tests, q)]

    # アブレーション（eq6）
    abl = {"base": quantum("results/cobyla_eq6_seedSEED.csv")}
    for name in ("way1", "pool_full", "way1_full", "init_Y1", "init_Z1", "init_2layer"):
        abl[name] = quantum(f"results_ablation/cobyla_eq6_seedSEED_{name}.csv")
    if all(v is not None for v in abl.values()):
        lines.append("\n## Ablation on eq6 (COBYLA, 10 seeds), vs base = way 2 / paper pool / init X1\n")
        lines.append("| config | test | params | p vs base | q |")
        lines.append("|---|---|---|---|---|")
        ps = []
        for name, v in abl.items():
            if name == "base":
                continue
            d = v["test"] - abl["base"]["test"]
            p = 1.0 if np.allclose(d, 0) else float(wilcoxon(v["test"], abl["base"]["test"])[1])
            ps.append((name, v, p))
        q = bh([p for *_, p in ps])
        lines.append(f"| base | {fmt(abl['base']['test'])} | {abl['base']['params']:.0f} | - | - |")
        for (name, v, p), qq in zip(ps, q):
            lines.append(f"| {name} | {fmt(v['test'])} | {v['params']:.0f} | {p:.4f} | {qq:.4f} |")
        summary["ablation"] = {k: dict(mean=float(v["test"].mean()), sd=float(v["test"].std(ddof=1)),
                                       params=v["params"], per_seed=v["test"].tolist())
                               for k, v in abl.items()}
    else:
        lines.append("\n(ablation results incomplete: " +
                     ", ".join(k for k, v in abl.items() if v is None) + ")")

    text = "\n".join(lines)
    print(text)
    open(f"{OUT}/table_fig2.md", "w").write(text + "\n")
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
