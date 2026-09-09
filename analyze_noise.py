"""雑音・実機の集計: results_noise/infer_*.json（脱分極, 実機雑音モデル）, results_noise/train_*.json（雑音下訓練）,
results_hardware/job_*.json（実機）。出力: results_model/noise_summary.md"""
import glob, json, os
import numpy as np
from scipy.stats import wilcoxon

lines = ["# Depolarising noise, noisy training, and hardware inference\n"]
for size, pat in (("d=4 (eq. 1, seeds 0-9)", "results_noise/infer_cobyla_eq6_*.json"),
                  ("d=12 (sepn12, seeds 30-49)", "results_noise/infer_cobyla_sepn12_*.json")):
    files = sorted(f for f in glob.glob(pat) if not f.endswith("_backend.json"))
    if not files:
        continue
    R = [json.load(open(f)) for f in files]
    for r, f in zip(R, files):                       # 実機雑音モデルの結果が別ファイルなら統合
        fb = f.replace(".json", "_backend.json")
        if os.path.exists(fb):
            r["levels"]["backend"] = json.load(open(fb))["levels"]["backend"]
    levels = list(R[0]["levels"].keys())
    ex = np.array([r["exact_test"] for r in R])
    lines.append(f"## {size}: n={len(R)}, exact (statevector) {ex.mean():.2f} ± {ex.std(ddof=1):.2f}, "
                 f"CZ per circuit {R[0]['levels']['0']['cz']:.1f}\n")
    lines.append("| p2 (2q depolarising) | test error | vs exact p |"); lines.append("|---|---|---|")
    for lv in levels:
        a = np.array([r["levels"][lv]["test"] for r in R if lv in r["levels"]])
        if len(a) != len(ex):
            continue
        p = wilcoxon(a, ex)[1] if not np.allclose(a, ex) else 1.0
        lab = "backend noise model" if lv == "backend" else lv
        lines.append(f"| {lab} | {a.mean():.2f} ± {a.std(ddof=1):.2f} | {p:.3f} |")
    lines.append("")
tr = sorted(glob.glob("results_noise/train_*.json"))
if tr:
    T = [json.load(open(f)) for f in tr]
    b = np.array([t["best"]["test"] for t in T])
    lines.append(f"## Noisy training, d=4 eq. 1: p2={T[0]['p2']}, {T[0]['shots']} shots, n={len(T)}\n")
    S = json.load(open("results_model/summary.json"))["eq6"]["avqkan"]["per_seed"]
    ref = np.array([S[t["seed"]] for t in T])
    lines.append(f"test error at min training cost: {b.mean():.2f} ± {b.std(ddof=1):.2f} "
                 f"(noise-free training, same seeds: {ref.mean():.2f} ± {ref.std(ddof=1):.2f}; paired Wilcoxon p = {wilcoxon(b, ref)[1]:.3f}); "
                 f"params {np.mean([t['n_params'] for t in T]):.0f}\n")
hw = sorted(glob.glob("results_hardware/job_*.json"))
if hw:
    lines.append("## Hardware inference\n")
    lines.append("| job | backend | model | CZ | depth | exact | hardware |"); lines.append("|---|---|---|---|---|---|---|")
    ex_h, hw_h, usage = [], [], 0
    for f in hw:
        J = json.load(open(f))
        usage += int(J.get("usage") or 0)
        for m in J["models"]:
            if "hardware_test" in m:
                ex_h.append(m["exact_test"]); hw_h.append(m["hardware_test"])
                lines.append(f"| {J['job_id'][:8]} | {J['backend']} | {os.path.basename(m['model'])} | {m['cz']:.0f} | {m['depth']:.0f} | {m['exact_test']:.2f} | {m['hardware_test']:.2f} |")
    if ex_h:
        d = np.array(hw_h) - np.array(ex_h)
        lines.append(f"\n{len(ex_h)} models: hardware − exact = {d.mean():+.3f} ± {d.std(ddof=1):.3f} "
                     f"(max |diff| {np.abs(d).max():.3f}), paired Wilcoxon p = {wilcoxon(hw_h, ex_h)[1]:.3f}; QPU usage {usage} s")
    lines.append("")
text = "\n".join(lines); print(text)
os.makedirs("results_model", exist_ok=True); open("results_model/noise_summary.md", "w").write(text + "\n")
