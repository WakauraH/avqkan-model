"""雑音・実機の集計: results_noise/infer_*.json（脱分極, 実機雑音モデル）, results_noise/train_*.json（雑音下訓練）,
results_hardware/job_*.json（実機）。出力: results_model/noise_summary.md"""
import glob, json, os
import numpy as np
from scipy.stats import wilcoxon

lines = ["# Depolarising noise, noisy training, and hardware inference\n"]
for size, pat in (("d=4 (eq. 1, seeds 0-9)", "results_noise/infer_cobyla_eq6_*.json"),
                  ("d=12 (sepn12, seeds 30-49)", "results_noise/infer_cobyla_sepn12_*.json")):
    files = sorted(glob.glob(pat))
    if not files:
        continue
    R = [json.load(open(f)) for f in files]
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
    lines.append(f"test error at min training cost: {b.mean():.2f} ± {b.std(ddof=1):.2f} "
                 f"(noise-free training: 11.48 ± 4.23); params {np.mean([t['n_params'] for t in T]):.0f}\n")
hw = sorted(glob.glob("results_hardware/job_*.json"))
if hw:
    lines.append("## Hardware inference\n")
    lines.append("| job | backend | model | exact | hardware |"); lines.append("|---|---|---|---|---|")
    for f in hw:
        J = json.load(open(f))
        for m in J["models"]:
            if "hardware_test" in m:
                lines.append(f"| {J['job_id'][:8]} | {J['backend']} | {os.path.basename(m['model'])} | {m['exact_test']:.2f} | {m['hardware_test']:.2f} |")
    lines.append("")
text = "\n".join(lines); print(text)
os.makedirs("results_model", exist_ok=True); open("results_model/noise_summary.md", "w").write(text + "\n")
