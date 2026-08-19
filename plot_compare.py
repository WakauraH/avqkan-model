"""比較結果の作図: tau vs テスト絶対距離の和（5 シードの中央値と範囲）。"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET = sys.argv[1] if len(sys.argv) > 1 else "log"
SEEDS = [0, 1, 2, 3, 4]
METHODS = [("legacy", "原実装 (legacy)", "#888888"),
           ("cobyla", "COBYLA 直接 (前報方式)", "#1f77b4"),
           ("gd",     "勾配降下 (計量なし)", "#2ca02c"),
           ("ite",    "虚時間発展 = 自然勾配 (修正版)", "#d62728")]

plt.rcParams["font.family"] = ["Hiragino Sans", "Arial Unicode MS", "sans-serif"]
plt.rcParams["font.size"] = 11
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for key, label, color in METHODS:
    curves = []
    for s in SEEDS:
        p = f"results/{key}_{TARGET}_seed{s}.csv"
        if not os.path.exists(p):
            continue
        rows = list(csv.DictReader(open(p)))
        curves.append([float(r["test_absdist"]) for r in rows])
    if not curves:
        continue
    n = min(len(c) for c in curves)
    Y = np.array([c[:n] for c in curves])
    tau = np.array([float(r["tau"]) for r in rows][:n])
    for ax, arr, ylab in ((axes[0], Y, "テスト 50 点の絶対距離の和"),
                          (axes[1], Y, None)):
        med = np.median(arr, axis=0)
        ax.plot(tau, med, color=color, label=label, lw=1.8)
        ax.fill_between(tau, arr.min(0), arr.max(0), color=color, alpha=0.12)

axes[0].set_xlabel(r"虚時間 $\tau$"); axes[0].set_ylabel("テスト 50 点の絶対距離の和")
axes[0].set_title(f"線形スケール (target = {TARGET})")
axes[1].set_yscale("log"); axes[1].set_xlabel(r"虚時間 $\tau$")
axes[1].set_title("対数スケール")
axes[0].legend(fontsize=9)
for ax in axes:
    ax.grid(alpha=0.25)
fig.tight_layout()
out = f"compare_{TARGET}.png"
fig.savefig(out, dpi=160)
print("wrote", out)
