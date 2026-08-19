"""ショットノイズ依存性と量子ビット数スケーリングの図。"""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Hiragino Sans", "Arial Unicode MS", "sans-serif"]
plt.rcParams["font.size"] = 11
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))

# --- (a) ショットノイズ ---
SH = [256, 1024, 4096, 16384]
COL = {"cobyla": "#1f77b4", "gd": "#2ca02c", "ite": "#d62728"}
LAB = {"cobyla": "COBYLA 直接", "gd": "勾配降下（計量なし）", "ite": "虚時間発展（自然勾配）"}
for m in ("cobyla", "gd", "ite"):
    mu, sd = [], []
    for s in SH:
        d = json.load(open(f"results_shots/summary_shots{s}.json"))[m]
        mu.append(d["best_mean"]); sd.append(d["best_std"])
    inf = json.load(open("results_shots/summary_shots0.json"))[m]
    ax[0].errorbar(SH, mu, yerr=sd, marker="o", color=COL[m], label=LAB[m], capsize=3)
    ax[0].axhline(inf["best_mean"], color=COL[m], ls=":", lw=1.2, alpha=0.8)
ax[0].set_xscale("log"); ax[0].set_xlabel("ショット数"); ax[0].set_ylabel("テスト絶対距離の和（best）")
ax[0].set_title("(a) ショットノイズ依存性 (target=log)\n点線 = 無限ショットの値")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)

# --- (b) 量子ビット数 vs 精度 ---
NQ = [4, 6, 8, 10]
for m in ("cobyla", "ite"):
    mu = [json.load(open(f"results_scaling/summary_nq{n}.json"))[m]["best_mean"] for n in NQ]
    sd = [json.load(open(f"results_scaling/summary_nq{n}.json"))[m]["best_std"] for n in NQ]
    ax[1].errorbar(NQ, mu, yerr=sd, marker="s", color=COL[m], label=LAB[m], capsize=3)
ax[1].set_xlabel("量子ビット数 $n_q$ (= 入力変数の数 $d$)")
ax[1].set_ylabel("テスト絶対距離の和（best）")
ax[1].set_title("(b) スケーリング：精度\n※ sep$_d$ 族は $d$ について難易度が非単調")
ax[1].set_xticks(NQ); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)

# --- (c) 量子ビット数 vs 回路評価回数 ---
for m in ("cobyla", "ite"):
    fc = [json.load(open(f"results_scaling/summary_nq{n}.json"))[m]["forward_calls"] for n in NQ]
    ax[2].plot(NQ, fc, marker="^", color=COL[m], label=LAB[m])
ax[2].set_xlabel("量子ビット数 $n_q$"); ax[2].set_ylabel("回路評価の総回数")
ax[2].set_title("(c) スケーリング：測定コスト")
ax[2].set_xticks(NQ); ax[2].set_yscale("log"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.25)

fig.tight_layout(); fig.savefig("compare_shots_scaling.png", dpi=160)
print("wrote compare_shots_scaling.png")
