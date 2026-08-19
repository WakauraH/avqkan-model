"""5 対象関数 × 10 シードの結果を集計し、対応のある検定と古典ベースライン比較を行う。

同じシード＝同じ訓練/テスト分割なので、手法間の比較は対応のあるデータになる。
サンプル数が 10 と少なく正規性も仮定できないため、Wilcoxon 符号順位検定を使う。
効果量は rank-biserial correlation（r = (W+ - W-)/(W+ + W-)）で報告する。
"""

import csv
import json
import os

import numpy as np
from scipy.stats import wilcoxon

TARGETS = ["eq6", "log", "frac", "radius", "expo"]
METHODS = ["legacy", "cobyla", "gd", "ite"]
SEEDS = list(range(10))
RES = "results"


def load(method, target, seed):
    p = f"{RES}/{method}_{target}_seed{seed}.csv"
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    te = [float(r["test_absdist"]) for r in rows]
    return dict(final=te[-1], best=min(te))


def collect():
    data = {}
    for t in TARGETS:
        data[t] = {}
        for m in METHODS:
            vals = [load(m, t, s) for s in SEEDS]
            if any(v is None for v in vals):
                continue
            data[t][m] = dict(final=np.array([v["final"] for v in vals]),
                              best=np.array([v["best"] for v in vals]))
    return data


def rank_biserial(x, y):
    d = x - y
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = np.argsort(np.argsort(np.abs(d))) + 1
    wp = r[d > 0].sum()
    wm = r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def main():
    data = collect()

    print("=" * 92)
    print("表 1  テスト 50 点の絶対距離の和（10 シード, mean ± std）")
    print("=" * 92)
    print(f"{'target':8s}" + "".join(f"{m:>20s}" for m in METHODS))
    for metric in ("final", "best"):
        print(f"-- {metric} --")
        for t in TARGETS:
            row = f"{t:8s}"
            for m in METHODS:
                if m not in data[t]:
                    row += f"{'-':>20s}"; continue
                a = data[t][m][metric]
                row += f"{a.mean():11.3f} ±{a.std():7.3f}"
            print(row)

    print()
    print("=" * 92)
    print("表 2  Wilcoxon 符号順位検定（対応あり, 両側, n=10）  ite を基準に比較")
    print("      負の中央値差 = ite の方が誤差が小さい（＝良い）")
    print("=" * 92)
    print(f"{'target':8s} {'metric':6s} {'比較':18s} {'中央値差':>10s} {'p':>10s} "
          f"{'効果量 r':>9s} {'判定':>8s}")
    stats = {}
    for t in TARGETS:
        stats[t] = {}
        for metric in ("final", "best"):
            for other in ("cobyla", "gd", "legacy"):
                if "ite" not in data[t] or other not in data[t]:
                    continue
                a, b = data[t]["ite"][metric], data[t][other][metric]
                diff = a - b
                if np.allclose(diff, 0):
                    continue
                W, p = wilcoxon(a, b)
                r = rank_biserial(a, b)
                verdict = ("ite 有意に良" if p < 0.05 and np.median(diff) < 0 else
                           "他が有意に良" if p < 0.05 else "有意差なし")
                print(f"{t:8s} {metric:6s} {'ite vs ' + other:18s} "
                      f"{np.median(diff):10.3f} {p:10.4f} {r:9.3f} {verdict:>8s}")
                stats[t][f"{metric}_ite_vs_{other}"] = dict(
                    median_diff=float(np.median(diff)), p=float(p), r=float(r))
        print()

    # 古典ベースライン
    cp = f"{RES}/summary_classical.json"
    if os.path.exists(cp):
        cls = json.load(open(cp))
        print("=" * 92)
        print("表 3  古典ベースラインとの比較（テスト絶対距離の和）")
        print("=" * 92)
        print(f"{'target':8s} {'古典KAN':>16s} {'MLP 8x8':>16s} {'MLP 16':>16s} "
              f"{'量子 ite(final)':>18s} {'量子 ite(best)':>16s}")
        for t in TARGETS:
            if t not in cls:
                continue
            c = cls[t]
            q = data[t].get("ite")
            qf = f"{q['final'].mean():8.3f} ±{q['final'].std():5.2f}" if q else "-"
            qb = f"{q['best'].mean():7.3f} ±{q['best'].std():5.2f}" if q else "-"
            print(f"{t:8s} {c['kan']['test_mean']:9.3f} ±{c['kan']['test_std']:5.2f} "
                  f"{c['mlp8x8']['test_mean']:9.3f} ±{c['mlp8x8']['test_std']:5.2f} "
                  f"{c['mlp16']['test_mean']:9.3f} ±{c['mlp16']['test_std']:5.2f} "
                  f"{qf:>18s} {qb:>16s}")
        print()
        print("パラメータ数: 古典KAN 19-37 / MLP8x8 105-121 / MLP16 65-97 / "
              "量子 ite 8-32（成長後）")

    json.dump(stats, open(f"{RES}/wilcoxon.json", "w"), indent=2)
    print(f"\n-> {RES}/wilcoxon.json に保存")


if __name__ == "__main__":
    main()
