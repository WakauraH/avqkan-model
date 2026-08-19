"""対照実験: 推定量の効果とショット雑音の効果を分離する。

無限ショットの `ite`(7.590) と有限ショットの `ite`(16.98) の差には 2 要因が混在する:
  (i) ショット雑音
  (ii) 推定量の違い — 無限ショット時は状態ベクトルから厳密ヤコビアン、
       有限ショット時は忠実度の 2 階中心差分
そこで **無限ショットのまま 2 階中心差分の推定量を使う** 条件を測り、(ii) 単独の
寄与を切り出す。
"""
import numpy as np
from avqkan import run
SEEDS = range(10)
fin, best, calls = [], [], []
for s in SEEDS:
    rows, _ = run("ite", "sep4", s, 40, 0.1, 10, "results_control", verbose=False,
                  shots=0, nq=4, readout_mode="single", force_fd_metric=True,
                  ite_reg=1e-3, eta_m=0.3, tag_suffix="_fdexact")
    te = [r["test_absdist"] for r in rows]
    fin.append(te[-1]); best.append(min(te)); calls.append(rows[-1]["opt_calls"])
print(f"ite(2階差分推定量, 無限ショット) final={np.mean(fin):8.3f}±{np.std(fin):6.3f} "
      f"best={np.mean(best):8.3f}±{np.std(best):6.3f} calls={int(np.mean(calls))}", flush=True)
