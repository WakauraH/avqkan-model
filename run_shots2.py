"""#12 ITE が優位な条件（sep4 / 読み出し single）でのショットノイズ比較。

§3 は target=log で実施したが、log は無限ショットでも ITE と COBYLA が同等
（p=0.375）な対象で、「ショットノイズが ITE の *優位* を壊すか」を検証できていない。
(a) の結果、ITE が等予算 COBYLA に有意に勝つのは sep4 × 読み出し single だった
ので、その条件で測り直す。ite_spsa のハイパラは tune_spsa.py の最良値。
"""
import sys
import numpy as np
from avqkan import run
SHOTS = int(sys.argv[1]); METHOD = sys.argv[2]
SEEDS = range(10)
kw = {}
mi = 1000
if METHOD == "cobyla":
    mi = 30
if METHOD == "ite_spsa":
    kw = dict(ite_reg=0.03, eta_m=0.3, spsa_resamplings=8)
fin, best, calls = [], [], []
for s in SEEDS:
    rows, _ = run(METHOD, "sep4", s, 40, 0.1, 10, "results_shots2", maxiter=mi,
                  verbose=False, shots=SHOTS, nq=4, readout_mode="single",
                  tag_suffix=f"_sh{SHOTS}", **kw)
    te = [r["test_absdist"] for r in rows]
    fin.append(te[-1]); best.append(min(te)); calls.append(rows[-1]["opt_calls"])
print(f"shots={SHOTS:6d} {METHOD:9s} final={np.mean(fin):8.3f}±{np.std(fin):6.3f} "
      f"best={np.mean(best):8.3f}±{np.std(best):6.3f} calls={int(np.mean(calls)):8d}", flush=True)
