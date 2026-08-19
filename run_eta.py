"""2 階差分推定量のバイアスを差分幅 eta_m で測る（無限ショット＝雑音ゼロ）。
eta_m -> 0 で厳密ヤコビアンの結果(7.590)に収束するはず。収束に必要な eta_m から、
実機で必要なショット数（F の雑音 << eta_m^2 より shots >> eta_m^-4）を逆算できる。"""
import sys
import numpy as np
from avqkan import run
eta = float(sys.argv[1]); s = int(sys.argv[2])
rows, _ = run("ite", "sep4", s, 40, 0.1, 10, "results_eta", verbose=False,
              shots=0, nq=4, readout_mode="single", force_fd_metric=True,
              ite_reg=1e-3, eta_m=eta, tag_suffix=f"_eta{eta}")
te=[r["test_absdist"] for r in rows]
print(f"eta={eta} seed{s} best={min(te):.3f}", flush=True)
