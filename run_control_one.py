import sys
import numpy as np
from avqkan import run
s = int(sys.argv[1])
rows, _ = run("ite", "sep4", s, 40, 0.1, 10, "results_control", verbose=False,
              shots=0, nq=4, readout_mode="single", force_fd_metric=True,
              ite_reg=1e-3, eta_m=0.3, tag_suffix="_fdexact")
te = [r["test_absdist"] for r in rows]
print(f"seed{s} final={te[-1]:.3f} best={min(te):.3f} calls={rows[-1]['opt_calls']}", flush=True)
