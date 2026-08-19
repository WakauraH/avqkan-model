"""1 条件（method, nq, seed）だけを実行する。シード単位の並列化用。"""
import sys
from avqkan import run
method, nq, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
mi = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
rows, meta = run(method, f"sep{nq}", seed, 40, 0.1, 10, "results_scaling",
                 maxiter=mi, verbose=False, nq=nq, readout_mode="all_pairs",
                 tag_suffix=f"_nq{nq}_{'ite' if method=='ite' else 'm'+str(mi)}")
te = [r["test_absdist"] for r in rows]
print(f"nq={nq} {method} m{mi} seed{seed} final={te[-1]:.3f} best={min(te):.3f} "
      f"calls={rows[-1]['opt_calls']} {meta['elapsed_sec']}s", flush=True)
