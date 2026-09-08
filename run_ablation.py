"""Adaptive VQKAN（COBYLA）のアブレーション: eq6、10 シード、60 ステップ、20 ステップごとに成長。

基準 (init X1, pool paper, way 2) は results/cobyla_eq6_seed*.csv を再利用する。
ここで回すのは
  way1        : 成長則 = 勾配最大（論文の way 1）
  pool_full   : 二体演算子 9 通りの拡張プール（way 2）
  way1_full   : way 1 + 拡張プール
  init_Y1 / init_Z1 : 初期アンザッツの依存性
  init_2layer : 2 フレームに X1（論文の「2 層」）
使い方: python run_ablation.py [n_workers]
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from avqkan import run

OUT = "results_ablation"
SEEDS = range(10)
CONFIGS = {
    "way1":        dict(select="grad"),
    "pool_full":   dict(pool_kind="full"),
    "way1_full":   dict(select="grad", pool_kind="full"),
    "init_Y1":     dict(init="Y1"),
    "init_Z1":     dict(init="Z1"),
    "init_2layer": dict(init="X1|X1"),
}


def job(args):
    name, seed = args
    rows, meta = run("cobyla", "eq6", seed, 60, 0.1, 20, OUT, maxiter=200,
                     verbose=False, tag_suffix=f"_{name}", **CONFIGS[name])
    return name, seed, rows[-1]["test_absdist"], rows[-1]["ansatz"], meta["elapsed_sec"]


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    jobs = [(n, s) for n in CONFIGS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for name, seed, te, ans, sec in ex.map(job, jobs):
            print(f"{name:12s} s{seed} final test={te:8.3f}  ansatz={ans}  ({sec}s)", flush=True)
