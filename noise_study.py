"""脱分極雑音の評価（qiskit Aer）。

推論モード（既定）: 保存済みモデル（models/d4, models/d12 の best パラメータ）を IBM 基底 {cz, rz, sx, x}・
線形結合に transpile し、1 量子ビットゲートに p1 = p2/10、2 量子ビットゲートに p2 の脱分極雑音を掛けて
50 テスト点の誤差を評価する。p2 ∈ {0, 1e-3, 3e-3, 1e-2, 3e-2}、shots 固定。
--backend-noise <name> を付けると実機のキャリブレーション雑音モデル（AerSimulator.from_backend）でも評価する。

訓練モード（--train）: d=4 eq6 を p2 と shots を固定した雑音下で COBYLA 訓練する（10 シード）。
出力: results_noise/*.json
"""
import argparse, glob, json, os, sys
import numpy as np
from qiskit import transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

import avqkan as A
from qiskit_bridge import build_circuit, readout_op, expval_from_counts, load_model

BASIS = ["cz", "rz", "sx", "x"]
OUT = "results_noise"; os.makedirs(OUT, exist_ok=True)


def depol_model(p2):
    nm = NoiseModel(basis_gates=BASIS)
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ["sx", "x"])
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cz"])
    return nm


class NoisyRunner:
    def __init__(self, nq, noise_model, shots, seed):
        # 雑音ありでは density_matrix が statevector（軌跡サンプリング）より 12 量子ビットで 60 倍速い
        self.sim = AerSimulator(noise_model=noise_model, seed_simulator=seed,
                                method="density_matrix" if nq <= 12 else "automatic")
        self.cmap = CouplingMap.from_line(nq)
        self.shots = shots
        self.n2q = []

    def expval(self, qc, op):
        tqc = transpile(qc, basis_gates=BASIS, coupling_map=self.cmap, optimization_level=1, seed_transpiler=0)
        self.n2q.append(tqc.count_ops().get("cz", 0))
        counts = self.sim.run(tqc, shots=self.shots).result().get_counts()
        return expval_from_counts(counts, op)


def infer(model_path, p2_list, shots, backend_nm=None):
    m, cfg = load_model(model_path)
    prob = A.Problem(A.Config(nq=cfg.nq, seed=cfg.seed, readout_mode=cfg.readout_mode, n_train=cfg.n_train),
                     m["target"])
    v, labels = np.array(m["best"]["vlist"]), m["best"]["ansatz"]
    op = readout_op(cfg)
    out = dict(model=model_path, target=m["target"], seed=cfg.seed, exact_test=m["best"]["test_absdist"], levels={})
    for p2 in p2_list:
        R = NoisyRunner(cfg.nq, depol_model(p2), shots, seed=cfg.seed + 5)
        h = np.array([R.expval(build_circuit(v, x, labels, cfg), op) for x in prob.Xf])
        out["levels"][str(p2)] = dict(test=float(np.sum(np.abs(h - prob.ff))), cz=float(np.mean(R.n2q)))
    if backend_nm is not None:
        R = NoisyRunner(cfg.nq, backend_nm, shots, seed=cfg.seed + 5)
        h = np.array([R.expval(build_circuit(v, x, labels, cfg), op) for x in prob.Xf])
        out["levels"]["backend"] = dict(test=float(np.sum(np.abs(h - prob.ff))), cz=float(np.mean(R.n2q)))
    return out


class NoisyProblem(A.Problem):
    """訓練中の出力を Aer（雑音 + 有限ショット）で評価する Problem。指標の報告は厳密値のまま。"""
    def __init__(self, cfg, target, p2, shots):
        super().__init__(cfg, target)
        self.R = NoisyRunner(cfg.nq, depol_model(p2), shots, seed=cfg.seed + 5)
        self.op = readout_op(cfg)
        self.labels_ref = None

    def outputs(self, vlist, cluster, X, ng=None, rng=None):
        if rng is None:                       # 指標記録用: 厳密値
            return super().outputs(vlist, cluster, X, ng, None)
        labels = self.labels_ref(cluster)
        return np.array([self.R.expval(build_circuit(np.asarray(vlist), x, labels, self.cfg), self.op) for x in X])


def train_noisy(seed, p2, shots, steps=60, grow=20, maxiter=200):
    """run() と同じ手順で、目的関数だけ雑音下の推定に置き換える（成長則の評価も雑音下）。"""
    from scipy.optimize import minimize as smin
    cfg = A.Config(nq=4, seed=seed, shots=shots)
    prob = NoisyProblem(cfg, "eq6", p2, shots)
    pool, labels = A.build_pool(4)
    prob.labels_ref = lambda cluster: [labels[pool.index(op)] for cl in cluster for op in cl]
    cluster = [[A.parse_op("X1")], [], []]
    vlist = np.zeros(A.n_params(cluster, cfg))
    mrng = np.random.default_rng(seed + 99991)
    best = dict(cost=np.inf); rows = []
    for step in range(steps):
        res = smin(lambda p: prob.cost(p, cluster, None, mrng)[0], vlist, method="COBYLA",
                   options={"maxiter": maxiter})
        vlist = np.array(res.x)
        C, h = prob.cost(vlist, cluster)                 # 厳密
        te = prob.test_absdist(vlist, cluster)
        rows.append(dict(step=step, cost=C, test=te))
        if C < best["cost"]:
            best = dict(cost=C, step=step, test=te)
        if grow and (step + 1) % grow == 0 and step + 1 < steps:
            vlist, cluster, added, _ = A.grow_ansatz(vlist, cluster, prob, pool, labels, 0, mrng)
    return dict(seed=seed, p2=p2, shots=shots, best=best, final=rows[-1],
                ansatz=prob.labels_ref(cluster), n_params=len(vlist))


def _one_wrapper(args):
    path, shots, backend_name = args
    bnm = None
    if backend_name:
        from qiskit_ibm_runtime import QiskitRuntimeService
        bnm = NoiseModel.from_backend(QiskitRuntimeService().backend(backend_name))
    r = infer(path, [0, 1e-3, 3e-3, 1e-2, 3e-2], shots, bnm)
    tag = os.path.basename(path).replace(".json", "")
    suffix = "_backend" if bnm is not None else ""
    json.dump(r, open(f"{OUT}/infer_{tag}{suffix}.json", "w"), indent=1)
    return tag, r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--p2", type=float, default=1e-2)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default="models/d4/*.json")
    ap.add_argument("--backend-noise", default=None)
    a = ap.parse_args()
    if a.train:
        r = train_noisy(a.seed, a.p2, a.shots)
        json.dump(r, open(f"{OUT}/train_eq6_p{a.p2}_s{a.shots}_seed{a.seed}.json", "w"), indent=1)
        print(f"train seed{a.seed} p2={a.p2} shots={a.shots}: best test={r['best']['test']:.3f} ansatz={r['ansatz']}")
    else:
        bnm = None
        if a.backend_noise:
            from qiskit_ibm_runtime import QiskitRuntimeService
            bnm = NoiseModel.from_backend(QiskitRuntimeService().backend(a.backend_noise))
        from concurrent.futures import ProcessPoolExecutor
        paths = sorted(glob.glob(a.models))
        def one(path):
            r = infer(path, [0, 1e-3, 3e-3, 1e-2, 3e-2], a.shots, bnm)
            tag = os.path.basename(path).replace(".json", "")
            suffix = "_backend" if bnm is not None else ""
            json.dump(r, open(f"{OUT}/infer_{tag}{suffix}.json", "w"), indent=1)
            return tag, r
        with ProcessPoolExecutor(max_workers=min(10, len(paths))) as ex:
            for tag, r in ex.map(_one_wrapper, [(p_, a.shots, a.backend_noise) for p_ in paths]):
                print(tag, "exact", round(r["exact_test"], 2), {k: round(v["test"], 2) for k, v in r["levels"].items()},
                      "cz/circuit", r["levels"]["0"]["cz"], flush=True)
