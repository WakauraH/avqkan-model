"""blueqat で訓練した Adaptive VQKAN を qiskit 回路に変換し、Aer（脱分極雑音）と IBM 実機で推論する。

KAN 層の入力 z_i = (<Z_i>+1)/2 は、状態準備が積状態 Ry(arccos x_i + π/2)|0> なので
<Z_i> = cos(arccos x_i + π/2) = -sqrt(1 - x_i^2) と解析的に決まる（測定不要）。
よって推論 1 点あたりの量子回路は「状態準備 → アンザッツ → 読み出し測定」の 1 本。

検証: `python qiskit_bridge.py --test` で、乱数パラメータに対し qiskit 状態ベクトルが blueqat と一致することを確認する。
"""
import argparse, json
import numpy as np
from scipy.interpolate import make_interp_spline
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import PauliEvolutionGate

import avqkan as A
from blueqat import pauli

_EPS = 1e-13


def layer_angles(vlist, x, ansatz_labels, cfg: A.Config):
    """kan_forward と同じ式でフレーム 0 の各演算子の角度 val_j を返す（1 層モデル用）。"""
    d = cfg.dim
    Sz = [-np.sqrt(max(0.0, 1 - float(x[j]) ** 2)) if j < d else 1.0 for j in range(cfg.nq)]
    r0 = np.linspace(-1, 1, cfg.grids); r = np.linspace(-1, 1, cfg.ng)
    vals = []
    for j in range(len(ansatz_labels)):
        coef = vlist[j * cfg.grids:(j + 1) * cfg.grids]
        spl = make_interp_spline(r0, coef, k=2)
        gv = spl(r); pmin, pmax = float(gv.min()), float(gv.max())
        denom = pmax + 1 / (1 + np.exp(-1)) - pmin
        xi = 0.0
        for w in range(d):
            xw = 0.5 * (Sz[w] + 1)
            phi = float(spl(xw))
            ratio = (phi + xw / (1 + np.exp(-xw)) - pmin) / denom + _EPS
            xi += 2 * np.arccos(np.clip(ratio, -1.0, 1.0))
        vals.append((xi % (2 * np.pi) - np.pi) / np.pi)
    return vals


def merged_terms(ansatz_labels, vals):
    """apply_ansatz は HD = Σ op·val を simplify してから項ごとに時間発展させる。
    同一 Pauli 文字列は係数が合算される。blueqat の順序を再現するため blueqat 自身で simplify する。"""
    HD = pauli.X[0] * pauli.X[0] * 0
    for lab, v in zip(ansatz_labels, vals):
        HD = HD + A.parse_op(lab) * v
    terms = []
    for term in HD.to_expr().simplify().terms:
        if not term.ops:
            continue
        coeff = float(np.real(term.coeff))
        s = ["I"] * 64
        for op in term.ops:
            s[op.n] = op.op
        terms.append(("".join(s), coeff))
    return terms


def build_circuit(vlist, x, ansatz_labels, cfg: A.Config, measure=True):
    nq = cfg.nq
    qc = QuantumCircuit(nq, nq if measure else 0)
    for j in range(cfg.dim):
        qc.ry(np.arccos(np.clip(x[j], -1.0, 1.0)) + 0.5 * np.pi, j)
    vals = layer_angles(vlist, x, ansatz_labels, cfg)
    for label, coeff in merged_terms(ansatz_labels, vals):
        # blueqat の get_time_evolution は exp(+i π coeff P)（Y が奇数個の文字列では符号が反転する）。
        # 数値検証済み（selftest）。qiskit の PauliEvolutionGate(P, time) は exp(-i time P)。
        lab = label[:nq][::-1]                       # qiskit のラベルは右端が qubit 0
        sign = 1.0 if lab.count("Y") % 2 == 1 else -1.0
        qc.append(PauliEvolutionGate(SparsePauliOp(lab), time=sign * np.pi * coeff), range(nq))
    if measure:
        qc.measure(range(nq), range(nq))
    return qc


def readout_op(cfg: A.Config):
    nq = cfg.nq
    if cfg.readout_mode == "all_pairs":
        npair = nq // 2
        return sum(SparsePauliOp.from_sparse_list([("ZZ", [2 * j, 2 * j + 1], 1.0 / npair)], nq) for j in range(npair)).simplify()
    return SparsePauliOp.from_sparse_list([("ZZ", list(cfg.readout), 1.0)], nq)


def expval_from_counts(counts, op: SparsePauliOp):
    """Z 対角オブザーバブルの期待値をビット列カウントから計算する（qiskit ビット列は右端が qubit 0）。"""
    tot = sum(counts.values()); val = 0.0
    for bits, n in counts.items():
        b = bits.replace(" ", "")[::-1]
        for lab, c in zip(op.paulis.to_labels(), op.coeffs):
            sgn = 1.0
            for q, ch in enumerate(lab[::-1]):
                if ch == "Z" and b[q] == "1":
                    sgn = -sgn
            val += float(np.real(c)) * sgn * n
    return val / tot


def load_model(path):
    m = json.load(open(path))
    cfgd = m["config"]
    cfg = A.Config(nq=cfgd["nq"], dim=cfgd["dim"], grids=cfgd["grids"], ng=cfgd["ng"], ndT=cfgd["ndT"],
                   readout=tuple(cfgd["readout"]), readout_mode=cfgd["readout_mode"], seed=cfgd["seed"])
    return m, cfg


def selftest():
    rng = np.random.default_rng(0)
    for nq, labels, mode in [(4, ["X1", "Z0Z1", "X2X3", "Y0", "X1", "Y2Y3", "X0Y1"], "single"), (12, ["X1", "X2X3", "X4Y5", "Y0Z11", "Y6Y7", "Z8Z9"], "all_pairs")]:
        cfg = A.Config(nq=nq, dim=nq, readout_mode=mode)
        v = rng.random(8 * len(labels)) * 0.7
        cluster = [[A.parse_op(l) for l in labels], [], []]
        for _ in range(3):
            x = rng.random(nq)
            psi_b = A.kan_forward(v, x, cluster, cfg)
            qc = build_circuit(v, x, labels, cfg, measure=False)
            psi_q = Statevector(qc).data
            ov = abs(np.vdot(psi_b, psi_q))
            hb = A.expval_diag(psi_b, A.diag_signs(list(cfg.readout), nq) if mode == "single"
                               else sum(A.diag_signs([2 * j, 2 * j + 1], nq) for j in range(nq // 2)) / (nq // 2))
            hq = float(np.real(Statevector(qc).expectation_value(readout_op(cfg))))
            print(f"nq={nq}: |<blueqat|qiskit>| = {ov:.8f}  <H> blueqat={hb:+.6f} qiskit={hq:+.6f}")
            assert ov > 1 - 1e-8 and abs(hb - hq) < 1e-8
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true"); a = ap.parse_args()
    if a.test:
        selftest()
