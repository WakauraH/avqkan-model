"""参照実装の検証テスト。

`python test_avqkan.py` で実行する（pytest でも可）。

検証項目:
  T1  量子ビット順序の規約が blueqat の statevector と一致すること
  T2  状態ベクトル内積で計算した |<psi1|psi2>|^2 が、原実装が使っていた
      swap test 回路の出力（無限ショット極限）と一致すること
  T3  Fubini-Study 計量が半正定値であること
  T4  解析勾配が回帰コストの数値微分と一致すること
  T5  spline_eval='grid'（原実装）が入力に対して階段関数になり、
      'direct'（修正）が滑らかであること
"""

import numpy as np
from blueqat import Circuit, pauli

import avqkan as A


def _ok(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + extra) if extra else ''}")
    assert cond, name


def t1_qubit_order():
    print("T1 量子ビット順序")
    for j in range(3):
        c = Circuit(3)
        c.x[j]
        psi = c.run(returns="statevector")
        for k in range(3):
            z = A.expval_diag(psi, A.diag_signs([k], 3))
            want = -1.0 if k == j else 1.0
            _ok(f"X on q{j}: <Z{k}> = {z:+.1f} (want {want:+.1f})", abs(z - want) < 1e-9)
    c = Circuit(4).x[0].x[2]
    psi = c.run(returns="statevector")
    z01 = A.expval_diag(psi, A.diag_signs([0, 1], 4))
    z23 = A.expval_diag(psi, A.diag_signs([2, 3], 4))
    _ok("X0 X2: <Z0Z1> = -1", abs(z01 + 1) < 1e-9)
    _ok("X0 X2: <Z2Z3> = -1", abs(z23 + 1) < 1e-9)


def _swap_test_circuit(psi1_prep, psi2_prep, nq):
    """原実装 `swaptestp()` と同じ構成の制御 SWAP テスト回路を組み、
    無限ショット極限の出力 2*p0-1 を返す。"""
    cs = Circuit(nq)
    psi1_prep(cs)
    cs = cs.x[2 * nq].x[2 * nq]           # レジスタ拡張（原実装と同じ手口）
    for i in range(nq):                    # 系 A を上位レジスタへ SWAP
        nm = i + nq
        cs = cs.cx[i, nm].cx[nm, i].cx[i, nm]
    psi2_prep(cs)
    cs = cs.h[2 * nq]
    for i in range(nq):
        cs = cs.cx[i + nq, i]
        cs = cs.toffoli[2 * nq, i, i + nq]
        cs = cs.cx[i + nq, i]
    cs = cs.h[2 * nq]
    v = cs.run(returns="statevector")
    nl = len(v) // 2
    p0 = float(np.real(np.vdot(v[:nl], v[:nl])))
    return 2.0 * p0 - 1.0


def t2_swap_test_equivalence():
    print("T2 swap test 回路 vs 状態ベクトル内積")
    nq = 2
    rng = np.random.default_rng(3)
    for trial in range(3):
        a = rng.random(nq) * np.pi
        b = rng.random(nq) * np.pi

        def prep1(c):
            for j in range(nq):
                c.ry(a[j])[j]

        def prep2(c):
            for j in range(nq):
                c.ry(b[j])[j]

        st = _swap_test_circuit(prep1, prep2, nq)
        c1 = Circuit(nq); prep1(c1); p1 = c1.run(returns="statevector")
        c2 = Circuit(nq); prep2(c2); p2 = c2.run(returns="statevector")
        exact = abs(np.vdot(p1, p2)) ** 2
        _ok(f"trial {trial}: swap={st:.6f} exact={exact:.6f}", abs(st - exact) < 1e-9)


def t3_metric_psd():
    print("T3 Fubini-Study 計量の半正定値性")
    cfg = A.Config(seed=0, grids=6, ng=6, ndT=3)
    prob = A.Problem(cfg, "log")
    cluster = [[pauli.X[1]], [], []]
    v = np.random.default_rng(0).random(A.n_params(cluster, cfg)) * 0.5
    P = len(v)
    Amat = np.zeros((P, P))
    for x in prob.X[:3]:
        psi, J = A.jacobian_and_state(v, x, cluster, prob)
        ov = J.conj().T @ psi
        Amat += np.real(J.conj().T @ J) - np.real(np.outer(ov, ov.conj()))
    ev = np.linalg.eigvalsh((Amat + Amat.T) / 2)
    _ok(f"最小固有値 = {ev.min():.3e} >= -1e-6", ev.min() > -1e-6)
    _ok("対称性", np.allclose(Amat, Amat.T, atol=1e-8))


def t4_gradient():
    print("T4 解析勾配 vs 数値微分")
    cfg = A.Config(seed=0, grids=6, ng=6, ndT=3)
    prob = A.Problem(cfg, "log")
    cluster = [[pauli.X[1]], [], []]
    v = np.random.default_rng(1).random(A.n_params(cluster, cfg)) * 0.5
    P = len(v)
    g = np.zeros(P)
    for m, x in enumerate(prob.X):
        psi, J = A.jacobian_and_state(v, x, cluster, prob)
        h = A.expval_diag(psi, prob.Hsign)
        dh = 2 * np.real(J.conj().T @ (psi * prob.Hsign))
        g += prob.w[m] * 2 * (h - prob.f[m]) * dh
    eps = 1e-5
    gnum = np.zeros(P)
    for i in range(P):
        vp = v.copy(); vp[i] += eps
        vm = v.copy(); vm[i] -= eps
        gnum[i] = (prob.cost(vp, cluster)[0] - prob.cost(vm, cluster)[0]) / (2 * eps)
    rel = np.linalg.norm(g - gnum) / max(np.linalg.norm(gnum), 1e-12)
    _ok(f"相対誤差 = {rel:.2e} < 1e-3", rel < 1e-3)


def t5_spline_quantization():
    print("T5 スプライン評価: grid（原実装）は階段、direct（修正）は滑らか")
    r0 = np.linspace(-1, 1, 8)
    from scipy.interpolate import make_interp_spline
    coef = np.linspace(-0.5, 0.5, 8)
    spl = make_interp_spline(r0, coef, k=2)
    r = np.linspace(-1, 1, 8)
    gv = spl(r)
    xs = np.linspace(0.0, 1.0, 200)
    grid_vals = np.array([gv[np.argmin(np.abs(r - x))] for x in xs])
    direct_vals = np.array([float(spl(x)) for x in xs])
    n_levels = len(np.unique(np.round(grid_vals, 12)))
    _ok(f"grid モードの出力段数 = {n_levels}（入力 200 点に対し離散的）", n_levels <= 8)
    _ok(f"direct モードの出力段数 = {len(np.unique(np.round(direct_vals, 12)))}",
        len(np.unique(np.round(direct_vals, 12))) > 150)


if __name__ == "__main__":
    for fn in (t1_qubit_order, t2_swap_test_equivalence, t3_metric_psd,
               t4_gradient, t5_spline_quantization):
        fn()
    print("\nすべて成功")
