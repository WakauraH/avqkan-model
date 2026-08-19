"""
Adaptive VQKAN — 再現可能な参照実装
==================================

`ITE_diagnosis.md` で指摘した欠陥を修正した実装と、比較のための原実装（legacy）の
両方を含む。すべての乱数はシードで固定される。

前進計算（状態準備 → KAN 層 → アンザッツ → 読み出し）は原ノートブックの
`kancore0()` / `ansatz()` に忠実。違いは最適化の目的関数と更新則にある。

最適化手法（`--method`）:
  legacy  : 原実装の再現。COBYLA が「e^{-δH}|ψ⟩ への忠実度のずれ」のみを最小化する。
            回帰対象 f_aim は目的関数に入らない（診断書 A-0）。
  ite     : 虚時間発展（McLachlan 変分原理 / 自然勾配流）を回帰コストに適用する修正版。
                A(θ) θ̇ = -∇C(θ),  C(θ) = Σ_m w_m (⟨H⟩_m - f_m)^2
                A_ij = Re[⟨∂_iψ|∂_jψ⟩ - ⟨∂_iψ|ψ⟩⟨ψ|∂_jψ⟩]   (Fubini-Study 計量)
            C = ⟨H⟩ のとき通常の VarQITE に帰着する。
  gd      : 同じ勾配を計量なしで使う（計量の寄与を切り分けるアブレーション）。
  cobyla  : 回帰コスト C を COBYLA で直接最小化（前報 arXiv:2503.21336 の方式）。

使い方の例:
    python avqkan.py --method ite --target log --seed 0 --steps 60
    python avqkan.py --method legacy --target log --seed 0 --steps 60
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import minimize as smin

from blueqat import Circuit, pauli

# --------------------------------------------------------------------------
# 対象関数
#
# 原ノートブックでは同一セル内に `losslastregin` が 3〜5 個定義され、最後の定義
# だけが有効になっていた（診断書 C-1）。ここでは名前で明示的に選択する。
# 定義は原コードの式をそのまま移したもの。
# --------------------------------------------------------------------------

_EPS = 1e-13


def _t_eq6(x):
    """論文 eq.(6): exp(sin(x0^2+x1^2) + sin(x2^2+x3^2)) — 4 変数"""
    return (np.exp(np.sin(x[0] ** 2 + x[1] ** 2) + np.sin(x[2] ** 2 + x[3] ** 2)) - 1) \
        / (np.exp(2.0) - 1) * 2 - 1


def _t_log(x):
    """対数関数 log(x0/x1) — 2 変数"""
    lo = np.log(_EPS / (1 + _EPS))
    hi = np.log((1 + _EPS) / _EPS)
    return (np.log(np.abs(x[0] + _EPS) / np.abs(x[1] + _EPS)) - lo) / (hi - lo) * 2 - 1


def _t_frac(x):
    """分数関数 1/(1+x0*x1) — 2 変数"""
    return ((1.0 / (1 + x[0] * x[1])) - 0.5) / (1 - 0.5) * 2 - 1


def _t_radius(x):
    """球の半径 sqrt(x0^2+x1^2+x2^2) — 3 変数"""
    return np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) / np.sqrt(3) * 2 - 1


def _t_expo(x):
    """指数関数 exp((x0-x1)^2 / 2 x2) — 3 変数"""
    return (np.exp(1 / 2 / (x[2] + 0.1) * (x[0] - x[1]) ** 2) - 1) \
        / (np.exp(1 / 2 / 0.1) - 1) * 2 - 1


def make_sepsin(d):
    """eq.(6) を d 変数に一般化した族:
        f_d(x) = [exp(Σ_{i<d/2} sin(x_{2i}^2 + x_{2i+1}^2)) - 1] / [exp(d/2) - 1] * 2 - 1
    d = 4 で eq.(6) に一致する。量子ビット数とともに問題の難しさが増すので、
    スケーリング実験ではこの族を使う。"""
    def f(x):
        ssum = sum(np.sin(x[2 * i] ** 2 + x[2 * i + 1] ** 2) for i in range(d // 2))
        return (np.exp(ssum) - 1) / (np.exp(d // 2) - 1) * 2 - 1
    return f, d


# 標準化 sep 族の定数（一様乱数 2e5 点のモンテカルロ、シード 20260819 で確定）
_SEPN_STATS = {4: (1.123063, 0.415513), 6: (1.684815, 0.509674),
               8: (2.244644, 0.587657), 10: (2.806651, 0.657629),
               12: (3.367388, 0.719816), 14: (3.928791, 0.777606),
               16: (4.487510, 0.834387)}


def make_sepsin_norm(d, k=2.5):
    """sep_d の標準化版: f = clip((Σ sin − μ_d)/(k σ_d), −1, 1)。
    μ_d, σ_d は上の定数。目標分布の位置・スケールが d に依らないため、
    定数予測器の誤差が d にほぼ非依存になり、サイズ間比較が意味を持つ。"""
    mu, sd = _SEPN_STATS[d]
    def f(x):
        z = sum(np.sin(x[2 * i] ** 2 + x[2 * i + 1] ** 2) for i in range(d // 2))
        return float(np.clip((z - mu) / (k * sd), -1.0, 1.0))
    return f, d


TARGETS = {
    "eq6":    (_t_eq6, 4),
    "sep4":   make_sepsin(4),
    "sep6":   make_sepsin(6),
    "sep8":   make_sepsin(8),
    "sep10":  make_sepsin(10),
    "sep12":  make_sepsin(12),
    "sep14":  make_sepsin(14),
    "sepn4":  make_sepsin_norm(4),
    "sepn6":  make_sepsin_norm(6),
    "sepn8":  make_sepsin_norm(8),
    "sepn10": make_sepsin_norm(10),
    "sepn12": make_sepsin_norm(12),
    "sepn14": make_sepsin_norm(14),
    "sepn16": make_sepsin_norm(16),
    "log":    (_t_log, 2),
    "frac":   (_t_frac, 2),
    "radius": (_t_radius, 3),
    "expo":   (_t_expo, 3),
}


# --------------------------------------------------------------------------
# 状態ベクトルのユーティリティ
# --------------------------------------------------------------------------

def diag_signs(ops, nq):
    """対角 Pauli 積（Z のみ）の符号ベクトルを返す。ops は量子ビット番号のリスト。"""
    idx = np.arange(2 ** nq)
    sign = np.ones(2 ** nq)
    for q in ops:
        sign *= 1.0 - 2.0 * ((idx >> q) & 1)   # qubit 0 = LSB (blueqat の規約)
    return sign


def expval_diag(psi, sign):
    return float(np.real(np.sum(np.abs(psi) ** 2 * sign)))


def sample_probs(psi, shots, rng):
    """Z 基底で shots 回測定した経験確率を返す。可換な対角オブザーバブルは
    すべてこの同一サンプルから評価する（実機で 1 回の測定設定に対応）。"""
    p = np.abs(psi) ** 2
    p = p / p.sum()
    return rng.multinomial(shots, p) / shots


def expval_shot(psi, sign, shots, rng):
    if not shots:
        return expval_diag(psi, sign)
    return float(np.sum(sample_probs(psi, shots, rng) * sign))


def fidelity_shot(psi_a, psi_b, shots, rng):
    """|<a|b>|^2 の推定。実機では compute-uncompute で全ビット 0 の確率として
    測る（成功確率 F の二項分布）。"""
    F = float(np.abs(np.vdot(psi_a, psi_b)) ** 2)
    if not shots:
        return F
    return float(rng.binomial(shots, min(max(F, 0.0), 1.0)) / shots)


# --------------------------------------------------------------------------
# 設定
# --------------------------------------------------------------------------

@dataclass
class Config:
    nq: int = 4
    dim: int = 2                 # 入力変数の数（対象関数から決まる）
    grids: int = 8               # スプライン制御点の数
    ng: int = 8                  # スプライン評価格子の点数
    ndT: int = 3                 # KAN のフレーム（層）数
    n_train: int = 10
    n_test: int = 50
    readout: tuple = (0, 1)      # "single" のときの読み出し Z_a Z_b
    readout_mode: str = "single"  # "single" | "all_pairs"（(2/nq)ΣZ_{2j}Z_{2j+1}）
    spline_eval: str = "direct"  # "direct"（修正）| "grid"（原実装の最近傍量子化）
    encoding: str = "code"       # "code"（実装どおり arccos(x), dim量子ビット）|
                                 # "paper"（本文式 arccos(2x-1)+π/2, 全量子ビット）
    ng_growth: bool = False      # True で原実装の ng = grids+4 変更を再現
    weighting: bool = True       # 原実装の重み (B - fn)/B
    shots: int = 0               # 0 = 無限ショット（状態ベクトル厳密）
    seed: int = 0


# --------------------------------------------------------------------------
# 前進計算（原実装 kancore0 / ansatz に忠実）
# --------------------------------------------------------------------------

def prep_circuit(x, cfg: Config) -> Circuit:
    c = Circuit(cfg.nq)
    if cfg.encoding == "paper":
        for j in range(cfg.nq):
            xv = x[j % len(x)]
            c.ry(np.arccos(np.clip(2 * xv - 1, -1.0, 1.0)) + 0.5 * np.pi)[j]
    else:
        for j in range(cfg.dim):
            c.ry(np.arccos(np.clip(x[j], -1.0, 1.0)) + 0.5 * np.pi)[j]
    return c


def apply_ansatz(c: Circuit, val, clus, ndT_arg=1):
    """原 `ansatz()`: HD = Σ_j op_j * val[j] を作り、各項に時間発展 π/ndT_arg を掛ける。"""
    if not clus:
        return c
    HD = pauli.X[0] * pauli.X[0] * 0
    for j, op in enumerate(clus):
        HD = HD + op * val[j]
    Htot = HD.to_expr().simplify()
    for term in Htot.terms:
        term.get_time_evolution()(c, 1 / ndT_arg * np.pi)
    return c


# 回路（状態ベクトル）評価の総回数。手法間の計算コストを公平に比較するために数える。
FORWARD_CALLS = {"n": 0}


def kan_forward(vlist, x, cluster, cfg: Config, ng=None, rng=None):
    """状態準備 → 各フレームで KAN スプライン → アンザッツ。最終状態ベクトルを返す。

    cfg.shots > 0 のとき、各層の <Z_j> は有限ショットで推定される（層ごとに独立な
    ショット予算。実機では層ごとに別の回路実行が必要なため）。"""
    FORWARD_CALLS["n"] += 1
    ng = cfg.ng if ng is None else ng
    r0 = np.linspace(-1, 1, cfg.grids)
    r = np.linspace(-1, 1, ng)
    c = prep_circuit(x, cfg)
    cursor = 0
    for t in range(cfg.ndT):
        clus = cluster[t]
        if not clus:
            continue
        psi = c.run(returns="statevector")
        if cfg.shots and rng is not None:
            emp = sample_probs(psi, cfg.shots, rng)
            Sz = [float(np.sum(emp * diag_signs([j], cfg.nq))) for j in range(cfg.nq)]
        else:
            Sz = [expval_diag(psi, diag_signs([j], cfg.nq)) for j in range(cfg.nq)]
        val = []
        for j in range(len(clus)):
            coef = vlist[cursor + j * cfg.grids: cursor + (j + 1) * cfg.grids]
            spl = make_interp_spline(r0, coef, k=2)
            grid_vals = spl(r)
            pmin, pmax = float(np.min(grid_vals)), float(np.max(grid_vals))
            denom = pmax + 1 / (1 + np.exp(-1)) - pmin
            xi = 0.0
            for w in range(cfg.dim):
                xw = 0.5 * (Sz[w] + 1)
                if cfg.spline_eval == "grid":
                    # 原実装: 評価格子上の最近傍を取る → 入力に対して ng 段の階段関数になる
                    phi = float(grid_vals[np.argmin(np.abs(r - xw))])
                else:
                    phi = float(spl(xw))
                ratio = (phi + xw / (1 + np.exp(-xw)) - pmin) / denom + _EPS
                xi += 2 * np.arccos(np.clip(ratio, -1.0, 1.0))
            val.append((xi % (2 * np.pi) - np.pi) / np.pi)
        c = apply_ansatz(c, val, clus, ndT_arg=1)
        cursor += len(clus) * cfg.grids
    return c.run(returns="statevector")


def n_params(cluster, cfg: Config) -> int:
    return sum(len(cl) for cl in cluster) * cfg.grids


# --------------------------------------------------------------------------
# 回帰コストと評価指標
# --------------------------------------------------------------------------

class Problem:
    def __init__(self, cfg: Config, target_name: str):
        self.cfg = cfg
        self.fn, nvar = TARGETS[target_name]
        cfg.dim = nvar
        self.target_name = target_name
        if cfg.readout_mode == "single_scaled":
            # 単項のまま出力レンジだけ半分にする（読み出し依存性の機構切り分け用）
            self.Hsign = 0.5 * diag_signs(list(cfg.readout), cfg.nq)
        elif cfg.readout_mode == "all_pairs":
            npair = cfg.nq // 2
            self.Hsign = sum(diag_signs([2 * j, 2 * j + 1], cfg.nq)
                             for j in range(npair)) / npair
        else:
            self.Hsign = diag_signs(list(cfg.readout), cfg.nq)

        rng = np.random.default_rng(cfg.seed)
        self.X = rng.random((cfg.n_train, max(4, nvar)))
        self.Xf = rng.random((cfg.n_test, max(4, nvar)))
        self.f = np.array([self.fn(x) for x in self.X])
        self.ff = np.array([self.fn(x) for x in self.Xf])

        # 原実装の重み: fn を最小値だけシフトし w = (B - fn)/B
        if cfg.weighting:
            B = 2 + 1 / cfg.n_train
            shifted = self.f - np.min(self.f)
            self.w = (B - shifted) / B
        else:
            self.w = np.ones(cfg.n_train)

    def outputs(self, vlist, cluster, X, ng=None, rng=None):
        """rng を渡すと cfg.shots に従った有限ショット推定になる。
        評価指標の報告には rng=None（厳密値）を使うこと。"""
        out = []
        for x in X:
            psi = kan_forward(vlist, x, cluster, self.cfg, ng, rng)
            if rng is not None and self.cfg.shots:
                out.append(expval_shot(psi, self.Hsign, self.cfg.shots, rng))
            else:
                out.append(expval_diag(psi, self.Hsign))
        return np.array(out)

    def cost(self, vlist, cluster, ng=None, rng=None):
        h = self.outputs(vlist, cluster, self.X, ng, rng)
        return float(np.sum(self.w * (h - self.f) ** 2)), h

    def train_absdist(self, h):
        return float(np.sum(np.abs(h - self.f)))

    def test_absdist(self, vlist, cluster, ng=None):
        h = self.outputs(vlist, cluster, self.Xf, ng)
        return float(np.sum(np.abs(h - self.ff)))


# --------------------------------------------------------------------------
# 原実装の目的関数（legacy）— 診断書 A-0, A-1, B-2, B-3 をそのまま再現する
# --------------------------------------------------------------------------

def legacy_objective(vlis, vlist, prob: Problem, cluster, delta, ng=None):
    """
    原 `Varqite.imaginary()` の再現。

    再現している欠陥:
      A-0 回帰対象 f_aim は重みにしか入らない（返り値は ITE 忠実度のずれのみ）
      A-1 swap test の出力（|<.|.>|^2）を振幅として足し、np.abs で符号を捨てる
      B-2 テイラー展開の結果を cosh/sinh で上書き（ここでは上書き後のみ実装）
      B-3 Ham の項ループで grad を上書き → 最後の項しか効かない
    """
    cfg = prob.cfg
    Se, So = np.cosh(delta), np.sinh(delta)
    Grad = 0.0
    B = 2 + 1 / cfg.n_train
    shifted = prob.f - np.min(prob.f)
    for m, x in enumerate(prob.X):
        psi_new = kan_forward(vlis, x, cluster, cfg, ng)
        psi_ref = kan_forward(vlist, x, cluster, cfg, ng)
        F = expval_diag(psi_ref, prob.Hsign)
        # 最後の Pauli 項（= 読み出しと同じ Z_a Z_b）のみが効く（B-3 の再現）
        P_psi_ref = psi_ref * prob.Hsign
        re0 = np.abs(np.vdot(psi_new, psi_ref)) ** 2      # swap test の出力
        re = np.abs(np.vdot(psi_new, P_psi_ref)) ** 2     # swap test の出力
        denom = np.sqrt(Se * Se + So * So - 2 * Se * So * F)
        grad = np.abs(1 - (-So * np.abs(re) + Se * np.abs(re0)) / denom)
        Grad += grad * (B - shifted[m]) / B
    return float(Grad)


# --------------------------------------------------------------------------
# 修正版: 虚時間発展 = 回帰コストに対する自然勾配流（McLachlan 変分原理）
# --------------------------------------------------------------------------

def jacobian_and_state(vlist, x, cluster, prob: Problem, eta=1e-4, ng=None):
    """中心差分で |∂_i ψ⟩ を作る。パラメータがスプライン係数経由で非線形に入るため
    パラメータシフト則が使えないので差分を用いる。"""
    cfg = prob.cfg
    P = len(vlist)
    psi = kan_forward(vlist, x, cluster, cfg, ng)
    J = np.zeros((psi.shape[0], P), dtype=complex)
    for i in range(P):
        vp = np.array(vlist, dtype=float); vp[i] += eta
        vm = np.array(vlist, dtype=float); vm[i] -= eta
        J[:, i] = (kan_forward(vp, x, cluster, cfg, ng)
                   - kan_forward(vm, x, cluster, cfg, ng)) / (2 * eta)
    return psi, J


def ite_step_shots(vlist, cluster, prob: Problem, delta, rng, reg=1e-3,
                   eta_g=0.05, eta_m=0.3, use_metric=True, ng=None):
    """有限ショット版の虚時間（自然勾配）1 ステップ。

    実機で実際に測れる量だけを使う:
      勾配   ∂_i C  … 有限ショットの ⟨H⟩ の中心差分
      計量   A_ij   … 忠実度 F(θ, θ+η) の 2 階中心差分
                      F は compute-uncompute の全ビット 0 確率（二項分布）で推定

    F(η) ≈ 1 − Σ_ij A_ij η_i η_j なので A_ij = −(1/2) ∂²F/∂η_i∂η_j。
    ショットノイズが η² のオーダーの信号に埋もれないよう、計量用の差分幅 eta_m は
    勾配用 eta_g より大きく取る必要がある（既定 0.3 vs 0.05）。
    """
    cfg = prob.cfg
    shots = cfg.shots
    P = len(vlist)
    v = np.array(vlist, dtype=float)
    g = np.zeros(P)
    A = np.zeros((P, P))

    def state(vv, x):
        return kan_forward(vv, x, cluster, cfg, ng, rng)

    for m, x in enumerate(prob.X):
        h = expval_shot(state(v, x), prob.Hsign, shots, rng)
        dh = np.zeros(P)
        for i in range(P):
            vp = v.copy(); vp[i] += eta_g
            vm = v.copy(); vm[i] -= eta_g
            hp = expval_shot(state(vp, x), prob.Hsign, shots, rng)
            hm = expval_shot(state(vm, x), prob.Hsign, shots, rng)
            dh[i] = (hp - hm) / (2 * eta_g)
        g += prob.w[m] * 2 * (h - prob.f[m]) * dh

        if use_metric:
            psi0 = state(v, x)
            Am = np.zeros((P, P))
            for i in range(P):
                for j in range(i, P):
                    F = {}
                    for si in (+1, -1):
                        for sj in (+1, -1):
                            vv = v.copy()
                            vv[i] += si * eta_m
                            vv[j] += sj * eta_m
                            F[(si, sj)] = fidelity_shot(psi0, state(vv, x), shots, rng)
                    d2 = (F[(1, 1)] - F[(1, -1)] - F[(-1, 1)] + F[(-1, -1)]) \
                        / (4 * eta_m ** 2)
                    Am[i, j] = Am[j, i] = -0.5 * d2
            A += prob.w[m] * Am

    if use_metric:
        # 推定した計量は雑音で不定値になりうるので固有値をクリップして正定化する
        As = (A + A.T) / 2
        w_, V = np.linalg.eigh(As)
        As = V @ np.diag(np.clip(w_, reg, None)) @ V.T
        step = np.linalg.solve(As, g)
    else:
        step = g
    return v - delta * step, float(np.linalg.norm(g))


def ite_step_spsa(vlist, cluster, prob: Problem, delta, rng, state,
                  reg=1e-2, eta_g=0.05, eps=0.3, resamplings=1, ng=None):
    """QN-SPSA（Gacon et al., *Quantum* **5**, 567 (2021)）による虚時間ステップ。

    §3/§4 で、忠実度の 2 階中心差分による計量推定が有限ショットで破綻することを
    確認した。その推定量は 1 ステップあたり 4·P(P+1)/2 回の忠実度測定を要し、
    信号は η² のオーダーしかない。SPSA-QFI はランダム 2 方向 Δ1, Δ2 ∈ {±1}^P を
    引いて

        δF = F(θ, θ+ε(Δ1+Δ2)) − F(θ, θ+εΔ1) − F(θ, θ+ε(−Δ1+Δ2)) + F(θ, θ−εΔ1)
        ĝ  = −(1/2) · (δF / 4ε²) · (Δ1Δ2ᵀ + Δ2Δ1ᵀ)/2

    と 1 サンプルあたり **4 回**の忠実度測定で不偏推定を作り、反復平均で分散を
    落とす。`state` に反復平均 `gbar` と反復回数 `k` を保持する（run() が持ち回る）。
    """
    cfg = prob.cfg
    shots = cfg.shots
    P = len(vlist)
    v = np.array(vlist, dtype=float)

    def st(vv, x):
        return kan_forward(vv, x, cluster, cfg, ng, rng)

    # --- 勾配は従来どおり ⟨H⟩ の中心差分（計量推定のみを入れ替えて切り分ける） ---
    g = np.zeros(P)
    for m, x in enumerate(prob.X):
        h = expval_shot(st(v, x), prob.Hsign, shots, rng)
        dh = np.zeros(P)
        for i in range(P):
            vp = v.copy(); vp[i] += eta_g
            vm = v.copy(); vm[i] -= eta_g
            dh[i] = (expval_shot(st(vp, x), prob.Hsign, shots, rng)
                     - expval_shot(st(vm, x), prob.Hsign, shots, rng)) / (2 * eta_g)
        g += prob.w[m] * 2 * (h - prob.f[m]) * dh

    # --- 計量は SPSA-QFI ---
    ghat = np.zeros((P, P))
    for _ in range(resamplings):
        d1 = rng.choice([-1.0, 1.0], size=P)
        d2 = rng.choice([-1.0, 1.0], size=P)
        dF = 0.0
        for m, x in enumerate(prob.X):
            psi0 = st(v, x)
            f = lambda vv: fidelity_shot(psi0, st(vv, x), shots, rng)
            dF += prob.w[m] * (f(v + eps * (d1 + d2)) - f(v + eps * d1)
                               - f(v + eps * (-d1 + d2)) + f(v - eps * d1))
        outer = (np.outer(d1, d2) + np.outer(d2, d1)) / 2
        ghat += -0.5 * (dF / (4 * eps ** 2)) * outer
    ghat /= resamplings

    # --- 反復平均 → 正定化 ---
    k = state["k"]
    state["gbar"] = (k / (k + 1)) * state["gbar"] + (1 / (k + 1)) * ghat
    state["k"] = k + 1
    A = (state["gbar"] + state["gbar"].T) / 2
    w_, V = np.linalg.eigh(A)
    A = V @ np.diag(np.maximum(np.abs(w_), reg)) @ V.T   # |固有値| を取り PSD 射影
    step = np.linalg.solve(A + reg * np.eye(P), g)
    return v - delta * step, float(np.linalg.norm(g))


def ite_step_ctl(vlist, cluster, prob: Problem, delta, reg=1e-3, eta_g=0.05,
                 eta=1e-4, ng=None):
    """交絡切り分け用の対照: 計量は厳密ヤコビアン（FD推定と同じ量の厳密値）、
    正則化は FD 経路と同じ固有値クリップ、勾配は FD 経路と同じ η_g=0.05 の
    ⟨H⟩ 中心差分（厳密期待値）。計量『推定量』だけを FD → 厳密に戻した条件。"""
    P = len(vlist)
    v = np.array(vlist, dtype=float)
    A = np.zeros((P, P))
    g = np.zeros(P)
    with np.errstate(all="ignore"):
        for m, x in enumerate(prob.X):
            psi, J = jacobian_and_state(v, x, cluster, prob, eta, ng)
            h = expval_diag(psi, prob.Hsign)
            dh = np.zeros(P)
            for i in range(P):
                vp = v.copy(); vp[i] += eta_g
                vm = v.copy(); vm[i] -= eta_g
                hp = expval_diag(kan_forward(vp, x, cluster, prob.cfg, ng), prob.Hsign)
                hm = expval_diag(kan_forward(vm, x, cluster, prob.cfg, ng), prob.Hsign)
                dh[i] = (hp - hm) / (2 * eta_g)
            g += prob.w[m] * 2 * (h - prob.f[m]) * dh
            ov = J.conj().T @ psi
            A += prob.w[m] * (np.real(J.conj().T @ J) - np.real(np.outer(ov, ov.conj())))
    As = (A + A.T) / 2
    w_, V = np.linalg.eigh(As)
    As = V @ np.diag(np.clip(w_, reg, None)) @ V.T
    step = np.linalg.solve(As, g)
    return v - delta * step, float(np.linalg.norm(g))


def ite_step(vlist, cluster, prob: Problem, delta, reg=1e-6, eta=1e-4,
             use_metric=True, ng=None):
    """自然勾配（虚時間）1 ステップ。更新後のパラメータと診断量を返す。"""
    P = len(vlist)
    A = np.zeros((P, P))
    g = np.zeros(P)
    h_all = np.zeros(prob.cfg.n_train)
    # macOS/Accelerate BLAS は複素行列積で偽の FPE 警告を出す（結果は vdot と厳密一致）
    with np.errstate(all="ignore"):
        for m, x in enumerate(prob.X):
            psi, J = jacobian_and_state(vlist, x, cluster, prob, eta, ng)
            h = expval_diag(psi, prob.Hsign)
            h_all[m] = h
            w = prob.w[m]
            # ∂_i ⟨H⟩ = 2 Re⟨∂_iψ|H|ψ⟩
            Hpsi = psi * prob.Hsign
            dh = 2 * np.real(J.conj().T @ Hpsi)
            g += w * 2 * (h - prob.f[m]) * dh
            if use_metric:
                ov = J.conj().T @ psi                  # ⟨∂_iψ|ψ⟩
                A += w * (np.real(J.conj().T @ J) - np.real(np.outer(ov, ov.conj())))
    if use_metric:
        step = np.linalg.solve(A + reg * np.eye(P), g)
    else:
        step = g
    new = np.array(vlist, dtype=float) - delta * step
    return new, h_all, float(np.linalg.norm(g))


# --------------------------------------------------------------------------
# アンザッツ成長（原 `adaptive()` の way 2。診断書 C-3 の para スコープ問題を修正）
# --------------------------------------------------------------------------

def build_pool(nq):
    pool, label = [], []
    for j in range(nq):
        for nm, op in (("X", pauli.X), ("Y", pauli.Y), ("Z", pauli.Z)):
            pool.append(op[j]); label.append(f"{nm}{j}")
        for k in range(j + 1, nq):
            a = [("X", pauli.X[j]), ("Y", pauli.Y[j]), ("Z", pauli.Z[j])]
            b = [("X", pauli.X[k]), ("Y", pauli.Y[k]), ("Z", pauli.Z[k])]
            for l in range(3):
                for m in range(l, 3):
                    pool.append(a[l][1] * b[m][1])
                    label.append(f"{a[l][0]}{j}{b[m][0]}{k}")
    return pool, label


def grow_ansatz(vlist, cluster, prob: Problem, pool, labels, frame=0, rng=None):
    """way 2: プールの各演算子を末尾に追加し、コストが最も下がるものを採用する。
    採用時のパラメータは原実装どおり 0 で初期化する。"""
    cfg = prob.cfg
    base, _ = prob.cost(vlist, cluster, None, rng)
    best, best_idx = base, None
    for i, op in enumerate(pool):
        trial_cluster = [list(cl) for cl in cluster]
        trial_cluster[frame] = trial_cluster[frame] + [op]
        # 新しい係数はフレーム frame のブロック末尾に挿入される
        offset = sum(len(cluster[t]) for t in range(frame + 1)) * cfg.grids
        trial_v = np.concatenate([vlist[:offset], np.zeros(cfg.grids), vlist[offset:]])
        c, _ = prob.cost(trial_v, trial_cluster, None, rng)
        if c < best:
            best, best_idx = c, i
    if best_idx is None:
        return vlist, cluster, None, base
    op = pool[best_idx]
    cluster = [list(cl) for cl in cluster]
    cluster[frame] = cluster[frame] + [op]
    offset = sum(len(cluster[t]) for t in range(frame)) * cfg.grids \
        + (len(cluster[frame]) - 1) * cfg.grids
    vlist = np.concatenate([vlist[:offset], np.zeros(cfg.grids), vlist[offset:]])
    return vlist, cluster, labels[best_idx], best


# --------------------------------------------------------------------------
# 実行ドライバ
# --------------------------------------------------------------------------

def run(method, target, seed, steps, delta, grow_every, out_dir,
        spline_eval="direct", ng_growth=False, kick=0.0, maxiter=1000,
        grids=8, ndT=3, verbose=True, shots=0, nq=4, tag_suffix="",
        readout_mode="single", ite_reg=1e-3, eta_m=0.3, spsa_resamplings=1,
        force_fd_metric=False, encoding="code"):
    cfg = Config(nq=nq, seed=seed, grids=grids, ng=grids, ndT=ndT,
                 spline_eval=spline_eval, ng_growth=ng_growth, shots=shots,
                 readout_mode=readout_mode, encoding=encoding)
    prob = Problem(cfg, target)
    rng = np.random.default_rng(seed + 12345)
    # 測定用の乱数は最適化用と分離する（ショット雑音の再現性のため）
    mrng = np.random.default_rng(seed + 99991) if shots else None

    pool, labels = build_pool(cfg.nq)
    cluster = [[pauli.X[1]], [], []][:cfg.ndT]
    vlist = np.zeros(n_params(cluster, cfg))
    ng = cfg.ng

    rows = []
    tau = 0.0
    report_calls = 0          # 指標の記録に使った評価（最適化コストからは除外する）
    spsa_state = None
    FORWARD_CALLS["n"] = 0
    t0 = time.time()
    for step in range(steps):
        if method == "legacy":
            start = np.array(vlist) + rng.random(len(vlist)) * kick
            res = smin(lambda p: legacy_objective(p, vlist, prob, cluster, delta, ng),
                       start, method="COBYLA",
                       options={"disp": False, "maxiter": maxiter})
            vlist = np.array(res.x)
            if ng_growth:
                ng = cfg.grids + 4
            gnorm = float("nan")
        elif method == "cobyla":
            start = np.array(vlist) + rng.random(len(vlist)) * kick
            res = smin(lambda p: prob.cost(p, cluster, ng, mrng)[0], start,
                       method="COBYLA", options={"disp": False, "maxiter": maxiter})
            vlist = np.array(res.x)
            gnorm = float("nan")
        elif method == "ite_ctl":
            vlist, gnorm = ite_step_ctl(vlist, cluster, prob, delta,
                                        reg=ite_reg, ng=ng)
        elif method == "ite_spsa":
            if spsa_state is None or spsa_state["gbar"].shape[0] != len(vlist):
                # アンザッツ成長でパラメータ数が変わったら反復平均を作り直す
                spsa_state = {"gbar": np.zeros((len(vlist), len(vlist))), "k": 0}
            vlist, gnorm = ite_step_spsa(vlist, cluster, prob, delta,
                                         mrng if shots else np.random.default_rng(seed + 7),
                                         spsa_state, reg=ite_reg, eps=eta_m,
                                         resamplings=spsa_resamplings, ng=ng)
        elif method in ("ite", "gd") and (shots or force_fd_metric):
            vlist, gnorm = ite_step_shots(vlist, cluster, prob, delta,
                                          mrng if shots else np.random.default_rng(seed + 11),
                                          reg=ite_reg, eta_m=eta_m,
                                          use_metric=(method == "ite"), ng=ng)
        elif method in ("ite", "gd"):
            vlist, _, gnorm = ite_step(vlist, cluster, prob, delta,
                                       use_metric=(method == "ite"), ng=ng)
        else:
            raise ValueError(method)

        _calls_before_report = FORWARD_CALLS["n"]
        C, h = prob.cost(vlist, cluster, ng)
        tr = prob.train_absdist(h)
        te = prob.test_absdist(vlist, cluster, ng)
        report_calls += FORWARD_CALLS["n"] - _calls_before_report
        tau += delta
        rows.append(dict(step=step, tau=round(tau, 6), cost=C,
                         forward_calls=FORWARD_CALLS["n"],
                         opt_calls=FORWARD_CALLS["n"] - report_calls,
                         train_absdist=tr, test_absdist=te,
                         n_params=len(vlist), grad_norm=gnorm,
                         ansatz=";".join(
                             labels[pool.index(op)] if op in pool else "?"
                             for cl in cluster for op in cl)))
        if verbose:
            print(f"[{method}/{target}/s{seed}] step {step:3d}  tau={tau:6.2f}  "
                  f"C={C:10.4f}  train={tr:8.4f}  test={te:8.4f}  P={len(vlist)}")

        if grow_every and (step + 1) % grow_every == 0 and step + 1 < steps:
            vlist, cluster, added, _ = grow_ansatz(vlist, cluster, prob, pool, labels, 0, mrng)
            if verbose:
                print(f"    -> ansatz grown with {added} (P={len(vlist)})")

    os.makedirs(out_dir, exist_ok=True)
    tag = f"{method}_{target}_seed{seed}{tag_suffix}"
    with open(os.path.join(out_dir, f"{tag}.csv"), "w", newline="") as fp:
        wtr = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)
    meta = dict(method=method, target=target, seed=seed, steps=steps, delta=delta,
                shots=shots, nq=nq, ite_reg=ite_reg, eta_m=eta_m,
                grow_every=grow_every, kick=kick, maxiter=maxiter,
                config=asdict(cfg), elapsed_sec=round(time.time() - t0, 2))
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as fp:
        json.dump(meta, fp, indent=2, default=str)
    return rows, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method",
                    choices=["legacy", "ite", "gd", "cobyla", "ite_spsa", "ite_ctl"],
                    default="ite")
    ap.add_argument("--target", choices=sorted(TARGETS), default="log")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--grow-every", type=int, default=20,
                    help="この間隔でアンザッツを 1 項成長させる。0 で成長なし")
    ap.add_argument("--kick", type=float, default=0.0,
                    help="COBYLA 再スタート時のランダム摂動幅。原実装は 0.1")
    ap.add_argument("--maxiter", type=int, default=1000)
    ap.add_argument("--grids", type=int, default=8)
    ap.add_argument("--ndT", type=int, default=3)
    ap.add_argument("--spline-eval", choices=["direct", "grid"], default="direct")
    ap.add_argument("--ng-growth", action="store_true")
    ap.add_argument("--shots", type=int, default=0, help="0 で無限ショット")
    ap.add_argument("--nq", type=int, default=4)
    ap.add_argument("--readout-mode",
                    choices=["single", "all_pairs", "single_scaled"],
                    default="single")
    ap.add_argument("--encoding", choices=["code", "paper"], default="code")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    run(a.method, a.target, a.seed, a.steps, a.delta, a.grow_every, a.out,
        a.spline_eval, a.ng_growth, a.kick, a.maxiter, a.grids, a.ndT,
        shots=a.shots, nq=a.nq, readout_mode=a.readout_mode, encoding=a.encoding)


if __name__ == "__main__":
    main()
