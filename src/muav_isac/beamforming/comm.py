"""Communication beamforming via fractional programming (paper Algorithm 1).

Given a trajectory (encoded in the channel snapshot) and a *fixed* set of sensing
beams ``I``, maximize the sum rate over the communication beam covariances
``G_v^u[n] = g_v^u (g_v^u)^H`` by Shen & Yu's Lagrangian-dual + quadratic
transform (paper P2 -> P3 -> P4):

    * closed-form auxiliary updates
        chi_v[n]*  = Tr(H_v G_v) / Theta_v            (= SINR_v)
        psi_v[n]*  = sqrt( tau (1+chi_v) Tr(H_v G_v) / (Tr(H_v G_v) + Theta_v) )
      where Theta_v = sum of interference powers (other comm beams + sensing) + sigma^2.

    * with chi, psi fixed, the G-update is convex (concave objective with a
      sqrt(Tr(H_v G_v)) term) and is solved per slot by cvxpy. The energy budget
      (8d) enters as a Lagrange multiplier ``lam`` on Tr(G_v), so each slot is
      independent once lam is fixed.

The SDR drops the rank-one constraint; rank-one beams are recovered afterward
(see :mod:`rank1`). With the default ``E_th`` the energy budget is slack and
``lam = 0`` (the per-slot Pmax binds instead).
"""

from __future__ import annotations

import os

import cvxpy as cp
import numpy as np

from ..scenario import ChannelSnapshot, Network
from . import rank1
from .cvxbuild import channel_covariances

# SCS-tuned defaults: the FP inner loop only needs modest accuracy (the outer BCD
# refines further), so a loose eps + warmup acceleration is ~3-4x faster. CLARABEL
# currently hits a numerical error on this SDP+SOCP structure, so SCS is the default.
DEFAULT_SOLVER_KWARGS = {"eps": 1e-3, "max_iters": 5000, "acceleration_lookback": 20}


def _solver_diagnostics_enabled() -> bool:
    return os.environ.get("MUAV_ISAC_SOLVER_DIAGNOSTICS", "").lower() in {"1", "true", "yes", "on"}


def _mrt_init(net: Network, snap: ChannelSnapshot, i_beams: dict, Pmax_comm: float) -> dict:
    """Initial G_v = power * (h_serving/||h||)(.)^H for every (u, v, n)."""
    p = net.p
    G = {}
    for u in range(p.U):
        Vu = net.Vu_list[u]
        per = Pmax_comm / max(Vu, 1)
        for lv, v in enumerate(net.user_of_uav[u]):
            h = snap.h[u, v]                                  # (N, M)
            nrm = np.linalg.norm(h, axis=1, keepdims=True)
            g = (h / np.maximum(nrm, 1e-12)) * np.sqrt(per)   # (N, M)
            G[(u, lv)] = g[..., :, None] * np.conj(g[..., None, :])  # (N, M, M)
    return G


def _sensing_interference(net: Network, HH: np.ndarray, n: int, i_beams: dict) -> np.ndarray:
    """Per-user constant sensing interference at slot n (from fixed I beams)."""
    p = net.p
    out = np.zeros(net.V)
    for uu in range(p.U):
        for lt, _k in enumerate(net.target_of_uav[uu]):
            i_vec = i_beams[(uu, lt)][n]                      # (M,) beam vector
            Icov = np.outer(i_vec, i_vec.conj())             # i i^H
            for v in range(net.V):
                out[v] += np.real(np.trace(HH[uu, v, n] @ Icov))
    return out


def _signal_and_interf(
    net: Network, HH: np.ndarray, n: int, Gslot: dict, interf_I_v: np.ndarray,
    noise_floor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-user (signal S_v, total interference+noise Theta_v) at slot n."""
    p = net.p
    S = np.zeros(net.V)
    inter_g = np.zeros(net.V)
    for v in range(net.V):
        su, lv = net.user_local(v)
        S[v] = np.real(np.trace(HH[su, v, n] @ Gslot[(su, lv)]))
        for uu in range(p.U):
            for lu, _gv in enumerate(net.user_of_uav[uu]):
                if uu == su and lu == lv:
                    continue
                inter_g[v] += np.real(np.trace(HH[uu, v, n] @ Gslot[(uu, lu)]))
    Theta = inter_g + interf_I_v + noise_floor   # noise-normalized (sigma2 -> noise_floor)
    return S, Theta


def _solve_slot(
    net: Network,
    HH: np.ndarray,
    n: int,
    chi: np.ndarray,
    psi: np.ndarray,
    interf_I_v: np.ndarray,
    sensing_pwr_u: np.ndarray,
    lam: float,
    solver: str,
    solver_kwargs: dict | None = None,
    noise_floor: float = 1.0,
) -> dict | None:
    """Convex G-update at slot n (all UAVs jointly). Returns {(u, lv): G} or None."""
    p = net.p
    M = p.M
    Pw = p.Pmax_W
    # Unit-budget variable G̃ (= G / Pw) keeps the solver variable O(1) at high Pmax.
    # cvxpy folds the Pw scalar into the trace coefficients, so SCS sees O(1) data
    # AND an O(1) variable instead of a 1e5-Watt variable with 1e-5-scaled data.
    Gtilde = {(u, lv): cp.Variable((M, M), hermitian=True)
              for u in range(p.U) for lv in range(net.Vu_list[u])}

    obj = 0.0
    for v in range(net.V):
        su, lv = net.user_local(v)
        S = Pw * cp.real(cp.trace(HH[su, v, n] @ Gtilde[(su, lv)]))
        interf = 0.0
        for uu in range(p.U):
            for lu, _gv in enumerate(net.user_of_uav[uu]):
                if uu == su and lu == lv:
                    continue
                interf = interf + Pw * cp.real(cp.trace(HH[uu, v, n] @ Gtilde[(uu, lu)]))
        P_v = S + interf + interf_I_v[v] + noise_floor   # noise-normalized
        coeff = 2.0 * psi[v] * np.sqrt(p.tau * (1.0 + chi[v]))
        obj = obj + coeff * cp.sqrt(S) - (psi[v] ** 2) * P_v

    for u in range(p.U):
        trGt = sum(cp.real(cp.trace(Gtilde[(u, lv)])) for lv in range(net.Vu_list[u]))
        obj = obj - lam * p.tau * Pw * trGt

    cons = [Gt >> 0 for Gt in Gtilde.values()]
    # for u in range(p.U):
    #     trGt = sum(cp.real(cp.trace(Gtilde[(u, lv)])) for lv in range(net.Vu_list[u]))
    #     cons.append(Pw * trGt <= p.Pmax_W - sensing_pwr_u[u])

    for u in range(p.U):
        trGt = sum(
            cp.real(cp.trace(Gtilde[(u, lv)]))
            for lv in range(net.Vu_list[u])
        )

        remaining_fraction = max(
            0.0,
            1.0 - sensing_pwr_u[u] / Pw
        )

        cons.append(trGt <= remaining_fraction)


    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver=solver, **(solver_kwargs if solver_kwargs is not None else DEFAULT_SOLVER_KWARGS))
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None

    # 先取出 SCS 得到的归一化通信协方差
    Gval = {
        k: np.asarray(Gtilde[k].value, dtype=complex)
        for k in Gtilde
    }

    # 对每架 UAV 检查总通信功率
    for u in range(p.U):

        keys_u = [
            (u, lv)
            for lv in range(net.Vu_list[u])
        ]

        total_trace = sum(
            max(
                float(np.real(np.trace(Gval[k]))),
                0.0
            )
            for k in keys_u
        )

        # 通信真正允许使用的归一化功率
        remaining_fraction = max(
            0.0,
            1.0 - sensing_pwr_u[u] / Pw
        )

        # SCS 若轻微超过功率约束，按比例投影回可行域
        if total_trace > remaining_fraction and total_trace > 0.0:

            scale_factor = remaining_fraction / total_trace

            if _solver_diagnostics_enabled():
                print(
                    f"[COMM PROJECT] slot={n}, "
                    f"uav={u}, "
                    f"power_fraction={total_trace:.6f}, "
                    f"limit={remaining_fraction:.6f}, "
                    f"scale={scale_factor:.6f}"
                )

            for k in keys_u:
                Gval[k] = Gval[k] * scale_factor

    return {
        k: Pw * Gval[k]
        for k in Gval
    }

def comm_beamforming(
    net: Network,
    snap: ChannelSnapshot,
    i_beams: dict,
    *,
    init_G: dict | None = None,
    max_iter: int = 8,
    tol: float = 1e-3,
    lam: float = 0.0,
    solver: str = cp.SCS,
    solver_kwargs: dict | None = None,
) -> tuple[dict, dict, list[float]]:
    """Run Algorithm 1. Returns ``(G_beams, g_beams, objective_history)``.

    * ``G_beams[(u, lv)]``: (N, M, M) PSD covariances.
    * ``g_beams[(u, lv)]``: (N, M) rank-one-recovered beam vectors.
    * ``objective_history``: sum-rate per FP iteration (should be non-decreasing).
    """
    p = net.p
    # Noise-normalize the channel covariances for numerical conditioning: the raw
    # Tr(H G) ~ alpha0 * P / d^2 ~ 1e-12 is badly scaled for the solver, while the
    # SINR (hence the optimal G) is invariant under H, sigma2 -> H/sigma2, 1.
    HH = channel_covariances(snap) / p.sigma2

    # Per-instance rescaling so the solver sees O(1) numbers at any power budget.
    # With Pmax up to 100 kW (80 dBm) the noise-normalized signal Tr(H G) ~ |h|^2
    # P / sigma^2 reaches ~1e6, and SCS (eps=1e-3) then cannot resolve the convex
    # sqrt term against the linear penalty -> garbage solutions / rate 0. Dividing
    # H by (Pmax * mean link trace) makes a full-power signal ~1; the noise floor
    # becomes 1/scale (the noise-to-signal ratio). SINR -- hence the optimal G and
    # the rate -- is invariant under this scaling (signal, interference, and noise
    # all divided by the same factor).
    mean_trace = float(np.real(np.trace(HH, axis1=-1, axis2=-2)).mean())
    scale = max(1.0, p.Pmax_W * mean_trace)
    HH = HH / scale
    noise_floor = 1.0 / scale

    # per-slot constants from the fixed sensing beams
    sensing_pwr = np.zeros((p.U, p.N))
    for u in range(p.U):
        for lt in range(net.Ku_list[u]):
            sensing_pwr[u] += (np.abs(i_beams[(u, lt)]) ** 2).sum(axis=1).real

    Pmax_comm = max(p.Pmax_W - sensing_pwr.max(initial=0.0), 1e-6)
    G = init_G if init_G is not None else _mrt_init(net, snap, i_beams, Pmax_comm)

    history: list[float] = []
    for _it in range(max_iter):
        # ---- closed-form chi / psi updates + objective, slot by slot ----
        chi = np.zeros((net.V, p.N))
        psi = np.zeros((net.V, p.N))
        obj_val = 0.0
        for n in range(p.N):
            Gslot = {(u, lv): G[(u, lv)][n] for u in range(p.U) for lv in range(net.Vu_list[u])}
            interf_I = _sensing_interference(net, HH, n, i_beams)
            S, Theta = _signal_and_interf(net, HH, n, Gslot, interf_I, noise_floor)
            sinr = np.maximum(S, 0.0) / np.maximum(Theta, 1e-12)
            chi[:, n] = sinr
            P_v = np.maximum(S, 0.0) + Theta
            psi[:, n] = (np.sqrt(np.maximum(p.tau * (1.0 + sinr) * np.maximum(S, 0.0), 0.0)) / np.maximum(P_v, 1e-12))
            obj_val += p.tau * np.sum(np.log2(1.0 + sinr))

        # ---- convex G-update per slot ----
        new_G = {k: np.empty((p.N, p.M, p.M), dtype=complex) for k in
                 [(u, lv) for u in range(p.U) for lv in range(net.Vu_list[u])]}
        ok = True
        for n in range(p.N):
            interf_I = _sensing_interference(net, HH, n, i_beams)
            res = _solve_slot(net, HH, n, chi[:, n], psi[:, n], interf_I, sensing_pwr[:, n], lam,
                              solver, solver_kwargs, noise_floor)
            if res is None:
                ok = False
                break
            for k in res:
                new_G[k][n] = res[k]
        if not ok:
            break

        G = new_G
        history.append(obj_val)
        if len(history) >= 2 and abs(history[-1] - history[-2]) <= tol * max(1.0, abs(history[-2])):
            break

    # rank-one recovery -> beam vectors
    rng = np.random.default_rng(p.seed)
    g_beams = {}
    for u in range(p.U):
        for lv in range(net.Vu_list[u]):
            arr = np.empty((p.N, p.M), dtype=complex)
            for n in range(p.N):
                arr[n] = rank1.recover_beam(G[(u, lv)][n], rng)
            g_beams[(u, lv)] = arr

    return G, g_beams, history
