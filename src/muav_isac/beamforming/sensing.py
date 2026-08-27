"""Sensing beamforming via fractional programming (paper Algorithm 2).

Given a trajectory and *fixed* communication beams, maximize the sum rate over
the sensing beam covariances ``I_k^u[n] = i_k^u (i_k^u)^H`` subject to the CRB
constraint (9b), the per-slot power budget (9c), and PSD.

Sensing beams only enter the rate through the interference term Theta_v, which
makes the rate convex-decreasing in the sensing power -- a non-concave objective.
Following the paper (P5 -> P6 -> P7) we introduce an auxiliary SINR variable
``iota_v[n]`` and upper-bound the bilinear constraint (14c) by (15):

    Tr(H_v G_v) >= Theta_v(I)^2 / (2 Omega_v) + iota_v^2 Omega_v / 2 ,

which is tight at ``Omega_v = Theta_v / iota_v``. With ``Omega`` fixed this is
convex (square of an affine Theta_v, square of iota_v), so cvxpy solves P7 per
slot. We alternate the closed-form ``Omega`` update with the convex (I, iota)
solve until convergence.

Noise-normalized exactly like :mod:`comm` (H -> H/sigma^2, noise -> 1); the CRB
constraint stays in actual units (AtA, I in Watts) since Eq.5 is dimensional.
"""

from __future__ import annotations

import os

import cvxpy as cp
import numpy as np

from ..scenario import ChannelSnapshot, Network
from . import rank1
from .comm import DEFAULT_SOLVER_KWARGS
from .cvxbuild import channel_covariances, crb_thresholds, sensing_fisher_matrices


def _solver_diagnostics_enabled() -> bool:
    return os.environ.get("MUAV_ISAC_SOLVER_DIAGNOSTICS", "").lower() in {"1", "true", "yes", "on"}


def _comm_covariances(net: Network, g_beams: dict) -> dict:
    """G[(u, lv)] = g g^H per slot, shape (N, M, M), from beam vectors."""
    out = {}
    for u in range(net.p.U):
        for lv in range(net.Vu_list[u]):
            g = g_beams[(u, lv)]                                  # (N, M)
            out[(u, lv)] = g[..., :, None] * np.conj(g[..., None, :])
    return out


def _comm_signal_and_interf(net: Network, HH: np.ndarray, n: int, Gslot: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-user (serving signal, comm-only interference) from fixed G at slot n."""
    p = net.p
    sig = np.zeros(net.V)
    inter = np.zeros(net.V)
    for v in range(net.V):
        su, lv = net.user_local(v)
        sig[v] = np.real(np.trace(HH[su, v, n] @ Gslot[(su, lv)]))
        for uu in range(p.U):
            for lu, _gv in enumerate(net.user_of_uav[uu]):
                if uu == su and lu == lv:
                    continue
                inter[v] += np.real(np.trace(HH[uu, v, n] @ Gslot[(uu, lu)]))
    return sig, inter


def _sensing_interf_expr(net: Network, HH: np.ndarray, n: int, Ivar: dict, scale: float = 1.0) -> dict:
    """Per-user affine sensing-interference expression at slot n (cvxpy).

    ``scale`` multiplies each trace (used to fold Pmax in when ``Ivar`` is a
    unit-budget variable, matching comm.py's variable rescaling).
    """
    p = net.p
    out = {}
    for v in range(net.V):
        e = 0.0
        for uu in range(p.U):
            for lt, _k in enumerate(net.target_of_uav[uu]):
                if (uu, lt) in Ivar:
                    e = e + scale * cp.real(cp.trace(HH[uu, v, n] @ Ivar[(uu, lt)]))
        out[v] = e
    return out


def _sensing_interf_val(net: Network, HH: np.ndarray, n: int, Islot: dict) -> np.ndarray:
    """Per-user sensing interference value at slot n (numpy)."""
    p = net.p
    out = np.zeros(net.V)
    for v in range(net.V):
        for uu in range(p.U):
            for lt, _k in enumerate(net.target_of_uav[uu]):
                out[v] += np.real(np.trace(HH[uu, v, n] @ Islot[(uu, lt)]))
    return out


def _crb_feasible_init(net: Network, AtA: np.ndarray, crb_thr: np.ndarray) -> dict:
    """Minimum-power rank-one I_k along A^H A's top eigenvector, meeting CRB."""
    p = net.p
    I = {k: np.empty((p.N, p.M, p.M), dtype=complex)
         for k in [(u, lt) for u in range(p.U) for lt in range(net.Ku_list[u])]}
    for u in range(p.U):
        for lt, gk in enumerate(net.target_of_uav[u]):
            for n in range(p.N):
                w, V = np.linalg.eigh((AtA[u, gk, n] + AtA[u, gk, n].conj().T) / 2)
                vmax = V[:, -1]
                fisher = max(w[-1], 1e-12)
                c = crb_thr[u, gk, n] / fisher
                I[(u, lt)][n] = c * np.outer(vmax, vmax.conj())
    return I

def crb_min_beams(
    net: Network,
    snap: ChannelSnapshot,
) -> dict:
    """
    根据当前轨迹生成满足CRB所需的最小功率感知波束。
    用于轨迹变化后，在通信波束优化前预留正确的感知功率。
    """
    p = net.p

    AtA = sensing_fisher_matrices(snap)
    crb_thr = crb_thresholds(snap, p)

    i_beams = {}

    for u in range(p.U):
        for lt, gk in enumerate(net.target_of_uav[u]):

            arr = np.empty(
                (p.N, p.M),
                dtype=complex
            )

            for n in range(p.N):

                A = AtA[u, gk, n]

                # 数值上保证Hermitian
                A = (
                    A + A.conj().T
                ) / 2.0

                eigvals, eigvecs = np.linalg.eigh(A)

                lambda_max = max(
                    float(np.real(eigvals[-1])),
                    1e-30
                )

                vmax = eigvecs[:, -1]

                # 满足CRB约束需要的最小功率
                required_power = (
                    float(crb_thr[u, gk, n])
                    / lambda_max
                )

                arr[n] = (
                    vmax
                    * np.sqrt(required_power)
                )

            i_beams[(u, lt)] = arr

    return i_beams


def _solve_slot(
    net: Network,
    HH: np.ndarray,
    AtA: np.ndarray,
    crb_thr: np.ndarray,
    n: int,
    Omega: np.ndarray,
    sig_G: np.ndarray,
    inter_G: np.ndarray,
    comm_pwr_u: np.ndarray,
    solver: str,
    solver_kwargs: dict | None,
    noise_floor: float = 1.0,
) -> dict | None:
    """Convex (I, iota) update at slot n (P7). Returns {(u,lt): I, '_iota': iota} or None."""
    p = net.p
    M = p.M
    Pw = p.Pmax_W
    # Unit-budget variable Ỉ (= I / Pw); see comm._solve_slot for the rationale
    # (keeps the solver variable O(1) at high Pmax so SCS stays accurate).
    Itilde = {(u, lt): cp.Variable((M, M), hermitian=True)
              for u in range(p.U) for lt in range(net.Ku_list[u])}
    iota = cp.Variable(net.V, nonneg=True)

    sens_interf = _sensing_interf_expr(net, HH, n, Itilde, scale=Pw)  # = Tr(H I), I in Watts
    obj = p.tau * cp.sum(cp.log(1 + iota)) / np.log(2.0)

    cons = [It >> 0 for It in Itilde.values()]
    for u in range(p.U):
        trIt = sum(cp.real(cp.trace(Itilde[(u, lt)])) for lt in range(net.Ku_list[u]))
        cons.append(Pw * trIt <= p.Pmax_W - comm_pwr_u[u])
    for u in range(p.U):
        for lt, gk in enumerate(net.target_of_uav[u]):
            cons.append(Pw * cp.real(cp.trace(AtA[u, gk, n] @ Itilde[(u, lt)])) >= crb_thr[u, gk, n])
    for v in range(net.V):
        Theta = inter_G[v] + sens_interf[v] + noise_floor         # noise-normalized
        rhs = cp.square(Theta) / (2.0 * Omega[v]) + cp.square(iota[v]) * Omega[v] / 2.0
        cons.append(sig_G[v] >= rhs)

    prob = cp.Problem(cp.Maximize(obj), cons)

    prob.solve(
        solver=solver,
        **(
            solver_kwargs
            if solver_kwargs is not None
            else DEFAULT_SOLVER_KWARGS
        )
    )

    diag = _solver_diagnostics_enabled()

    if prob.status not in ("optimal", "optimal_inaccurate"):

        if diag:
            print(
                f"[SENS FAIL] slot={n}, "
                f"status={prob.status}, "
                f"objective={prob.value}"
            )

            print("========== SENS DIAGNOSTIC ==========")

        for u in range(p.U):

            # 当前 UAV 的通信功率
            comm_power = float(comm_pwr_u[u])

            # 留给感知的剩余功率
            remaining_power = float(
                p.Pmax_W - comm_power
            )

            # 满足 CRB 所需的理论最低感知功率
            crb_required_power = 0.0

            for lt, gk in enumerate(net.target_of_uav[u]):
                A = AtA[u, gk, n]

                # 保证数值上是 Hermitian 矩阵
                A = (
                            A + A.conj().T
                    ) / 2.0

                eigvals = np.linalg.eigvalsh(A)

                lambda_max = max(
                    float(np.real(eigvals[-1])),
                    1e-30
                )

                required_this_target = (
                        float(crb_thr[u, gk, n])
                        / lambda_max
                )

                crb_required_power += required_this_target

                if diag:
                    print(
                        f"  UAV={u}, target={gk}, "
                        f"lambda_max={lambda_max:.6e}, "
                        f"CRB_thr={crb_thr[u, gk, n]:.6e}, "
                        f"CRB_min_power={required_this_target:.6f} W"
                    )

            power_gap = (
                    remaining_power
                    - crb_required_power
            )

            if diag:
                print(
                    f"[UAV {u}] "
                    f"Pmax={p.Pmax_W:.6f} W, "
                    f"comm_power={comm_power:.6f} W, "
                    f"remaining_sensing_power={remaining_power:.6f} W, "
                    f"CRB_min_required={crb_required_power:.6f} W, "
                    f"gap={power_gap:.6f} W"
                )

            if diag:
                if remaining_power < 0:
                    print(
                        "  >>> CAUSE: communication power already exceeds Pmax"
                    )

                elif crb_required_power > remaining_power:
                    print(
                        "  >>> CAUSE: not enough remaining power to satisfy CRB"
                    )

                else:
                    print(
                        "  >>> POWER/CRB CHECK PASSED: "
                        "likely SINR/FP/SCS numerical infeasibility"
                    )

        if diag:
            print("=====================================")

        return None

    elif prob.status == "optimal_inaccurate" and diag:
        print(
            f"[SENS WARNING] slot={n}, "
            f"status={prob.status}, "
            f"objective={prob.value}"
        )

    return {
        k: Pw * np.asarray(Itilde[k].value)
        for k in Itilde
    } | {
        "_iota": np.asarray(iota.value)
    }


def sensing_beamforming(
    net: Network,
    snap: ChannelSnapshot,
    g_beams: dict,
    *,
    max_iter: int = 8,
    tol: float = 1e-3,
    solver: str = cp.SCS,
    solver_kwargs: dict | None = None,
) -> tuple[dict, dict, list[float]]:
    """Run Algorithm 2. Returns ``(I_beams_cov, i_beams, objective_history)``.

    * ``I_beams_cov[(u, lt)]``: (N, M, M) PSD covariances.
    * ``i_beams[(u, lt)]``: (N, M) rank-one-recovered sensing beam vectors.
    * ``objective_history``: sum-rate per FP iteration (non-decreasing).
    """
    p = net.p
    HH = channel_covariances(snap) / p.sigma2
    # Same per-instance rescaling as comm.py (see there): keep the solver's data
    # O(1) at high Pmax. SINR is invariant, so the optimal I and the rate are too.
    mean_trace = float(np.real(np.trace(HH, axis1=-1, axis2=-2)).mean())
    scale = max(1.0, p.Pmax_W * mean_trace)
    HH = HH / scale
    noise_floor = 1.0 / scale

    AtA = sensing_fisher_matrices(snap)
    crb_thr = crb_thresholds(snap, p)
    G = _comm_covariances(net, g_beams)

    # fixed comm power per (u, n)
    comm_pwr = np.zeros((p.U, p.N))
    for u in range(p.U):
        for lv in range(net.Vu_list[u]):
            comm_pwr[u] += (np.abs(g_beams[(u, lv)]) ** 2).sum(axis=1).real

    I = _crb_feasible_init(net, AtA, crb_thr)

    history: list[float] = []
    for _it in range(max_iter):
        # closed-form Omega update + objective from current (I, iota~SINR)
        Omega = np.zeros((net.V, p.N))
        obj_val = 0.0
        for n in range(p.N):
            Gslot = {(u, lv): G[(u, lv)][n] for u in range(p.U) for lv in range(net.Vu_list[u])}
            Islot = {(u, lt): I[(u, lt)][n] for u in range(p.U) for lt in range(net.Ku_list[u])}
            sig_G, inter_G = _comm_signal_and_interf(net, HH, n, Gslot)
            sens_I = _sensing_interf_val(net, HH, n, Islot)
            Theta = inter_G + sens_I + noise_floor
            sinr = np.maximum(sig_G, 0.0) / np.maximum(Theta, 1e-12)
            # guard Omega > 0 (a slightly negative Theta from sloppy SDR beams would
            # otherwise flip the square(Theta)/(2*Omega) curvature -> DCPError)
            Omega[:, n] = np.maximum(Theta, 1e-9) / np.maximum(sinr, 1e-6)
            obj_val += p.tau * np.sum(np.log2(1.0 + sinr))

        # convex (I, iota) update per slot
        new_I = {k: np.empty((p.N, p.M, p.M), dtype=complex) for k in
                 [(u, lt) for u in range(p.U) for lt in range(net.Ku_list[u])]}
        ok = True
        for n in range(p.N):
            Gslot = {(u, lv): G[(u, lv)][n] for u in range(p.U) for lv in range(net.Vu_list[u])}
            sig_G, inter_G = _comm_signal_and_interf(net, HH, n, Gslot)
            res = _solve_slot(net, HH, AtA, crb_thr, n, Omega[:, n], sig_G, inter_G,
                              comm_pwr[:, n], solver, solver_kwargs, noise_floor)
            if res is None:
                ok = False
                break
            for u in range(p.U):
                for lt in range(net.Ku_list[u]):
                    new_I[(u, lt)][n] = res[(u, lt)]
        if not ok:
            break
        I = new_I
        history.append(obj_val)
        if len(history) >= 2 and abs(history[-1] - history[-2]) <= tol * max(1.0, abs(history[-2])):
            break

    rng = np.random.default_rng(p.seed)
    i_beams = {}
    for u in range(p.U):
        for lt, gk in enumerate(net.target_of_uav[u]):
            arr = np.empty((p.N, p.M), dtype=complex)
            for n in range(p.N):
                g = rank1.recover_beam(I[(u, lt)][n], rng)
                # rank-one recovery can under-satisfy the CRB (enforced on I in
                # the SDR, not on g). Scaling up only works when the recovered
                # beam has nonzero Fisher; if it is Fisher-orthogonal to the
                # sensing channel (top eigenvector of I not aligned with A^H A),
                # scaling is futile -> realign to A^H A's top eigenvector (the
                # min-power direction that hits the CRB) and scale to the threshold.
                fisher = float(np.real(np.vdot(g, AtA[u, gk, n] @ g)))
                if fisher < crb_thr[u, gk, n]:
                    A = (AtA[u, gk, n] + AtA[u, gk, n].conj().T) / 2
                    w_eig, V_eig = np.linalg.eigh(A)
                    lam = max(float(w_eig[-1]), 1e-30)
                    g = V_eig[:, -1] * np.sqrt(crb_thr[u, gk, n] / lam)
                arr[n] = g
            i_beams[(u, lt)] = arr

    return I, i_beams, history
