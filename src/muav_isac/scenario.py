"""Scenario layout, channel snapshot, and end-to-end rate / CRB / energy assembly.

This module wires the physical primitives (``geometry``, ``channel``, ``sensing``,
``energy``, ``rate``) into the multi-UAV topology used by the paper:

* ``Network`` — static layout: UAV start/end, user and target positions, the
  disjoint per-UAV user/target assignment, and the fixed small-scale fading
  draws (one CN(0,I) realization per link, held over the whole flight).
* ``init_trajectory`` — straight-line start -> end at the mid altitude.
* ``ChannelSnapshot`` + ``compute_channels`` — precompute the full cross-channel
  tensor h[u, v, n] and the sensing matrices C, A, beta for a trajectory.
* ``sum_rate`` / ``crb_table`` / ``uav_energy`` — evaluate Eq.3 / Eq.5 / Eq.6-7.

Beam bookkeeping uses local per-UAV indices because V_u and K_u differ across
UAVs: ``g_beams[(u, lv)]`` and ``i_beams[(u, lt)]`` are (N, M) complex arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import channel as ch
from . import energy as en
from . import geometry as geom
from . import rate as rate_mod
from . import sensing as sens
from .config import Params


# --------------------------------------------------------------------------
# Network topology
# --------------------------------------------------------------------------
@dataclass
class Network:
    p: Params
    user_xy: np.ndarray                    # (V, 2)
    target_xy: np.ndarray                  # (K, 2)
    uav_start: np.ndarray                  # (U, 2)
    uav_end: np.ndarray                    # (U, 2)
    user_of_uav: list[list[int]]           # u -> global user indices it serves
    target_of_uav: list[list[int]]         # u -> global target indices it senses
    serving_uav: np.ndarray                # (V,) global user -> serving u
    g_nlos: np.ndarray                     # (U, V, M)  comm NLoS draws
    nlos_mat: np.ndarray                   # (U, K, M, M) sensing NLoS draws

    @property
    def V(self) -> int:
        return self.user_xy.shape[0]

    @property
    def K(self) -> int:
        return self.target_xy.shape[0]

    @property
    def Vu_list(self) -> list[int]:
        return [len(g) for g in self.user_of_uav]

    @property
    def Ku_list(self) -> list[int]:
        return [len(g) for g in self.target_of_uav]

    def user_local(self, global_v: int) -> tuple[int, int]:
        """Global user index -> (serving uav u, local user index lv)."""
        u = int(self.serving_uav[global_v])
        lv = self.user_of_uav[u].index(global_v)
        return u, lv

    def target_local(self, global_k: int) -> tuple[int, int]:
        """Global target index -> (owning uav u, local target index lt)."""
        for u, lst in enumerate(self.target_of_uav):
            if global_k in lst:
                return u, lst.index(global_k)
        raise IndexError(f"target {global_k} not found")


def sample_network(p: Params, rng: np.random.Generator) -> Network:
    """Randomly place UAVs/users/targets and the fixed fading draws."""
    U, M = p.U, p.M

    # per-UAV user/target counts, sampled in the given ranges
    Vu_list = [int(rng.integers(p.Vu_range[0], p.Vu_range[1] + 1)) for _ in range(U)]
    Ku_list = [int(rng.integers(p.Ku_range[0], p.Ku_range[1] + 1)) for _ in range(U)]
    V = sum(Vu_list)
    K = sum(Ku_list)

    # users and targets uniformly in the square area
    user_xy = rng.uniform(0.05 * p.area_m, 0.95 * p.area_m, size=(V, 2))
    target_xy = rng.uniform(0.05 * p.area_m, 0.95 * p.area_m, size=(K, 2))

    # disjoint per-UAV assignment (first Vu_0 users -> UAV 0, etc.)
    user_of_uav: list[list[int]] = []
    idx = 0
    for vu in Vu_list:
        user_of_uav.append(list(range(idx, idx + vu)))
        idx += vu
    target_of_uav: list[list[int]] = []
    idx = 0
    for ku in Ku_list:
        target_of_uav.append(list(range(idx, idx + ku)))
        idx += ku

    serving_uav = np.empty(V, dtype=int)
    for u, lst in enumerate(user_of_uav):
        for v in lst:
            serving_uav[v] = u

    # Paper Sect.IV: "all UAVs share identical starting and destination points".
    # All U UAVs start at one common corner and end at the opposite common corner
    # -- this is what makes trajectory optimization pay off: the common straight
    # line serves no single UAV's user cluster well, so each must detour to its own
    # users and reconverge. (Per-UAV spread start/end makes each UAV's diagonal
    # already pass through its own region, so the straight line is near-optimal
    # and BFWOT ties proposed -- the gap in Fig.1b vanishes.)
    start_pt = np.array([0.1 * p.area_m, 0.1 * p.area_m])
    end_pt = np.array([0.9 * p.area_m, 0.9 * p.area_m])
    uav_start = np.broadcast_to(start_pt, (U, 2)).copy()
    uav_end = np.broadcast_to(end_pt, (U, 2)).copy()

    # fixed small-scale fading draws (one per link, reused across all slots)
    g_nlos = (rng.standard_normal((U, V, M)) + 1j * rng.standard_normal((U, V, M))) / np.sqrt(2.0)
    nlos_mat = np.empty((U, K, M, M), dtype=complex)
    for u in range(U):
        for k in range(K):
            nlos_mat[u, k] = ch.cn0_matrix(rng, M)

    return Network(
        p=p,
        user_xy=user_xy,
        target_xy=target_xy,
        uav_start=uav_start,
        uav_end=uav_end,
        user_of_uav=user_of_uav,
        target_of_uav=target_of_uav,
        serving_uav=serving_uav,
        g_nlos=g_nlos,
        nlos_mat=nlos_mat,
    )


# --------------------------------------------------------------------------
# Trajectory
# --------------------------------------------------------------------------
def init_trajectory(net: Network) -> np.ndarray:
    """Straight-line start -> end at the mid altitude, shape (U, N, 3)."""
    p = net.p
    N, U = p.N, p.U
    H_mid = 0.5 * (p.H_range[0] + p.H_range[1])
    t = np.linspace(0.0, 1.0, N)
    traj = np.empty((U, N, 3))
    for u in range(U):
        traj[u, :, 0] = net.uav_start[u, 0] + t * (net.uav_end[u, 0] - net.uav_start[u, 0])
        traj[u, :, 1] = net.uav_start[u, 1] + t * (net.uav_end[u, 1] - net.uav_start[u, 1])
        traj[u, :, 2] = H_mid
    return traj


def project_trajectory(net: Network, traj: np.ndarray) -> np.ndarray:
    """Project a trajectory onto the hard feasibility constraints of paper P8:
    pin the first waypoint to ``uav_start`` and the last to ``uav_end``
    (``q[0] = q_I``, ``q[N] = q_F``), then clamp xy to the area box and altitude
    to ``H_range``.

    The endpoint correction is applied as a smooth affine warp -- the per-slot
    displacement varies linearly from the start correction to the end correction
    -- so the interior shape (e.g. a DDPG detour toward served users) is preserved
    as far as possible rather than kinked. The endpoints are equalities in the
    trajectory subproblem, so enforcing them by projection in the BCD outer loop
    is more reliable than a soft RL penalty (which is drowned out by the dense
    per-slot rate reward).
    """
    p = net.p
    t = traj.copy()
    N = t.shape[1]
    w = np.linspace(0.0, 1.0, N)[:, None]             # (N, 1) interpolation weight
    for u in range(p.U):
        d0 = net.uav_start[u] - t[u, 0, :2]           # correction at first waypoint
        d1 = net.uav_end[u] - t[u, -1, :2]            # correction at last waypoint
        t[u, :, :2] += (1.0 - w) * d0 + w * d1
        t[u, :, 2] = np.clip(t[u, :, 2], p.H_range[0], p.H_range[1])
    t[:, :, 0] = np.clip(t[:, :, 0], 0.0, p.area_m)
    t[:, :, 1] = np.clip(t[:, :, 1], 0.0, p.area_m)
    return t


# --------------------------------------------------------------------------
# Channel snapshot for a trajectory
# --------------------------------------------------------------------------
@dataclass
class ChannelSnapshot:
    h: np.ndarray          # (U, V, N, M) comm channel from each UAV to each user
    C: np.ndarray          # (U, K, N, M, M) sensing signal matrix
    A: np.ndarray          # (U, K, N, M, M) dC/dphi derivative
    beta: np.ndarray       # (U, K, N) round-trip sensing gain
    traj: np.ndarray       # (U, N, 3)


def compute_channels(net: Network, traj: np.ndarray) -> ChannelSnapshot:
    """Precompute all channels / sensing matrices for ``traj`` (U, N, 3)."""
    p = net.p
    U, V, K, N, M = p.U, net.V, net.K, p.N, p.M

    h = np.empty((U, V, N, M), dtype=complex)
    C = np.empty((U, K, N, M, M), dtype=complex)
    A = np.empty((U, K, N, M, M), dtype=complex)
    beta = np.empty((U, K, N))

    for u in range(U):
        for n in range(N):
            uav_xy = traj[u, n, :2]
            H = float(traj[u, n, 2])
            for v in range(V):
                h[u, v, n] = ch.comm_channel(p, uav_xy, net.user_xy[v], H, net.g_nlos[u, v])
            for k in range(K):
                C[u, k, n] = ch.sensing_channel_matrix(
                    p, uav_xy, net.target_xy[k], H, net.nlos_mat[u, k]
                )
                A[u, k, n] = sens.A_matrix(p, uav_xy, net.target_xy[k], H, net.nlos_mat[u, k])
                d = float(geom.distance3d(uav_xy, net.target_xy[k], H))
                beta[u, k, n] = sens.beta(p, d)

    return ChannelSnapshot(h=h, C=C, A=A, beta=beta, traj=traj)


# --------------------------------------------------------------------------
# Evaluation: rate, CRB, energy
# --------------------------------------------------------------------------
def per_user_rates(
    net: Network, snap: ChannelSnapshot, g_beams: dict, i_beams: dict
) -> np.ndarray:
    """Per-user per-slot rate R_v[n] (without tau), shape (V, N)."""
    p = net.p
    V, N = net.V, p.N
    R = np.empty((V, N))
    for v in range(V):
        su, lv = net.user_local(v)
        for n in range(N):
            h_cross = snap.h[:, v, n]                     # (U, M)
            g_own = g_beams[(su, lv)][n]
            signal = rate_mod.received_power_vec(h_cross[su], g_own)

            interf = 0.0
            for uu in range(p.U):
                for lu, _v in enumerate(net.user_of_uav[uu]):
                    if uu == su and lu == lv:
                        continue
                    interf += rate_mod.received_power_vec(h_cross[uu], g_beams[(uu, lu)][n])
                for lt, _k in enumerate(net.target_of_uav[uu]):
                    interf += rate_mod.received_power_vec(h_cross[uu], i_beams[(uu, lt)][n])

            R[v, n] = rate_mod.rate(signal, interf, p.sigma2)
    return R


def sum_rate(
    net: Network, snap: ChannelSnapshot, g_beams: dict, i_beams: dict
) -> float:
    """Total system sum-rate sum_{n,u,v} tau * R_v^u[n] (paper Eq.3 objective)."""
    R = per_user_rates(net, snap, g_beams, i_beams)
    return float(net.p.tau * R.sum())


# --------------------------------------------------------------------------
# Single-position helpers (used by the DDPG trajectory environment)
# --------------------------------------------------------------------------
def comm_channels_at(net: Network, uav_xy: np.ndarray, uav_H: np.ndarray) -> np.ndarray:
    """Communication channels from every UAV to every user at one position set.

    ``uav_xy``: (U, 2); ``uav_H``: (U,). Returns h of shape (U, V, M).
    """
    p = net.p
    U, V, M = p.U, net.V, p.M
    h = np.empty((U, V, M), dtype=complex)
    for u in range(U):
        for v in range(V):
            h[u, v] = ch.comm_channel(p, uav_xy[u], net.user_xy[v], float(uav_H[u]), net.g_nlos[u, v])
    return h


def comm_channels(net: Network, traj: np.ndarray) -> np.ndarray:
    """Communication channels h[u, v, n] over a whole trajectory (U, V, N, M).

    Lighter than :func:`compute_channels` (skips the sensing C/A/beta used only
    by the CRB), so DDPG's per-episode rate evaluation stays cheap.
    """
    p = net.p
    U, V, N, M = p.U, net.V, p.N, p.M
    h = np.empty((U, V, N, M), dtype=complex)
    for u in range(U):
        for n in range(N):
            uav_xy = traj[u, n, :2]
            H = float(traj[u, n, 2])
            for v in range(V):
                h[u, v, n] = ch.comm_channel(p, uav_xy, net.user_xy[v], H, net.g_nlos[u, v])
    return h


def sum_rate_h(net: Network, h: np.ndarray, g_beams: dict, i_beams: dict) -> float:
    """Sum rate (with tau) from a precomputed ``h`` (U, V, N, M)."""
    p = net.p
    V, N = net.V, p.N
    total = 0.0
    for v in range(V):
        su, lv = net.user_local(v)
        for n in range(N):
            h_cross = h[:, v, n]
            signal = rate_mod.received_power_vec(h_cross[su], g_beams[(su, lv)][n])
            interf = 0.0
            for uu in range(p.U):
                for lu, _gv in enumerate(net.user_of_uav[uu]):
                    if uu == su and lu == lv:
                        continue
                    interf += rate_mod.received_power_vec(h_cross[uu], g_beams[(uu, lu)][n])
                for lt, _k in enumerate(net.target_of_uav[uu]):
                    interf += rate_mod.received_power_vec(h_cross[uu], i_beams[(uu, lt)][n])
            total += rate_mod.rate(signal, interf, p.sigma2)
    return p.tau * total


def slot_sum_rate(
    net: Network, h_uvm: np.ndarray, g_beams: dict, i_beams: dict
) -> float:
    """Sum rate (with tau) at one slot, given channels ``h_uvm`` (U, V, M).

    ``g_beams[(u, lv)]`` / ``i_beams[(u, lt)]`` are single-slot (M,) vectors.
    """
    p = net.p
    total = 0.0
    for v in range(net.V):
        su, lv = net.user_local(v)
        h_cross = h_uvm[:, v]                                  # (U, M)
        signal = rate_mod.received_power_vec(h_cross[su], g_beams[(su, lv)])
        interf = 0.0
        for uu in range(p.U):
            for lu, _gv in enumerate(net.user_of_uav[uu]):
                if uu == su and lu == lv:
                    continue
                interf += rate_mod.received_power_vec(h_cross[uu], g_beams[(uu, lu)])
            for lt, _k in enumerate(net.target_of_uav[uu]):
                interf += rate_mod.received_power_vec(h_cross[uu], i_beams[(uu, lt)])
        total += rate_mod.rate(signal, interf, p.sigma2)
    return p.tau * total


def realign_comm_beams(net: Network, h: np.ndarray, g_stale: dict) -> dict:
    """Re-point each comm beam to its serving channel, keeping the FP power.

    The trajectory subproblem (DDPG reward and best-trajectory selection) must be
    scored on *position* quality, not on how well the stale FP beams -- solved for
    the previous trajectory -- still couple to the channel. For each beam (u, lv)
    serving global user v we replace its direction with the maximum-ratio direction
    ``h[u, v] / ||h[u, v]||`` and rescale to ``||g_stale[(u, lv)]||``, so per-beam
    transmit power is unchanged but the beam always faces where the channel now is.
    Without this, the surrogate rate rewards "stay where the old beams still point"
    -- a straight-line attractor -- and DDPG has no gradient toward better positions.

    ``h`` is either (U, V, M) (one position set, as from :func:`comm_channels_at`)
    or (U, V, N, M) (a full trajectory, as from :func:`comm_channels`); the returned
    dict matches ``g_stale``'s per-value shape. Sensing beams are left stale -- their
    mis-aim enters only as a second-order interference term.
    """
    g = {}
    for u in range(net.p.U):
        for lv, v in enumerate(net.user_of_uav[u]):
            hdir = h[u, v]                                       # (M,) or (N, M)
            nrm = np.linalg.norm(hdir, axis=-1, keepdims=True)
            power = np.linalg.norm(g_stale[(u, lv)], axis=-1, keepdims=True)
            g[(u, lv)] = (hdir / np.maximum(nrm, 1e-12)) * power
    return g


def crb_table(net: Network, snap: ChannelSnapshot, i_beams: dict) -> dict:
    """CRB(phi) per (u, local target lt, slot n) -> {(u, lt): (N,)} in [rad^2]."""
    p = net.p
    out: dict[tuple[int, int], np.ndarray] = {}
    for u in range(p.U):
        for lt, gk in enumerate(net.target_of_uav[u]):
            arr = np.empty(p.N)
            for n in range(p.N):
                arr[n] = sens.crb(p, snap.A[u, gk, n], i_beams[(u, lt)][n], snap.beta[u, gk, n])
            out[(u, lt)] = arr
    return out


def uav_energy(
    net: Network, snap: ChannelSnapshot, g_beams: dict, i_beams: dict
) -> dict[str, np.ndarray]:
    """Per-UAV energy accounting.

    Returns ``{"cs": (U, N)}`` transmit energy and ``{"fly": (U, N-1)}`` flying
    energy over slot transitions. Sum each over slots for the per-UAV totals
    used in constraint (8d).
    """
    p = net.p
    U, N = p.U, p.N
    cs = np.zeros((U, N))
    for u in range(U):
        for n in range(N):
            tx = 0.0
            for lu, _v in enumerate(net.user_of_uav[u]):
                tx += float((np.abs(g_beams[(u, lu)][n]) ** 2).sum())
            for lt, _k in enumerate(net.target_of_uav[u]):
                tx += float((np.abs(i_beams[(u, lt)][n]) ** 2).sum())
            cs[u, n] = en.E_cs(p, tx)
    fly = en.trajectory_flying_energy(p, snap.traj)        # (U, N-1)
    return {"cs": cs, "fly": fly}
