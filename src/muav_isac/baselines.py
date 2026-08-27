"""Separated-design baselines for Fig.1b: TWOBF and BFWOT.

* **TWOBF** (Trajectory-Without-(optimized)-Beamforming): trajectory is optimized
  by DDPG, but the beams use a cheap heuristic (maximum-ratio comm + equal-power
  sensing) instead of the FP solve. Captures the loss from skipping beamforming
  optimization.
* **BFWOT** (Beamforming-Without-Trajectory): beams are FP-optimized, but the
  trajectory is frozen at the straight-line initialization. Captures the loss
  from skipping trajectory optimization.

Both return a :class:`bcd.BCDResult` so they share an interface with the proposed
joint method.
"""

from __future__ import annotations

import numpy as np

from . import scenario as sc
from .bcd import BCDResult, _crb_feasible_beams, _zero_sensing
from .beamforming import comm, sensing
from .scenario import Network
from .trajectory.ddpg import train_ddpg
from .trajectory.env import TrajectoryEnv


def _mrt_beams(net: Network, snap, per_beam_power) -> dict:
    """Comm beam aligned to the serving channel (no FP), equal power per user.

    ``per_beam_power`` is the *total* comm transmit power per UAV, broadcastable
    to (U, N): a scalar (same power everywhere), a (U,) vector (per-UAV, constant
    over slots), or a (U, N) array (per-UAV per-slot). It is split equally across
    that UAV's served users, so each user's beam carries ``per_beam_power / V_u``.
    """
    p = net.p
    power = np.broadcast_to(np.asarray(per_beam_power, dtype=float), (p.U, p.N))
    g = {}
    for u in range(p.U):
        per_user = power[u] / max(net.Vu_list[u], 1)           # (N,)
        for lv, v in enumerate(net.user_of_uav[u]):
            h = snap.h[u, v]                                   # (N, M)
            nrm = np.linalg.norm(h, axis=1, keepdims=True)     # (N, 1)
            g[(u, lv)] = (h / np.maximum(nrm, 1e-12)) * np.sqrt(per_user)[:, None]
    return g


def _sensing_power(net: Network, i_beams: dict) -> np.ndarray:
    """Per-UAV per-slot sensing transmit power ||i_{u,lt}[n]||^2 summed over
    targets, shape (U, N). Used to size the comm beam to the *remaining* budget."""
    p = net.p
    out = np.zeros((p.U, p.N))
    for u in range(p.U):
        for lt, _k in enumerate(net.target_of_uav[u]):
            out[u] += (np.abs(i_beams[(u, lt)]) ** 2).sum(axis=1)   # sum over M -> (N,)
    return out


def twobf(
    net: Network,
    init_traj: np.ndarray | None = None,
    *,
    max_iter: int = 4,
    ddpg_episodes: int = 300,
    reward_scale: float = 10.0,
    seed: int = 0,
    verbose: bool = False,
) -> BCDResult:
    """Trajectory-optimized, heuristic (non-FP) beamforming.

    The comm beam is maximum-ratio (no FP interference management) but uses the
    *full remaining power budget* ``Pmax - sensing_power`` per UAV per slot -- so
    the gap to the proposed method reflects the loss from skipping FP beamforming,
    not an artificial power cap. (An earlier version hardcoded 0.1 W = 20 dBm,
    which against a 70 dBm budget left TWOBF starved and near rate-0.)
    """
    p = net.p
    traj = init_traj.copy() if init_traj is not None else sc.init_trajectory(net)
    rate_history: list[float] = []
    best = {"rate": -np.inf, "traj": traj.copy(), "g": None, "i": None}

    for it in range(max_iter):
        snap = sc.compute_channels(net, traj)
        i = _crb_feasible_beams(net, snap)        # naive but CRB-feasible sensing
        comm_power = np.maximum(p.Pmax_W - _sensing_power(net, i), 1e-6)  # (U, N)
        g = _mrt_beams(net, snap, comm_power)     # MRT comm on the remaining budget
        rate = sc.sum_rate(net, snap, g, i)
        rate_history.append(rate)
        if rate > best["rate"]:
            best = {"rate": rate, "traj": traj.copy(), "g": g, "i": i}
        if verbose:
            print(f"[TWOBF] iter {it}: sum rate = {rate:.4f}")

        env = TrajectoryEnv(net, g, i, reward_scale=reward_scale)
        eval_fn = lambda t, gb=g, ib=i: sc.sum_rate_h(net, sc.comm_channels(net, t), gb, ib)
        _a, _r, traj = train_ddpg(
            env, episodes=ddpg_episodes, seed=seed + it, eval_rate_fn=eval_fn
        )

    return BCDResult(
        traj=best["traj"], g=best["g"], i=best["i"],
        rate_history=rate_history, final_rate=rate_history[-1],
    )


def bfwot(
    net: Network,
    init_traj: np.ndarray | None = None,
    *,
    bf_iters: int = 6,
    verbose: bool = False,
) -> BCDResult:
    """FP-optimized beamforming on a *fixed* straight-line trajectory (no DDPG).

    The beamforming block (Alg.1 + Alg.2) is iterated to convergence -- this is
    the paper's "beamforming without trajectory" baseline. A single comm->sensing
    pass is not enough: the comm beams are designed for the *current* sensing
    interference, so they must be re-solved once sensing adds its interference.
    Iterating the two FP subproblems (as the full BCD does across its outer
    loops) lets FP overtake a heuristic MRT+CRB-feasible design.
    """
    traj = init_traj.copy() if init_traj is not None else sc.init_trajectory(net)
    snap = sc.compute_channels(net, traj)

    # Robust baseline (always feasible): MRT comm + CRB-feasible sensing. At very
    # tight Gamma the FP sensing subproblem can be infeasible (needs more power
    # than Pmax to hit the CRB) or cvxpy can return SolverError -- without this
    # fallback BFWOT would return rate 0 there. The FP loop improves from here.
    g_best = _mrt_beams(net, snap, 0.1)
    i_best = _crb_feasible_beams(net, snap)
    best_rate = sc.sum_rate(net, snap, g_best, i_best)
    rate_history = [best_rate]

    i = {k: v.copy() for k, v in i_best.items()}
    g = g_best
    for it in range(bf_iters):
        try:
            _, g_new, _ = comm.comm_beamforming(net, snap, i, max_iter=bf_iters)
            _, i_new, _ = sensing.sensing_beamforming(net, snap, g_new, max_iter=bf_iters)
            g, i = g_new, i_new
            rate = sc.sum_rate(net, snap, g, i)
        except Exception:  # noqa: BLE001 - FP can fail on infeasible CRB / numerics
            rate = best_rate
        if rate > best_rate:
            best_rate, g_best, i_best = rate, g, i
        rate_history.append(best_rate)                       # monotone
        if verbose:
            print(f"[BFWOT] bf iter {it}: sum rate = {rate:.4f} (best {best_rate:.4f})")
        if len(rate_history) >= 2 and abs(rate_history[-1] - rate_history[-2]) <= 1e-3 * max(1.0, abs(rate_history[-2])):
            break
    return BCDResult(traj=traj, g=g_best, i=i_best, rate_history=rate_history, final_rate=rate_history[-1])
