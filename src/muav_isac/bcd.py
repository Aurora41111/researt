"""Outer block-coordinate-descent loop (paper Algorithm 1 + 2 + 3).

Alternate the three subproblems until the sum rate converges:

    given trajectory o  ->  comm BF  (Alg.1)  ->  sensing BF (Alg.2)
    given beams g, i    ->  trajectory via DDPG (Alg.3)

The rate is measured *after* the two FP solves on the current trajectory, so
``rate_history`` is the convergence curve plotted in Fig.1c. The FP subproblems
are monotone, and DDPG is run with rate-based best-trajectory selection, so the
outer rate is (stochastically) non-decreasing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import scenario as sc
from .beamforming import comm, sensing
from .beamforming.cvxbuild import crb_thresholds, sensing_fisher_matrices
from .config import Params
from .scenario import Network
from .trajectory.ddpg import train_ddpg
from .trajectory.env import TrajectoryEnv


@dataclass
class BCDResult:
    traj: np.ndarray                       # (U, N, 3) best trajectory
    g: dict                                # best comm beams {(u, lv): (N, M)}
    i: dict                                # best sensing beams {(u, lt): (N, M)}
    rate_history: list[float] = field(default_factory=list)
    final_rate: float = 0.0


def _zero_sensing(net: Network) -> dict:
    p = net.p
    return {(u, lt): np.zeros((p.N, p.M), dtype=complex)
            for u in range(p.U) for lt in range(net.Ku_list[u])}


def bcd_solve(
    net: Network,
    init_traj: np.ndarray | None = None,
    *,
    max_iter: int = 6,
    bf_iters: int = 6,
    ddpg_episodes: int = 300,
    ddpg_kwargs: dict | None = None,
    reward_scale: float = 10.0,
    seed: int = 0,
    tol: float = 1e-3,
    early_stop: bool = True,
    verbose: bool = False,
) -> BCDResult:
    """Run the full BCD. Returns a :class:`BCDResult` with rate history.

    Robust monotone BCD: ``rate_history`` records the best rate seen so far, so it
    is non-decreasing even when DDPG (stochastic) or the FP solver (numerical,
    can fail on extreme trajectories) misbehaves. The FP solve is wrapped so that
    on failure we fall back to the best beams; the DDPG proposal is clipped to the
    area/altitude bounds and judged by the next iteration's re-solve.
    """
    p = net.p
    traj = init_traj.copy() if init_traj is not None else sc.init_trajectory(net)

    # Initial best (always feasible): MRT comm beams + CRB-feasible sensing beams
    # on the init trajectory. Using *zero* sensing here would be infeasible (it
    # violates the CRB) yet would score a higher comm rate than any feasible point
    # at tight Gamma, corrupting the monotone track -- so the baseline must itself
    # satisfy the CRB. The FP/DDPG loop then improves from here.
    snap0 = sc.compute_channels(net, traj)
    g_init = _mrt_beams(net, snap0)
    i_init = _crb_feasible_beams(net, snap0)
    best = {
        "rate": sc.sum_rate(net, snap0, g_init, i_init),
        "traj": traj.copy(),
        "g": g_init,
        "i": i_init,
    }
    i_beams = {k: v.copy() for k, v in i_init.items()}
    rate_history: list[float] = []

    for it in range(max_iter):
        snap = sc.compute_channels(net, traj)
        try:
            _, g_beams, _ = comm.comm_beamforming(net, snap, i_beams, max_iter=bf_iters)
            _, i_beams, _ = sensing.sensing_beamforming(net, snap, g_beams, max_iter=bf_iters)
            rate = sc.sum_rate(net, snap, g_beams, i_beams)
        except Exception:  # noqa: BLE001 - FP can fail many ways on extreme trajectories
            # (cvxpy DCP/SolverError, numpy linalg, ...); fall back to the best so far.
            g_beams, i_beams, rate = best["g"], best["i"], best["rate"]

        if rate > best["rate"]:
            best = {"rate": rate, "traj": traj.copy(), "g": g_beams, "i": i_beams}


        if verbose:
            print(f"[BCD] iter {it}: sum rate = {rate:.4f} (best {best['rate']:.4f})")



        # ---- trajectory update via DDPG (beams fixed to the best so far) ----
        env = TrajectoryEnv(net, best["g"], best["i"], reward_scale=reward_scale)

        # # Select the best trajectory by rate with comm beams RE-POINTED to each
        # # candidate trajectory's own channels (keeping FP per-beam power). Using the
        # # stale beams here would make the straight-line init -- the beams' home -- the
        # # surrogate optimum, so DDPG could never beat it and proposed would tie BFWOT.
        # def _eval_rate(t, gb=best["g"], ib=best["i"]):
        #     h = sc.comm_channels(net, t)
        #     return sc.sum_rate_h(net, h, sc.realign_comm_beams(net, h, gb), ib)

        def _eval_rate(t, gb=best["g"], ib=best["i"]):
            # DDPG候选轨迹最终一定会经过这个投影，
            # 因此候选轨迹在选择阶段也必须先投影再评价
            t = sc.project_trajectory(net, t)

            h = sc.comm_channels(net, t)

            return sc.sum_rate_h(
                net,
                h,
                sc.realign_comm_beams(net, h, gb),
                ib
            )
        eval_fn = _eval_rate
        # _agent, _rets, traj_new = train_ddpg(
        #     env, episodes=ddpg_episodes, seed=seed + it, eval_rate_fn=eval_fn,
        #     **(ddpg_kwargs or {}),
        # )
        # # Enforce the hard endpoint/box constraints by projection (paper P8:
        # # q[0]=q_I, q[N]=q_F). DDPG is only a surrogate; feasibility is the outer
        # # loop's job, not the RL reward's.
        # traj = sc.project_trajectory(net, traj_new)

        # 当前历史最优轨迹在同一个代理评价函数下的速率
        surrogate_current = eval_fn(best["traj"])

        # DDPG训练；候选轨迹在train_ddpg内部也会通过eval_fn先投影再评价
        _agent, _rets, traj_new = train_ddpg(
            env,
            episodes=ddpg_episodes,
            seed=seed + it,
            eval_rate_fn=eval_fn,
            log_every=100,
            **(ddpg_kwargs or {}),
        )
        # DDPG最终返回的轨迹真正进入BCD前也进行一次投影
        traj_projected = sc.project_trajectory(net, traj_new)

        # 新轨迹在相同代理目标下的速率
        surrogate_new = eval_fn(traj_projected)

        if verbose:
            print(
                f"[DDPG] iter={it}, "
                f"surrogate_current={surrogate_current:.4f}, "
                f"surrogate_new={surrogate_new:.4f}"
            )

        # 默认认为DDPG候选轨迹无效
        candidate_valid = False
        true_candidate_rate = -np.inf
        g_candidate = None
        i_candidate = None

        # 只有代理目标先变好，才值得做昂贵的真实BF验证
        if surrogate_new > surrogate_current:

            snap_candidate = sc.compute_channels(
                net,
                traj_projected
            )

            # 根据 DDPG 新轨迹重新计算 CRB 可行的感知初始波束
            i_candidate_init = _crb_feasible_beams(
                net,
                snap_candidate
            )

            try:
                _, g_candidate, comm_hist_candidate = comm.comm_beamforming(
                    net,
                    snap_candidate,
                    i_candidate_init,  # 不再使用旧轨迹 best["i"]
                    max_iter=bf_iters,
                )

                _, i_candidate, sens_hist_candidate = sensing.sensing_beamforming(
                    net,
                    snap_candidate,
                    g_candidate,
                    max_iter=bf_iters,
                )

                candidate_valid = (
                        len(comm_hist_candidate) > 0
                        and len(sens_hist_candidate) > 0
                )



                if candidate_valid:
                    true_candidate_rate = sc.sum_rate(
                        net,
                        snap_candidate,
                        g_candidate,
                        i_candidate,
                    )

            except Exception:
                candidate_valid = False
                true_candidate_rate = -np.inf

            if verbose:
                print(
                    f"[DDPG TRUE CHECK] iter={it}, "
                    f"current_true={best['rate']:.4f}, "
                    f"candidate_true={true_candidate_rate:.4f}, "
                    f"valid={candidate_valid}"
                )

        # ==========================================
        # 最终决定是否接受DDPG轨迹
        # ==========================================

        if candidate_valid and true_candidate_rate > best["rate"]:

            # DDPG候选是真正可行且真实速率更高
            best = {
                "rate": true_candidate_rate,
                "traj": traj_projected.copy(),
                "g": g_candidate,
                "i": i_candidate,
            }

            traj = traj_projected.copy()

            # 下一轮通信优化用新的感知波束
            i_beams = {
                k: v.copy()
                for k, v in i_candidate.items()
            }

            if verbose:
                print(
                    f"[DDPG ACCEPT] iter={it}, "
                    f"new_best={true_candidate_rate:.4f}"
                )

        else:

            # DDPG轨迹不合格或真实性能没提升
            traj = best["traj"].copy()

            i_beams = {
                k: v.copy()
                for k, v in best["i"].items()
            }

            if verbose:
                print(
                    f"[DDPG REJECT] iter={it}"
                )
        # ==========================================
        # 本轮BCD全部完成以后，再记录历史最优值
        # ==========================================
        rate_history.append(best["rate"])

        if verbose:
            print(
                f"[BCD FINAL] iter={it}, "
                f"best_rate={best['rate']:.4f}"
            )

        # 整个BCD轮次完成以后再判断收敛
        if (
            early_stop
            and len(rate_history) >= 4
            and abs(rate_history[-1] - rate_history[-2])
            <= tol * max(1.0, abs(rate_history[-2]))
        ):
            break


    return BCDResult(
        traj=best["traj"], g=best["g"], i=best["i"],
        rate_history=rate_history, final_rate=rate_history[-1],
    )


def _mrt_beams(net: Network, snap, per_beam_power: float = 0.1) -> dict:
    """Maximum-ratio comm beams (aligned to the serving channel) -- a stable fallback."""
    p = net.p
    g = {}
    for u in range(p.U):
        per = per_beam_power / max(net.Vu_list[u], 1)
        for lv, v in enumerate(net.user_of_uav[u]):
            h = snap.h[u, v]
            nrm = np.linalg.norm(h, axis=1, keepdims=True)
            g[(u, lv)] = (h / np.maximum(nrm, 1e-12)) * np.sqrt(per)
    return g


def _crb_feasible_beams(net: Network, snap) -> dict:
    """CRB-feasible sensing beams: along A^H A's top eigenvector, scaled to just
    meet the CRB threshold (Eq.5/9b), power-clamped to Pmax. A feasible baseline."""
    p = net.p
    AtA = sensing_fisher_matrices(snap)                      # (U, K, N, M, M)
    thr = crb_thresholds(snap, p)                            # (U, K, N)
    beams = {}
    for u in range(p.U):
        for lt, gk in enumerate(net.target_of_uav[u]):
            arr = np.empty((p.N, p.M), dtype=complex)
            for n in range(p.N):
                a_mat = (AtA[u, gk, n] + AtA[u, gk, n].conj().T) / 2
                w, V = np.linalg.eigh(a_mat)
                vmax = V[:, -1]
                lam = max(w[-1], 1e-12)
                scale = np.sqrt(min(thr[u, gk, n] / lam, p.Pmax_W))   # meet CRB, clamp power
                arr[n] = scale * vmax
            beams[(u, lt)] = arr
    return beams
