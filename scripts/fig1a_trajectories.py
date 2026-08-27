"""Fig.1(a): UAV trajectories -- DDPG vs. A2C.

Builds one scenario, solves the beamforming once, then optimizes the trajectory
with DDPG (Alg.3) and with A2C, and plots both in 2D (top-down, with users and
targets) and 3D (showing altitude). Expected: DDPG paths stay closer to the
served users than A2C (paper Fig.1a).

Usage:
    uv run python scripts/fig1a_trajectories.py [--N 20] [--episodes 400] [--sigma2-dBm -80]
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from muav_isac import scenario as sc
from muav_isac.beamforming import comm, sensing
from muav_isac.config import default_params
from muav_isac.trajectory import a2c, ddpg
from muav_isac.trajectory.env import TrajectoryEnv

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--T", type=float, default=60.0,
                    help="flight period [s]; default 60 (so T=60, N=20 gives tau=3s).")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--bf-iters", type=int, default=5)
    ap.add_argument("--sigma2-dBm", type=float, default=default_params().sigma2_dBm,
                    help="AWGN power [dBm]. The paper omits this; default is paper-calibrated.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    p = dataclasses.replace(default_params(), N=args.N, T=args.T, sigma2_dBm=args.sigma2_dBm)
    net = sc.sample_network(p, np.random.default_rng(args.seed))
    traj0 = sc.init_trajectory(net)
    snap = sc.compute_channels(net, traj0)

    i0 = {(u, lt): np.zeros((p.N, p.M), dtype=complex)
          for u in range(p.U) for lt in range(net.Ku_list[u])}
    _, g, _ = comm.comm_beamforming(net, snap, i0, max_iter=args.bf_iters)
    _, i, _ = sensing.sensing_beamforming(net, snap, g, max_iter=args.bf_iters)

    def rate_of(t, gb=g, ib=i):
        return sc.sum_rate_h(net, sc.comm_channels(net, t), gb, ib)

    env = TrajectoryEnv(net, g, i, reward_scale=10.0)
    print("training DDPG ...")
    _ag, _ret, traj_ddpg = ddpg.train_ddpg(
        env, episodes=args.episodes, seed=args.seed, eval_rate_fn=rate_of
    )
    print("training A2C ...")
    _m, _r, traj_a2c = a2c.train_a2c(
        env, episodes=args.episodes, seed=args.seed, eval_rate_fn=rate_of
    )
    # Enforce hard endpoint/box constraints (paper P8) by projection -- the RL
    # surrogate is under no obligation to return to base on its own. Rates are
    # reported on the projected (feasible) trajectories.
    traj_ddpg = sc.project_trajectory(net, traj_ddpg)
    traj_a2c = sc.project_trajectory(net, traj_a2c)
    print(f"DDPG rate={rate_of(traj_ddpg):.2f}  A2C rate={rate_of(traj_a2c):.2f}")

    fig = plt.figure(figsize=(9.5, 4.2))

    # --- 2D top-down ---
    ax2d = fig.add_subplot(1, 2, 1)
    _plot_topdown(ax2d, net, traj_ddpg, traj_a2c)

    # --- 3D (with altitude) ---
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    _plot_3d(ax3d, net, traj_ddpg, traj_a2c)

    fig.suptitle("Fig. 1(a): UAV trajectories — DDPG (solid) vs A2C (dashed)")
    fig.tight_layout()
    out_png = os.path.join(RESULTS, "fig1a_trajectories.png")
    fig.savefig(out_png, dpi=130)
    np.savez(
        os.path.join(RESULTS, "fig1a_trajectories.npz"),
        ddpg=traj_ddpg, a2c=traj_a2c,
        user_xy=net.user_xy, target_xy=net.target_xy,
        uav_start=net.uav_start, uav_end=net.uav_end,
        N=args.N, T=args.T, sigma2_dBm=args.sigma2_dBm,
    )
    print(f"saved {out_png}")


def _plot_topdown(ax, net, traj_d, traj_a):
    for u in range(net.p.U):
        c = COLORS[u % len(COLORS)]
        ax.plot(traj_d[u, :, 0], traj_d[u, :, 1], "-", color=c, label=f"UAV{u + 1} DDPG")
        ax.plot(traj_a[u, :, 0], traj_a[u, :, 1], "--", color=c, label=f"UAV{u + 1} A2C")
        ax.scatter(*net.uav_start[u], marker="s", color=c, edgecolor="k", zorder=5)
        ax.scatter(*net.uav_end[u], marker="*", color=c, edgecolor="k", s=90, zorder=5)
        # users / targets colored by their serving (owning) UAV, so the disjoint
        # assignment is visible: each UAV's trajectory matches its own users/targets.
        us = net.user_xy[net.user_of_uav[u]]
        ax.scatter(us[:, 0], us[:, 1], c=[c], s=55, marker="o", edgecolor="k", zorder=4)
        tg = net.target_xy[net.target_of_uav[u]]
        ax.scatter(tg[:, 0], tg[:, 1], c=[c], s=75, marker="x", linewidths=2, zorder=4)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("2D — ○ users, × targets (colored by serving UAV)")
    ax.set_xlim(0, net.p.area_m); ax.set_ylim(0, net.p.area_m)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.legend(fontsize=6, loc="upper right", ncol=2)


def _plot_3d(ax, net, traj_d, traj_a):
    for u in range(net.p.U):
        c = COLORS[u % len(COLORS)]
        ax.plot(traj_d[u, :, 0], traj_d[u, :, 1], traj_d[u, :, 2], "-", color=c)
        ax.plot(traj_a[u, :, 0], traj_a[u, :, 1], traj_a[u, :, 2], "--", color=c)
        us = net.user_xy[net.user_of_uav[u]]
        ax.scatter(us[:, 0], us[:, 1], np.zeros(len(us)), c=[c], s=40, marker="o", edgecolor="k")
        tg = net.target_xy[net.target_of_uav[u]]
        ax.scatter(tg[:, 0], tg[:, 1], np.zeros(len(tg)), c=[c], s=55, marker="x", linewidths=2)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("altitude (m)")
    ax.set_title("3D — ○ users, × targets (colored by serving UAV)")
    ax.set_zlim(0, net.p.H_range[1])   # 0..200: ground users/targets (z=0) + full altitude band


if __name__ == "__main__":
    main()
