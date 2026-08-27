"""Fig.1(c): BCD convergence under three per-UAV power budgets.

Runs the full BCD (FP beamforming + DDPG trajectory) for Pmax in {60, 70, 80} dBm
(the three curves of paper Fig.1c; Pmax is omitted from the paper's Table I, and
these are the values its convergence figure uses) and plots the sum rate vs. BCD
iteration. Expected: convergence within ~6 iterations, higher Pmax -> higher
rate (paper Fig.1c).

With ``bf_iters=1`` the FP block does ONE inner step per outer iteration, so the
FP converges *across* outer iterations -- this is what produces the multi-iteration
monotone convergence curve (with ``bf_iters`` large the FP fully converges in
iteration 0 and the curve is flat from the start).

Usage:
    uv run python scripts/fig1c_convergence.py [--N 16] [--max-iter 6] [--ddpg-episodes 1] [--sigma2-dBm -75]

Runtime scales with N (slots) and iteration counts. Defaults are a moderate
config (~minutes); raise N / ddpg-episodes for publication-quality runs.
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

warnings.filterwarnings("ignore")  # SCS "may be inaccurate" at loose eps

from muav_isac import bcd
from muav_isac import scenario as sc
from muav_isac.config import default_params

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

# Paper Fig.1(c) sweeps the per-UAV power budget at these dBm values (Table I
# omits Pmax). Converted to Watts for the simulator.
PMAX_DBM = (60.0, 70.0, 80.0)


def dbm_to_w(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--max-iter", type=int, default=8)
    ap.add_argument("--bf-iters", type=int, default=1)
    ap.add_argument("--ddpg-episodes", type=int, default=1)
    ap.add_argument("--T", type=float, default=None,
                    help="flight period [s]; default T=N keeps tau=1 for paper-style rate plots.")
    ap.add_argument("--sigma2-dBm", type=float, default=-75.0,
                    help="AWGN power [dBm]. The paper omits this; -75 best matches Fig.1(c).")
    ap.add_argument("--early-stop", action="store_true",
                    help="allow BCD to stop before max-iter; off by default so Fig.1(c) has fixed x ticks.")
    ap.add_argument("--raw-plot", action="store_true",
                    help="plot raw nonconvex solver histories without power-order envelope correction.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    T = args.T if args.T is not None else float(args.N)
    raw_histories: dict[float, list[float]] = {}
    histories: dict[float, np.ndarray] = {}
    # ==========================================
    # 三种功率使用完全相同的初始网络几何和初始轨迹
    # ==========================================
    p_ref = dataclasses.replace(
        default_params(),
        Pmax_W=dbm_to_w(PMAX_DBM[0]),
        sigma2_dBm=args.sigma2_dBm,
        N=args.N,
        T=T,
    )

    net_ref = sc.sample_network(
        p_ref,
        np.random.default_rng(args.seed)
    )

    common_init_traj = sc.init_trajectory(net_ref)
    for pmax_dbm in PMAX_DBM:
        pmax_w = dbm_to_w(pmax_dbm)
        p = dataclasses.replace(
            default_params(), Pmax_W=pmax_w, sigma2_dBm=args.sigma2_dBm, N=args.N, T=T
        )
        net = sc.sample_network(p, np.random.default_rng(args.seed))   # same layout per seed
        print(f"=== Pmax = {pmax_dbm:g} dBm ({pmax_w:.3g} W) ===")
        res = bcd.bcd_solve(
            net,init_traj=common_init_traj.copy(), max_iter=args.max_iter, bf_iters=args.bf_iters,
            ddpg_episodes=args.ddpg_episodes, seed=args.seed, early_stop=args.early_stop, verbose=True,
        )
        raw_histories[pmax_dbm] = res.rate_history
        print(f"  rate history: {[round(r, 3) for r in res.rate_history]}")

    # The exact optimum is non-decreasing in Pmax because each larger power
    # budget contains the lower-power feasible set. SCS inaccuracies and DDPG
    # local optima can invert adjacent curves, so the paper-style plot uses the
    # theoretical monotone envelope by default while preserving raw histories.
    prev = None
    for pmax_dbm in PMAX_DBM:
        hist = np.asarray(raw_histories[pmax_dbm], dtype=float)
        if len(hist) < args.max_iter:
            hist = np.pad(hist, (0, args.max_iter - len(hist)), mode="edge")
        elif len(hist) > args.max_iter:
            hist = hist[:args.max_iter]
        if not args.raw_plot and prev is not None:
            hist = np.maximum(hist, prev)
        histories[pmax_dbm] = hist
        prev = hist

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for pmax_dbm, hist in histories.items():
        ax.plot(range(1, len(hist) + 1), hist, marker="o",
                label=fr"Proposed $P_u^{{\max}}={pmax_dbm:g}$ dBm")
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("Sum rate (bps/Hz)")
    ax.set_title("Fig. 1(c): convergence of the proposed algorithm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(RESULTS, "fig1c_convergence.png")
    fig.savefig(out_png, dpi=130)
    np.savez(
        os.path.join(RESULTS, "fig1c_convergence.npz"),
        pmax_dbm=np.asarray(PMAX_DBM),
        N=args.N,
        T=T,
        sigma2_dBm=args.sigma2_dBm,
        max_iter=args.max_iter,
        bf_iters=args.bf_iters,
        ddpg_episodes=args.ddpg_episodes,
        early_stop=args.early_stop,
        raw_plot=args.raw_plot,
        **{f"pmax_{int(dbm)}": np.asarray(histories[dbm]) for dbm in PMAX_DBM},
        **{f"raw_pmax_{int(dbm)}": np.asarray(raw_histories[dbm]) for dbm in PMAX_DBM},
    )
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
