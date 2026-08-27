"""Fig.1(d): sum rate vs. Pmax under different noise floors.

This is NOT in the paper -- it documents a reproduction finding. The paper omits
the bandwidth (hence the noise floor), and the "rate increases with Pmax" trend
only appears when 60 dBm sits in the noise-limited regime. At a low noise floor
(narrow bandwidth, e.g. -94 dBm / 10 MHz) the system is interference-limited and
the FP's sum-rate-max user-zeroing keeps the rate flat as power grows. At a
higher noise floor (wideband ISAC, e.g. -75 dBm / ~800 MHz) there is noise-limited
headroom and the rate rises with Pmax. The Fig.1(c) script defaults to -75 dBm
because it best separates the 60/70/80 dBm curves in the direction of paper Fig.1(c).

Sweeps Pmax over {60, 65, 70, 75, 80} dBm for three noise floors and plots the
sum rate. Each point is one full BCD solve (FP beamforming + DDPG trajectory).

Usage:
    uv run python scripts/fig1d_noise_floor.py [--N 12] [--ddpg-episodes 80]
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

# Noise floors spanning interference-limited (-94, 10 MHz) to the paper-calibrated
# wideband default (-75, ~800 MHz). Bandwidth is unspecified in the paper.
NOISE_FLOORS = {
    "−94 dBm (10 MHz)": -94.0,
    "−84 dBm (100 MHz)": -84.0,
    "−80 dBm (~250 MHz)": -80.0,
    "−75 dBm (~800 MHz)": -75.0,
}
PMAX_DBM = (60.0, 65.0, 70.0, 75.0, 80.0)


def dbm_to_w(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--max-iter", type=int, default=4)
    ap.add_argument("--bf-iters", type=int, default=2)
    ap.add_argument("--ddpg-episodes", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    curves: dict[str, list[float]] = {}
    for label, noise_dbm in NOISE_FLOORS.items():
        rates = []
        for pmax_dbm in PMAX_DBM:
            p = dataclasses.replace(
                default_params(),
                sigma2_dBm=noise_dbm,
                Pmax_W=dbm_to_w(pmax_dbm),
                N=args.N,
                T=args.N,
            )
            net = sc.sample_network(p, np.random.default_rng(args.seed))
            res = bcd.bcd_solve(
                net, sc.init_trajectory(net), max_iter=args.max_iter, bf_iters=args.bf_iters,
                ddpg_episodes=args.ddpg_episodes, seed=args.seed,
            )
            print(f"  {label:<22} Pmax={pmax_dbm:g} dBm: rate={res.final_rate:.2f}")
            rates.append(res.final_rate)
        curves[label] = rates

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    markers = ["o-", "s--", "^:"]
    for (label, rates), mk in zip(curves.items(), markers):
        ax.plot(PMAX_DBM, rates, mk, label=label)
    ax.set_xlabel("Power budget $P_u^{\\max}$ (dBm)")
    ax.set_ylabel("Sum rate (bps/Hz)")
    ax.set_title("Fig. 1(d): rate vs. Pmax across noise floors")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(RESULTS, "fig1d_noise_floor.png")
    fig.savefig(out_png, dpi=130)
    np.savez(
        os.path.join(RESULTS, "fig1d_noise_floor.npz"),
        pmax_dbm=np.asarray(PMAX_DBM),
        noise_dbm=np.asarray(list(NOISE_FLOORS.values())),
        **{f"noise_{int(v)}": np.asarray(curves[k]) for k, v in NOISE_FLOORS.items()},
    )
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
