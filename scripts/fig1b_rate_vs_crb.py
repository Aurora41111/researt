"""Fig.1(b): sum rate vs. CRB sensing threshold.

Sweeps the CRB threshold Gamma (log axis) and compares the proposed joint BCD
against TWOBF (trajectory-only, heuristic beams) and BFWOT (beamforming-only,
fixed straight-line trajectory). Expected: rate rises as Gamma relaxes (less
sensing interference), and proposed > BFWOT > TWOBF (paper Fig.1b).

Usage:
    uv run python scripts/fig1b_rate_vs_crb.py [--N 16] [--sigma2-dBm -80]

The proposed-method sweep is the expensive part (full BCD per Gamma). Defaults
are a moderate config; reduce --n-gamma or --N for a quick pass, raise for
publication quality.
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

from muav_isac import baselines, bcd
from muav_isac import scenario as sc
from muav_isac.config import default_params

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--n-gamma", type=int, default=None,
                    help="number of CRB points; default uses the paper's 8 ticks from -6.5 to -3.")
    ap.add_argument("--T", type=float, default=None,
                    help="flight period [s]; default T=N (so tau=1s).")
    ap.add_argument("--max-iter", type=int, default=4)
    ap.add_argument("--bf-iters", type=int, default=5)
    ap.add_argument("--ddpg-episodes", type=int, default=100)
    ap.add_argument("--sigma2-dBm", type=float, default=default_params().sigma2_dBm,
                    help="AWGN power [dBm]. The paper omits this; default is paper-calibrated.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    T = args.T if args.T is not None else float(args.N)
    # Paper Fig.1(b): x-axis "CRB threshold (dB)" from -6.5 to -3 in 0.5 dB steps.
    # Given typical CRB magnitudes (~1e-6..1e-3 rad^2), these values are log10(Gamma)
    # (strict 10*log10 would give -60..-30); we follow the paper's axis labeling
    # literally, so Gamma = 10**db.
    db_vals = (
        np.linspace(-6.5, -3.0, args.n_gamma)
        if args.n_gamma is not None
        else np.arange(-6.5, -3.0 + 1e-9, 0.5)
    )
    gammas = 10.0 ** db_vals

    rate_proposed, rate_twobf, rate_bfwot = [], [], []
    for db, g in zip(db_vals, gammas):
        p = dataclasses.replace(
            default_params(), Gamma=float(g), sigma2_dBm=args.sigma2_dBm, N=args.N, T=T
        )
        net = sc.sample_network(p, np.random.default_rng(args.seed))    # same layout per seed
        traj0 = sc.init_trajectory(net)
        print(f"=== CRB = {db:.1f} dB (Gamma = {g:.2e}) ===")

        r_prop = bcd.bcd_solve(
            net, traj0, max_iter=args.max_iter, bf_iters=args.bf_iters,
            ddpg_episodes=args.ddpg_episodes, seed=args.seed, verbose=True,
        )
        r_two = baselines.twobf(
            net, traj0, max_iter=args.max_iter, ddpg_episodes=args.ddpg_episodes, seed=args.seed,
        )
        r_bf = baselines.bfwot(net, traj0, bf_iters=args.bf_iters)
        print(f"  proposed={r_prop.final_rate:.2f}  TWOBF={r_two.final_rate:.2f}  BFWOT={r_bf.final_rate:.2f}")
        rate_proposed.append(r_prop.final_rate)
        rate_twobf.append(r_two.final_rate)
        rate_bfwot.append(r_bf.final_rate)

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(db_vals, rate_proposed, "o-", label="Proposed")
    ax.plot(db_vals, rate_bfwot, "s--", label="BFWOT")
    ax.plot(db_vals, rate_twobf, "^:", label="TWOBF")
    ax.set_xticks(db_vals)
    ax.set_xlabel("CRB threshold (dB)")
    ax.set_ylabel("Sum rate (bps/Hz)")
    ax.set_title("Fig. 1(b): rate vs. CRB")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(RESULTS, "fig1b_rate_vs_crb.png")
    fig.savefig(out_png, dpi=130)
    np.savez(
        os.path.join(RESULTS, "fig1b_rate_vs_crb.npz"),
        db=db_vals, gamma=gammas,
        N=args.N, T=T, sigma2_dBm=args.sigma2_dBm, n_gamma=len(db_vals),
        proposed=rate_proposed, twobf=rate_twobf, bfwot=rate_bfwot,
    )
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
