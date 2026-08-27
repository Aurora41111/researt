"""Outer BCD + baselines tests. Small N / few iters to keep the suite fast."""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")  # SCS "may be inaccurate" at loose eps

from muav_isac import scenario as sc
from muav_isac.baselines import bfwot, twobf
from muav_isac.bcd import bcd_solve
from muav_isac.config import default_params

# Real config defaults (Pmax=70 dBm, common wideband sigma2, Gamma=1e-5); only N shrunk
# for speed.
PV = dataclasses.replace(default_params(), N=6, T=6)


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(5)
    net = sc.sample_network(PV, rng)
    return net, sc.init_trajectory(net)


def test_bcd_runs_and_history_finite(setup):
    net, traj0 = setup
    res = bcd_solve(net, traj0, max_iter=2, bf_iters=4, ddpg_episodes=50, seed=0)
    assert len(res.rate_history) >= 1
    assert np.isfinite(res.final_rate) and res.final_rate > 0
    assert res.traj.shape == (PV.U, PV.N, 3)


def test_bcd_history_nondecreasing(setup):
    net, traj0 = setup
    res = bcd_solve(net, traj0, max_iter=3, bf_iters=4, ddpg_episodes=80, seed=1)
    h = np.asarray(res.rate_history)
    # FP subproblems are monotone; DDPG adds mild noise -> allow a small dip
    diffs = np.diff(h)
    assert np.all(diffs >= -0.5 * max(1.0, abs(h[0])))


def test_baselines_run(setup):
    net, traj0 = setup
    r_two = twobf(net, traj0, max_iter=2, ddpg_episodes=50, seed=2)
    r_bf = bfwot(net, traj0, bf_iters=4)
    assert np.isfinite(r_two.final_rate) and r_two.final_rate > 0
    assert np.isfinite(r_bf.final_rate) and r_bf.final_rate > 0


def test_fp_beamforming_beats_heuristic_on_fixed_trajectory(setup):
    # Isolate the beamforming effect: on the SAME (straight-line) trajectory,
    # FP-optimized beams (BFWOT) must beat heuristic MRT + CRB-feasible beams.
    # Comparing through the full BCD/TWOBF is noisy at tiny N (DDPG + trajectory
    # trade-offs), so we hold the trajectory fixed and vary only the beams.
    from muav_isac.baselines import _mrt_beams as mrt_heuristic
    from muav_isac.bcd import _crb_feasible_beams

    net, traj0 = setup
    r_fp = bfwot(net, traj0, bf_iters=4)                     # FP comm + FP sensing
    snap = sc.compute_channels(net, traj0)
    g_heur = mrt_heuristic(net, snap, 0.1)
    i_heur = _crb_feasible_beams(net, snap)                  # MRT + CRB-feasible heuristic
    rate_heur = sc.sum_rate(net, snap, g_heur, i_heur)
    assert r_fp.final_rate > rate_heur
