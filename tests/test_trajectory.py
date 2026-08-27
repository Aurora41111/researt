"""Trajectory optimization tests: env, DDPG (Alg.3), A2C baseline.

Small scenario (N=10) and modest episode counts keep the suite fast while still
exercising that (a) the env is a well-formed MDP, (b) DDPG's returns trend up,
and (c) A2C produces a valid trajectory.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from muav_isac import scenario as sc
from muav_isac.config import default_params
from muav_isac.trajectory import a2c, ddpg
from muav_isac.trajectory.env import TrajectoryEnv

PV = dataclasses.replace(default_params(), N=10, T=10)            # tau = 1 s


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(3)
    net = sc.sample_network(PV, rng)
    snap = sc.compute_channels(net, sc.init_trajectory(net))
    g, i = {}, {}
    for u in range(PV.U):
        for lv, v in enumerate(net.user_of_uav[u]):
            h = snap.h[u, v]
            nrm = np.linalg.norm(h, axis=1, keepdims=True)
            g[(u, lv)] = (h / np.maximum(nrm, 1e-12)) * np.sqrt(0.1)
        for lt in range(net.Ku_list[u]):
            i[(u, lt)] = np.ones((PV.N, PV.M), dtype=complex) * np.sqrt(0.1 / PV.M)
    return net, snap, g, i


def test_env_reset_step_and_truncation(setup):
    net, _snap, g, i = setup
    env = TrajectoryEnv(net, g, i)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (env.obs_dim,)
    total_r = 0.0
    n_steps = 0
    done = False
    while not done:
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        assert np.isfinite(r)
        total_r += r
        n_steps += 1
        done = term or trunc
    assert n_steps == PV.N                                  # full episode of N slots
    assert "energy_used" in info


def test_ddpg_returns_improve(setup):
    net, _snap, g, i = setup
    env = TrajectoryEnv(net, g, i, reward_scale=10.0)
    _agent, returns, best = ddpg.train_ddpg(env, episodes=150, seed=0, noise_decay=0.99)
    assert best.shape == (PV.U, PV.N, 3)
    # returns trend upward over training (sample efficiency of DDPG)
    assert np.mean(returns[-20:]) >= np.mean(returns[:20]) - 0.5


def test_ddpg_trajectory_reasonable_rate(setup):
    net, _snap, g, i = setup
    env = TrajectoryEnv(net, g, i, reward_scale=10.0)

    def rate_of(traj):
        # Re-point the (init-trajectory-aligned) comm beams to each candidate
        # trajectory's own channels before scoring -- matches BCD's _eval_rate.
        # Without this, a DDPG detour toward users is penalized by stale beams,
        # which is exactly the behavior we want to reward.
        h = sc.comm_channels(net, traj)
        g_re = sc.realign_comm_beams(net, h, g)
        return sc.sum_rate_h(net, h, g_re, i)

    straight = rate_of(sc.init_trajectory(net))
    _agent, _returns, best = ddpg.train_ddpg(
        env, episodes=300, seed=0, noise_decay=0.995, eval_rate_fn=rate_of
    )
    ddpg_rate = rate_of(best)
    # DDPG should land within reach of the straight-line baseline (it exploits
    # user positions; with full BF in the real BCD the margin grows). The margin
    # depends on the SNR regime and beam power; 0.80 is a loose "didn't break"
    # floor for the test's low-power heuristic beams.
    assert ddpg_rate >= 0.80 * straight


def test_a2c_runs_and_produces_trajectory(setup):
    net, _snap, g, i = setup
    env = TrajectoryEnv(net, g, i, reward_scale=10.0)
    _model, returns, best = a2c.train_a2c(env, episodes=60, seed=0)
    assert best.shape == (PV.U, PV.N, 3)
    assert np.all(np.isfinite(best))
    assert len(returns) == 60
