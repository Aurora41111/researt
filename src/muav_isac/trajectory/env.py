"""Gymnasium environment for the UAV trajectory subproblem (paper P8 / Alg.3).

Casts "maximize sum rate over the trajectory subject to speed, altitude,
collision, energy, and endpoint constraints" into an MDP solved by DDPG.

* **State** -- normalized UAV positions, user positions, target positions, and
  slot progress. (Beamforming is fixed during one BCD iteration, so its effect
  enters through the reward, not the state.)
* **Action** -- per UAV: horizontal speed a_h in [10, 20], direction theta in
  [-5pi/12, 5pi/12], altitude H in [150, 200]. The policy emits tanh in [-1, 1]
  and we affine-map each component to its range.
* **Dynamics** -- x[n+1] = x[n] + tau*a_h*cos(theta), y likewise, H[n+1] = H.
* **Reward** -- per-slot average user rate minus penalties for collision,
  boundary, accumulated energy, and (terminal) endpoint deviation. CRB is *not*
  penalized here: the beamforming step re-solves CRB each BCD iteration, so
  trajectory drift is corrected downstream (kept fast -- per-step A-matrices
  would dominate DDPG training time).
"""

from __future__ import annotations

from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .. import energy as en
from .. import scenario as sc
from ..config import Params
from ..scenario import Network


class TrajectoryEnv(gym.Env):
    """Fixed-beamforming trajectory environment for one BCD iteration."""

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        net: Network,
        g_beams_last: dict,
        i_beams_last: dict,
        params: Params | None = None,
        reward_scale: float = 10.0,
        w_collision: float = 2.0,
        w_boundary: float = 1.0,
        w_energy: float = 0.05,
        w_endpoint: float = 5.0,
    ) -> None:
        super().__init__()
        self.net = net
        self.p = params or net.p
        # (u, lv) -> (N, M) beam vectors from the last BF solve; the env slices the
        # current slot's beam to score the new position (BCD: BF fixed, indexed by slot).
        self.g_beams_last = g_beams_last
        self.i_beams_last = i_beams_last
        self.reward_scale = reward_scale
        self.w_collision = w_collision
        self.w_boundary = w_boundary
        self.w_energy = w_energy
        self.w_endpoint = w_endpoint

        self.obs_dim = self.p.U * 3 + self.net.V * 2 + self.net.K * 2 + 1
        self.act_dim = self.p.U * 3
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)

        self.state: np.ndarray | None = None        # UAV positions (U, 3) in meters
        self.slot = 0
        self.energy_used = 0.0
        self._prev_xy = None

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        H_mid = 0.5 * (self.p.H_range[0] + self.p.H_range[1])
        self.state = np.concatenate(
            [self.net.uav_start, np.full((self.p.U, 1), H_mid)], axis=1
        ).astype(float)                               # (U, 3)
        self.slot = 0
        self.energy_used = 0.0
        self._prev_xy = self.state[:, :2].copy()
        return self._obs(), {}

    def step(self, action: np.ndarray):
        assert self.state is not None, "call reset() before step()"
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(self.p.U, 3), -1.0, 1.0)
        ah = self.p.ah_range[0] + 0.5 * (self.p.ah_range[1] - self.p.ah_range[0]) * (a[:, 0] + 1)
        # symmetric map a[:,1] in [-1,1] -> theta in [-t_max, t_max]
        theta = self.p.theta_range[1] * a[:, 1]
        H_new = 0.5 * (self.p.H_range[0] + self.p.H_range[1]) + 0.5 * (
            self.p.H_range[1] - self.p.H_range[0]
        ) * a[:, 2]

        xy_new = self.state[:, :2] + self.p.tau * ah[:, None] * np.stack(
            [np.cos(theta), np.sin(theta)], axis=1
        )

        # --- reward: average user rate at the new positions (fixed BF, slot n) ---
        n = self.slot
        i_slot = {(u, lt): self.i_beams_last[(u, lt)][n] for u in range(self.p.U) for lt in range(self.net.Ku_list[u])}
        h_uvm = sc.comm_channels_at(self.net, xy_new, H_new)
        # Re-point the stale (previous-trajectory-aligned) comm beams to the channels
        # at the *candidate* positions, keeping per-beam FP power. Otherwise the
        # surrogate rewards "stay where the old beams still point" -- a straight-line
        # attractor -- instead of position quality. Sensing beams stay stale (their
        # mis-aim is a second-order interference effect).
        g_slot_stale = {(u, lv): self.g_beams_last[(u, lv)][n] for u in range(self.p.U) for lv in range(self.net.Vu_list[u])}
        g_slot = sc.realign_comm_beams(self.net, h_uvm, g_slot_stale)
        sr = sc.slot_sum_rate(self.net, h_uvm, g_slot, i_slot)
        avg_rate = sr / (self.p.tau * max(self.net.V, 1))    # bps/Hz per user
        reward = self.reward_scale * avg_rate

        # --- penalties ---
        reward -= self.w_collision * self._collision_penalty(xy_new)
        reward -= self.w_boundary * self._boundary_penalty(xy_new)
        # flying energy over this transition
        V_horiz = np.hypot(xy_new[:, 0] - self._prev_xy[:, 0], xy_new[:, 1] - self._prev_xy[:, 1]) / self.p.tau
        ac = np.abs(H_new - self.state[:, 2]) / self.p.tau
        self.energy_used += float(np.sum([en.E_fly(self.p, V, c) for V, c in zip(V_horiz, ac)]))
        eth = self._energy_budget()
        if self.energy_used > eth:
            reward -= self.w_energy * (self.energy_used - eth) / max(eth, 1.0)

        # advance state
        self.state = np.concatenate([xy_new, H_new[:, None]], axis=1)
        self._prev_xy = xy_new.copy()
        self.slot += 1

        terminated = False
        truncated = self.slot >= self.p.N
        if truncated:
            reward -= self.w_endpoint * self._endpoint_penalty()

        return self._obs(), float(reward), terminated, truncated, self._info()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        assert self.state is not None
        s = self.state / self.p.area_m
        users = self.net.user_xy / self.p.area_m
        targets = self.net.target_xy / self.p.area_m
        prog = np.array([self.slot / self.p.N])
        return np.concatenate([s.reshape(-1), users.reshape(-1), targets.reshape(-1), prog]).astype(np.float32)

    def _collision_penalty(self, xy: np.ndarray) -> float:
        pen = 0.0
        for i in range(self.p.U):
            for j in range(i + 1, self.p.U):
                d = float(np.hypot(*(xy[i] - xy[j])))
                if d < self.p.dmin:
                    pen += (self.p.dmin - d) / self.p.dmin
        return pen

    def _boundary_penalty(self, xy: np.ndarray) -> float:
        pen = 0.0
        for u in range(self.p.U):
            for d in range(2):
                if xy[u, d] < 0:
                    pen += -xy[u, d] / self.p.area_m
                elif xy[u, d] > self.p.area_m:
                    pen += (xy[u, d] - self.p.area_m) / self.p.area_m
        return pen

    def _endpoint_penalty(self) -> float:
        assert self.state is not None
        return float(np.mean(np.hypot(
            self.state[:, 0] - self.net.uav_end[:, 0],
            self.state[:, 1] - self.net.uav_end[:, 1],
        )) / self.p.area_m)

    def _energy_budget(self) -> float:
        """E_th from the paper's Eth_factor (full-power TX + full-speed hover/fly)."""
        p = self.p
        e_tx_max = p.tau * p.Pmax_W * p.U * p.N
        Vcruise = 0.5 * (p.ah_range[0] + p.ah_range[1])
        e_fly_max = p.tau * en.propulsion_power(p, Vcruise, 0.0) * p.U * p.N
        return p.Eth_factor * (e_tx_max + e_fly_max)

    def _info(self) -> dict:
        return {"slot": self.slot, "energy_used": self.energy_used}

    # ------------------------------------------------------------------
    # Trajectory extraction (for the caller / plotting)
    # ------------------------------------------------------------------
    def rollout(self, policy_fn) -> np.ndarray:
        """Run a full deterministic episode with ``policy_fn(obs) -> action``; return (U, N, 3)."""
        obs, _ = self.reset()
        traj = np.empty((self.p.U, self.p.N, 3))
        for n in range(self.p.N):
            a = policy_fn(obs)
            obs, _r, _t, trunc, _info = self.step(a)
            traj[:, n] = self.state
            if trunc:
                break
        return traj
