"""DDPG agent for multi-UAV trajectory optimization (paper Algorithm 3).

Standard Deep Deterministic Policy Gradient: an actor mu(s) and critic Q(s,a),
each with a softly-updated target network, trained off-policy from a replay
buffer with Ornstein-Uhlenbeck exploration. Hyperparameters follow Table I
(replay buffer 1600, batch 32) and DDPG conventions.

The networks are small (obs ~ a few dozen dims), so CPU is the default device;
MPS is supported but rarely faster for this size and can be less numerically
stable for some ops.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch: int):
        idx = np.random.randint(0, len(self.buf), size=batch)
        s, a, r, s2, done = zip(*(self.buf[i] for i in idx))
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.float32),
            np.array(r, dtype=np.float32),
            np.array(s2, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


class OUNoise:
    """Ornstein-Uhlenbeck process for deterministic-policy exploration."""

    def __init__(self, shape, theta: float = 0.15, sigma: float = 0.2, seed: int = 0):
        self.shape = shape
        self.theta = theta
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(shape)

    def reset(self):
        self.state = np.zeros(self.shape)

    def sample(self) -> np.ndarray:
        self.state += self.theta * (-self.state) + self.sigma * self.rng.standard_normal(self.shape)
        return self.state


class DDPGAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        gamma: float = 0.95,
        tau: float = 0.005,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        buffer_size: int = 1600,
        batch_size: int = 32,
        hidden: int = 256,
        device: str = "cpu",
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = torch.device(device)
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.actor = Actor(obs_dim, act_dim, hidden).to(self.device)
        self.actor_target = Actor(obs_dim, act_dim, hidden).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.noise = OUNoise(act_dim, seed=seed)

    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        with torch.no_grad():
            s = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            a = self.actor(s).cpu().numpy()
        if explore:
            a = a + self.noise.sample()
        return np.clip(a, -1.0, 1.0)

    def remember(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return
        s, a, r, s2, done = self.buffer.sample(self.batch_size)
        s = torch.as_tensor(s, device=self.device)
        a = torch.as_tensor(a, device=self.device)
        r = torch.as_tensor(r, device=self.device)
        s2 = torch.as_tensor(s2, device=self.device)
        done = torch.as_tensor(done, device=self.device)

        with torch.no_grad():
            a2 = self.actor_target(s2)
            q2 = self.critic_target(s2, a2)
            y = r + self.gamma * (1.0 - done) * q2
        q = self.critic(s, a)
        critic_loss = F.mse_loss(q, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.actor, self.actor_target)

    def _soft_update(self, src: nn.Module, tgt: nn.Module):
        for sp, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * sp.data)


def train_ddpg(
    env,
    *,
    episodes: int = 300,
    agent: DDPGAgent | None = None,
    seed: int = 0,
    noise_decay: float = 0.99,
    eval_rate_fn=None,
    log_every: int = 0,
    device: str = "cpu",
):
    """Train DDPG on ``env``. Returns ``(agent, returns, best_traj)``.

    OU exploration noise decays by ``noise_decay`` each episode (steadily
    shifting from exploration to exploitation). If ``eval_rate_fn(traj)`` is
    given, ``best_traj`` is the highest-*rate* trajectory seen (the true
    objective); otherwise it is the highest-return trajectory.
    """
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = agent or DDPGAgent(obs_dim, act_dim, device=device, seed=seed)
    sigma0 = agent.noise.sigma

    # returns: list[float] = []
    # best_score = -np.inf
    # best_traj = None
    # for ep in range(episodes):
    #     obs, _ = env.reset(seed=seed + ep)
    #     agent.noise.reset()
    #     ep_ret = 0.0
    #     traj = np.empty((env.p.U, env.p.N, 3))
    #     done = False
    #     n = 0
    #     while not done:
    #         a = agent.act(obs, explore=True)
    #         obs2, r, term, trunc, _ = env.step(a)
    #         done = term or trunc
    #         agent.remember(obs, a, r, obs2, float(done))
    #         agent.learn()
    #         obs = obs2
    #         ep_ret += r
    #         traj[:, n] = env.state
    #         n += 1
    #     returns.append(ep_ret)
    #     score = eval_rate_fn(traj) if eval_rate_fn is not None else ep_ret
    #     if score > best_score:
    #         best_score = score
    #         best_traj = traj.copy()
    #     agent.noise.sigma = max(sigma0 * noise_decay**ep, 0.01 * sigma0)
    #     if log_every and (ep % log_every == 0):
    #         print(f"ep {ep:3d}  return {ep_ret:.3f}  (best score {best_score:.3f})")
    #
    # return agent, returns, best_traj
    returns: list[float] = []
    best_score = -np.inf
    best_traj = None

    for ep in range(episodes):

        obs, _ = env.reset(seed=seed + ep)
        agent.noise.reset()

        ep_ret = 0.0
        traj = np.empty((env.p.U, env.p.N, 3))

        done = False
        n = 0

        while not done:

            # 训练阶段保留探索噪声
            a = agent.act(obs, explore=True)

            obs2, r, term, trunc, _ = env.step(a)

            done = term or trunc

            agent.remember(
                obs,
                a,
                r,
                obs2,
                float(done)
            )

            agent.learn()

            obs = obs2
            ep_ret += r

            traj[:, n] = env.state
            n += 1

        returns.append(ep_ret)

        # 训练过程中仍记录表现最好的带噪轨迹
        score = (
            eval_rate_fn(traj)
            if eval_rate_fn is not None
            else ep_ret
        )

        if score > best_score:
            best_score = score
            best_traj = traj.copy()

        # OU噪声逐步衰减
        agent.noise.sigma = max(
            sigma0 * noise_decay**ep,
            0.01 * sigma0
        )

        if log_every and ep % log_every == 0:
            print(
                f"ep {ep:3d}  "
                f"return {ep_ret:.3f}  "
                f"(best score {best_score:.3f})"
            )

    # =========================================================
    # 新增：训练完成后关闭探索噪声
    # 用最终Actor确定性生成一条轨迹
    # =========================================================
    if eval_rate_fn is not None:

        deterministic_traj = env.rollout(
            lambda obs: agent.act(
                obs,
                explore=False
            )
        )

        deterministic_score = eval_rate_fn(
            deterministic_traj
        )

        if log_every:
            print(
                f"[DDPG FINAL] "
                f"best_noisy_score={best_score:.4f}, "
                f"deterministic_score={deterministic_score:.4f}"
            )

        # noisy最优轨迹和最终Actor确定性轨迹二选一
        if deterministic_score > best_score:

            best_score = deterministic_score

            best_traj = deterministic_traj.copy()

    return agent, returns, best_traj