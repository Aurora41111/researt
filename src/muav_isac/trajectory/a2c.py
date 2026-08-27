"""A2C baseline for UAV trajectory optimization (used only for Fig.1a).

On-policy Advantage Actor-Critic with a Gaussian continuous policy. The paper
notes A2C is less sample-efficient than DDPG (it updates from current-episode
data only), which shows up as noisier, less user-aligned trajectories -- the
contrast Fig.1a is meant to highlight. Kept deliberately simpler than the DDPG
implementation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128, init_std: float = 0.5):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mean = nn.Linear(hidden, act_dim)
        self.value = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(np.log(init_std) * torch.ones(act_dim, dtype=torch.float32))

    def forward(self, s: torch.Tensor):
        h = self.trunk(s)
        return torch.tanh(self.mean(h)), self.value(h).squeeze(-1)


def train_a2c(
    env,
    *,
    episodes: int = 300,
    lr: float = 3e-4,
    gamma: float = 0.99,
    seed: int = 0,
    eval_rate_fn=None,
    device: str = "cpu",
    log_every: int = 0,
):
    """Train A2C on-policy. Returns ``(model, returns, best_traj)``."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    device = torch.device(device)
    model = ActorCritic(obs_dim, act_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    returns: list[float] = []
    best_score = -np.inf
    best_traj = None
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        log_probs, values, rewards = [], [], []
        traj = np.empty((env.p.U, env.p.N, 3))
        done = False
        n = 0
        while not done:
            s = torch.as_tensor(obs, dtype=torch.float32, device=device)
            mean, v = model(s)
            dist = torch.distributions.Normal(mean, model.log_std.exp())
            a = dist.sample()
            log_probs.append(dist.log_prob(a).sum())
            values.append(v)
            action = torch.tanh(a).cpu().numpy()
            obs2, r, term, trunc, _ = env.step(action)
            done = term or trunc
            rewards.append(r)
            obs = obs2
            traj[:, n] = env.state
            n += 1

        # discounted returns
        T = len(rewards)
        rets = np.zeros(T, dtype=np.float32)
        running = 0.0
        for t in reversed(range(T)):
            running = rewards[t] + gamma * running
            rets[t] = running
        rets_t = torch.as_tensor(rets, device=device)
        vals_t = torch.stack(values)
        logp_t = torch.stack(log_probs)
        adv = rets_t - vals_t.detach()

        actor_loss = -(logp_t * adv).mean()
        critic_loss = adv.pow(2).mean()
        loss = actor_loss + critic_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        ep_ret = float(sum(rewards))
        returns.append(ep_ret)
        score = eval_rate_fn(traj) if eval_rate_fn is not None else ep_ret
        if score > best_score:
            best_score = score
            best_traj = traj.copy()
        if log_every and ep % log_every == 0:
            print(f"a2c ep {ep:3d}  return {ep_ret:.3f}  (best {best_score:.3f})")

    return model, returns, best_traj
