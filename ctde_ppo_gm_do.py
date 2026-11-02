#!/usr/bin/env python3
"""
CTDE-PPO gain scheduling for the GM-driven distributed optimization example.

This is a self-contained implementation that uses the same 5-agent directed
graph from Fig. 1 of the paper:

    1 -> 5 (0.2)
    2 -> 1 (0.3)
    3 -> 2 (0.5)
    4 -> 3 (0.6)
    5 -> 4 (0.8)

The script supports two modes:
  - baseline: the original heuristic alpha_dot_i = ||epsilon_i||^2 update
  - rl:       a minimal CTDE-PPO style actor-critic scheduler

If PyTorch is available, the script uses a shared MLP Gaussian actor and a
centralized MLP critic under CTDE-PPO. If CUDA cannot initialize in the current
session, the torch path falls back to CPU automatically. Otherwise, it falls
back to a lightweight NumPy implementation so the file still runs in minimal
environments.
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-config")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

try:
    import networkx as nx
except Exception as exc:  # pragma: no cover
    nx = None


EPS = 1e-12
PAPER_INITIAL_X = np.array([4.0, 4.8, 6.0, 2.0, 0.4], dtype=float)
PAPER_ALPHA_INIT = 4.0


def resolve_torch_device(device: str | None) -> torch.device:
    """Resolve a torch device, falling back to CPU if CUDA cannot initialize."""
    requested = "cuda" if device is None else device
    resolved = torch.device(requested)
    if resolved.type != "cuda":
        return resolved

    try:
        # Probe actual CUDA allocation rather than relying on is_available(),
        # which can return a false negative when the runtime is in a bad session.
        probe = torch.ones(1, device=resolved)
        probe.add_(1.0)
        return resolved
    except Exception as exc:
        warnings.warn(
            "\n".join(
                [
                    f"CUDA probe failed for requested device {requested!r}; continuing on CPU.",
                    f"torch.__version__ = {torch.__version__}",
                    f"torch.version.cuda = {getattr(torch.version, 'cuda', None)}",
                    f"probe_error = {exc!r}",
                    "If you expected GPU execution, start a fresh shell/IDE session, verify `nvidia-smi`,",
                    "and make sure `CUDA_VISIBLE_DEVICES` is set before launching Python.",
                ]
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return np.log1p(np.exp(x))


def gaussian_log_prob(action: np.ndarray, mean: np.ndarray, log_std: np.ndarray) -> np.ndarray:
    var = np.exp(2.0 * log_std)
    return -0.5 * np.sum(((action - mean) ** 2) / var + 2.0 * log_std + np.log(2.0 * np.pi), axis=-1)


def build_paper_adjacency() -> np.ndarray:
    """Return the exact weighted digraph used in Fig. 1."""
    A = np.zeros((5, 5), dtype=float)
    A[0, 4] = 0.2  # 1 -> 5
    A[1, 0] = 0.3  # 2 -> 1
    A[2, 1] = 0.5  # 3 -> 2
    A[3, 2] = 0.6  # 4 -> 3
    A[4, 3] = 0.8  # 5 -> 4
    return A


def laplacian_from_adjacency(A: np.ndarray) -> np.ndarray:
    return np.diag(A.sum(axis=1)) - A


def left_eigenvector_from_laplacian(L: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eig(L.T)
    idx = int(np.argmin(np.abs(eigvals)))
    w = np.real(eigvecs[:, idx])
    w = np.abs(w)
    w = np.maximum(w, EPS)
    return w / w.sum()


def local_cost_and_grad(agent_idx: int, x: float) -> Tuple[float, float]:
    """Paper's simulation costs, all valid for x > 0."""
    x = float(np.clip(x, 1e-8, None))
    if agent_idx == 0:
        return 5.0, 0.0
    if agent_idx == 1:
        return 3.0 / x**3, -9.0 / x**4
    if agent_idx == 2:
        return 4.0 / x**2, -8.0 / x**3
    if agent_idx == 3:
        return 2.0 * x, 2.0
    if agent_idx == 4:
        return x**4, 4.0 * x**3
    raise IndexError("agent_idx out of range")


def global_cost_vector(x_vec: np.ndarray) -> float:
    return float(sum(local_cost_and_grad(i, float(x_vec[i]))[0] for i in range(len(x_vec))))


def consensus_cost_and_grad_scalar(x: float) -> Tuple[float, float]:
    cost = 0.0
    grad = 0.0
    for i in range(5):
        c, g = local_cost_and_grad(i, x)
        cost += c
        grad += g
    return cost, grad


def solve_global_optimum() -> Tuple[float, float]:
    """Find x* for the consensus objective via bisection on the derivative."""
    lo, hi = 1e-4, 10.0
    _, g_lo = consensus_cost_and_grad_scalar(lo)
    _, g_hi = consensus_cost_and_grad_scalar(hi)
    while g_lo * g_hi > 0:
        hi *= 2.0
        _, g_hi = consensus_cost_and_grad_scalar(hi)
        if hi > 1e6:
            raise RuntimeError("Failed to bracket the global optimum.")

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        _, g_mid = consensus_cost_and_grad_scalar(mid)
        if g_lo * g_mid <= 0:
            hi = mid
            g_hi = g_mid
        else:
            lo = mid
            g_lo = g_mid

    x_star = 0.5 * (lo + hi)
    f_star = global_cost_vector(np.full(5, x_star))
    return x_star, f_star


def paper_graph_positions() -> Dict[int, Tuple[float, float]]:
    return {
        1: (0.0, 1.0),
        2: (1.0, 0.15),
        3: (0.55, -0.95),
        4: (-0.55, -0.95),
        5: (-1.0, 0.15),
    }


def plot_paper_graph(path: Path) -> None:
    if nx is None:
        return
    A = build_paper_adjacency()
    G = nx.DiGraph()
    edges = [(1, 5, 0.2), (2, 1, 0.3), (3, 2, 0.5), (4, 3, 0.6), (5, 4, 0.8)]
    G.add_weighted_edges_from(edges)
    pos = paper_graph_positions()

    fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=180)
    ax.set_facecolor("white")
    nx.draw_networkx_nodes(G, pos, node_color="#f7f7f7", edgecolors="black", node_size=850, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=14, font_family="serif", ax=ax)
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        width=2.0,
        edge_color="black",
        connectionstyle="arc3,rad=0.0",
        min_source_margin=12,
        min_target_margin=12,
    )
    edge_labels = {(u, v): f"{w:.1f}" for u, v, w in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=11, font_family="serif", ax=ax)
    ax.set_axis_off()
    ax.set_title("Fig. 1 graph used in the paper", fontsize=12, family="serif")
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


class GMDOEnv:
    """Discrete-time simulation of the GM-driven DO dynamics in log coordinates."""

    def __init__(
        self,
        dt: float = 0.05,
        horizon: int = 100,
        seed: int = 42,
        lambda1: float = 0.2,
        lambda2: float = 0.01,
        consensus_lambda: float = 0.35,
        K_min: float = 0.05,
        K_max: float = 3.0,
        delta_alpha_max: float = 5.0,
        alpha_floor: float = 0.05,
        alpha_max: float = 20.0,
        nominal_k1: float = 4.0,
        nominal_k2: float = 0.05,
        nominal_delta_alpha: float = 0.5,
        delta_k1_scale: float = 1.0,
        delta_k2_scale: float = 0.05,
        delta_alpha_scale: float = 0.75,
        initial_x: np.ndarray | None = None,
    ) -> None:
        self.A = build_paper_adjacency()
        self.L = laplacian_from_adjacency(self.A)
        self.w_left = left_eigenvector_from_laplacian(self.L)
        self.N = 5
        self.dt = dt
        self.horizon = horizon
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.consensus_lambda = consensus_lambda
        self.K_min = K_min
        self.K_max = K_max
        self.delta_alpha_max = delta_alpha_max
        self.alpha_floor = alpha_floor
        self.alpha_max = alpha_max
        self.nominal_k1 = nominal_k1
        self.nominal_k2 = nominal_k2
        self.nominal_delta_alpha = nominal_delta_alpha
        self.delta_k1_scale = delta_k1_scale
        self.delta_k2_scale = delta_k2_scale
        self.delta_alpha_scale = delta_alpha_scale
        self.x_min = 1e-8
        self.x_max = 50.0
        self.v_min = 1e-8
        self.v_max = 50.0
        self.rng = np.random.default_rng(seed)
        self.x_star, self.f_star = solve_global_optimum()
        if initial_x is None:
            initial_x = PAPER_INITIAL_X.copy()
        self.initial_x = np.asarray(initial_x, dtype=float)
        if self.initial_x.shape != (self.N,):
            raise ValueError(f"initial_x must have shape ({self.N},), got {self.initial_x.shape}.")
        self.reset()

    @property
    def obs_dim(self) -> int:
        return 6

    @property
    def act_dim(self) -> int:
        return 3

    def reset(self, x0: np.ndarray | None = None) -> np.ndarray:
        if x0 is None:
            x0 = self.initial_x
        self.x = np.clip(np.asarray(x0, dtype=float), self.x_min, self.x_max)
        self.p = np.log(self.x)
        self.v = self.x.copy()
        self.q = np.log(self.v)
        self.alpha = np.full(self.N, PAPER_ALPHA_INIT, dtype=float)
        self.beta = np.zeros(self.N, dtype=float)
        self.t = 0
        return self._obs()

    def _consensus_term(self, z: np.ndarray) -> np.ndarray:
        # Row-wise sum_j a_ij (z_j - z_i)
        return np.sum(self.A * (z[None, :] - z[:, None]), axis=1)

    def _log_error(self) -> np.ndarray:
        # epsilon_i = w_i * sum_j a_ij (p_i - p_j)
        return self.w_left * np.sum(self.A * (self.p[:, None] - self.p[None, :]), axis=1)

    def _grad_x(self) -> np.ndarray:
        return np.array([local_cost_and_grad(i, float(self.x[i]))[1] for i in range(self.N)], dtype=float)

    def _obs(self) -> np.ndarray:
        x_bar = float(np.mean(self.x))
        eps = self._log_error()
        grad_x = self._grad_x()
        time_feature = np.full(self.N, self.t / max(1, self.horizon - 1), dtype=float)
        return np.column_stack(
            [
                eps,
                grad_x,
                np.abs(self.x - x_bar),
                self.alpha,
                self.beta,
                time_feature,
            ]
        )

    def transform_raw_actions(self, raw_actions: np.ndarray) -> np.ndarray:
        """Map unconstrained Gaussian actions to bounded deltas around nominal gains."""
        raw_actions = np.asarray(raw_actions, dtype=float)
        k1 = np.clip(self.nominal_k1 + self.delta_k1_scale * np.tanh(raw_actions[:, 0]), self.K_min, self.K_max)
        k2 = np.clip(self.nominal_k2 + self.delta_k2_scale * np.tanh(raw_actions[:, 1]), self.K_min, self.K_max)
        delta_alpha = np.clip(
            self.nominal_delta_alpha + self.delta_alpha_scale * np.tanh(raw_actions[:, 2]),
            0.0,
            self.delta_alpha_max,
        )
        return np.column_stack([k1, k2, delta_alpha])

    def baseline_actions(
        self,
        obs: np.ndarray,
        k1_fixed: float | None = None,
        k2_fixed: float | None = None,
    ) -> np.ndarray:
        eps = obs[:, 0]
        if k1_fixed is None:
            k1_fixed = self.nominal_k1
        if k2_fixed is None:
            k2_fixed = self.nominal_k2
        delta_alpha = np.clip(eps**2, 0.0, self.delta_alpha_max)
        return np.column_stack(
            [
                np.full(self.N, k1_fixed, dtype=float),
                np.full(self.N, k2_fixed, dtype=float),
                delta_alpha,
            ]
        )

    def step(self, gains: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        gains = np.asarray(gains, dtype=float)
        k1 = np.clip(gains[:, 0], self.K_min, self.K_max)
        k2 = np.clip(gains[:, 1], self.K_min, self.K_max)
        delta_alpha = np.clip(gains[:, 2], 0.0, self.delta_alpha_max)

        # In the paper beta_i tracks the alpha update rate. Here we use the RL
        # output as that rate and project alpha into a safe interval.
        self.beta = delta_alpha
        gain_sum = np.clip(self.alpha + self.beta, self.alpha_floor, self.alpha_max)

        consensus_p = self._consensus_term(self.p)
        consensus_q = self._consensus_term(self.q)
        grad_x = self._grad_x()
        grad_log = self.x * grad_x

        # Discretized version of the transformed dynamics (9) / (10).
        p_dot = k1 * gain_sum * self.w_left * consensus_p + self.w_left * consensus_q - k2 * grad_log
        q_dot = k1 * gain_sum * self.w_left * (-consensus_p)

        self.p = self.p + self.dt * p_dot
        self.q = self.q + self.dt * q_dot

        self.x = np.clip(np.exp(self.p), self.x_min, self.x_max)
        self.p = np.log(self.x)
        self.v = np.clip(np.exp(self.q), self.v_min, self.v_max)
        self.q = np.log(self.v)

        self.alpha = np.clip(self.alpha + self.dt * self.beta, self.alpha_floor, self.alpha_max)
        self.t += 1

        obs = self._obs()
        global_cost = global_cost_vector(self.x)
        overshoot = max(0.0, global_cost - self.f_star)
        x_bar = float(np.mean(self.x))
        eps = np.abs(self._log_error())
        consensus_penalty = np.abs(self.x - x_bar) + eps
        per_agent_reward = (
            -np.abs(self.x - self.x_star)
            - self.lambda1 * overshoot
            - self.lambda2 * np.abs(k1)
            - self.consensus_lambda * consensus_penalty
        )
        team_reward = float(np.mean(per_agent_reward))
        done = self.t >= self.horizon

        info = {
            "global_cost": global_cost,
            "overshoot": overshoot,
            "consensus_penalty": float(np.mean(consensus_penalty)),
            "x_error_norm": float(np.linalg.norm(self.x - self.x_star)),
            "mean_alpha": float(np.mean(self.alpha)),
            "mean_beta": float(np.mean(self.beta)),
            "mean_k1": float(np.mean(k1)),
            "mean_k2": float(np.mean(k2)),
        }
        return obs, team_reward, done, info


class LinearGaussianActor:
    """Shared linear Gaussian actor used by all agents."""

    def __init__(self, obs_dim: int, act_dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W = rng.normal(scale=0.08, size=(act_dim, obs_dim))
        self.b = np.zeros(act_dim, dtype=float)
        self.log_std = np.log(np.array([0.2, 0.2, 0.3], dtype=float))

    def mean(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=float)
        return obs @ self.W.T + self.b

    def sample(self, obs: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu = self.mean(obs)
        std = np.exp(self.log_std)
        raw = mu + rng.normal(size=mu.shape) * std
        logp = gaussian_log_prob(raw, mu, self.log_std)
        return raw, mu, logp

    def log_prob(self, obs: np.ndarray, raw_actions: np.ndarray) -> np.ndarray:
        mu = self.mean(obs)
        return gaussian_log_prob(raw_actions, mu, self.log_std)

    def update(
        self,
        obs: np.ndarray,
        raw_actions: np.ndarray,
        old_logp: np.ndarray,
        advantages: np.ndarray,
        clip_ratio: float,
        lr: float,
        epochs: int = 4,
    ) -> float:
        obs = np.asarray(obs, dtype=float)
        raw_actions = np.asarray(raw_actions, dtype=float)
        old_logp = np.asarray(old_logp, dtype=float)
        advantages = np.asarray(advantages, dtype=float)

        # Normalize advantages for a more stable update.
        adv = advantages.copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        std = np.exp(self.log_std)
        var = std**2
        batch_size = float(len(obs))
        last_loss = 0.0

        for _ in range(epochs):
            mu = self.mean(obs)
            new_logp = gaussian_log_prob(raw_actions, mu, self.log_std)
            ratio = np.exp(new_logp - old_logp)

            # PPO clipped surrogate. We only backprop through the unclipped part.
            active = np.ones_like(ratio, dtype=float)
            active[(adv >= 0.0) & (ratio > (1.0 + clip_ratio))] = 0.0
            active[(adv < 0.0) & (ratio < (1.0 - clip_ratio))] = 0.0

            coeff = adv * ratio * active
            dmu = -(coeff[:, None] * (raw_actions - mu) / var[None, :])

            grad_W = dmu.T @ obs / batch_size
            grad_b = dmu.mean(axis=0)

            self.W -= lr * np.clip(grad_W, -5.0, 5.0)
            self.b -= lr * np.clip(grad_b, -5.0, 5.0)

            clipped = np.where(
                adv >= 0.0,
                np.minimum(ratio, 1.0 + clip_ratio),
                np.maximum(ratio, 1.0 - clip_ratio),
            )
            surrogate = np.minimum(ratio * adv, clipped * adv)
            last_loss = float(-np.mean(surrogate))

        return last_loss


class LinearCritic:
    """Centralized critic V(s) over the concatenated global state."""

    def __init__(self, state_dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed + 1)
        self.w = rng.normal(scale=0.05, size=(state_dim,))
        self.b = 0.0

    def value(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        return state @ self.w + self.b

    def update(self, states: np.ndarray, targets: np.ndarray, lr: float, epochs: int = 8) -> float:
        states = np.asarray(states, dtype=float)
        targets = np.asarray(targets, dtype=float)
        batch_size = float(len(states))
        last_loss = 0.0
        for _ in range(epochs):
            values = self.value(states)
            err = values - targets
            grad_w = states.T @ err / batch_size
            grad_b = float(np.mean(err))
            self.w -= lr * np.clip(grad_w, -5.0, 5.0)
            self.b -= lr * np.clip(grad_b, -5.0, 5.0)
            last_loss = float(0.5 * np.mean(err**2))
        return last_loss


@dataclass
class TrajectoryBatch:
    obs: np.ndarray
    global_states: np.ndarray
    raw_actions: np.ndarray
    logp: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    infos: List[Dict[str, float]]


class CTDEPPOTrainer:
    def __init__(
        self,
        env: GMDOEnv,
        actor: LinearGaussianActor,
        critic: LinearCritic,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        actor_lr: float = 0.001,
        critic_lr: float = 0.002,
        clip_ratio: float = 0.1,
        mini_batch_size: int = 64,
        entropy_coef: float = 0.01,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_ratio = clip_ratio
        self.mini_batch_size = mini_batch_size
        self.entropy_coef = entropy_coef
        self.rng = np.random.default_rng(seed)

    def collect_episode(self, deterministic: bool = False) -> TrajectoryBatch:
        obs = self.env.reset()
        done = False

        obs_hist: List[np.ndarray] = []
        global_state_hist: List[np.ndarray] = []
        raw_hist: List[np.ndarray] = []
        logp_hist: List[np.ndarray] = []
        value_hist: List[float] = []
        reward_hist: List[float] = []
        info_hist: List[Dict[str, float]] = []

        while not done:
            global_state = obs.reshape(-1)
            value = float(self.critic.value(global_state))
            if deterministic:
                raw = self.actor.mean(obs)
                logp = self.actor.log_prob(obs, raw)
            else:
                raw, _, logp = self.actor.sample(obs, self.rng)

            gains = self.env.transform_raw_actions(raw)
            next_obs, reward, done, info = self.env.step(gains)

            obs_hist.append(obs.copy())
            global_state_hist.append(global_state.copy())
            raw_hist.append(raw.copy())
            logp_hist.append(logp.copy())
            value_hist.append(value)
            reward_hist.append(float(reward))
            info_hist.append(info)

            obs = next_obs

        return TrajectoryBatch(
            obs=np.asarray(obs_hist, dtype=float),
            global_states=np.asarray(global_state_hist, dtype=float),
            raw_actions=np.asarray(raw_hist, dtype=float),
            logp=np.asarray(logp_hist, dtype=float),
            values=np.asarray(value_hist, dtype=float),
            rewards=np.asarray(reward_hist, dtype=float),
            infos=info_hist,
        )

    def compute_gae(self, rewards: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray(rewards, dtype=float)
        values = np.asarray(values, dtype=float)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + values
        return advantages, returns

    def update(self, batch: TrajectoryBatch, actor_epochs: int = 4, critic_epochs: int = 8) -> Dict[str, float]:
        advantages, returns = self.compute_gae(batch.rewards, batch.values)

        # Centralized critic update on the global state sequence.
        critic_loss = self.critic.update(batch.global_states, returns, lr=self.critic_lr, epochs=critic_epochs)

        # Shared actor update. All agents at time t share the same team advantage.
        T, N, obs_dim = batch.obs.shape
        act_dim = batch.raw_actions.shape[-1]
        obs_flat = batch.obs.reshape(T * N, obs_dim)
        raw_flat = batch.raw_actions.reshape(T * N, act_dim)
        logp_flat = batch.logp.reshape(T * N)
        adv_flat = np.repeat(advantages, N)
        actor_loss = self.actor.update(
            obs=obs_flat,
            raw_actions=raw_flat,
            old_logp=logp_flat,
            advantages=adv_flat,
            clip_ratio=self.clip_ratio,
            lr=self.actor_lr,
            epochs=actor_epochs,
        )

        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "episode_reward": float(np.sum(batch.rewards)),
            "final_global_cost": float(batch.infos[-1]["global_cost"]),
            "final_x_error": float(batch.infos[-1]["x_error_norm"]),
            "mean_alpha": float(np.mean([info["mean_alpha"] for info in batch.infos])),
            "mean_beta": float(np.mean([info["mean_beta"] for info in batch.infos])),
            "mean_consensus_penalty": float(np.mean([info["consensus_penalty"] for info in batch.infos])),
        }


def evaluate_policy(
    env: GMDOEnv,
    actor: LinearGaussianActor | None,
    deterministic: bool = True,
) -> Dict[str, np.ndarray]:
    obs = env.reset()
    done = False

    x_hist = []
    cost_hist = []
    reward_hist = []
    k1_hist = []
    k2_hist = []
    alpha_hist = []
    beta_hist = []

    while not done:
        if actor is None:
            gains = env.baseline_actions(obs)
        else:
            raw = actor.mean(obs) if deterministic else actor.sample(obs, env.rng)[0]
            gains = env.transform_raw_actions(raw)
        obs, reward, done, info = env.step(gains)
        x_hist.append(env.x.copy())
        cost_hist.append(info["global_cost"])
        reward_hist.append(reward)
        k1_hist.append(info["mean_k1"])
        k2_hist.append(info["mean_k2"])
        alpha_hist.append(info["mean_alpha"])
        beta_hist.append(info["mean_beta"])

    return {
        "x": np.asarray(x_hist),
        "cost": np.asarray(cost_hist),
        "reward": np.asarray(reward_hist),
        "k1": np.asarray(k1_hist),
        "k2": np.asarray(k2_hist),
        "alpha": np.asarray(alpha_hist),
        "beta": np.asarray(beta_hist),
    }


def summarize_convergence(
    traj: Dict[str, np.ndarray],
    x_star: float,
    dt: float,
    rel_band: float = 0.02,
) -> Dict[str, float | bool]:
    x_hist = np.asarray(traj["x"], dtype=float)
    cost_hist = np.asarray(traj["cost"], dtype=float)
    if x_hist.size == 0 or cost_hist.size == 0:
        return {
            "settled": False,
            "step": 0.0,
            "time": 0.0,
            "cost": float("nan"),
            "x_error": float("nan"),
        }

    tol = rel_band * abs(float(x_star))
    settled_mask = np.all(np.abs(x_hist - float(x_star)) <= tol, axis=1)
    # Count convergence only when the trajectory enters the tolerance band and
    # stays there for the remainder of the rollout. This avoids false positives
    # from trajectories that briefly touch x* and then diverge later.
    settled_suffix = np.logical_and.accumulate(settled_mask[::-1])[::-1]
    settled_idx = np.flatnonzero(settled_suffix)
    if settled_idx.size > 0:
        idx = int(settled_idx[0])
    else:
        idx = int(len(cost_hist) - 1)

    step = float(idx + 1)
    return {
        "settled": bool(settled_idx.size > 0),
        "step": step,
        "time": step * float(dt),
        "cost": float(cost_hist[idx]),
        "x_error": float(np.linalg.norm(x_hist[idx] - float(x_star))),
    }


def plot_trajectories(
    out_path: Path,
    baseline: Dict[str, np.ndarray],
    rl: Dict[str, np.ndarray] | None,
    x_star: float,
    f_star: float,
    title: str,
    dt: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=180)
    axes = axes.ravel()

    t_base = np.arange(len(baseline["x"])) * dt
    for i in range(baseline["x"].shape[1]):
        axes[0].plot(t_base, baseline["x"][:, i], label=f"agent {i+1}")
    axes[0].axhline(x_star, color="black", linestyle="--", linewidth=1.2, label="x*")
    axes[0].set_title("Baseline: agent states")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("x_i")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_base, baseline["cost"], color="#1f77b4", label="baseline")
    axes[1].axhline(f_star, color="black", linestyle="--", linewidth=1.2, label="f(x*)")
    axes[1].set_title("Baseline: global cost")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("cost")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    if rl is not None:
        t_rl = np.arange(len(rl["x"])) * dt
        for i in range(rl["x"].shape[1]):
            axes[2].plot(t_rl, rl["x"][:, i], label=f"agent {i+1}")
        axes[2].axhline(x_star, color="black", linestyle="--", linewidth=1.2, label="x*")
        axes[2].set_title("RL: agent states")
        axes[2].set_xlabel("time")
        axes[2].set_ylabel("x_i")
        axes[2].legend(fontsize=8, ncol=2)
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(t_rl, rl["cost"], color="#d62728", label="rl")
        axes[3].axhline(f_star, color="black", linestyle="--", linewidth=1.2, label="f(x*)")
        axes[3].set_title("RL: global cost")
        axes[3].set_xlabel("time")
        axes[3].set_ylabel("cost")
        axes[3].legend(fontsize=8)
        axes[3].grid(True, alpha=0.3)
    else:
        axes[2].axis("off")
        axes[3].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_gains(out_path: Path, rl: Dict[str, np.ndarray] | None, dt: float) -> None:
    if rl is None:
        return
    t = np.arange(len(rl["k1"])) * dt
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), dpi=180, sharex=True)
    axes[0].plot(t, rl["k1"], color="#2ca02c")
    axes[0].set_ylabel("K1")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, rl["k2"], color="#ff7f0e")
    axes[1].set_ylabel("K2")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(t, rl["alpha"], color="#9467bd", label="alpha")
    axes[2].plot(t, rl["beta"], color="#8c564b", linestyle="--", label="beta")
    axes[2].set_ylabel("gain")
    axes[2].set_xlabel("time")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_reward(out_path: Path, logs: List[Dict[str, float]], window: int = 10) -> None:
    if not logs:
        return

    steps = np.array([entry["step_count"] for entry in logs], dtype=float)
    rewards = np.array([entry["episode_reward"] for entry in logs], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=180)
    ax.plot(
        steps,
        rewards,
        marker="o",
        markersize=3.0,
        linewidth=1.0,
        alpha=0.35,
        color="#1f77b4",
        label="reward",
    )

    if len(rewards) >= window:
        kernel = np.ones(window, dtype=float) / float(window)
        smoothed = np.convolve(rewards, kernel, mode="valid")
        smooth_steps = steps[window - 1 :]
        ax.plot(
            smooth_steps,
            smoothed,
            color="#d62728",
            linewidth=2.0,
            label=f"{window}-episode moving average",
        )

    ax.set_title("Training reward over steps")
    ax.set_xlabel("training step")
    ax.set_ylabel("reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if TORCH_AVAILABLE:
    class TorchGaussianActor(nn.Module):
        """Shared Gaussian actor for decentralized execution."""

        def __init__(self, obs_dim: int, act_dim: int, seed: int = 0) -> None:
            super().__init__()
            torch.manual_seed(seed)
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, act_dim),
            )
            self.log_std = nn.Parameter(torch.full((act_dim,), -0.8))

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return self.net(obs)

        def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
            mean = self.forward(obs)
            std = torch.exp(torch.clamp(self.log_std, -4.0, 2.0)).expand_as(mean)
            return torch.distributions.Normal(mean, std)


    class TorchCritic(nn.Module):
        """Centralized critic V(s) over the concatenated global state."""

        def __init__(self, state_dim: int, seed: int = 0) -> None:
            super().__init__()
            torch.manual_seed(seed + 1)
            self.net = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return self.net(state).squeeze(-1)


    class TorchCTDEPPOTrainer:
        def __init__(
            self,
            env: GMDOEnv,
            actor: TorchGaussianActor,
            critic: TorchCritic,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            actor_lr: float = 0.001,
            critic_lr: float = 0.002,
            clip_ratio: float = 0.1,
            mini_batch_size: int = 64,
            entropy_coef: float = 0.01,
            seed: int = 0,
            device: str | None = None,
        ) -> None:
            self.env = env
            self.actor = actor
            self.critic = critic
            self.gamma = gamma
            self.gae_lambda = gae_lambda
            self.actor_lr = actor_lr
            self.critic_lr = critic_lr
            self.clip_ratio = clip_ratio
            self.mini_batch_size = mini_batch_size
            self.entropy_coef = entropy_coef
            self.device = resolve_torch_device(device)
            torch.manual_seed(seed)
            self.actor.to(self.device)
            self.critic.to(self.device)
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        def collect_episode(self, deterministic: bool = False) -> TrajectoryBatch:
            obs = self.env.reset()
            done = False

            obs_hist: List[np.ndarray] = []
            global_state_hist: List[np.ndarray] = []
            raw_hist: List[np.ndarray] = []
            logp_hist: List[np.ndarray] = []
            value_hist: List[float] = []
            reward_hist: List[float] = []
            info_hist: List[Dict[str, float]] = []

            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                state_t = torch.as_tensor(obs.reshape(-1), dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    dist = self.actor.dist(obs_t)
                    raw = dist.mean if deterministic else dist.sample()
                    logp = dist.log_prob(raw).sum(dim=-1)
                    value = self.critic(state_t)

                raw_np = raw.detach().cpu().numpy()
                gains = self.env.transform_raw_actions(raw_np)
                next_obs, reward, done, info = self.env.step(gains)

                obs_hist.append(obs.copy())
                global_state_hist.append(obs.reshape(-1).copy())
                raw_hist.append(raw_np.copy())
                logp_hist.append(logp.detach().cpu().numpy().copy())
                value_hist.append(float(value.item()))
                reward_hist.append(float(reward))
                info_hist.append(info)
                obs = next_obs

            return TrajectoryBatch(
                obs=np.asarray(obs_hist, dtype=float),
                global_states=np.asarray(global_state_hist, dtype=float),
                raw_actions=np.asarray(raw_hist, dtype=float),
                logp=np.asarray(logp_hist, dtype=float),
                values=np.asarray(value_hist, dtype=float),
                rewards=np.asarray(reward_hist, dtype=float),
                infos=info_hist,
            )

        def compute_gae(self, rewards: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            rewards = np.asarray(rewards, dtype=float)
            values = np.asarray(values, dtype=float)
            advantages = np.zeros_like(rewards)
            gae = 0.0
            next_value = 0.0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * next_value - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
                advantages[t] = gae
                next_value = values[t]
            returns = advantages + values
            return advantages, returns

        def update(self, batch: TrajectoryBatch, actor_epochs: int = 4, critic_epochs: int = 8) -> Dict[str, float]:
            advantages, returns = self.compute_gae(batch.rewards, batch.values)

            states = torch.as_tensor(batch.global_states, dtype=torch.float32, device=self.device)
            returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
            critic_loss = 0.0
            critic_loss_sum = 0.0
            critic_loss_count = 0
            critic_batch = min(self.mini_batch_size, len(states))
            for _ in range(critic_epochs):
                perm = torch.randperm(len(states), device=self.device)
                for start in range(0, len(states), critic_batch):
                    idx = perm[start : start + critic_batch]
                    self.critic_opt.zero_grad(set_to_none=True)
                    values = self.critic(states[idx])
                    loss = F.mse_loss(values, returns_t[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.75)
                    self.critic_opt.step()
                    critic_loss = float(loss.item())
                    critic_loss_sum += critic_loss
                    critic_loss_count += 1

            T, N, obs_dim = batch.obs.shape
            act_dim = batch.raw_actions.shape[-1]
            obs = torch.as_tensor(batch.obs.reshape(T * N, obs_dim), dtype=torch.float32, device=self.device)
            raw = torch.as_tensor(batch.raw_actions.reshape(T * N, act_dim), dtype=torch.float32, device=self.device)
            old_logp = torch.as_tensor(batch.logp.reshape(T * N), dtype=torch.float32, device=self.device)
            adv = torch.as_tensor(np.repeat(advantages, N), dtype=torch.float32, device=self.device)
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

            actor_loss = 0.0
            actor_loss_sum = 0.0
            actor_loss_count = 0
            actor_batch = min(self.mini_batch_size, len(obs))
            for _ in range(actor_epochs):
                perm = torch.randperm(len(obs), device=self.device)
                for start in range(0, len(obs), actor_batch):
                    idx = perm[start : start + actor_batch]
                    dist = self.actor.dist(obs[idx])
                    new_logp = dist.log_prob(raw[idx]).sum(dim=-1)
                    ratio = torch.exp(new_logp - old_logp[idx])
                    clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                    entropy = dist.entropy().sum(dim=-1).mean()
                    loss = -torch.mean(torch.min(ratio * adv[idx], clipped * adv[idx])) - self.entropy_coef * entropy
                    self.actor_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.75)
                    self.actor_opt.step()
                    actor_loss = float(loss.item())
                    actor_loss_sum += actor_loss
                    actor_loss_count += 1

            return {
                "critic_loss": critic_loss_sum / max(1, critic_loss_count),
                "actor_loss": actor_loss_sum / max(1, actor_loss_count),
                "episode_reward": float(np.sum(batch.rewards)),
                "final_global_cost": float(batch.infos[-1]["global_cost"]),
                "final_x_error": float(batch.infos[-1]["x_error_norm"]),
                "mean_alpha": float(np.mean([info["mean_alpha"] for info in batch.infos])),
                "mean_beta": float(np.mean([info["mean_beta"] for info in batch.infos])),
                "mean_consensus_penalty": float(np.mean([info["consensus_penalty"] for info in batch.infos])),
            }


    @torch.no_grad()
    def evaluate_policy_torch(
        env: GMDOEnv,
        actor: TorchGaussianActor,
        deterministic: bool = True,
    ) -> Dict[str, np.ndarray]:
        device = next(actor.parameters()).device
        obs = env.reset()
        done = False

        x_hist = []
        cost_hist = []
        reward_hist = []
        k1_hist = []
        k2_hist = []
        alpha_hist = []
        beta_hist = []

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            dist = actor.dist(obs_t)
            raw = dist.mean if deterministic else dist.sample()
            gains = env.transform_raw_actions(raw.detach().cpu().numpy())
            obs, reward, done, info = env.step(gains)
            x_hist.append(env.x.copy())
            cost_hist.append(info["global_cost"])
            reward_hist.append(reward)
            k1_hist.append(info["mean_k1"])
            k2_hist.append(info["mean_k2"])
            alpha_hist.append(info["mean_alpha"])
            beta_hist.append(info["mean_beta"])

        return {
            "x": np.asarray(x_hist),
            "cost": np.asarray(cost_hist),
            "reward": np.asarray(reward_hist),
            "k1": np.asarray(k1_hist),
            "k2": np.asarray(k2_hist),
            "alpha": np.asarray(alpha_hist),
            "beta": np.asarray(beta_hist),
        }


def train_and_evaluate_torch(args: argparse.Namespace) -> None:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        plot_horizon = max(args.horizon, int(round(100.0 / args.dt)))

        env = GMDOEnv(
            dt=args.dt,
            horizon=args.horizon,
            seed=args.seed,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            consensus_lambda=args.consensus_lambda,
            K_min=args.k_min,
            K_max=args.k_max,
            delta_alpha_max=args.delta_alpha_max,
            alpha_floor=args.alpha_floor,
            alpha_max=args.alpha_max,
            nominal_k1=args.nominal_k1,
            nominal_k2=args.nominal_k2,
            nominal_delta_alpha=args.nominal_delta_alpha,
            delta_k1_scale=args.delta_k1_scale,
            delta_k2_scale=args.delta_k2_scale,
            delta_alpha_scale=args.delta_alpha_scale,
        )
        print(f"Initial state x0 = {np.array2string(PAPER_INITIAL_X, precision=3, separator=', ')}")

        plot_paper_graph(outdir / "paper_graph.png")

        baseline_env = GMDOEnv(
            dt=args.dt,
            horizon=plot_horizon,
            seed=args.seed,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            consensus_lambda=args.consensus_lambda,
            K_min=args.k_min,
            K_max=args.k_max,
            delta_alpha_max=args.delta_alpha_max,
            alpha_floor=args.alpha_floor,
            alpha_max=args.alpha_max,
            nominal_k1=args.nominal_k1,
            nominal_k2=args.nominal_k2,
            nominal_delta_alpha=args.nominal_delta_alpha,
            delta_k1_scale=args.delta_k1_scale,
            delta_k2_scale=args.delta_k2_scale,
            delta_alpha_scale=args.delta_alpha_scale,
        )
        baseline_result = evaluate_policy(baseline_env, actor=None, deterministic=True)
        baseline_summary = summarize_convergence(baseline_result, env.x_star, args.dt)
        print(
            f"Baseline convergence cost: {baseline_summary['cost']:.4f} "
            f"at t={baseline_summary['time']:.2f} "
            f"(x_err={baseline_summary['x_error']:.4f}, settled={baseline_summary['settled']})"
        )

        if args.mode == "baseline":
            plot_trajectories(
                outdir / "baseline_only.png",
                baseline_result,
                None,
                env.x_star,
                env.f_star,
                "Baseline GM-driven DO on the paper graph",
                args.dt,
            )
            return

        actor = TorchGaussianActor(env.obs_dim, env.act_dim, seed=args.seed)
        critic = TorchCritic(env.N * env.obs_dim, seed=args.seed)
        trainer = TorchCTDEPPOTrainer(
            env=env,
            actor=actor,
            critic=critic,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            clip_ratio=args.clip_ratio,
            mini_batch_size=args.mini_batch_size,
            entropy_coef=args.entropy_coef,
            seed=args.seed,
            device=args.device,
        )

        logs: List[Dict[str, float]] = []
        print(f"Using torch backend on {trainer.device}")
        global_steps = 0
        episode = 0
        while global_steps < args.total_steps:
            remaining = args.total_steps - global_steps
            original_horizon = env.horizon
            env.horizon = min(original_horizon, remaining)
            batch = trainer.collect_episode(deterministic=False)
            env.horizon = original_horizon
            stats = trainer.update(batch, actor_epochs=args.actor_epochs, critic_epochs=args.critic_epochs)
            global_steps += len(batch.rewards)
            episode += 1
            stats["step_count"] = float(global_steps)
            logs.append(stats)
            if episode % args.log_every == 0 or episode == 1:
                print(
                    f"episode {episode:4d} | "
                    f"steps {global_steps:5d} | "
                    f"reward {stats['episode_reward']:+.3f} | "
                    f"cost {stats['final_global_cost']:.4f} | "
                    f"x_err {stats['final_x_error']:.4f} | "
                    f"cons {stats['mean_consensus_penalty']:.4f} | "
                    f"actor {stats['actor_loss']:.4f} | critic {stats['critic_loss']:.4f}"
                )

        eval_env = GMDOEnv(
            dt=args.dt,
            horizon=plot_horizon,
            seed=args.seed,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            consensus_lambda=args.consensus_lambda,
            K_min=args.k_min,
            K_max=args.k_max,
            delta_alpha_max=args.delta_alpha_max,
            alpha_floor=args.alpha_floor,
            alpha_max=args.alpha_max,
            nominal_k1=args.nominal_k1,
            nominal_k2=args.nominal_k2,
            nominal_delta_alpha=args.nominal_delta_alpha,
            delta_k1_scale=args.delta_k1_scale,
            delta_k2_scale=args.delta_k2_scale,
            delta_alpha_scale=args.delta_alpha_scale,
        )
        rl_result = evaluate_policy_torch(eval_env, actor=actor, deterministic=True)
        rl_summary = summarize_convergence(rl_result, env.x_star, args.dt)
        print(
            f"RL convergence cost: {rl_summary['cost']:.4f} "
            f"at t={rl_summary['time']:.2f} "
            f"(x_err={rl_summary['x_error']:.4f}, settled={rl_summary['settled']})"
        )

        plot_trajectories(
            outdir / "trajectories.png",
            baseline_result,
            rl_result,
            env.x_star,
            env.f_star,
            "Baseline vs CTDE-PPO on the paper graph",
            args.dt,
        )
        plot_gains(outdir / "rl_gains.png", rl_result, args.dt)
        plot_training_reward(outdir / "reward_vs_steps.png", logs)
        torch.save(actor.state_dict(), outdir / "actor_torch.pt")
        torch.save(critic.state_dict(), outdir / "critic_torch.pt")
        np.savez(
            outdir / "training_logs.npz",
            episodes=np.array([d["episode_reward"] for d in logs], dtype=float),
            episode_rewards=np.array([d["episode_reward"] for d in logs], dtype=float),
            costs=np.array([d["final_global_cost"] for d in logs], dtype=float),
            errors=np.array([d["final_x_error"] for d in logs], dtype=float),
            steps=np.array([d["step_count"] for d in logs], dtype=float),
        )


def train_and_evaluate_numpy(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_horizon = max(args.horizon, int(round(100.0 / args.dt)))

    env = GMDOEnv(
        dt=args.dt,
        horizon=args.horizon,
        seed=args.seed,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        consensus_lambda=args.consensus_lambda,
        K_min=args.k_min,
        K_max=args.k_max,
        delta_alpha_max=args.delta_alpha_max,
        alpha_floor=args.alpha_floor,
        alpha_max=args.alpha_max,
        nominal_k1=args.nominal_k1,
        nominal_k2=args.nominal_k2,
        nominal_delta_alpha=args.nominal_delta_alpha,
        delta_k1_scale=args.delta_k1_scale,
        delta_k2_scale=args.delta_k2_scale,
        delta_alpha_scale=args.delta_alpha_scale,
    )
    print(f"Using hardcoded initial state x0 = {np.array2string(PAPER_INITIAL_X, precision=3, separator=', ')}")

    plot_paper_graph(outdir / "paper_graph.png")

    # Baseline rollout.
    baseline_env = GMDOEnv(
        dt=args.dt,
        horizon=plot_horizon,
        seed=args.seed,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        consensus_lambda=args.consensus_lambda,
        K_min=args.k_min,
        K_max=args.k_max,
        delta_alpha_max=args.delta_alpha_max,
        alpha_floor=args.alpha_floor,
        alpha_max=args.alpha_max,
        nominal_k1=args.nominal_k1,
        nominal_k2=args.nominal_k2,
        nominal_delta_alpha=args.nominal_delta_alpha,
        delta_k1_scale=args.delta_k1_scale,
        delta_k2_scale=args.delta_k2_scale,
        delta_alpha_scale=args.delta_alpha_scale,
    )
    baseline_result = evaluate_policy(baseline_env, actor=None, deterministic=True)
    baseline_summary = summarize_convergence(baseline_result, env.x_star, args.dt)
    print(
        f"Baseline convergence cost: {baseline_summary['cost']:.4f} "
        f"at t={baseline_summary['time']:.2f} "
        f"(x_err={baseline_summary['x_error']:.4f}, settled={baseline_summary['settled']})"
    )

    if args.mode == "baseline":
        plot_trajectories(
            outdir / "baseline_only.png",
            baseline_result,
            None,
            env.x_star,
            env.f_star,
            "Baseline GM-driven DO on the paper graph",
            args.dt,
        )
        return

    actor = LinearGaussianActor(env.obs_dim, env.act_dim, seed=args.seed)
    critic = LinearCritic(env.N * env.obs_dim, seed=args.seed)
    trainer = CTDEPPOTrainer(
        env=env,
        actor=actor,
        critic=critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        clip_ratio=args.clip_ratio,
        mini_batch_size=args.mini_batch_size,
        entropy_coef=args.entropy_coef,
        seed=args.seed,
    )

    logs: List[Dict[str, float]] = []
    global_steps = 0
    episode = 0
    while global_steps < args.total_steps:
        remaining = args.total_steps - global_steps
        original_horizon = env.horizon
        env.horizon = min(original_horizon, remaining)
        batch = trainer.collect_episode(deterministic=False)
        env.horizon = original_horizon
        stats = trainer.update(batch, actor_epochs=args.actor_epochs, critic_epochs=args.critic_epochs)
        global_steps += len(batch.rewards)
        episode += 1
        stats["step_count"] = float(global_steps)
        logs.append(stats)
        if episode % args.log_every == 0 or episode == 1:
            print(
                f"episode {episode:4d} | "
                f"steps {global_steps:5d} | "
                f"reward {stats['episode_reward']:+.3f} | "
                f"cost {stats['final_global_cost']:.4f} | "
                f"x_err {stats['final_x_error']:.4f} | "
                f"cons {stats['mean_consensus_penalty']:.4f} | "
                f"actor {stats['actor_loss']:.4f} | critic {stats['critic_loss']:.4f}"
            )

    # Evaluate the trained actor deterministically.
    eval_env = GMDOEnv(
        dt=args.dt,
        horizon=plot_horizon,
        seed=args.seed,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        consensus_lambda=args.consensus_lambda,
        K_min=args.k_min,
        K_max=args.k_max,
        delta_alpha_max=args.delta_alpha_max,
        alpha_floor=args.alpha_floor,
        alpha_max=args.alpha_max,
        nominal_k1=args.nominal_k1,
        nominal_k2=args.nominal_k2,
        nominal_delta_alpha=args.nominal_delta_alpha,
        delta_k1_scale=args.delta_k1_scale,
        delta_k2_scale=args.delta_k2_scale,
        delta_alpha_scale=args.delta_alpha_scale,
    )
    rl_result = evaluate_policy(eval_env, actor=actor, deterministic=True)
    rl_summary = summarize_convergence(rl_result, env.x_star, args.dt)
    print(
        f"RL convergence cost: {rl_summary['cost']:.4f} "
        f"at t={rl_summary['time']:.2f} "
        f"(x_err={rl_summary['x_error']:.4f}, settled={rl_summary['settled']})"
    )

    # Save figures and model parameters.
    plot_trajectories(
        outdir / "trajectories.png",
        baseline_result,
        rl_result,
        env.x_star,
        env.f_star,
        "Baseline vs CTDE-PPO on the paper graph",
        args.dt,
    )
    plot_gains(outdir / "rl_gains.png", rl_result, args.dt)
    plot_training_reward(outdir / "reward_vs_steps.png", logs)
    np.savez(outdir / "actor_linear.npz", W=actor.W, b=actor.b, log_std=actor.log_std)
    np.savez(outdir / "critic_linear.npz", w=critic.w, b=np.array([critic.b]))
    np.savez(
        outdir / "training_logs.npz",
        episodes=np.array([d["episode_reward"] for d in logs], dtype=float),
        episode_rewards=np.array([d["episode_reward"] for d in logs], dtype=float),
        costs=np.array([d["final_global_cost"] for d in logs], dtype=float),
        errors=np.array([d["final_x_error"] for d in logs], dtype=float),
        steps=np.array([d["step_count"] for d in logs], dtype=float),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CTDE-PPO gain scheduling for the paper's 5-agent graph.")
    parser.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device name, e.g. cuda, cuda:0, or cpu; falls back to CPU if CUDA cannot initialize.",
    )
    parser.add_argument("--mode", choices=["baseline", "rl"], default="rl")
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--total-steps", type=int, default=15000)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="results_ctde_ppo")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--actor-lr", type=float, default=0.001)
    parser.add_argument("--critic-lr", type=float, default=0.002)
    parser.add_argument("--clip-ratio", type=float, default=0.1)
    parser.add_argument("--actor-epochs", type=int, default=4)
    parser.add_argument("--critic-epochs", type=int, default=8)
    parser.add_argument("--mini-batch-size", type=int, default=64)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--lambda1", type=float, default=0.2)
    parser.add_argument("--lambda2", type=float, default=0.01)
    parser.add_argument("--consensus-lambda", type=float, default=0.35)
    parser.add_argument("--k-min", type=float, default=0.05)
    parser.add_argument("--k-max", type=float, default=3.0)
    parser.add_argument("--delta-alpha-max", type=float, default=5.0)
    parser.add_argument("--alpha-floor", type=float, default=0.05)
    parser.add_argument("--alpha-max", type=float, default=20.0)
    parser.add_argument("--nominal-k1", type=float, default=4.0)
    parser.add_argument("--nominal-k2", type=float, default=0.05)
    parser.add_argument("--nominal-delta-alpha", type=float, default=0.5)
    parser.add_argument("--delta-k1-scale", type=float, default=1.0)
    parser.add_argument("--delta-k2-scale", type=float, default=0.05)
    parser.add_argument("--delta-alpha-scale", type=float, default=0.75)
    return parser.parse_args()


def train_and_evaluate(args: argparse.Namespace) -> None:
    backend = args.backend
    if backend == "auto":
        backend = "torch" if TORCH_AVAILABLE else "numpy"
    if backend == "torch":
        if not TORCH_AVAILABLE:
            print("Torch backend requested but torch is not available here; falling back to NumPy.")
            backend = "numpy"
        else:
            return train_and_evaluate_torch(args)
    return train_and_evaluate_numpy(args)


def main() -> None:
    args = parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
