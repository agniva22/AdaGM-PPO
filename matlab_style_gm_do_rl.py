#!/usr/bin/env python3
"""
Matlab-style GM-driven distributed optimization with CTDE-PPO gain scheduling.

This file is a more faithful Python port of the earlier Matlab baseline in
`fn_DO/fn_DO.m` and `fn_DO/main_DO.m`:

* same 5-agent digraph
* same initial condition x(0) = [4, 4.8, 6, 2, 0.4]^T
* same positive left eigenvector weights used in the Matlab code
* same continuous-time x/v/alpha structure from equation (7)

Two modes are provided:

* baseline: fixed K1/K2 and the heuristic beta = ||epsilon||^2 update
* rl: CTDE-PPO learns decentralized schedules for K1, K2, and the alpha-rate

This file is now fully self-contained: it vendors the graph, dynamics,
plotting, and PPO trainer used by the Matlab-style experiment.
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

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
except Exception:  # pragma: no cover
    nx = None


EPS = 1e-12


MATLAB_INITIAL_X = np.array([4.0, 4.8, 6.0, 2.0, 0.4], dtype=float)
MATLAB_INITIAL_ALPHA = 0.1

# Matches the Matlab line:
#   w = n^2 * [0.3774, 0.2516, 0.1509, 0.1258, 0.0943]'
MATLAB_LEFT_EIGENVECTOR = np.array(
    [9.43396226, 6.28930818, 3.77358491, 3.14465409, 2.35849057],
    dtype=float,
)


def build_paper_adjacency() -> np.ndarray:
    """Return the exact weighted digraph used in Fig. 1."""
    A = np.zeros((5, 5), dtype=float)
    A[0, 4] = 0.2  # 1 -> 5
    A[1, 0] = 0.3  # 2 -> 1
    A[2, 1] = 0.5  # 3 -> 2
    A[3, 2] = 0.6  # 4 -> 3
    A[4, 3] = 0.8  # 5 -> 4
    return A


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


def laplacian_from_adjacency(A: np.ndarray) -> np.ndarray:
    return np.diag(A.sum(axis=1)) - A


def left_eigenvector_from_laplacian(L: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eig(L.T)
    idx = int(np.argmin(np.abs(eigvals)))
    w = np.real(eigvecs[:, idx])
    w = np.abs(w)
    w = np.maximum(w, EPS)
    return w / w.sum()


def paper_graph_positions() -> Dict[int, Tuple[float, float]]:
    return {
        1: (0.0, 1.0),
        2: (1.0, 0.15),
        3: (0.55, -0.95),
        4: (-0.55, -0.95),
        5: (-1.0, 0.15),
    }


def resolve_torch_device(device: str | None) -> torch.device:
    """Resolve a torch device, falling back to CPU if CUDA cannot initialize."""
    requested = "cuda" if device is None else device
    resolved = torch.device(requested)
    if resolved.type != "cuda":
        return resolved

    try:
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


class MatlabStyleGMDOEnv:
    """Continuous-time GM-driven DO with a Matlab-style step integrator."""

    def __init__(
        self,
        control_dt: float = 0.1,
        sim_dt: float = 0.001,
        horizon: int = 100,
        seed: int = 42,
        lambda1: float = 0.2,
        lambda2: float = 0.01,
        consensus_lambda: float = 0.35,
        k_min: float = 0.05,
        k_max: float = 5.0,
        beta_max: float = 50.0,
        alpha_floor: float = 1e-12,
        alpha_max: float = 1e6,
        baseline_k1: float = 2.0,
        baseline_k2: float = 2.0,
        nominal_k1: float = 2.0,
        nominal_k2: float = 2.0,
        nominal_delta_alpha: float = 10.0,
        delta_k1_scale: float = 2.0,
        delta_k2_scale: float = 2.0,
        delta_alpha_scale: float = 10.0,
        initial_x: np.ndarray | None = None,
        beta_source: str = "policy",
        integrator: str = "rk4",
    ) -> None:
        if beta_source not in {"policy", "heuristic"}:
            raise ValueError("beta_source must be 'policy' or 'heuristic'.")
        if integrator not in {"rk4", "euler"}:
            raise ValueError("integrator must be 'rk4' or 'euler'.")

        self.A = build_paper_adjacency()
        self.N = 5
        self.w_left = MATLAB_LEFT_EIGENVECTOR.copy()
        self.control_dt = float(control_dt)
        self.sim_dt = float(sim_dt)
        self.horizon = int(horizon)
        self.n_substeps = max(1, int(round(self.control_dt / self.sim_dt)))
        self.sub_dt = self.control_dt / float(self.n_substeps)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.consensus_lambda = consensus_lambda
        self.k_min = k_min
        self.k_max = k_max
        self.beta_max = beta_max
        self.alpha_floor = alpha_floor
        self.alpha_max = alpha_max
        self.baseline_k1 = baseline_k1
        self.baseline_k2 = baseline_k2
        self.nominal_k1 = nominal_k1
        self.nominal_k2 = nominal_k2
        self.nominal_delta_alpha = nominal_delta_alpha
        self.delta_k1_scale = delta_k1_scale
        self.delta_k2_scale = delta_k2_scale
        self.delta_alpha_scale = delta_alpha_scale
        self.beta_source = beta_source
        self.integrator = integrator
        self.x_min = 1e-12
        self.x_max = 1e6
        self.v_min = 1e-12
        self.v_max = 1e6
        self.rng = np.random.default_rng(seed)
        self.x_star, self.f_star = solve_global_optimum()

        if initial_x is None:
            initial_x = MATLAB_INITIAL_X.copy()
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
        self.v = self.x.copy()
        self.alpha = np.full(self.N, MATLAB_INITIAL_ALPHA, dtype=float)
        self.beta = np.zeros(self.N, dtype=float)
        self.step_index = 0
        self.time = 0.0
        self.prev_global_cost = global_cost_vector(self.x)
        return self._obs()

    def _clip_positive(self, z: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return np.clip(np.asarray(z, dtype=float), lo, hi)

    def _log(self, z: np.ndarray) -> np.ndarray:
        return np.log(self._clip_positive(z, self.x_min, self.x_max))

    def _consensus_term(self, z: np.ndarray) -> np.ndarray:
        # Row-wise sum_j a_ij (z_j - z_i)
        return np.sum(self.A * (z[None, :] - z[:, None]), axis=1)

    def _epsilon(self, x: np.ndarray) -> np.ndarray:
        # epsilon_i = w_i * sum_j a_ij (ln x_i - ln x_j)
        log_x = self._log(x)
        return self.w_left * np.sum(self.A * (log_x[:, None] - log_x[None, :]), axis=1)

    def _grad_x(self, x: np.ndarray) -> np.ndarray:
        return np.array([local_cost_and_grad(i, float(x[i]))[1] for i in range(self.N)], dtype=float)

    def _current_beta(self, x: np.ndarray, gains: np.ndarray) -> np.ndarray:
        if self.beta_source == "heuristic":
            return np.square(self._epsilon(x))
        return np.clip(gains[:, 2], 0.0, self.beta_max)

    def _rhs(
        self,
        x: np.ndarray,
        v: np.ndarray,
        alpha: np.ndarray,
        gains: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        k1 = np.clip(gains[:, 0], self.k_min, self.k_max)
        k2 = np.clip(gains[:, 1], self.k_min, self.k_max)
        beta = self._current_beta(x, gains)

        log_x = self._log(x)
        log_v = self._log(v)
        consensus_x = self._consensus_term(log_x)
        consensus_v = self._consensus_term(log_v)
        grad_x = self._grad_x(x)
        gain_sum = alpha + beta

        # This is the scalar-agent version of equation (7) / (10).
        x_dot = (
            k1 * gain_sum * self.w_left * x * consensus_x
            + x * self.w_left * consensus_v
            - k2 * np.square(x) * grad_x
        )
        v_dot = k1 * gain_sum * self.w_left * v * (-consensus_x)
        alpha_dot = beta
        return x_dot, v_dot, alpha_dot, beta

    def _project_state(self, x: np.ndarray, v: np.ndarray, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self._clip_positive(x, self.x_min, self.x_max)
        v = self._clip_positive(v, self.v_min, self.v_max)
        alpha = self._clip_positive(alpha, self.alpha_floor, self.alpha_max)
        return x, v, alpha

    def _integrate_interval(self, gains: np.ndarray) -> None:
        x = self.x.copy()
        v = self.v.copy()
        alpha = self.alpha.copy()

        for _ in range(self.n_substeps):
            if self.integrator == "euler":
                dx, dv, da, _ = self._rhs(x, v, alpha, gains)
                x = x + self.sub_dt * dx
                v = v + self.sub_dt * dv
                alpha = alpha + self.sub_dt * da
            else:
                k1_x, k1_v, k1_a, _ = self._rhs(x, v, alpha, gains)
                k2_x, k2_v, k2_a, _ = self._rhs(
                    x + 0.5 * self.sub_dt * k1_x,
                    v + 0.5 * self.sub_dt * k1_v,
                    alpha + 0.5 * self.sub_dt * k1_a,
                    gains,
                )
                k3_x, k3_v, k3_a, _ = self._rhs(
                    x + 0.5 * self.sub_dt * k2_x,
                    v + 0.5 * self.sub_dt * k2_v,
                    alpha + 0.5 * self.sub_dt * k2_a,
                    gains,
                )
                k4_x, k4_v, k4_a, _ = self._rhs(
                    x + self.sub_dt * k3_x,
                    v + self.sub_dt * k3_v,
                    alpha + self.sub_dt * k3_a,
                    gains,
                )
                x = x + (self.sub_dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
                v = v + (self.sub_dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
                alpha = alpha + (self.sub_dt / 6.0) * (k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a)

            x, v, alpha = self._project_state(x, v, alpha)

        self.x = x
        self.v = v
        self.alpha = alpha

    def _obs(self) -> np.ndarray:
        x_bar = float(np.mean(self.x))
        eps = self._epsilon(self.x)
        grad_x = self._grad_x(self.x)
        time_feature = np.full(self.N, self.step_index / max(1, self.horizon - 1), dtype=float)
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
        """Map unconstrained actor outputs to bounded positive gains.

        For the alpha-rate, we keep the learned policy in a residual band around
        the Matlab-style heuristic beta = ||epsilon||^2. That keeps exploration
        inside the same stability envelope as the baseline while still letting
        RL accelerate or slow down the adaptation online.
        """
        raw_actions = np.asarray(raw_actions, dtype=float)
        k1 = np.clip(self.nominal_k1 + self.delta_k1_scale * np.tanh(raw_actions[:, 0]), self.k_min, self.k_max)
        k2 = np.clip(self.nominal_k2 + self.delta_k2_scale * np.tanh(raw_actions[:, 1]), self.k_min, self.k_max)
        if self.beta_source == "heuristic":
            beta = np.square(self._epsilon(self.x))
        else:
            beta_ref = np.clip(np.square(self._epsilon(self.x)), 0.0, self.beta_max)
            beta_scale = 1.0 + 0.5 * np.tanh(raw_actions[:, 2])
            beta = np.clip(beta_ref * beta_scale, 0.0, self.beta_max)
        return np.column_stack([k1, k2, beta])

    def baseline_actions(
        self,
        obs: np.ndarray,
        k1_fixed: float | None = None,
        k2_fixed: float | None = None,
    ) -> np.ndarray:
        eps = obs[:, 0]
        if k1_fixed is None:
            k1_fixed = self.baseline_k1
        if k2_fixed is None:
            k2_fixed = self.baseline_k2
        # Matched to the heuristic alpha_dot = ||epsilon||^2 used in the Matlab code.
        beta = np.square(eps)
        if self.beta_source == "policy":
            beta = np.clip(beta, 0.0, self.beta_max)
        return np.column_stack(
            [
                np.full(self.N, k1_fixed, dtype=float),
                np.full(self.N, k2_fixed, dtype=float),
                beta,
            ]
        )

    def step(self, gains: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        gains = np.asarray(gains, dtype=float)
        self._integrate_interval(gains)
        self.step_index += 1
        self.time = self.step_index * self.control_dt

        # The instantaneous beta visible to the next observation.
        self.beta = self._current_beta(self.x, gains)

        obs = self._obs()
        global_cost = global_cost_vector(self.x)
        overshoot = max(0.0, global_cost - self.f_star)
        x_bar = float(np.mean(self.x))
        eps = np.abs(self._epsilon(self.x))
        consensus_penalty = np.abs(self.x - x_bar) + eps
        prev_cost = float(self.prev_global_cost)
        cost_progress = (prev_cost - global_cost) / max(1.0, prev_cost)

        k1 = np.clip(gains[:, 0], self.k_min, self.k_max)
        k2 = np.clip(gains[:, 1], self.k_min, self.k_max)
        state_term = np.mean(np.abs(self.x - self.x_star)) / max(1.0, abs(float(self.x_star)))
        overshoot_term = overshoot / max(1.0, float(self.f_star))
        gain_term = (np.abs(k1 - self.nominal_k1) + np.abs(k2 - self.nominal_k2)) / max(1.0, self.k_max)
        consensus_scale = max(1.0, float(np.mean(np.abs(MATLAB_INITIAL_X - float(self.x_star)))))
        consensus_term = consensus_penalty / consensus_scale
        settled_bonus = 0.15 if np.all(np.abs(self.x - self.x_star) <= 0.02 * abs(float(self.x_star))) else 0.0
        per_agent_reward = (
            cost_progress
            - 0.2 * state_term
            - self.lambda1 * overshoot_term
            - self.lambda2 * gain_term
            - self.consensus_lambda * consensus_term
            - 0.01
            + settled_bonus
        )
        team_reward = float(np.clip(np.mean(per_agent_reward), -5.0, 2.0))
        self.prev_global_cost = float(global_cost)
        done = self.step_index >= self.horizon

        info = {
            "global_cost": global_cost,
            "overshoot": overshoot,
            "consensus_penalty": float(np.mean(consensus_penalty)),
            "x_error_norm": float(np.linalg.norm(self.x - self.x_star)),
            "mean_alpha": float(np.mean(self.alpha)),
            "mean_beta": float(np.mean(self.beta)),
            "mean_k1": float(np.mean(k1)),
            "mean_k2": float(np.mean(k2)),
            "time": self.time,
        }
        return obs, team_reward, done, info


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


def evaluate_policy(
    env: MatlabStyleGMDOEnv,
    actor: "LinearGaussianActor" | None,
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


if TORCH_AVAILABLE:
    @torch.no_grad()
    def evaluate_policy_torch(
        env: MatlabStyleGMDOEnv,
        actor: "TorchGaussianActor",
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


def _style_paper_axis(ax: plt.Axes, tick_size: int = 11) -> None:
    ax.set_facecolor("white")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.75)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_size,
        width=1.1,
        length=4,
        direction="in",
        top=True,
        right=True,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=0.9,
        length=2,
        direction="in",
        top=True,
        right=True,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def _paper_zoom_window(
    t: np.ndarray,
    series: np.ndarray,
    center: float,
    zoom_start: float = 20.0,
    zoom_end: float = 50.0,
) -> Tuple[float, float, float, float]:
    if t.size == 0 or series.size == 0:
        pad = max(0.02, 0.05 * max(1.0, abs(float(center))))
        return zoom_start, zoom_end, center - pad, center + pad

    x0 = float(np.clip(zoom_start, float(t[0]), float(t[-1])))
    x1 = float(np.clip(zoom_end, float(t[0]), float(t[-1])))
    if x1 <= x0:
        x0 = float(t[0])
        x1 = float(t[-1])

    mask = (t >= x0) & (t <= x1)
    window = np.asarray(series)[mask]
    if window.size == 0:
        pad = max(0.02, 0.05 * max(1.0, abs(float(center))))
        return x0, x1, center - pad, center + pad

    y_min = float(np.min(window))
    y_max = float(np.max(window))
    span = y_max - y_min
    pad = max(0.02, 0.15 * max(span, 1e-6))
    return x0, x1, y_min - pad, y_max + pad


def _add_state_inset(
    ax: plt.Axes,
    t: np.ndarray,
    x_hist: np.ndarray,
    x_star: float,
    colors: List[str],
    linestyles: List[str],
) -> None:
    axins = inset_axes(ax, width="38%", height="34%", loc="upper right", borderpad=1.0)
    for i in range(x_hist.shape[1]):
        axins.plot(
            t,
            x_hist[:, i],
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            linewidth=1.0,
        )
    axins.axhline(x_star, color="black", linestyle="--", linewidth=0.9)
    zoom_start, zoom_end, y_min, y_max = _paper_zoom_window(
        t,
        x_hist,
        center=float(x_star),
    )
    axins.set_xlim(zoom_start, zoom_end)
    axins.set_ylim(y_min, y_max)
    axins.set_xticks([20.0, 50.0] if zoom_end > zoom_start else [zoom_end])
    axins.set_yticks(np.linspace(y_min, y_max, 3))
    _style_paper_axis(axins, tick_size=8)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="black", lw=0.8)


def _add_cost_inset(
    ax: plt.Axes,
    t: np.ndarray,
    cost: np.ndarray,
    f_star: float,
    color: str,
) -> None:
    axins = inset_axes(ax, width="38%", height="34%", loc="upper right", borderpad=1.0)
    axins.plot(t, cost, color=color, linestyle="--", linewidth=1.0)
    axins.axhline(f_star, color="black", linestyle="--", linewidth=0.9)
    zoom_start, zoom_end, y_min, y_max = _paper_zoom_window(
        t,
        cost,
        center=float(f_star),
    )
    axins.set_xlim(zoom_start, zoom_end)
    axins.set_ylim(y_min, y_max)
    axins.set_xticks([20.0, 50.0] if zoom_end > zoom_start else [zoom_end])
    axins.set_yticks(np.linspace(y_min, y_max, 3))
    _style_paper_axis(axins, tick_size=8)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="black", lw=0.8)


def _plot_state_panel(
    ax: plt.Axes,
    t: np.ndarray,
    x_hist: np.ndarray,
    x_star: float,
) -> None:
    state_colors = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]
    state_styles = ["--", ":", "-.", "-", (0, (5, 1))]

    for i in range(x_hist.shape[1]):
        ax.plot(
            t,
            x_hist[:, i],
            label=str(i + 1),
            color=state_colors[i % len(state_colors)],
            linestyle=state_styles[i % len(state_styles)],
            linewidth=1.6,
        )
    ax.axhline(x_star, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlim(float(t[0]) if t.size else 0.0, float(t[-1]) if t.size else 1.0)
    y_min = float(min(np.min(x_hist), x_star))
    y_max = float(max(np.max(x_hist), x_star))
    y_pad = max(0.05, 0.12 * max(1.0, y_max - y_min))
    ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)
    ax.set_xticks(np.linspace(float(t[0]) if t.size else 0.0, float(t[-1]) if t.size else 1.0, 3))
    ax.set_xlabel("Time", fontsize=14, fontweight="bold")
    ax.set_ylabel(r"$x_i$", fontsize=14, fontweight="bold")
    ax.legend(
        loc="upper left",
        ncol=5,
        bbox_to_anchor=(0.0, 1.02),
        frameon=True,
        fontsize=7,
        handlelength=1.7,
        columnspacing=0.6,
        borderpad=0.2,
    )
    _style_paper_axis(ax)
    _add_state_inset(ax, t, x_hist, x_star, state_colors, state_styles)


def _plot_cost_panel(
    ax: plt.Axes,
    t: np.ndarray,
    cost: np.ndarray,
    f_star: float,
    color: str,
) -> None:
    ax.plot(t, cost, color=color, linestyle="--", linewidth=1.6)
    ax.axhline(f_star, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlim(float(t[0]) if t.size else 0.0, float(t[-1]) if t.size else 1.0)
    y_min = float(min(np.min(cost), f_star))
    y_max = float(max(np.max(cost), f_star))
    y_pad = max(0.08, 0.10 * max(1.0, y_max - y_min))
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xticks(np.linspace(float(t[0]) if t.size else 0.0, float(t[-1]) if t.size else 1.0, 3))
    ax.set_xlabel("Time", fontsize=14, fontweight="bold")
    ax.set_ylabel(r"$f$", fontsize=14, fontweight="bold")
    _style_paper_axis(ax)
    _add_cost_inset(ax, t, cost, f_star, color)


def plot_trajectories(
    out_dir: Path,
    baseline: Dict[str, np.ndarray],
    rl: Dict[str, np.ndarray] | None,
    x_star: float,
    f_star: float,
    title: str,
    dt: float,
) -> None:
    out_dir = Path(out_dir)
    t_base = np.arange(len(baseline["x"])) * dt
    if rl is None:
        fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=180)
        _plot_state_panel(ax, t_base, baseline["x"], x_star)
        fig.tight_layout()
        fig.savefig(out_dir / "baseline_states.eps", bbox_inches="tight", format="eps")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=180)
        _plot_cost_panel(ax, t_base, baseline["cost"], f_star, color="#1f77b4")
        fig.tight_layout()
        fig.savefig(out_dir / "baseline_cost.eps", bbox_inches="tight", format="eps")
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=180)
        _plot_state_panel(ax, t_base, baseline["x"], x_star)
        fig.tight_layout()
        fig.savefig(out_dir / "baseline_states.eps", bbox_inches="tight", format="eps")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=180)
        _plot_cost_panel(ax, t_base, baseline["cost"], f_star, color="#1f77b4")
        fig.tight_layout()
        fig.savefig(out_dir / "baseline_cost.eps", bbox_inches="tight", format="eps")
        plt.close(fig)

        t_rl = np.arange(len(rl["x"])) * dt
        fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=180)
        _plot_state_panel(ax, t_rl, rl["x"], x_star)
        fig.tight_layout()
        fig.savefig(out_dir / "rl_states.eps", bbox_inches="tight", format="eps")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=180)
        _plot_cost_panel(ax, t_rl, rl["cost"], f_star, color="#1f77b4")
        fig.tight_layout()
        fig.savefig(out_dir / "rl_cost.eps", bbox_inches="tight", format="eps")
        plt.close(fig)


def plot_gains(out_path: Path, rl: Dict[str, np.ndarray] | None, dt: float) -> None:
    if rl is None:
        return
    t = np.arange(len(rl["k1"])) * dt
    fig, axes = plt.subplots(3, 1, figsize=(6.0, 4.0), dpi=180, sharex=True)
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
    fig.savefig(out_path, bbox_inches="tight", format="eps")
    plt.close(fig)


def plot_training_reward(out_path: Path, logs: List[Dict[str, float]], window: int = 10) -> None:
    if not logs:
        return

    steps = np.array([entry["step_count"] for entry in logs], dtype=float)
    rewards = np.array([entry["episode_reward"] for entry in logs], dtype=float)

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=180)

    ax.plot(
        steps,
        rewards,
        marker="o",
        markersize=3.0,
        linewidth=1.0,
        color="#9ecae1",
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
    fig.savefig(out_path, bbox_inches="tight", format="eps")
    plt.close(fig)


def plot_paper_graph(path: Path) -> None:
    if nx is None:
        return
    G = nx.DiGraph()
    edges = [(1, 5, 0.2), (2, 1, 0.3), (3, 2, 0.5), (4, 3, 0.6), (5, 4, 0.8)]
    G.add_weighted_edges_from(edges)
    pos = paper_graph_positions()

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=180)
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
    fig.savefig(path, bbox_inches="tight", format="eps")
    plt.close(fig)


def mode_output_dir(base_outdir: Path, mode: str) -> Path:
    return base_outdir / mode


def append_reward_trace(
    reward_trace: List[Dict[str, float]],
    rewards: np.ndarray,
    step_offset: int,
    chunk_size: int = 100,
) -> None:
    """Convert a per-step reward vector into fixed-size logging chunks.

    We log the mean reward per step inside each chunk, not the chunk return,
    so the training curve stays readable even when episode lengths are long.
    """
    rewards = np.asarray(rewards, dtype=float)
    if rewards.size == 0:
        return

    for start in range(0, len(rewards), chunk_size):
        end = min(start + chunk_size, len(rewards))
        reward_trace.append(
            {
                "step_count": float(step_offset + end),
                "episode_reward": float(np.mean(rewards[start:end])),
            }
        )


class LinearGaussianActor:
    """Shared linear Gaussian actor used by all agents."""

    def __init__(self, obs_dim: int, act_dim: int, seed: int = 0) -> None:
        _ = np.random.default_rng(seed)
        # Start exactly at the nominal baseline gains so RL learns residual
        # improvements instead of recovering from an unstable random policy.
        self.W = np.zeros((act_dim, obs_dim), dtype=float)
        self.b = np.zeros(act_dim, dtype=float)
        self.log_std = np.log(np.array([0.08, 0.08, 0.12], dtype=float))

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
    bootstrap_value: float


class CTDEPPOTrainer:
    def __init__(
        self,
        env: MatlabStyleGMDOEnv,
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

        bootstrap_value = float(self.critic.value(obs.reshape(-1)))

        return TrajectoryBatch(
            obs=np.asarray(obs_hist, dtype=float),
            global_states=np.asarray(global_state_hist, dtype=float),
            raw_actions=np.asarray(raw_hist, dtype=float),
            logp=np.asarray(logp_hist, dtype=float),
            values=np.asarray(value_hist, dtype=float),
            rewards=np.asarray(reward_hist, dtype=float),
            infos=info_hist,
            bootstrap_value=bootstrap_value,
        )

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        bootstrap_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray(rewards, dtype=float)
        values = np.asarray(values, dtype=float)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        next_value = float(bootstrap_value)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + values
        return advantages, returns

    def update(self, batch: TrajectoryBatch, actor_epochs: int = 4, critic_epochs: int = 8) -> Dict[str, float]:
        advantages, returns = self.compute_gae(batch.rewards, batch.values, batch.bootstrap_value)

        critic_loss = self.critic.update(batch.global_states, returns, lr=self.critic_lr, epochs=critic_epochs)

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
            with torch.no_grad():
                nn.init.zeros_(self.net[-1].weight)
                nn.init.zeros_(self.net[-1].bias)
            self.log_std = nn.Parameter(torch.full((act_dim,), -1.6))

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
            env: MatlabStyleGMDOEnv,
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

            with torch.no_grad():
                bootstrap_value = float(
                    self.critic(torch.as_tensor(obs.reshape(-1), dtype=torch.float32, device=self.device)).item()
                )

            return TrajectoryBatch(
                obs=np.asarray(obs_hist, dtype=float),
                global_states=np.asarray(global_state_hist, dtype=float),
                raw_actions=np.asarray(raw_hist, dtype=float),
                logp=np.asarray(logp_hist, dtype=float),
                values=np.asarray(value_hist, dtype=float),
                rewards=np.asarray(reward_hist, dtype=float),
                infos=info_hist,
                bootstrap_value=bootstrap_value,
            )

        def compute_gae(
            self,
            rewards: np.ndarray,
            values: np.ndarray,
            bootstrap_value: float,
        ) -> Tuple[np.ndarray, np.ndarray]:
            rewards = np.asarray(rewards, dtype=float)
            values = np.asarray(values, dtype=float)
            advantages = np.zeros_like(rewards)
            gae = 0.0
            next_value = float(bootstrap_value)
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * next_value - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
                advantages[t] = gae
                next_value = values[t]
            returns = advantages + values
            return advantages, returns

        def update(self, batch: TrajectoryBatch, actor_epochs: int = 4, critic_epochs: int = 8) -> Dict[str, float]:
            advantages, returns = self.compute_gae(batch.rewards, batch.values, batch.bootstrap_value)

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
def train_and_evaluate_numpy(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    run_dir = mode_output_dir(outdir, args.mode)
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_horizon = int(round(100.0 / args.dt))

    env = MatlabStyleGMDOEnv(
        control_dt=args.dt,
        sim_dt=args.sim_dt,
        horizon=args.horizon,
        seed=args.seed,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        consensus_lambda=args.consensus_lambda,
        k_min=args.k_min,
        k_max=args.k_max,
        beta_max=args.beta_max,
        alpha_floor=args.alpha_floor,
        alpha_max=args.alpha_max,
        baseline_k1=args.baseline_k1,
        baseline_k2=args.baseline_k2,
        nominal_k1=args.nominal_k1,
        nominal_k2=args.nominal_k2,
        nominal_delta_alpha=args.nominal_delta_alpha,
        delta_k1_scale=args.delta_k1_scale,
        delta_k2_scale=args.delta_k2_scale,
        delta_alpha_scale=args.delta_alpha_scale,
        beta_source="policy",
        integrator=args.integrator,
    )
    print(f"Initial state x0 = {np.array2string(MATLAB_INITIAL_X, precision=3, separator=', ')}")
    print(f"Saving {args.mode} outputs under {run_dir}")

    plot_paper_graph(run_dir / "paper_graph.eps")

    baseline_env = MatlabStyleGMDOEnv(
        control_dt=args.dt,
        sim_dt=args.sim_dt,
        horizon=eval_horizon,
        seed=args.seed,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        consensus_lambda=args.consensus_lambda,
        k_min=args.k_min,
        k_max=args.k_max,
        beta_max=args.beta_max,
        alpha_floor=args.alpha_floor,
        alpha_max=args.alpha_max,
        baseline_k1=args.baseline_k1,
        baseline_k2=args.baseline_k2,
        nominal_k1=args.nominal_k1,
        nominal_k2=args.nominal_k2,
        nominal_delta_alpha=args.nominal_delta_alpha,
        delta_k1_scale=args.delta_k1_scale,
        delta_k2_scale=args.delta_k2_scale,
        delta_alpha_scale=args.delta_alpha_scale,
        beta_source="heuristic",
        integrator=args.integrator,
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
            run_dir,
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
    reward_trace: List[Dict[str, float]] = []
    global_steps = 0
    episode = 0
    while global_steps < args.total_steps:
        remaining = args.total_steps - global_steps
        original_horizon = env.horizon
        env.horizon = min(original_horizon, remaining)
        batch = trainer.collect_episode(deterministic=False)
        env.horizon = original_horizon
        stats = trainer.update(batch, actor_epochs=args.actor_epochs, critic_epochs=args.critic_epochs)
        append_reward_trace(reward_trace, batch.rewards, global_steps, chunk_size=args.reward_plot_interval)
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

    eval_env = MatlabStyleGMDOEnv(
        control_dt=args.dt,
        sim_dt=args.sim_dt,
        horizon=eval_horizon,
        seed=args.seed,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        consensus_lambda=args.consensus_lambda,
        k_min=args.k_min,
        k_max=args.k_max,
        beta_max=args.beta_max,
        alpha_floor=args.alpha_floor,
        alpha_max=args.alpha_max,
        baseline_k1=args.baseline_k1,
        baseline_k2=args.baseline_k2,
        nominal_k1=args.nominal_k1,
        nominal_k2=args.nominal_k2,
        nominal_delta_alpha=args.nominal_delta_alpha,
        delta_k1_scale=args.delta_k1_scale,
        delta_k2_scale=args.delta_k2_scale,
        delta_alpha_scale=args.delta_alpha_scale,
        beta_source="policy",
        integrator=args.integrator,
    )
    rl_result = evaluate_policy(eval_env, actor=actor, deterministic=True)
    rl_summary = summarize_convergence(rl_result, env.x_star, args.dt)
    print(
        f"RL convergence cost: {rl_summary['cost']:.4f} "
        f"at t={rl_summary['time']:.2f} "
        f"(x_err={rl_summary['x_error']:.4f}, settled={rl_summary['settled']})"
    )
    print(
        f"Comparison: baseline cost {baseline_summary['cost']:.4f} vs "
        f"RL cost {rl_summary['cost']:.4f} "
        f"(delta {rl_summary['cost'] - baseline_summary['cost']:+.4f})"
    )

    plot_trajectories(
        run_dir,
        baseline_result,
        rl_result,
        env.x_star,
        env.f_star,
        "Baseline vs CTDE-PPO on the Matlab-style GM dynamics",
        args.dt,
    )
    plot_gains(run_dir / "rl_gains.eps", rl_result, args.dt)
    plot_training_reward(run_dir / "reward_vs_steps.eps", logs)
    np.savez(run_dir / "actor_linear.npz", W=actor.W, b=actor.b, log_std=actor.log_std)
    np.savez(run_dir / "critic_linear.npz", w=critic.w, b=np.array([critic.b]))
    np.savez(
        run_dir / "training_logs.npz",
        episodes=np.array([d["episode_reward"] for d in logs], dtype=float),
        episode_rewards=np.array([d["episode_reward"] for d in logs], dtype=float),
        costs=np.array([d["final_global_cost"] for d in logs], dtype=float),
        errors=np.array([d["final_x_error"] for d in logs], dtype=float),
        steps=np.array([d["step_count"] for d in logs], dtype=float),
        reward_trace_steps=np.array([d["step_count"] for d in reward_trace], dtype=float),
        reward_trace_rewards=np.array([d["episode_reward"] for d in reward_trace], dtype=float),
    )


def train_and_evaluate_torch(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    run_dir = mode_output_dir(outdir, args.mode)
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_horizon = int(round(100.0 / args.dt))

    env = MatlabStyleGMDOEnv(
        control_dt=args.dt,
        sim_dt=args.sim_dt,
        horizon=args.horizon,
        seed=args.seed,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        consensus_lambda=args.consensus_lambda,
        k_min=args.k_min,
        k_max=args.k_max,
        beta_max=args.beta_max,
        alpha_floor=args.alpha_floor,
        alpha_max=args.alpha_max,
        baseline_k1=args.baseline_k1,
        baseline_k2=args.baseline_k2,
        nominal_k1=args.nominal_k1,
        nominal_k2=args.nominal_k2,
        nominal_delta_alpha=args.nominal_delta_alpha,
        delta_k1_scale=args.delta_k1_scale,
        delta_k2_scale=args.delta_k2_scale,
        delta_alpha_scale=args.delta_alpha_scale,
        beta_source="policy",
        integrator=args.integrator,
    )
    print(f"Initial state x0 = {np.array2string(MATLAB_INITIAL_X, precision=3, separator=', ')}")
    print(f"Saving {args.mode} outputs under {run_dir}")

    plot_paper_graph(run_dir / "paper_graph.eps")

    baseline_env = MatlabStyleGMDOEnv(
        control_dt=args.dt,
        sim_dt=args.sim_dt,
        horizon=eval_horizon,
        seed=args.seed,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        consensus_lambda=args.consensus_lambda,
        k_min=args.k_min,
        k_max=args.k_max,
        beta_max=args.beta_max,
        alpha_floor=args.alpha_floor,
        alpha_max=args.alpha_max,
        baseline_k1=args.baseline_k1,
        baseline_k2=args.baseline_k2,
        nominal_k1=args.nominal_k1,
        nominal_k2=args.nominal_k2,
        nominal_delta_alpha=args.nominal_delta_alpha,
        delta_k1_scale=args.delta_k1_scale,
        delta_k2_scale=args.delta_k2_scale,
        delta_alpha_scale=args.delta_alpha_scale,
        beta_source="heuristic",
        integrator=args.integrator,
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
            run_dir,
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
    reward_trace: List[Dict[str, float]] = []
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
        append_reward_trace(reward_trace, batch.rewards, global_steps, chunk_size=args.reward_plot_interval)
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

    eval_env = MatlabStyleGMDOEnv(
        control_dt=args.dt,
        sim_dt=args.sim_dt,
        horizon=eval_horizon,
        seed=args.seed,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        consensus_lambda=args.consensus_lambda,
        k_min=args.k_min,
        k_max=args.k_max,
        beta_max=args.beta_max,
        alpha_floor=args.alpha_floor,
        alpha_max=args.alpha_max,
        baseline_k1=args.baseline_k1,
        baseline_k2=args.baseline_k2,
        nominal_k1=args.nominal_k1,
        nominal_k2=args.nominal_k2,
        nominal_delta_alpha=args.nominal_delta_alpha,
        delta_k1_scale=args.delta_k1_scale,
        delta_k2_scale=args.delta_k2_scale,
        delta_alpha_scale=args.delta_alpha_scale,
        beta_source="policy",
        integrator=args.integrator,
    )
    rl_result = evaluate_policy_torch(eval_env, actor=actor, deterministic=True)
    rl_summary = summarize_convergence(rl_result, env.x_star, args.dt)
    print(
        f"RL convergence cost: {rl_summary['cost']:.4f} "
        f"at t={rl_summary['time']:.2f} "
        f"(x_err={rl_summary['x_error']:.4f}, settled={rl_summary['settled']})"
    )
    print(
        f"Comparison: baseline cost {baseline_summary['cost']:.4f} vs "
        f"RL cost {rl_summary['cost']:.4f} "
        f"(delta {rl_summary['cost'] - baseline_summary['cost']:+.4f})"
    )

    plot_trajectories(
        run_dir,
        baseline_result,
        rl_result,
        env.x_star,
        env.f_star,
        "Baseline vs CTDE-PPO on the Matlab-style GM dynamics",
        args.dt,
    )
    plot_gains(run_dir / "rl_gains.eps", rl_result, args.dt)
    plot_training_reward(run_dir / "reward_vs_steps.eps", logs)
    os.makedirs(run_dir, exist_ok=True)
    import torch  # local import to keep the file importable without torch

    torch.save(actor.state_dict(), run_dir / "actor_torch.pt")
    torch.save(critic.state_dict(), run_dir / "critic_torch.pt")
    np.savez(
        run_dir / "training_logs.npz",
        episodes=np.array([d["episode_reward"] for d in logs], dtype=float),
        episode_rewards=np.array([d["episode_reward"] for d in logs], dtype=float),
        costs=np.array([d["final_global_cost"] for d in logs], dtype=float),
        errors=np.array([d["final_x_error"] for d in logs], dtype=float),
        steps=np.array([d["step_count"] for d in logs], dtype=float),
        reward_trace_steps=np.array([d["step_count"] for d in reward_trace], dtype=float),
        reward_trace_rewards=np.array([d["episode_reward"] for d in reward_trace], dtype=float),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matlab-faithful GM-driven DO with CTDE-PPO scheduling.")
    parser.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device name, e.g. cuda, cuda:0, or cpu; falls back to CPU if CUDA cannot initialize.",
    )
    parser.add_argument("--mode", choices=["baseline", "rl"], default="rl")
    parser.add_argument("--total-steps", type=int, default=15000)
    parser.add_argument("--horizon", type=int, default=100, help="Training horizon per episode.")
    parser.add_argument("--dt", type=float, default=0.1, help="Control interval used for plotting and RL steps.")
    parser.add_argument("--sim-dt", type=float, default=0.001, help="Internal integration step within each control interval.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="results_matlab_style")
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
    parser.add_argument("--reward-plot-interval", type=int, default=100, help="Reward logging interval in environment steps.")
    parser.add_argument("--lambda1", type=float, default=0.2)
    parser.add_argument("--lambda2", type=float, default=0.01)
    parser.add_argument("--consensus-lambda", type=float, default=0.35)
    parser.add_argument("--k-min", type=float, default=0.05)
    parser.add_argument("--k-max", type=float, default=5.0)
    parser.add_argument("--beta-max", type=float, default=50.0)
    parser.add_argument("--alpha-floor", type=float, default=1e-12)
    parser.add_argument("--alpha-max", type=float, default=1e6)
    parser.add_argument("--baseline-k1", type=float, default=2.0)
    parser.add_argument("--baseline-k2", type=float, default=2.0)
    parser.add_argument("--nominal-k1", type=float, default=2.0)
    parser.add_argument("--nominal-k2", type=float, default=2.0)
    parser.add_argument("--nominal-delta-alpha", type=float, default=10.0)
    parser.add_argument("--delta-k1-scale", type=float, default=2.0)
    parser.add_argument("--delta-k2-scale", type=float, default=2.0)
    parser.add_argument("--delta-alpha-scale", type=float, default=10.0)
    parser.add_argument("--integrator", choices=["rk4", "euler"], default="rk4")
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
