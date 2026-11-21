#!/usr/bin/env python3
"""
AdaGM-DO: GM-driven distributed optimization with CTDE-PPO gain scheduling.

Self-contained implementation of the paper's method:

* problems (5-agent digraph and 10-agent ring), Phase 1 left-eigenvector estimate
* the GM-driven DO dynamics with the adaptive gain multiplying the integral and gradient terms
* the decentralized observation / action layout and the reward
* the CTDE-PPO trainer (shared actor, centralized critic)

Modes:

* rl: train the policy that schedules K2^i(t) and the adaptive-gain rate
* baseline: fixed nominal gains and the heuristic rate ||epsilon^i||^2
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import time
import warnings
import zipfile
from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-config")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

# Tuned so the 2016 variant settles inside the comparison window.
MATLAB_GM_2016_SPEED_SCALE = 10.0

# Upper limit on RK4 sub-steps per control step when the dynamics become stiff.
MAX_SUBSTEPS = 20000

# Cap on the heuristic increment ||epsilon||^2 used by the [21]-style baseline.
HEURISTIC_DELTA_ALPHA_CAP = 10.0

# Random initial conditions (training ablation and tests): each agent's nominal
# x_i(t0) is scaled by a log-uniform factor in [1/RANDOM_INIT_FACTOR, RANDOM_INIT_FACTOR].
# Wider ranges make the gradient term too stiff for the fixed-step integrator.
RANDOM_INIT_FACTOR = 1.5


@dataclass
class Problem:
    """Power-law DO problem: f_i(x) = coeff_i * x**power_i on a weighted digraph."""

    name: str
    A: np.ndarray
    coeff: np.ndarray
    power: np.ndarray
    x0: np.ndarray

    @property
    def N(self) -> int:
        return int(self.A.shape[0])

    def costs(self, x: np.ndarray) -> np.ndarray:
        return self.coeff * np.power(x, self.power)

    def grads(self, x: np.ndarray) -> np.ndarray:
        # power 0 gives a constant cost with zero gradient
        return np.where(self.power == 0.0, 0.0, self.coeff * self.power * np.power(x, self.power - 1.0))

    def global_cost(self, x: np.ndarray) -> float:
        return float(np.sum(self.costs(np.asarray(x, dtype=float))))

    def solve_optimum(self) -> Tuple[float, float]:
        """x* of sum_i f_i(x) via bisection on the derivative (log-convex, so unique)."""
        dg = lambda z: float(np.sum(self.grads(np.full(self.N, z))))
        lo, hi = 1e-4, 10.0
        while dg(lo) * dg(hi) > 0:
            hi *= 2.0
            if hi > 1e6:
                raise RuntimeError("Failed to bracket the global optimum.")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if dg(lo) * dg(mid) <= 0:
                hi = mid
            else:
                lo = mid
        x_star = 0.5 * (lo + hi)
        return x_star, self.global_cost(np.full(self.N, x_star))


def paper_problem() -> Problem:
    """The 5-agent example of Section V (Fig. 2)."""
    return Problem(
        name="paper5",
        A=build_paper_adjacency(),
        coeff=np.array([5.0, 3.0, 4.0, 2.0, 1.0]),
        power=np.array([0.0, -3.0, -2.0, 1.0, 4.0]),
        x0=MATLAB_INITIAL_X.copy(),
    )


def ring10_problem() -> Problem:
    """A 10-agent weight-unbalanced, strongly connected digraph with power-law costs.

    Directed ring i-1 -> i plus chords i+3 -> i, with fixed random weights.
    """
    N = 10
    rng = np.random.default_rng(2026)
    A = np.zeros((N, N), dtype=float)
    for i in range(N):
        A[i, (i - 1) % N] = rng.uniform(0.2, 1.0)
    for i in range(0, N, 3):
        A[i, (i + 3) % N] = rng.uniform(0.2, 1.0)
    return Problem(
        name="ring10",
        A=A,
        coeff=np.array([3.0, 4.0, 2.0, 2.0, 1.0, 1.0, 5.0, 3.0, 2.0, 1.0]),
        power=np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 4.0, -1.0, 1.0, -2.0, 2.0]),
        x0=np.round(rng.uniform(0.2, 6.0, size=N), 1),
    )


PROBLEMS = {"paper5": paper_problem, "ring10": ring10_problem}


def estimate_left_eigenvector(A: np.ndarray, phi: float = 1.0, duration: float = 50.0) -> np.ndarray:
    """Phase 1: run the distributed estimator (1) for a finite time.

    Stacking the agents' estimates as rows of W gives W_dot = -phi L W with
    W(0) = I, so W(t) = expm(-phi L t) and agent i keeps entry (i, i).
    """
    from scipy.linalg import expm

    L = laplacian_from_adjacency(A)
    return np.diag(expm(-phi * duration * L)).copy()


def random_initial_x(rng: np.random.Generator, x_nominal: np.ndarray) -> np.ndarray:
    log_f = np.log(RANDOM_INIT_FACTOR)
    return np.asarray(x_nominal, dtype=float) * np.exp(rng.uniform(-log_f, log_f, size=len(x_nominal)))


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
        horizon: int = 1000,
        seed: int = 42,
        lambda1: float = 0.2,
        lambda2: float = 0.002,
        consensus_lambda: float = 0.35,
        k_min: float = 0.05,
        k_max: float = 1.45,
        beta_max: float = 50.0,
        alpha_floor: float = 1e-12,
        alpha_max: float = 1e6,
        baseline_k1: float = 2.0,
        baseline_k2: float = 2.0,
        nominal_k1: float = 2.0,
        nominal_k2: float = 0.2,
        nominal_delta_alpha: float = 1.5,
        delta_k1_scale: float = 2.0,
        delta_k2_scale: float = 0.7,
        delta_alpha_scale: float = 2.0,
        delta_alpha_max: float = 4.5,
        initial_x: np.ndarray | None = None,
        beta_source: str = "heuristic",
        integrator: str = "rk4",
        problem: Problem | None = None,
        w_source: str = "estimate",
        phase1_phi: float = 1.0,
        phase1_duration: float = 50.0,
        delta_alpha_min: float = 0.5,
        lambda_alpha: float = 0.01,
        random_init: bool = False,
        k2_on_integral: bool = True,
        xstar_error: float = 0.0,
    ) -> None:
        if beta_source not in {"policy", "heuristic"}:
            raise ValueError("beta_source must be 'policy' or 'heuristic'.")
        if integrator not in {"rk4", "euler"}:
            raise ValueError("integrator must be 'rk4' or 'euler'.")

        if w_source not in {"estimate", "exact"}:
            raise ValueError("w_source must be 'estimate' or 'exact'.")
        self.problem = paper_problem() if problem is None else problem
        self.A = self.problem.A
        self.N = self.problem.N
        self.L = laplacian_from_adjacency(self.A)
        # Any positive multiple of w_l keeps W L a balanced Laplacian; the N^2
        # factor matches the original Matlab experiments.
        if w_source == "estimate":
            self.w_hat = estimate_left_eigenvector(self.A, phase1_phi, phase1_duration)
        else:
            self.w_hat = left_eigenvector_from_laplacian(laplacian_from_adjacency(self.A))
        self.w_left = float(self.N**2) * self.w_hat
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
        self.delta_alpha_max = delta_alpha_max
        self.delta_alpha_min = delta_alpha_min
        self.lambda_alpha = lambda_alpha
        # True: K2^i scales both the integral and gradient terms of (5a), so the
        # equilibrium is x* for any K2^i(t) > 0. False: original GM-DO of [21].
        self.k2_on_integral = bool(k2_on_integral)
        self.random_init = random_init
        self.beta_source = beta_source
        self.integrator = integrator
        self.fixed_k1 = float(nominal_k1)
        self.x_min = 1e-12
        self.x_max = 1e6
        self.v_min = 1e-12
        self.v_max = 1e6
        self.rng = np.random.default_rng(seed)
        self.x_star, self.f_star = self.problem.solve_optimum()
        # Targets used by the reward; an approximation of (x*, f*) when xstar_error != 0.
        self.x_star_reward = (1.0 + xstar_error) * self.x_star
        self.f_star_reward = (1.0 + xstar_error) * self.f_star

        if initial_x is None:
            initial_x = self.problem.x0.copy()
        self.initial_x = np.asarray(initial_x, dtype=float)
        if self.initial_x.shape != (self.N,):
            raise ValueError(f"initial_x must have shape ({self.N},), got {self.initial_x.shape}.")

        self.reset()

    @property
    def obs_dim(self) -> int:
        return 6

    @property
    def act_dim(self) -> int:
        return 2

    def reset(self, x0: np.ndarray | None = None) -> np.ndarray:
        if x0 is None:
            x0 = random_initial_x(self.rng, self.initial_x) if self.random_init else self.initial_x
        self.x = np.clip(np.asarray(x0, dtype=float), self.x_min, self.x_max)
        self.v = self.x.copy()
        self.alpha = np.full(self.N, MATLAB_INITIAL_ALPHA, dtype=float)
        self.beta = np.zeros(self.N, dtype=float)
        self.step_index = 0
        self.time = 0.0
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

    def _neighbor_mean(self, x: np.ndarray) -> np.ndarray:
        # x_bar_i = (1/|N_i|) sum_{j in N_i} x_j, from one-hop communication only
        mask = (self.A > 0).astype(float)
        return (mask @ x) / np.maximum(mask.sum(axis=1), 1.0)

    def _grad_x(self, x: np.ndarray) -> np.ndarray:
        return self.problem.grads(self._clip_positive(x, 1e-8, self.x_max))

    def _current_beta(self, x: np.ndarray, gains: np.ndarray) -> np.ndarray:
        _ = gains
        return np.square(self._epsilon(x))

    def _rhs(
        self,
        x: np.ndarray,
        v: np.ndarray,
        alpha: np.ndarray,
        gains: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        k2 = np.maximum(gains[:, 0], 0.0)
        delta_alpha = np.maximum(gains[:, 1], 0.0)

        # sum_j a_ij (z_j - z_i) = -(L z)_i
        Lx = self.L @ np.log(np.minimum(np.maximum(x, self.x_min), self.x_max))
        Lv = self.L @ np.log(np.minimum(np.maximum(v, self.v_min), self.v_max))
        eps = self.w_left * Lx
        beta = np.square(eps)
        grad_x = self.problem.grads(np.minimum(np.maximum(x, 1e-8), self.x_max))
        gain = self.fixed_k1 * (alpha + beta) * self.w_left

        # Scalar-agent version of (5a)-(5b).
        integral_gain = k2 if self.k2_on_integral else 1.0
        x_dot = -gain * x * Lx - integral_gain * x * self.w_left * Lv - k2 * np.square(x) * grad_x
        v_dot = gain * v * Lx
        return x_dot, v_dot, delta_alpha, beta

    def _project_state(self, x: np.ndarray, v: np.ndarray, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.minimum(np.maximum(x, self.x_min), self.x_max)
        v = np.minimum(np.maximum(v, self.v_min), self.v_max)
        alpha = np.minimum(np.maximum(alpha, self.alpha_floor), self.alpha_max)
        return x, v, alpha

    def _stiffness(self, gains: np.ndarray) -> float:
        """Gershgorin-type bound on the log-domain Jacobian of (5) over one control step."""
        k2 = np.maximum(gains[:, 0], 0.0)
        delta_alpha = np.maximum(gains[:, 1], 0.0)
        alpha_end = self.alpha + delta_alpha * self.control_dt
        beta = np.square(self._epsilon(self.x))
        row = 2.0 * self.w_left * np.diag(self.L)
        integral_gain = np.max(k2) if self.k2_on_integral else 1.0
        consensus = self.fixed_k1 * np.max((alpha_end + 3.0 * beta) * row) + integral_gain * np.max(row)
        # d/d(eta) of x * grad f for f = c x^p is c p^2 x^p
        x = np.minimum(np.maximum(self.x, 1e-8), self.x_max)
        gradient = np.max(k2 * np.abs(self.problem.coeff * self.problem.power**2 * np.power(x, self.problem.power)))
        return float(consensus + gradient)

    def _integrate_interval(self, gains: np.ndarray) -> None:
        x = self.x.copy()
        v = self.v.copy()
        alpha = self.alpha.copy()

        # Refine the fixed step when the gains make (5) stiff, keeping
        # sub_dt * lambda_max <= 1, well inside the RK4 stability region.
        n_sub = self.n_substeps
        if self.integrator == "rk4":
            n_sub = int(min(MAX_SUBSTEPS, max(n_sub, np.ceil(self.control_dt * self._stiffness(gains)))))
        sub_dt = self.control_dt / float(n_sub)

        for _ in range(n_sub):
            if self.integrator == "euler":
                dx, dv, da, _ = self._rhs(x, v, alpha, gains)
                x = x + sub_dt * dx
                v = v + sub_dt * dv
                alpha = alpha + sub_dt * da
            else:
                k1_x, k1_v, k1_a, _ = self._rhs(x, v, alpha, gains)
                k2_x, k2_v, k2_a, _ = self._rhs(
                    x + 0.5 * sub_dt * k1_x,
                    v + 0.5 * sub_dt * k1_v,
                    alpha + 0.5 * sub_dt * k1_a,
                    gains,
                )
                k3_x, k3_v, k3_a, _ = self._rhs(
                    x + 0.5 * sub_dt * k2_x,
                    v + 0.5 * sub_dt * k2_v,
                    alpha + 0.5 * sub_dt * k2_a,
                    gains,
                )
                k4_x, k4_v, k4_a, _ = self._rhs(
                    x + sub_dt * k3_x,
                    v + sub_dt * k3_v,
                    alpha + sub_dt * k3_a,
                    gains,
                )
                x = x + (sub_dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
                v = v + (sub_dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
                alpha = alpha + (sub_dt / 6.0) * (k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a)

            x, v, alpha = self._project_state(x, v, alpha)

        self.x = x
        self.v = v
        self.alpha = alpha

    def _obs(self) -> np.ndarray:
        x_bar = self._neighbor_mean(self.x)
        eps = self._epsilon(self.x)
        grad_x = self._grad_x(self.x)
        grad_p = self.x * grad_x
        time_feature = np.full(self.N, self.step_index / max(1, self.horizon), dtype=float)
        # Squash each feature to O(1): alpha grows without bound and x*grad f
        # is large early on, which otherwise saturates the actor.
        return np.column_stack(
            [
                np.tanh(eps),
                np.tanh(grad_p),
                np.tanh(np.abs(self.x - x_bar)),
                np.log1p(self.alpha),
                np.log1p(self.beta),
                time_feature,
            ]
        )

    def transform_raw_actions(self, raw_actions: np.ndarray) -> np.ndarray:
        """Map unconstrained actor outputs to bounded positive gains.

        The policy schedules the agent-specific gradient gain K2 and a
        residual correction around a nominal alpha-rate. This keeps the
        learned controller close to a known-stable operating point while
        still allowing agent-wise adaptation.
        """
        raw_actions = np.asarray(raw_actions, dtype=float)
        k2 = np.clip(self.nominal_k2 + self.delta_k2_scale * np.tanh(raw_actions[:, 0]), self.k_min, self.k_max)
        delta_alpha = np.clip(
            self.nominal_delta_alpha + self.delta_alpha_scale * np.tanh(raw_actions[:, 1]),
            self.delta_alpha_min,
            self.delta_alpha_max,
        )
        return np.column_stack([k2, delta_alpha])

    def baseline_actions(
        self,
        obs: np.ndarray,
        k2_fixed: float | None = None,
    ) -> np.ndarray:
        _ = obs
        # Read epsilon from the state, since the observation is squashed.
        eps = self._epsilon(self.x)
        if k2_fixed is None:
            k2_fixed = self.baseline_k2
        # Baseline follows the paper's heuristic increment Δα = ||epsilon||^2.
        delta_alpha = np.clip(np.square(eps), 0.0, HEURISTIC_DELTA_ALPHA_CAP)
        return np.column_stack(
            [
                np.full(self.N, k2_fixed, dtype=float),
                delta_alpha,
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
        global_cost = self.problem.global_cost(self.x)
        overshoot = max(0.0, global_cost - self.f_star_reward)
        x_bar = self._neighbor_mean(self.x)
        eps = np.abs(self._epsilon(self.x))
        consensus_penalty = np.abs(self.x - x_bar) + eps
        k1 = np.full(self.N, self.fixed_k1, dtype=float)
        k2 = np.maximum(gains[:, 0], 0.0)
        delta_alpha = np.maximum(gains[:, 1], 0.0)
        # Paper reward (8) plus an adaptation-effort penalty on the alpha-rate
        per_agent_reward = (
            -np.abs(self.x - self.x_star_reward)
            - self.lambda1 * overshoot
            - self.lambda2 * np.abs(k2 - self.nominal_k2)
            - self.consensus_lambda * consensus_penalty
            - self.lambda_alpha * delta_alpha
        )
        team_reward = float(np.mean(per_agent_reward))
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
            "mean_delta_alpha": float(np.mean(delta_alpha)),
            "time": self.time,
        }
        return obs, team_reward, done, info



def make_env(args: argparse.Namespace, horizon: int, **overrides) -> MatlabStyleGMDOEnv:
    """Build the environment from the command-line settings."""
    kwargs = dict(
        control_dt=args.dt,
        sim_dt=args.sim_dt,
        horizon=horizon,
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
        delta_alpha_max=args.delta_alpha_max,
        delta_alpha_min=args.delta_alpha_min,
        lambda_alpha=args.lambda_alpha,
        beta_source="heuristic",
        integrator=args.integrator,
        problem=PROBLEMS[args.problem](),
        w_source=args.w_source,
        phase1_duration=args.phase1_duration,
        random_init=args.random_init,
        xstar_error=args.xstar_error,
    )
    kwargs.update(overrides)
    return MatlabStyleGMDOEnv(**kwargs)

def summarize_convergence(
    traj: Dict[str, np.ndarray],
    x_star: float,
    dt: float,
    rel_band: float = 0.02,
    start_at_zero: bool = False,
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

    step = float(idx if start_at_zero else idx + 1)
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
    if actor is None:
        # the heuristic baseline is the original GM-DO method with a uniform K2
        env.k2_on_integral = False
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


class LoadedGaussianActor:
    """NumPy implementation of the saved torch Gaussian policy."""

    def __init__(self, state_dict: Dict[str, np.ndarray]) -> None:
        self.w1 = np.asarray(state_dict["net.0.weight"], dtype=float)
        self.b1 = np.asarray(state_dict["net.0.bias"], dtype=float)
        self.w2 = np.asarray(state_dict["net.2.weight"], dtype=float)
        self.b2 = np.asarray(state_dict["net.2.bias"], dtype=float)
        self.w3 = np.asarray(state_dict["net.4.weight"], dtype=float)
        self.b3 = np.asarray(state_dict["net.4.bias"], dtype=float)
        self.log_std = np.asarray(state_dict["log_std"], dtype=float)

    def mean(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=float)
        h = np.tanh(x @ self.w1.T + self.b1)
        h = np.tanh(h @ self.w2.T + self.b2)
        return h @ self.w3.T + self.b3

    def sample(self, obs: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu = self.mean(obs)
        std = np.exp(np.clip(self.log_std, -4.0, 2.0))
        raw = mu + rng.normal(size=mu.shape) * std
        logp = gaussian_log_prob(raw, mu, self.log_std)
        return raw, mu, logp


def _torch_storage_dtype(storage_type: object) -> np.dtype:
    name = getattr(storage_type, "__name__", "")
    mapping = {
        "FloatStorage": np.float32,
        "DoubleStorage": np.float64,
        "HalfStorage": np.float16,
        "LongStorage": np.int64,
        "IntStorage": np.int32,
        "ShortStorage": np.int16,
        "CharStorage": np.int8,
        "ByteStorage": np.uint8,
        "BoolStorage": np.bool_,
    }
    if name not in mapping:
        raise TypeError(f"Unsupported torch storage type: {name}")
    return np.dtype(mapping[name])


def load_saved_torch_actor_numpy(actor_path: Path) -> LoadedGaussianActor:
    """Load the saved torch policy weights without depending on torch."""
    actor_path = Path(actor_path)
    prefix = actor_path.stem
    storage_cache: Dict[str, np.ndarray] = {}
    class_cache: Dict[str, type] = {}

    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        arr = np.asarray(storage).reshape(-1)
        size_t = tuple(int(s) for s in size)
        stride_t = tuple(int(s) for s in stride)
        view = np.lib.stride_tricks.as_strided(
            arr[int(storage_offset) :],
            shape=size_t,
            strides=tuple(int(s) * arr.dtype.itemsize for s in stride_t),
        )
        return np.array(view, copy=True)

    def _rebuild_parameter(tensor, requires_grad, backward_hooks):
        return tensor

    class Loader(pickle.Unpickler):
        def persistent_load(self, pid):
            if not isinstance(pid, tuple) or len(pid) < 5 or pid[0] != "storage":
                raise pickle.UnpicklingError(f"Unsupported persistent id: {pid!r}")
            storage_type = pid[1]
            key = str(pid[2])
            size = int(pid[4])
            if key not in storage_cache:
                dtype = _torch_storage_dtype(storage_type)
                raw = archive.read(f"{prefix}/data/{key}")
                storage_cache[key] = np.frombuffer(raw, dtype=dtype, count=size).copy()
            return storage_cache[key]

        def find_class(self, module, name):
            if module == "collections" and name == "OrderedDict":
                return OrderedDict
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if module == "torch._utils" and name == "_rebuild_parameter":
                return _rebuild_parameter
            if module == "torch" and name.endswith("Storage"):
                return class_cache.setdefault(name, type(name, (), {}))
            return super().find_class(module, name)

    with zipfile.ZipFile(actor_path) as archive:
        raw = archive.read(f"{prefix}/data.pkl")
        state_dict = Loader(io.BytesIO(raw)).load()
    return LoadedGaussianActor(state_dict)


def load_saved_actor_for_evaluation(actor_path: Path) -> LoadedGaussianActor:
    """Load the saved RL actor in a torch-free way for plotting/evaluation."""
    return load_saved_torch_actor_numpy(actor_path)


def load_training_reward_logs(log_path: Path) -> List[Dict[str, float]]:
    """Rebuild the reward-vs-steps plot input from the saved training logs."""
    log_path = Path(log_path)
    if not log_path.exists():
        return []

    with np.load(log_path) as data:
        if "steps" in data and "episode_rewards" in data:
            steps = np.asarray(data["steps"], dtype=float)
            rewards = np.asarray(data["episode_rewards"], dtype=float)
        elif "reward_trace_steps" in data and "reward_trace_rewards" in data:
            steps = np.asarray(data["reward_trace_steps"], dtype=float)
            rewards = np.asarray(data["reward_trace_rewards"], dtype=float)
        else:
            return []

    return [
        {"step_count": float(step), "episode_reward": float(reward)}
        for step, reward in zip(steps, rewards)
    ]


def _style_paper_axis(ax: plt.Axes, tick_size: int = 11) -> None:
    ax.set_facecolor("white")
    ax.grid(False)
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
    zoom_start: float = 20.0,
    zoom_end: float = 50.0,
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
        zoom_start=zoom_start,
        zoom_end=zoom_end,
    )
    axins.set_xlim(zoom_start, zoom_end)
    axins.set_ylim(y_min, y_max)
    axins.set_xticks([zoom_start, zoom_end] if zoom_end > zoom_start else [zoom_end])
    axins.set_yticks(np.linspace(y_min, y_max, 3))
    _style_paper_axis(axins, tick_size=8)


def _add_cost_inset(
    ax: plt.Axes,
    t: np.ndarray,
    cost: np.ndarray,
    f_star: float,
    color: str,
    zoom_start: float = 20.0,
    zoom_end: float = 50.0,
) -> None:
    axins = inset_axes(ax, width="38%", height="34%", loc="upper right", borderpad=1.0)
    axins.plot(t, cost, color=color, linestyle="--", linewidth=1.0)
    axins.axhline(f_star, color="black", linestyle="--", linewidth=0.9)
    zoom_start, zoom_end, y_min, y_max = _paper_zoom_window(
        t,
        cost,
        center=float(f_star),
        zoom_start=zoom_start,
        zoom_end=zoom_end,
    )
    axins.set_xlim(zoom_start, zoom_end)
    axins.set_ylim(y_min, y_max)
    axins.set_xticks([zoom_start, zoom_end] if zoom_end > zoom_start else [zoom_end])
    axins.set_yticks(np.linspace(y_min, y_max, 3))
    _style_paper_axis(axins, tick_size=8)


def _plot_state_panel(
    ax: plt.Axes,
    t: np.ndarray,
    x_hist: np.ndarray,
    x_star: float,
    zoom_start: float = 20.0,
    zoom_end: float = 50.0,
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
    ax.set_xlabel("time", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"$x_i$", fontsize=12, fontweight="bold")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=5,
        frameon=True,
        fontsize=9,
        prop={"weight": "bold"},
        handlelength=1.4,
        columnspacing=0.5,
        borderpad=0.25,
    )
    _style_paper_axis(ax)
    _add_state_inset(
        ax,
        t,
        x_hist,
        x_star,
        state_colors,
        state_styles,
        zoom_start=zoom_start,
        zoom_end=zoom_end,
    )


def _plot_cost_panel(
    ax: plt.Axes,
    t: np.ndarray,
    cost: np.ndarray,
    f_star: float,
    color: str,
    zoom_start: float = 20.0,
    zoom_end: float = 50.0,
) -> None:
    ax.plot(t, cost, color=color, linestyle="--", linewidth=1.6, label="cost")
    ax.axhline(f_star, color="black", linestyle="--", linewidth=1.2, label=r"$f^*$")
    ax.set_xlim(float(t[0]) if t.size else 0.0, float(t[-1]) if t.size else 1.0)
    y_min = float(min(np.min(cost), f_star))
    y_max = float(max(np.max(cost), f_star))
    y_pad = max(0.08, 0.10 * max(1.0, y_max - y_min))
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xticks(np.linspace(float(t[0]) if t.size else 0.0, float(t[-1]) if t.size else 1.0, 3))
    ax.set_xlabel("time", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"$f$", fontsize=12, fontweight="bold")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        frameon=True,
        fontsize=9,
        prop={"weight": "bold"},
        handlelength=1.4,
        columnspacing=0.6,
        borderpad=0.25,
    )
    _style_paper_axis(ax)
    _add_cost_inset(ax, t, cost, f_star, color, zoom_start=zoom_start, zoom_end=zoom_end)


def plot_trajectories(
    out_dir: Path,
    method_a: Dict[str, np.ndarray],
    method_b: Dict[str, np.ndarray] | None,
    x_star: float,
    f_star: float,
    title: str,
    dt: float,
    method_a_name: str = "baseline",
    method_b_name: str = "rl",
    method_a_zoom: Tuple[float, float] = (20.0, 50.0),
    method_b_zoom: Tuple[float, float] = (10.0, 40.0),
    method_a_cost_zoom: Tuple[float, float] = (20.0, 50.0),
    method_b_cost_zoom: Tuple[float, float] = (20.0, 50.0),
) -> None:
    out_dir = Path(out_dir)
    t_base = np.arange(len(method_a["x"])) * dt
    if method_b is None:
        fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=180)
        _plot_state_panel(
            ax,
            t_base,
            method_a["x"],
            x_star,
            zoom_start=method_a_zoom[0],
            zoom_end=method_a_zoom[1],
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"{method_a_name}_states.eps", bbox_inches="tight", format="eps")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=180)
        _plot_cost_panel(
            ax,
            t_base,
            method_a["cost"],
            f_star,
            color="#1f77b4",
            zoom_start=method_a_cost_zoom[0],
            zoom_end=method_a_cost_zoom[1],
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"{method_a_name}_cost.eps", bbox_inches="tight", format="eps")
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=180)
        _plot_state_panel(
            ax,
            t_base,
            method_a["x"],
            x_star,
            zoom_start=method_a_zoom[0],
            zoom_end=method_a_zoom[1],
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"{method_a_name}_states.eps", bbox_inches="tight", format="eps")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=180)
        _plot_cost_panel(
            ax,
            t_base,
            method_a["cost"],
            f_star,
            color="#1f77b4",
            zoom_start=method_a_cost_zoom[0],
            zoom_end=method_a_cost_zoom[1],
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"{method_a_name}_cost.eps", bbox_inches="tight", format="eps")
        plt.close(fig)

        t_rl = np.arange(len(method_b["x"])) * dt
        fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=180)
        _plot_state_panel(
            ax,
            t_rl,
            method_b["x"],
            x_star,
            zoom_start=method_b_zoom[0],
            zoom_end=method_b_zoom[1],
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"{method_b_name}_states.eps", bbox_inches="tight", format="eps")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=180)
        _plot_cost_panel(
            ax,
            t_rl,
            method_b["cost"],
            f_star,
            color="#1f77b4",
            zoom_start=method_b_cost_zoom[0],
            zoom_end=method_b_cost_zoom[1],
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"{method_b_name}_cost.eps", bbox_inches="tight", format="eps")
        plt.close(fig)


def _cumulative_deviation(cost: np.ndarray, f_star: float, dt: float) -> np.ndarray:
    """Return the accumulated positive gap above the optimal cost."""
    cost = np.asarray(cost, dtype=float)
    gap = np.maximum(cost - float(f_star), 0.0)
    return np.cumsum(gap) * float(dt)


def plot_cumulative_deviation(
    out_path: Path,
    method_a: Dict[str, np.ndarray],
    method_b: Dict[str, np.ndarray] | None,
    f_star: float,
    dt: float,
    method_a_name: str = "baseline",
    method_b_name: str = "RL",
) -> None:
    fig, ax = plt.subplots(figsize=(4, 3), dpi=180)

    t_base = np.arange(len(method_a["cost"])) * dt
    baseline_dev = _cumulative_deviation(method_a["cost"], f_star, dt)
    ax.plot(t_base, baseline_dev, color="#1f77b4", linewidth=1.6, label=method_a_name)

    if method_b is not None:
        t_rl = np.arange(len(method_b["cost"])) * dt
        rl_dev = _cumulative_deviation(method_b["cost"], f_star, dt)
        ax.plot(t_rl, rl_dev, color="#d62728", linewidth=1.6, label=method_b_name)

    ax.set_xlabel("time", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative deviation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=11, prop={"weight": "bold"})
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", format="eps")
    plt.close(fig)


def plot_gains(out_path: Path, rl: Dict[str, np.ndarray] | None, dt: float) -> None:
    if rl is None:
        return
    t = np.arange(len(rl["k1"])) * dt
    fig, axes = plt.subplots(3, 1, figsize=(4.0, 3.0), dpi=180, sharex=True)
    axes[0].plot(t, rl["k1"], color="#2ca02c")
    axes[0].set_ylabel("K1 (fixed)", fontsize=12, fontweight="bold")
    axes[0].grid(False)
    axes[1].plot(t, rl["k2"], color="#ff7f0e")
    axes[1].set_ylabel("K2(t)", fontsize=12, fontweight="bold")
    axes[1].grid(False)
    axes[2].plot(t, rl["alpha"], color="#9467bd", label="alpha")
    axes[2].plot(t, rl["beta"], color="#8c564b", linestyle="--", label="beta")
    axes[2].set_ylabel("gain", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("time", fontsize=12, fontweight="bold")
    axes[2].legend(fontsize=11, prop={"weight": "bold"})
    axes[2].grid(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", format="eps")
    plt.close(fig)


def plot_training_reward(out_path: Path, logs: List[Dict[str, float]], window: int = 10) -> None:
    if not logs:
        return

    steps = np.array([entry["step_count"] for entry in logs], dtype=float)
    rewards = np.array([entry["episode_reward"] for entry in logs], dtype=float)

    fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=180)

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

    ax.set_xlabel("training step", fontsize=12, fontweight="bold")
    ax.set_ylabel("reward", fontsize=12, fontweight="bold")
    ax.grid(False)
    ax.legend(fontsize=11, prop={"weight": "bold"})
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", format="eps")
    plt.close(fig)


def plot_paper_graph(path: Path, A: np.ndarray | None = None) -> None:
    if nx is None:
        return
    if A is None:
        A = build_paper_adjacency()
    G = nx.DiGraph()
    # a_ij > 0 means agent i receives from j; drawn as i -> j as in Fig. 2
    edges = [(int(i) + 1, int(j) + 1, float(A[i, j])) for i, j in zip(*np.nonzero(A))]
    G.add_weighted_edges_from(edges)
    pos = paper_graph_positions() if A.shape[0] == 5 else nx.circular_layout(G)

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
    ax.set_title(f"{A.shape[0]}-agent communication graph", fontsize=12, family="serif")
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight", format="eps")
    plt.close(fig)


def _rk4_step(state: np.ndarray, dt: float, rhs) -> np.ndarray:
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _matlab_gm_rhs(method: str, state: np.ndarray, A: np.ndarray, w: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    row_sign = np.sign(A.sum(axis=1))
    if method == "gm2016":
        x = np.clip(np.asarray(state[:n], dtype=float), EPS, None)
        z = np.clip(np.asarray(state[n : 2 * n], dtype=float), EPS, None)
        grad = np.array([local_cost_and_grad(i, float(x[i]))[1] for i in range(n)], dtype=float)
        log_x = np.log(x)
        log_z = np.log(z)
        consensus_x = np.sum(A * (log_x[:, None] - log_x[None, :]), axis=1)
        consensus_z = np.sum(A * (log_z[:, None] - log_z[None, :]), axis=1)
        gain = 2.0 ** 1.5
        x_dot = (
            -np.square(x) * row_sign * grad
            - w * x * consensus_z
            - w * gain * x * consensus_x
        ) * MATLAB_GM_2016_SPEED_SCALE
        z_dot = w * z * consensus_x * MATLAB_GM_2016_SPEED_SCALE
        return np.concatenate([x_dot, z_dot])

    x = np.clip(np.asarray(state[:n], dtype=float), EPS, None)
    z = np.clip(np.asarray(state[n : 2 * n], dtype=float), EPS, None)
    alpha = np.clip(np.asarray(state[2 * n : 3 * n], dtype=float), EPS, None)

    grad = np.array([local_cost_and_grad(i, float(x[i]))[1] for i in range(n)], dtype=float)
    log_x = np.log(x)
    log_z = np.log(z)
    consensus_x = np.sum(A * (log_x[:, None] - log_x[None, :]), axis=1)
    consensus_z = np.sum(A * (log_z[:, None] - log_z[None, :]), axis=1)
    beta = np.square(w * consensus_x)

    if method == "gm2018":
        x_dot = (
            -np.square(x) * row_sign * grad
            - x * w * consensus_z
            - (alpha + beta) * w * x * consensus_x
        ) * 2.0
    elif method == "gm2015":
        x_dot = (
            -np.square(x) * row_sign * grad
            - x * log_z
            - (alpha + beta) * w * x * consensus_x
        ) * 2.0
    else:
        raise ValueError("method must be 'gm2015', 'gm2016' or 'gm2018'.")

    z_dot = w * z * consensus_x * 2.0
    alpha_dot = beta
    return np.concatenate([x_dot, z_dot, alpha_dot])


def simulate_matlab_gm_method(
    method: str,
    duration: float = 100.0,
    sample_dt: float = 0.01,
    sim_dt: float = 0.001,
    initial_x: np.ndarray | None = None,
    initial_alpha: float = MATLAB_INITIAL_ALPHA,
) -> Dict[str, np.ndarray]:
    if method not in {"gm2015", "gm2016", "gm2018"}:
        raise ValueError("method must be 'gm2015', 'gm2016' or 'gm2018'.")

    A = build_paper_adjacency()
    w = MATLAB_LEFT_EIGENVECTOR.copy()
    x0 = MATLAB_INITIAL_X.copy() if initial_x is None else np.asarray(initial_x, dtype=float).copy()
    z0 = x0.copy()
    if method == "gm2016":
        state = np.concatenate([x0, z0])
    else:
        alpha0 = np.full_like(x0, float(initial_alpha), dtype=float)
        state = np.concatenate([x0, z0, alpha0])

    n_steps = int(round(duration / sample_dt))
    n_substeps = max(1, int(round(sample_dt / sim_dt)))
    sub_dt = sample_dt / float(n_substeps)

    x_hist = [x0.copy()]
    cost_hist = [global_cost_vector(x0)]

    rhs = lambda s: _matlab_gm_rhs(method, s, A, w)
    for _ in range(n_steps):
        for _ in range(n_substeps):
            state = _rk4_step(state, sub_dt, rhs)
            state = np.clip(state, EPS, None)

        x_now = np.asarray(state[: A.shape[0]], dtype=float).copy()
        x_hist.append(x_now)
        cost_hist.append(global_cost_vector(x_now))

    return {
        "x": np.asarray(x_hist, dtype=float),
        "cost": np.asarray(cost_hist, dtype=float),
    }


def compare_matlab_methods(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    run_dir = mode_output_dir(outdir, args.mode)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Simulating df_GM_2018, df_GM_2016 and df_GM_2015 (Matlab baselines)")
    plot_paper_graph(run_dir / "paper_graph.eps", PROBLEMS[args.problem]().A)

    x_star, f_star = solve_global_optimum()
    eval_horizon = int(round(100.0 / args.dt))

    def make_eval_env(horizon: int) -> MatlabStyleGMDOEnv:
        return make_env(args, horizon=horizon, random_init=False)

    # Recreate the previous baseline/RL plots alongside the Matlab-style comparison.
    baseline_env = make_eval_env(eval_horizon)
    baseline_result = evaluate_policy(baseline_env, actor=None, deterministic=True)
    baseline_summary = summarize_convergence(baseline_result, baseline_env.x_star, args.dt)
    print(
        f"Baseline convergence cost: {baseline_summary['cost']:.4f} "
        f"at t={baseline_summary['time']:.2f} "
        f"(x_err={baseline_summary['x_error']:.4f}, settled={baseline_summary['settled']})"
    )

    rl_result = None
    rl_summary = None
    actor_path = outdir / "rl" / "actor_torch.pt"
    if actor_path.exists():
        try:
            rl_actor = load_saved_actor_for_evaluation(actor_path)
            rl_env = make_eval_env(eval_horizon)
            rl_result = evaluate_policy(rl_env, actor=rl_actor, deterministic=True)
            rl_summary = summarize_convergence(rl_result, rl_env.x_star, args.dt)
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
        except Exception as exc:
            print(f"Could not load the saved RL actor from {actor_path}: {exc}")
    else:
        print(f"Saved RL actor not found at {actor_path}; baseline plots will still be generated.")

    reward_logs = load_training_reward_logs(outdir / "rl" / "training_logs.npz")

    # Always regenerate the baseline plots so they follow the current styling.
    plot_trajectories(
        run_dir,
        baseline_result,
        None,
        baseline_env.x_star,
        baseline_env.f_star,
        "Baseline GM-driven DO on the paper graph",
        args.dt,
        method_a_name="baseline",
    )
    plot_cumulative_deviation(
        run_dir / "baseline_rl_cumulative_deviation.eps",
        baseline_result,
        None,
        baseline_env.f_star,
        args.dt,
        method_a_name="baseline",
    )

    if rl_result is not None:
        plot_trajectories(
            run_dir,
            baseline_result,
            rl_result,
            baseline_env.x_star,
            baseline_env.f_star,
            "Baseline vs CTDE-PPO on the Matlab-style GM dynamics",
            args.dt,
            method_a_name="baseline",
            method_b_name="rl",
        )
        plot_cumulative_deviation(
            run_dir / "baseline_rl_cumulative_deviation.eps",
            baseline_result,
            rl_result,
            baseline_env.f_star,
            args.dt,
            method_a_name="baseline",
            method_b_name="RL",
        )
        plot_cumulative_deviation(
            run_dir / "cumulative_deviation.eps",
            baseline_result,
            rl_result,
            baseline_env.f_star,
            args.dt,
            method_a_name="baseline",
            method_b_name="RL",
        )
        plot_gains(run_dir / "rl_gains.eps", rl_result, args.dt)
    else:
        plot_cumulative_deviation(
            run_dir / "cumulative_deviation.eps",
            baseline_result,
            None,
            baseline_env.f_star,
            args.dt,
            method_a_name="baseline",
        )

    if reward_logs:
        plot_training_reward(run_dir / "reward_vs_steps.eps", reward_logs)

    gm2018 = simulate_matlab_gm_method(
        "gm2018",
        duration=args.compare_duration,
        sample_dt=args.compare_dt,
        sim_dt=args.compare_sim_dt,
    )
    gm2016 = simulate_matlab_gm_method(
        "gm2016",
        duration=args.compare_duration,
        sample_dt=args.compare_dt,
        sim_dt=args.compare_sim_dt,
    )
    gm2015 = simulate_matlab_gm_method(
        "gm2015",
        duration=args.compare_duration,
        sample_dt=args.compare_dt,
        sim_dt=args.compare_sim_dt,
    )

    gm2018_summary = summarize_convergence(gm2018, x_star, args.compare_dt, start_at_zero=True)
    gm2016_summary = summarize_convergence(gm2016, x_star, args.compare_dt, start_at_zero=True)
    gm2015_summary = summarize_convergence(gm2015, x_star, args.compare_dt, start_at_zero=True)
    print(
        f"df_GM_2018 convergence cost: {gm2018_summary['cost']:.4f} "
        f"at t={gm2018_summary['time']:.2f} "
        f"(x_err={gm2018_summary['x_error']:.4f}, settled={gm2018_summary['settled']})"
    )
    print(
        f"df_GM_2016 convergence cost: {gm2016_summary['cost']:.4f} "
        f"at t={gm2016_summary['time']:.2f} "
        f"(x_err={gm2016_summary['x_error']:.4f}, settled={gm2016_summary['settled']})"
    )
    print(
        f"df_GM_2015 convergence cost: {gm2015_summary['cost']:.4f} "
        f"at t={gm2015_summary['time']:.2f} "
        f"(x_err={gm2015_summary['x_error']:.4f}, settled={gm2015_summary['settled']})"
    )
    print(
        f"Comparison: df_GM_2018 cost {gm2018_summary['cost']:.4f} vs "
        f"df_GM_2016 cost {gm2016_summary['cost']:.4f} vs "
        f"df_GM_2015 cost {gm2015_summary['cost']:.4f} "
        f"(2016 delta {gm2016_summary['cost'] - gm2018_summary['cost']:+.4f}, "
        f"2015 delta {gm2015_summary['cost'] - gm2018_summary['cost']:+.4f})"
    )

    plot_trajectories(
        run_dir,
        gm2018,
        gm2015,
        x_star,
        f_star,
        "df_GM_2018 vs df_GM_2015 on the Matlab-style GM dynamics",
        args.compare_dt,
        method_a_name="df_GM_2018",
        method_b_name="df_GM_2015",
        method_a_zoom=(40.0, 100.0),
        method_b_zoom=(40.0, 100.0),
        method_a_cost_zoom=(40.0, 100.0),
        method_b_cost_zoom=(40.0, 100.0),
    )
    plot_trajectories(
        run_dir,
        gm2016,
        None,
        x_star,
        f_star,
        "df_GM_2016 on the Matlab-style GM dynamics",
        args.compare_dt,
        method_a_name="df_GM_2016",
        method_a_zoom=(20.0, 80.0),
        method_a_cost_zoom=(20.0, 80.0),
    )
    plot_cumulative_deviation(
        run_dir / "df_GM_2016_cumulative_deviation.eps",
        gm2016,
        None,
        f_star,
        args.compare_dt,
        method_a_name="df_GM_2016",
    )
    plot_cumulative_deviation(
        run_dir / "df_gm_cumulative_deviation.eps",
        gm2018,
        gm2015,
        f_star,
        args.compare_dt,
        method_a_name="df_GM_2018",
        method_b_name="df_GM_2015",
    )


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
        # Start at the nominal K2 and alpha-rate residuals so RL learns
        # improvements instead of recovering from an unstable random policy.
        self.W = np.zeros((act_dim, obs_dim), dtype=float)
        self.b = np.zeros(act_dim, dtype=float)
        self.log_std = np.log(np.array([0.08, 0.12], dtype=float))

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

    env = make_env(args, horizon=args.horizon)
    print(f"Problem {args.problem}: x0 = {np.array2string(env.initial_x, precision=3, separator=', ')}, random_init={args.random_init}")
    print(f"Saving {args.mode} outputs under {run_dir}")

    plot_paper_graph(run_dir / "paper_graph.eps", PROBLEMS[args.problem]().A)

    baseline_env = make_env(args, horizon=eval_horizon, random_init=False)
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
        plot_cumulative_deviation(
            run_dir / "cumulative_deviation.eps",
            baseline_result,
            None,
            env.f_star,
            args.dt,
            method_a_name="baseline",
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

    eval_env = make_env(args, horizon=eval_horizon, random_init=False)
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
        method_a_name="baseline",
        method_b_name="rl",
    )
    plot_cumulative_deviation(
        run_dir / "cumulative_deviation.eps",
        baseline_result,
        rl_result,
        env.f_star,
        args.dt,
        method_a_name="baseline",
        method_b_name="RL",
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

    env = make_env(args, horizon=args.horizon)
    print(f"Problem {args.problem}: x0 = {np.array2string(env.initial_x, precision=3, separator=', ')}, random_init={args.random_init}")
    print(f"Saving {args.mode} outputs under {run_dir}")

    plot_paper_graph(run_dir / "paper_graph.eps", PROBLEMS[args.problem]().A)

    baseline_env = make_env(args, horizon=eval_horizon, random_init=False)
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
        plot_cumulative_deviation(
            run_dir / "cumulative_deviation.eps",
            baseline_result,
            None,
            env.f_star,
            args.dt,
            method_a_name="baseline",
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
    train_start = time.perf_counter()
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

    eval_env = make_env(args, horizon=eval_horizon, random_init=False)
    train_seconds = time.perf_counter() - train_start
    print(f"Training wall time: {train_seconds:.1f} s ({1e3 * train_seconds / max(1, global_steps):.2f} ms/step)")
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
        method_a_name="baseline",
        method_b_name="rl",
    )
    plot_cumulative_deviation(
        run_dir / "cumulative_deviation.eps",
        baseline_result,
        rl_result,
        env.f_star,
        args.dt,
        method_a_name="baseline",
        method_b_name="RL",
    )
    plot_gains(run_dir / "rl_gains.eps", rl_result, args.dt)
    plot_training_reward(run_dir / "reward_vs_steps.eps", logs)
    os.makedirs(run_dir, exist_ok=True)
    import torch  # local import to keep the file importable without torch

    torch.save(actor.state_dict(), run_dir / "actor_torch.pt")
    torch.save(critic.state_dict(), run_dir / "critic_torch.pt")
    with open(run_dir / "args.json", "w") as fh:
        json.dump(vars(args), fh, indent=2)
    np.savez(
        run_dir / "training_logs.npz",
        train_seconds=np.array([train_seconds]),
        episodes=np.array([d["episode_reward"] for d in logs], dtype=float),
        episode_rewards=np.array([d["episode_reward"] for d in logs], dtype=float),
        costs=np.array([d["final_global_cost"] for d in logs], dtype=float),
        errors=np.array([d["final_x_error"] for d in logs], dtype=float),
        steps=np.array([d["step_count"] for d in logs], dtype=float),
        reward_trace_steps=np.array([d["step_count"] for d in reward_trace], dtype=float),
        reward_trace_rewards=np.array([d["episode_reward"] for d in reward_trace], dtype=float),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K2-scheduled GM-driven DO with CTDE-PPO scheduling.")
    parser.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device name, e.g. cuda, cuda:0, or cpu; falls back to CPU if CUDA cannot initialize.",
    )
    parser.add_argument("--mode", choices=["baseline", "rl", "matlab_compare"], default="rl")
    parser.add_argument("--total-steps", type=int, default=300000)
    parser.add_argument(
        "--horizon",
        type=int,
        default=1000,
        help="Training horizon per episode; keep it long enough to cover settling.",
    )
    parser.add_argument("--dt", type=float, default=0.1, help="Control interval used for plotting and RL steps.")
    parser.add_argument("--sim-dt", type=float, default=0.001, help="Internal integration step within each control interval.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="results/run")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--actor-lr", type=float, default=0.003)
    parser.add_argument("--critic-lr", type=float, default=0.002)
    parser.add_argument("--clip-ratio", type=float, default=0.1)
    parser.add_argument("--actor-epochs", type=int, default=4)
    parser.add_argument("--critic-epochs", type=int, default=8)
    parser.add_argument("--mini-batch-size", type=int, default=64)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--reward-plot-interval", type=int, default=100, help="Reward logging interval in environment steps.")
    parser.add_argument("--lambda1", type=float, default=0.2)
    parser.add_argument("--lambda2", type=float, default=0.002)
    parser.add_argument("--consensus-lambda", type=float, default=0.35)
    parser.add_argument("--k-min", type=float, default=0.05)
    parser.add_argument("--k-max", type=float, default=1.45)
    parser.add_argument("--beta-max", type=float, default=50.0)
    parser.add_argument("--alpha-floor", type=float, default=1e-12)
    parser.add_argument("--alpha-max", type=float, default=1e6)
    parser.add_argument("--baseline-k1", type=float, default=2.0)
    parser.add_argument("--baseline-k2", type=float, default=2.0)
    parser.add_argument("--nominal-k1", type=float, default=2.0)
    parser.add_argument("--nominal-k2", type=float, default=0.2)
    parser.add_argument("--nominal-delta-alpha", type=float, default=1.5)
    parser.add_argument("--delta-k1-scale", type=float, default=2.0)
    parser.add_argument("--delta-k2-scale", type=float, default=0.7)
    parser.add_argument("--delta-alpha-scale", type=float, default=2.0)
    parser.add_argument("--delta-alpha-min", type=float, default=0.5, help="Lower clip in (7b); > 0 so Theorem 1 holds for any policy.")
    parser.add_argument("--delta-alpha-max", type=float, default=4.5)
    parser.add_argument("--lambda-alpha", type=float, default=0.01, help="Adaptation-effort penalty on the alpha-rate.")
    parser.add_argument("--problem", choices=sorted(PROBLEMS), default="paper5")
    parser.add_argument("--w-source", choices=["estimate", "exact"], default="estimate", help="Phase 1 estimate (1) or exact left eigenvector.")
    parser.add_argument("--phase1-duration", type=float, default=50.0)
    parser.add_argument("--random-init", action="store_true", help="Randomly perturb x(t0) (factor in [1/1.5, 1.5]) every training episode.")
    parser.add_argument("--xstar-error", type=float, default=0.0, help="Relative error of the (x*, f*) used by the reward.")
    parser.add_argument("--threads", type=int, default=1, help="Torch CPU threads.")
    parser.add_argument("--integrator", choices=["rk4", "euler"], default="rk4")
    parser.add_argument("--compare-duration", type=float, default=100.0, help="Duration for the Matlab comparison mode.")
    parser.add_argument("--compare-dt", type=float, default=0.01, help="Sampling interval for the Matlab comparison mode.")
    parser.add_argument("--compare-sim-dt", type=float, default=0.001, help="Internal RK4 step for the Matlab comparison mode.")
    return parser.parse_args()


def train_and_evaluate(args: argparse.Namespace) -> None:
    if args.mode == "matlab_compare":
        return compare_matlab_methods(args)

    backend = args.backend
    if backend == "auto":
        backend = "torch"
    if backend == "torch":
        if not TORCH_AVAILABLE:
            raise SystemExit(
                "PyTorch is not available in this Python, so the paper's MLP actor cannot be trained. "
                "Activate an environment with torch (e.g. myenv), or pass --backend numpy for the linear test actor."
            )
        else:
            torch.set_num_threads(max(1, args.threads))
            return train_and_evaluate_torch(args)
    return train_and_evaluate_numpy(args)


def main() -> None:
    args = parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
