"""
AM-driven baseline: D. Yue, S. Baldi, J. Cao, B. De Schutter,
"Distributed adaptive optimization with weight-balancing",
IEEE Trans. Autom. Control 67(4):2068-2075, 2022.

Two phases, as in the paper:
1. Finite-time weight balancing (12a)-(12b) along a directed spanning tree (DST),
   followed by step ii) of Section IV (negative tree weights become reversed edges).
2. Adaptive saddle-point-like dynamics (4a)-(4b) with DST-adaptive gains (5a)-(5b)
   on the balanced digraph G*.

Graph convention (same as the paper and as adagm_do.Problem.A):
W[i, j] > 0 iff j -> i (j is an in-neighbour of i); Laplacian L = diag(W 1) - W.
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Dict, Tuple

import numpy as np


def directed_spanning_tree(W: np.ndarray, root: int = 0) -> np.ndarray:
    """BFS DST rooted at `root`; parent[v] is the unique in-neighbour of v in the tree (-1 for the root)."""
    N = W.shape[0]
    parent = np.full(N, -1)
    seen = np.zeros(N, dtype=bool)
    seen[root] = True
    queue = deque([root])
    while queue:
        u = queue.popleft()
        for v in range(N):  # edge u -> v exists iff W[v, u] > 0
            if not seen[v] and W[v, u] > 0:
                seen[v] = True
                parent[v] = u
                queue.append(v)
    if not seen.all():
        raise ValueError("digraph is not strongly connected from the root")
    return parent


def _imbalance(B: np.ndarray) -> np.ndarray:
    """in-degree minus out-degree of every node: sum_p B[i,p] - sum_c B[c,i]."""
    return B.sum(axis=1) - B.sum(axis=0)


def balance_weights(W: np.ndarray, gamma3: float = 0.1, q: float = 0.1, dt: float = 1e-3,
                    t_max: float = 400.0, tol: float = 1e-9) -> Tuple[np.ndarray, Dict]:
    """Run the balancing law (12) on the DST weights, then apply step ii) of Section IV."""
    N = W.shape[0]
    parent = directed_spanning_tree(W)
    tree = [(v, parent[v]) for v in range(N) if parent[v] >= 0]  # edge parent -> v, weight B[v, parent]
    B = W.astype(float).copy()
    t = 0.0
    while t < t_max:
        d = _imbalance(B)
        if np.max(np.abs([d[v] for v, _ in tree])) < tol:
            break
        for v, p in tree:  # (12b): b_dot = -gamma3 * sig^q(in - out) of node v
            B[v, p] -= dt * gamma3 * np.sign(d[v]) * np.abs(d[v]) ** q
        t += dt
    b_star = {(v, p): B[v, p] for v, p in tree}
    # step ii): turn the pseudo-balanced digraph into an actual balanced digraph G*
    Ws = B.copy()
    for (v, p), b in b_star.items():
        if abs(b) < 1e-9:
            Ws[v, p] = 0.0
            if W[p, v] == 0:
                Ws[v, p] = 1.0
                Ws[p, v] += 1.0
        elif b < 0:
            Ws[v, p] = 0.0
            Ws[p, v] += abs(b)
    info = {"balance_time": t, "b_star": b_star, "dst": parent,
            "imbalance_after": float(np.max(np.abs(_imbalance(Ws))))}
    return Ws, info


def make_rhs(Ws: np.ndarray, grad: Callable[[np.ndarray], np.ndarray], gamma1: float, gamma2: float):
    """(4)-(5) for scalar decisions on the balanced digraph Ws. State = [x, y, a_bar]."""
    N = Ws.shape[0]
    parent = directed_spanning_tree(Ws)
    tree_nodes = np.array([v for v in range(N) if parent[v] >= 0])
    tree_par = parent[tree_nodes]
    children = [np.flatnonzero(parent == v) for v in range(N)]
    A_fixed = Ws.copy()
    A_fixed[tree_nodes, tree_par] = 0.0  # tree weights are replaced by the adaptive a_bar

    def rhs(s: np.ndarray) -> np.ndarray:
        x, y, a = s[:N], s[N:2 * N], s[2 * N:]
        At = A_fixed.copy()
        At[tree_nodes, tree_par] = a  # (5a)
        cons_x = At.sum(1) * x - At @ x  # sum_j alpha_ij (x_i - x_j)
        cons_y = Ws.sum(1) * y - Ws @ y  # sum_j w_ij (y_i - y_j)
        x_dot = -gamma1 * grad(x) - cons_x - cons_y  # (4a)
        y_dot = cons_x  # (4b)
        # (5b): a_dot_{v,p} = gamma2 * ((x_p - x_v) - sum_{c in children(v)} (x_v - x_c)) * (x_p - x_v)
        e = x[tree_par] - x[tree_nodes]
        child_sum = np.array([np.sum(x[v] - x[children[v]]) for v in tree_nodes])
        a_dot = gamma2 * (e - child_sum) * e
        return np.concatenate([x_dot, y_dot, a_dot])

    a0 = Ws[tree_nodes, tree_par].copy()  # A(0) = W
    return rhs, a0
