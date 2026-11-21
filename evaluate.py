#!/usr/bin/env python3
"""
Multi-initial-condition / multi-seed evaluation (Table I of the paper).

Every method is run on the same test set: the problem's nominal x(t0) plus
10 random initial conditions (each x_i(t0) scaled by a log-uniform factor in
[1/1.5, 1.5]) drawn with a seed that training never uses.

Methods
* AdaGM-DO (RL), one entry per trained run directory
* fixed nominal gains (the RL policy with zero residual, i.e. the best fixed
  gains of the grid search)
* heuristic alpha-rate ||epsilon||^2 in (5), K2 = 2
* [21]  GM-driven adaptive DO (Matlab df_GM_2018)
* [12]  AM-driven DO, Kia et al. 2015 (Matlab df_AM_2015)
* [13]  AM-driven adaptive DO, Li et al. TAC 2018 (Matlab df_AM_2018, method key 'am14')
* [Yue22] AM-driven adaptive DO with weight balancing, Yue et al. TAC 2022 (yue2022.py)

Metrics
* settling time: first time after which all agents stay within +-2% of x*
* overshoot: 100 * max_{t >= t1} (f - f*) / f*, with t1 the first time f <= f*
  (t1 = t0 when f(x(t0)) <= f*; 0 if the cost never reaches f*); this ignores
  a large initial cost when the agents start above the optimum
* cost IAE: int |f(x(t)) - f*| dt
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import adagm_do as m
import yue2022

DT = 0.1
SIM_DT = 0.001
DURATION = 100.0
N_RANDOM_TESTS = 10
TEST_SEED = 12345
# Yue et al. 2022 [TAC]: gains gamma1, gamma2 (tuned on the nominal x(t0) of paper5, see tune_yue22)
YUE_GAMMA1 = 0.02
YUE_GAMMA2 = 1000.0
_BALANCED: Dict[str, np.ndarray] = {}


def test_initial_conditions(problem: m.Problem) -> List[np.ndarray]:
    rng = np.random.default_rng(TEST_SEED + problem.N)
    return [problem.x0.copy()] + [m.random_initial_x(rng, problem.x0) for _ in range(N_RANDOM_TESTS)]


def metrics(x_hist: np.ndarray, cost: np.ndarray, x_star: float, f_star: float) -> Dict[str, float]:
    """x_hist / cost start at t0 (index k <-> time k * DT)."""
    finite = bool(np.all(np.isfinite(x_hist)) and np.all(np.isfinite(cost)) and np.max(np.abs(x_hist)) < 1e5)
    if not finite:
        return {"settle": np.inf, "overshoot": np.inf, "cost_iae": np.inf, "final_err": np.inf, "settled": False}
    s = m.summarize_convergence({"x": x_hist, "cost": cost}, x_star, DT, start_at_zero=True)
    below = np.flatnonzero(cost <= f_star)
    overshoot = 0.0 if below.size == 0 else 100.0 * max(0.0, float(np.max(cost[below[0]:])) - f_star) / f_star
    return {
        "settle": s["time"] if s["settled"] else np.inf,
        "overshoot": overshoot,
        "cost_iae": float(np.sum(np.abs(cost - f_star)) * DT),
        "final_err": float(np.max(np.abs(x_hist[-1] - x_star))),
        "settled": bool(s["settled"]),
    }


# ---------------------------------------------------------------- baselines


def _baseline_rhs(method: str, state: np.ndarray, P: m.Problem, w: np.ndarray) -> np.ndarray:
    """Ports of the original Matlab implementations of the baselines."""
    n = P.N
    A = P.A
    row_sign = np.sign(A.sum(axis=1))
    x, z = state[:n], state[n : 2 * n]
    with np.errstate(all="ignore"):
        grad = P.grads(x)
    if method == "gm21":
        x = np.clip(x, m.EPS, None)
        z = np.clip(z, m.EPS, None)
        alpha = state[2 * n :]
        lx, lz = np.log(x), np.log(z)
        s1 = np.sum(A * (lx[:, None] - lx[None, :]), axis=1)
        s2 = np.sum(A * (lz[:, None] - lz[None, :]), axis=1)
        beta = np.square(w * s1)
        x_dot = 2.0 * (-np.square(x) * row_sign * grad - x * w * s2 - (alpha + beta) * w * x * s1)
        z_dot = 2.0 * w * z * s1
        return np.concatenate([x_dot, z_dot, beta])
    s1 = np.sum(A * (x[:, None] - x[None, :]), axis=1)
    if method == "am12":
        g3 = 2.0**1.5
        x_dot = 2.0 * (-row_sign * grad - z - w * g3 * s1)
        z_dot = 2.0 * w * s1
        return np.concatenate([x_dot, z_dot])
    if method == "am14":
        alpha = state[2 * n :]
        s2 = np.sum(A * (z[:, None] - z[None, :]), axis=1)
        beta = np.square(s1)
        x_dot = 0.58 * (-row_sign * grad / w - s2 - (alpha + beta) * s1)
        z_dot = 0.58 * (alpha + beta) * s1
        return np.concatenate([x_dot, z_dot, beta])
    raise ValueError(method)


def balanced_graph(P: m.Problem) -> np.ndarray:
    """Phase 1 of Yue et al. 2022: finite-time weight balancing (12) + step ii)."""
    if P.name not in _BALANCED:
        _BALANCED[P.name] = yue2022.balance_weights(P.A)[0]
    return _BALANCED[P.name]


def simulate_baseline(method: str, P: m.Problem, w: np.ndarray, x0: np.ndarray, sim_dt: float = SIM_DT,
                      gammas: Tuple[float, float] | None = None) -> Tuple[np.ndarray, np.ndarray]:
    n = P.N
    if method == "yue22":
        g1, g2 = gammas or (YUE_GAMMA1, YUE_GAMMA2)
        with np.errstate(all="ignore"):
            yue_rhs, a0 = yue2022.make_rhs(balanced_graph(P), P.grads, g1, g2)
        state = np.concatenate([x0, np.zeros(n), a0])
        rhs = lambda s: yue_rhs(s)
        return _integrate_baseline(state, rhs, P, x0, sim_dt, positive=False)
    # Initial conditions as in the original Matlab implementations
    if method == "gm21":
        state = np.concatenate([x0, x0, np.full(n, m.MATLAB_INITIAL_ALPHA)])
    elif method == "am12":
        state = np.concatenate([x0, np.zeros(n)])
    else:
        state = np.concatenate([x0, x0, np.full(n, m.MATLAB_INITIAL_ALPHA)])
    rhs = lambda s: _baseline_rhs(method, s, P, w)
    return _integrate_baseline(state, rhs, P, x0, sim_dt, positive=(method == "gm21"))


def _integrate_baseline(state, rhs, P: m.Problem, x0: np.ndarray, sim_dt: float, positive: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Accurate stiff integration (LSODA, like Matlab ode15s/ode45 with tight tolerances), sampled every DT.

    `sim_dt` and `positive` are kept for the call signature; the step size is chosen by the solver.
    A trajectory that leaves the domain (non-finite state) is reported as diverged from that time on.
    """
    from scipy.integrate import solve_ivp
    n = P.N
    t_eval = np.arange(0.0, DURATION + 1e-9, DT)
    with np.errstate(all="ignore"):
        sol = solve_ivp(lambda t, s: rhs(s), (0.0, DURATION), state, t_eval=t_eval,
                        method="LSODA", rtol=1e-7, atol=1e-9)
        xs = sol.y[:n].T if sol.y.size else np.empty((0, n))
        if xs.shape[0] < t_eval.size:  # solver failed: pad as diverged
            xs = np.vstack([xs, np.full((t_eval.size - xs.shape[0], n), np.nan)])
        cs = np.array([P.global_cost(r) if np.all(np.isfinite(r)) else np.nan for r in xs])
    return xs, cs


# ---------------------------------------------------------------- env-based methods


def load_run_args(run_dir: Path) -> argparse.Namespace:
    with open(run_dir / "args.json") as fh:
        return argparse.Namespace(**json.load(fh))


class ZeroActor:
    def mean(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros((obs.shape[0], 2))


def simulate_env(args: argparse.Namespace, actor, P: m.Problem, x0: np.ndarray, **overrides) -> Tuple[np.ndarray, np.ndarray, Dict]:
    env = m.make_env(
        args,
        horizon=int(round(DURATION / DT)),
        problem=P,
        initial_x=x0,
        random_init=False,
        xstar_error=0.0,
        **overrides,
    )
    r = m.evaluate_policy(env, actor=actor, deterministic=True)
    r["x"] = np.vstack([x0[None, :], r["x"]])
    r["cost"] = np.concatenate([[P.global_cost(x0)], r["cost"]])
    extra = {"w_err": float(np.max(np.abs(env.w_hat - m.left_eigenvector_from_laplacian(m.laplacian_from_adjacency(P.A)))))}
    return r["x"], r["cost"], extra


def run_job(job: Dict) -> Dict:
    P = m.PROBLEMS[job["problem"]]()
    x0 = test_initial_conditions(P)[job["ic"]]
    x_star, f_star = P.solve_optimum()
    extra: Dict = {}
    if job["kind"] == "baseline":
        w = float(P.N**2) * m.estimate_left_eigenvector(P.A)
        x_hist, cost = simulate_baseline(job["method"], P, w, x0)
        extra["solver"] = "LSODA rtol=1e-7 atol=1e-9"
    else:
        args = load_run_args(Path(job["run_dir"]))
        if job["kind"] == "rl":
            actor = m.load_saved_actor_for_evaluation(Path(job["run_dir"]) / "actor_torch.pt")
        elif job["kind"] == "fixed":
            actor = ZeroActor()
        else:  # heuristic alpha-rate
            actor = None
        x_hist, cost, extra = simulate_env(args, actor, P, x0, **job.get("overrides", {}))
    out = dict(job)
    out.update(metrics(x_hist, cost, x_star, f_star))
    out.update(extra)
    if job["ic"] == 0 and job.get("save_traj", False):
        out["traj"] = {"x": x_hist.tolist(), "cost": cost.tolist()}
    return out


# ---------------------------------------------------------------- driver


def fmt(values: List[float]) -> str:
    v = np.asarray(values, dtype=float)
    n_fail = int(np.sum(~np.isfinite(v)))
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "diverged / not settled"
    s = f"{v.mean():.2f} ± {v.std():.2f}"
    return s + (f" ({n_fail} not settled)" if n_fail else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all methods on the shared test set.")
    parser.add_argument("--root", type=str, default="results")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    parser.add_argument("--only-baselines", nargs="+", default=None,
                        help="run only these baseline methods and merge them into an existing evaluation")
    args = parser.parse_args()
    root = Path(args.root)

    runs = sorted(p.parent for p in root.glob("*/rl/actor_torch.pt"))
    groups: Dict[str, List[Path]] = {}
    for rd in runs:
        tag = rd.parent.name.rsplit("_s", 1)[0]
        groups.setdefault(tag, []).append(rd)
    print("run groups:", {k: len(v) for k, v in groups.items()})

    jobs: List[Dict] = []
    for problem in ["paper5", "ring10"]:
        n_ic = 1 + N_RANDOM_TESTS
        # a reference run supplies the nominal gains for the fixed / heuristic rows
        ref_tag = "paper5_fixed" if "paper5_fixed" in groups else next(iter(groups))
        ref = str(groups[ref_tag][0])
        for ic in range(n_ic):
            for method in ["gm21", "am12", "am14", "yue22"]:
                jobs.append(dict(problem=problem, ic=ic, kind="baseline", method=method, label={"gm21": "[21]", "am12": "[12]", "am14": "[13]", "yue22": "[Yue22]"}[method], save_traj=True))
            jobs.append(dict(problem=problem, ic=ic, kind="fixed", run_dir=ref, label="fixed nominal gains", save_traj=True))
            jobs.append(dict(problem=problem, ic=ic, kind="heuristic", run_dir=ref, label="heuristic alpha-rate in (5)", save_traj=True))
            for tag, dirs in groups.items():
                trained_on = load_run_args(dirs[0]).problem
                label = f"RL {tag}" if trained_on == problem else f"RL {tag} (zero-shot)"
                if trained_on != problem and tag != "paper5_fixed":
                    continue
                for rd in dirs:
                    jobs.append(dict(problem=problem, ic=ic, kind="rl", run_dir=str(rd), label=label, save_traj=(rd == dirs[0])))
    # Phase 1 sensitivity: finite-time estimate of w_l
    if "paper5_fixed" in groups:
        for dur in [10.0, 20.0, 50.0]:
            for rd in groups["paper5_fixed"]:
                jobs.append(dict(problem="paper5", ic=0, kind="rl", run_dir=str(rd), label=f"phase1 T_w={dur:g}",
                                 overrides={"phase1_duration": dur}, phase1=True))

    if args.only_baselines:
        jobs = [j for j in jobs if j.get("method") in args.only_baselines]
    print(f"{len(jobs)} jobs on {args.workers} workers")
    with Pool(args.workers) as pool:
        results = pool.map(run_job, jobs, chunksize=1)

    out_dir = root / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    trajs = {}
    if args.only_baselines:  # merge into the previous evaluation
        new_labels = {r["label"] for r in results}
        old = json.load(open(out_dir / "results.json"))
        results = [r for r in old if r["label"] not in new_labels] + results
        trajs = {k: v for k, v in json.load(open(out_dir / "trajectories_nominal_x0.json")).items()
                 if k.split("|", 1)[1] not in new_labels}
    for r in results:
        if "traj" in r:
            trajs[f"{r['problem']}|{r['label']}"] = r.pop("traj")
    with open(out_dir / "results.json", "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    with open(out_dir / "trajectories_nominal_x0.json", "w") as fh:
        json.dump(trajs, fh)

    lines = ["# Evaluation on the shared test set", "",
             f"Test set: nominal x(t0) + {N_RANDOM_TESTS} random x(t0) = x_nom * factor, factor in [1/{m.RANDOM_INIT_FACTOR}, {m.RANDOM_INIT_FACTOR}] "
             f"(seed {TEST_SEED}). Mean ± std over initial conditions and training seeds.", ""]
    for problem in ["paper5", "ring10"]:
        for subset, ics in [("nominal x(t0) only", [0]), ("10 unseen random x(t0)", list(range(1, 1 + N_RANDOM_TESTS)))]:
            lines += [f"## {problem}: {subset}", "",
                      "| method | runs | settling time [s] | overshoot [%] | cost IAE | final max error |",
                      "|---|---|---|---|---|---|"]
            labels = list(dict.fromkeys(r["label"] for r in results if r["problem"] == problem and not r.get("phase1")))
            for lab in labels:
                rs = [r for r in results if r["problem"] == problem and r["label"] == lab and r["ic"] in ics and not r.get("phase1")]
                n_runs = len({r.get("run_dir", r.get("method")) for r in rs})
                lines.append(f"| {lab} | {n_runs} | {fmt([r['settle'] for r in rs])} | {fmt([r['overshoot'] for r in rs])} | "
                             f"{fmt([r['cost_iae'] for r in rs])} | {fmt([r['final_err'] for r in rs])} |")
            lines.append("")
    ph = [r for r in results if r.get("phase1")]
    if ph:
        lines += ["## Phase 1 sensitivity (paper5, nominal x(t0), RL paper5_fixed seeds)", "",
                  "| label | max abs w error | settling time [s] | final max error |", "|---|---|---|---|"]
        for lab in dict.fromkeys(r["label"] for r in ph):
            rs = [r for r in ph if r["label"] == lab]
            lines.append(f"| {lab} | {rs[0]['w_err']:.1e} | {fmt([r['settle'] for r in rs])} | {fmt([r['final_err'] for r in rs])} |")
        lines.append("")
    # training cost
    lines += ["## Training cost", "", "| run | steps | wall time [h] | ms / step |", "|---|---|---|---|"]
    for rd in runs:
        with np.load(rd / "training_logs.npz") as d:
            if "train_seconds" in d:
                sec = float(d["train_seconds"][0]); steps = float(d["steps"][-1])
                lines.append(f"| {rd.parent.name} | {steps:.0f} | {sec / 3600:.2f} | {1e3 * sec / steps:.2f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
