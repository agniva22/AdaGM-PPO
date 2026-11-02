# RL-Scheduled Gain Tuning Insert

## Contribution Sentence
We replace the paper's fixed scalar gains `K1`, `K2` and the heuristic adaptive law `alpha_dot_i = ||epsilon_i||^2` with a decentralized CTDE-PPO policy that schedules `K1_i`, `K2_i`, and `Delta alpha_i` online, improving adaptation speed while preserving the asymptotic convergence guarantee of Theorem 1.

## Problem Formulation
Each agent `i` maintains a local policy `pi_i(s_i) -> a_i`, where the local observation is

```text
s_i(t) = [ epsilon_i(t), grad f_i(x_i), ||x_i - x_bar||, alpha_i(t), beta_i(t), t ]
```

The term `epsilon_i` is already computed inside Eq. (7c), so it adds no extra communication cost.

The action is

```text
a_i(t) = [ K1_i(t), K2_i(t), Delta alpha_i(t) ]
```

with bounded gains

```text
K_min <= K1_i(t), K2_i(t) <= K_max
```

and `Delta alpha_i` replacing the fixed `||epsilon_i||^2` update.

## CTDE-PPO Training
Use centralized training with decentralized execution:

```text
s = [s_1, s_2, ..., s_N],   V(s) = centralized critic
```

The actor is shared across agents because the dynamics are identical; agent-specific differences enter through `grad f_i(x_i)` in the observation.

## Reward

```text
r_i(t) = - ||x_i(t) - x*|| - lambda1 * max(0, f(x(t)) - f(x*)) - lambda2 * |K1_i|
```

This reward penalizes tracking error, cost overshoot, and overly aggressive gains.

## Suggested Figure Caption
CTDE-PPO gain scheduling for GM-driven distributed optimization. Local observations feed a shared actor, a centralized critic trains offline using the global state, and the resulting actions schedule `K1_i`, `K2_i`, and `Delta alpha_i` online. The RL layer improves gain selection while preserving the asymptotic convergence guarantee of Theorem 1.
