# AdaGM-DO: RL-based gain scheduling for GM-driven distributed optimization

A decentralized policy, trained with CTDE-PPO, schedules the gradient gain and the adaptive-gain rate of each agent from local observations only. The adaptive gain multiplies both the integral and the gradient terms, so the closed loop converges to the global optimum for any admissible policy.

## Repository structure

```
adagm_do.py        method: problems, GM-driven DO dynamics, environment, CTDE-PPO training
yue2022.py         baseline: Yue et al., IEEE TAC 2022
evaluate.py        all methods on 10 unseen initial conditions (Table I)
make_figures.py    state and cost trajectories (Fig. 3)
run_all.sh         train every policy, then evaluate
results/
  <run>/rl/        trained policies (actor, critic, arguments, training logs)
  evaluation/      results.json, summary.md, trajectories
  figures/         figures of the paper
```

## Setup

```bash
pip install -r requirements.txt
```

`make_figures.py` also needs Ghostscript and `pdftops` (poppler-utils) to export font-free EPS files.

## Reproducing the results

The trained policies are included, so the table and figures can be reproduced without training:

```bash
python evaluate.py        # writes results/evaluation/summary.md
python make_figures.py    # writes results/figures/*.eps
```

To retrain all policies (9 runs, about 1 h each on a GPU) and evaluate:

```bash
bash run_all.sh [max_parallel_runs]
```

A single policy:

```bash
python adagm_do.py --mode rl --problem paper5 --seed 0 --outdir results/paper5_fixed_s0
```

| Run | Setting |
|---|---|
| `paper5_fixed_s{0..4}` | 5-agent network, nominal initial condition |
| `paper5_random_s{0..2}` | 5-agent network, random initial conditions |
| `ring10_fixed_s0` | 10-agent network |
