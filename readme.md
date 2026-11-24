# IAEML — RoboTaxi Navigation

Implementation of the two practical assignments for **Introduction to Adaptive and Efficient Machine Learning (IAEML)**. The project develops a JAX-based 2D vehicle environment and investigates reinforcement learning for navigation among static and moving obstacles.

## Overview

- **Assignment 1 — Environment:** continuous acceleration and steering, kinematic bicycle dynamics, rolling resistance, drag, ray-based perception, reward shaping and episode termination. A classical Pure Pursuit controller demonstrates navigation on map 0.
- **Assignment 2 — Reinforcement learning:** PPO with a shared actor-critic network, parallel environments and compiled rollouts. Two experiments compare the initial training setup with route-conditioned observations, curriculum learning, revised rewards and action repetition.

## Results

| Metric | Assignment 1: Pure Pursuit | PPO baseline (A) | Improved PPO (B) |
|---|---:|---:|---:|
| Training physics steps | — | 1,048,576 | 1,559,065 |
| Training policy decisions | — | 1,048,576 | 262,144 |
| Last reported training success rate | — | 0% | 8.5% |
| Peak full-route reported training success | — | 0% | 12.9% |
| Last reported mean original training return | — | -18.24 | -9.82 |
| Final training throughput (physics steps/s) | — | 1,594 | 2,608 |
| Successful demonstration physics steps | 1,118 | — | 442 |
| Original return in successful demonstration | 34.46 | — | 41.09 |

The baseline policy learned to remain stationary. Experiment B learned active navigation and reached the goal in **1 of 10 stochastic evaluation episodes** (9 collisions, no timeouts). Its single deterministic evaluation ended in a collision. The successful stochastic episode was episode 5 using seed 123, with 74 policy decisions and a shaped return of 129.19.

The observed training throughput increased by approximately **1.64×**. This is an end-to-end comparison of different training configurations, not an isolated benchmark of action repetition or JIT. Training success rates refer to reporting windows rather than independent evaluation, and shaped returns are not directly comparable with the original reward.

### Reports and demonstrations

| Assignment | Report | Video |
|---|---|---|
| 1 — Environment | [Assignment 1 report](report_A1.md) | [Pure Pursuit demonstration](assignment1_demo.mp4) |
| 2 — Reinforcement learning | [Assignment 2 report](report_A2.md) | [Successful PPO episode](results/ppo_improved/assignment2_stochastic_success.mp4) |

The PPO video shows a selected successful episode, not typical performance across all evaluation runs.

Additional artifacts: [baseline video](results/ppo/assignment2_demo.mp4), [baseline metrics](results/ppo/metrics.csv), [baseline checkpoint](results/ppo/model.eqx), [improved checkpoint](results/ppo_improved/model.eqx), [training configuration](results/ppo_improved/config.json), [improved training metrics](results/ppo_improved/metrics.csv), [deterministic evaluation](results/ppo_improved/evaluation_deterministic.csv), [stochastic evaluation](results/ppo_improved/evaluation_stochastic.csv).

## Repository structure

```text
.
├── instructions/
│   ├── readme.md
│   └── stage_1.md
├── src/
│   ├── main.py
│   ├── algorithms/
│   │   ├── path_controller.py
│   │   ├── ppo.py
│   │   ├── rl_env.py
│   │   ├── train_ppo.py
│   │   ├── train_ppo_b.py
│   │   ├── visualize.py
│   │   └── readme.md
│   ├── env/
│   │   ├── base.py
│   │   ├── robotaxi.py
│   │   ├── tasks.py
│   │   └── __init__.py
│   ├── eval/
│   │   ├── evaluate_ppo.py
│   │   ├── evaluate_ppo_b.py
│   │   └── test_robotaxi.py
│   └── utils/
│       ├── autoreset.py
│       ├── renderer.py
│       ├── utils.py
│       └── __init__.py
├── results/
│   ├── ppo/                       # Baseline experiment
│   │   ├── assignment2_demo.mp4
│   │   ├── metrics.csv
│   │   └── model.eqx
│   ├── ppo_baseline/              # Archived baseline (including nested backup)
│   ├── ppo_b_smoke/               # Improved PPO smoke test
│   │   ├── config.json
│   │   ├── metrics.csv
│   │   └── model.eqx
│   └── ppo_improved/
│       ├── assignment2_deterministic.mp4
│       ├── assignment2_stochastic.mp4
│       ├── assignment2_stochastic_success.mp4
│       ├── config.json
│       ├── evaluation_deterministic.csv
│       ├── evaluation_stochastic.csv
│       ├── metrics.csv
│       └── model.eqx
├── assignment1_demo.mp4
├── report_A1.md
├── report_A2.md
├── requirements.txt
├── .gitignore
└── README.md
```

The tree reflects the supplied directory listing, with local virtual environment files, Python caches, the tree export and duplicate backup contents omitted for clarity. Training and evaluation commands below recreate these outputs; keep the submitted checkpoints and experimental logs for reproducibility.

## Installation

Run commands from the repository root. Python 3.13 was used for the recorded Windows runs. A CPU is sufficient; JAX can also use a supported accelerator when installed appropriately.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux/macOS, activate the environment with `source .venv/bin/activate` instead.

The scripts were moved into `src/algorithms` and `src/eval`. Set the Python module search path from the repository root to support the original imports (such as `env.robotaxi`, `ppo` and `rl_env`):

```powershell
$env:PYTHONPATH = "$PWD/src;$PWD/src/algorithms"
```

On Linux/macOS:

```bash
export PYTHONPATH="$PWD/src:$PWD/src/algorithms"
```

Repeat this step in each new terminal. Keep the repository root as the working directory so relative result paths resolve correctly. If dependencies are missing from `requirements.txt`, install them with `python -m pip install jax equinox chex numpy optax matplotlib`.

Install FFmpeg and ensure it is available on `PATH` for MP4 export. On Windows, for example:

```powershell
winget install --id Gyan.FFmpeg -e
ffmpeg -version
```

Check the JAX device:

```bash
python -c "import jax; print(jax.devices())"
```

The visualization scripts use the non-interactive Matplotlib `Agg` backend, so a Tkinter window is not required.

## Reproduce Assignment 1

```bash
python src/eval/test_robotaxi.py
python src/algorithms/visualize.py
```

The first command runs a basic environment smoke test. The second runs the classical controller and exports `assignment1_demo.mp4` in the repository root. The recorded successful run on map 0 reached the goal in 1,118 physics steps with no collision and an original return of 34.46.

## Reproduce Assignment 2

### Experiment A — Baseline PPO

To train without overwriting the submitted results, use a fresh output directory:

```bash
python src/algorithms/train_ppo.py --steps 1048576 --envs 16 --rollout 128 --seed 42 --out results/ppo_reproduced
python src/eval/evaluate_ppo.py --checkpoint results/ppo_reproduced/model.eqx --episodes 1 --seed 123 --deterministic --video
```

To evaluate the saved baseline checkpoint directly:

```bash
python src/eval/evaluate_ppo.py --episodes 1 --seed 123 --deterministic --video
```

### Experiment B — Improved PPO

```bash
python src/algorithms/train_ppo_b.py --steps 262144 --envs 16 --rollout 128 --seed 42 --out results/ppo_improved_reproduced
python src/eval/evaluate_ppo_b.py --checkpoint results/ppo_improved_reproduced/model.eqx --episodes 10 --seed 123 --video
```

To reproduce evaluation using the submitted checkpoint:

```bash
python src/eval/evaluate_ppo_b.py --episodes 1 --seed 123 --deterministic
python src/eval/evaluate_ppo_b.py --episodes 10 --seed 123 --video
```

To explicitly record episode 5 of the submitted stochastic evaluation:

```bash
python src/eval/evaluate_ppo_b.py --episodes 10 --seed 123 --video --video-episode 5
```

The evaluation script automatically selects the first successful episode for video export unless an episode number is specified. Evaluation uses the original starting position on map 0. Repeated training may produce different results across hardware or dependency versions; the selected episode is reproducible with the recorded checkpoint and evaluation seed in the original setup. Evaluation writes CSVs and videos next to the checkpoint, so copy the checkpoint directory first if preserving existing artifacts is important.

### Training configuration

| Setting | Experiment A | Experiment B |
|---|---:|---:|
| Training seed | 42 | 42 |
| Parallel environments | 16 | 16 |
| Rollout length | 128 | 128 |
| Observation size | 38 | 25 |
| Hidden layers | 2 × 64 | 2 × 64 |
| Discount factor | 0.99 | 0.995 |
| GAE lambda | 0.95 | 0.95 |
| PPO clipping | 0.2 | 0.2 |
| Learning rate | 0.0003 | 0.0003 |
| Physics steps per action | 1 | Up to 6 |
| Curriculum | None | Near goal → middle → original spawn |

In Experiment B, `--steps` counts policy decisions, not physics frames. Action repetition stops early at terminal transitions. The curriculum uses the first 15% of training near the goal, the next 30% at an intermediate position, and the remaining training from the original spawn.

## Computational efficiency

- `jax.jit` compiles simulation and training computations.
- `jax.vmap` batches independent environment transitions and policy inference.
- `jax.lax.scan` collects rollouts and supports repeated simulation steps within compiled loops.
- Action repetition reduces the frequency of neural network inference relative to physics simulation.
- Equinox serialization stores model parameters; Optax provides the Adam optimizer.

## Limitations

The collision convention follows the starter environment's simplified center-distance threshold, rather than exact overlap of two radius-one circles. The classical controller and improved PPO use a manually defined reference route specific to map 0. PPO independently selects acceleration and steering; it does not use expert controller actions. Reliable collision avoidance and generalization to other maps have not been established. The stochastic evaluation sample is small and uses one map and a fixed initial environment. The successful demonstration should therefore be interpreted as proof of feasible learned navigation, not a production-ready autonomous driving policy.

## References

- [IAEML starter repository — InnoRL](https://github.com/InnoRL/IAEML-assignment)
- [Proximal Policy Optimization Algorithms — Schulman et al.](https://arxiv.org/abs/1707.06347)
- [JAX documentation](https://docs.jax.dev/)
- [Equinox documentation](https://docs.kidger.site/equinox/)
- [Optax documentation](https://optax.readthedocs.io/)
