# UAV Swarm Coordination -- 3D Project (8 approaches + proposed HRL-H)

Decentralized, SOC-aware UAV swarm coordination in a **3D** environment
((x, y, z) positions with an altitude/climb-power energy model), comparing
**eight** coordination approaches on the same metrics as the 2D project:

* **Random** -- uniformly random valid action (sanity floor)
* **Greedy-Nearest** -- always heads to the nearest active task, ignores battery until critical
* **CBBA** -- Consensus-Based Bundle Algorithm, SOC-aware bidding (including climb cost) over a lossy/stale comm channel
* **MAPPO** -- Multi-Agent PPO (NumPy reference implementation) **ported to 3D**, BC-warm-started + PPO fine-tuned in the notebook
* **MADDPG** -- discrete-action MADDPG (softmax actors, centralized critic), BC-warm-started + off-policy replay
* **QMIX** -- joint value decomposition with a monotone mixing network, stabilized via reward normalization / target clamp / BC regularization
* **DMPC** -- distributed model-predictive control using the known 3D dynamics (incl. climb energy) + CBBA consensus
* **HRL-H (proposed)** -- hierarchical RL + heuristic navigation: a high-level allocator (MAPPO or QMIX scores) drives a deterministic commit-and-fly, battery-safe, deconflicted executor

## Four evaluation environments

Every trained / baseline policy is evaluated in **four** environments (saved as a
**2x2** snapshot that can be re-plotted later from JSON):

| Id | Vertical layout | No-fly zones |
|---|---|---|
| **open** | Open volume (tasks scattered through the altitude range) | none |
| **layered** | Discrete altitude bands | none |
| **open_nfz** | Open volume | 4 cuboid no-fly zones |
| **layered_nfz** | Discrete altitude bands | 4 cuboid no-fly zones |

Training of MAPPO / MADDPG / QMIX is unchanged (one checkpoint, layered, obstacle-free).
Checkpoints are reused zero-shot in all four environments.

## Research extensions

On top of the 8-approach core, the 3D environment and notebook add three
backward-compatible research extensions. Each is **disabled by default** on the
training config:

* **Time-critical tasks** (`task_mode="time_critical"`, `task_deadline_steps`,
  `task_value_decay_per_step`, `task_spawn_window`). Swept per environment.
* **Coupled (multi-agent) tasks** (`task_requirements`). Swept per environment.
* **Obstacles / no-fly zones** (`obstacles_enabled`, `n_obstacles`, `obstacle_half`).
  First-class as `open_nfz` / `layered_nfz`; an extra density sweep (0 / 2 / 4
  cuboids) is also saved for the open and layered layouts.

## Contents

```
uav_swarm_marl_3d.ipynb     Main notebook: train 3D MAPPO/MADDPG/QMIX, evaluate all 8
                            approaches in 4 environments, save JSON/CSV + figures
code/env_3d/                3D environment, baselines, plotting, eval I/O helpers
code/mappo_3d/              NumPy MAPPO ported to 3D
code/marl_algorithms/       MADDPG + QMIX
code/model_based/           DMPC
code/hrl_h/                 HRL-H + ablations
outputs/                    Evaluation JSON + CSV + checkpoints
outputs/by_environment/     Per-environment copies (open, layered, open_nfz, layered_nfz)
outputs/env_snapshots/      Environment layouts for re-plotting the 2x2 figure
figures/                    Publication figures (600 dpi)
requirements.txt            numpy, matplotlib, jupyter/ipykernel
```

## How to run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Start Jupyter from this directory and open the notebook:

```
jupyter notebook uav_swarm_marl_3d.ipynb
```

3. Run all cells (Kernel > Restart & Run All).

Runtime is about **45-60 minutes** on a single CPU core for training; full
4-environment evaluation is longer because every sweep is repeated per
environment. Re-running training is fast because checkpoints are cached in
`outputs/`.

## Sweeps (every environment, every policy)

Base configuration is **8 UAVs, 8 tasks**. Independently, in each of the four
environments:

| Sweep | Levels |
|---|---|
| Task-density | 4, 8, 16, 32 tasks |
| UAV-count | 4, 8, 16 UAVs |
| Communication-range | 6, 12, 25, 50 m |
| Packet-loss | 0%, 20%, 40%, 60% |
| UAV-dropout | 0%, 25%, 50% |

Results are written as JSON **and** CSV (tables only; no sweep graphs):

* combined: `outputs/eval_<sweep>.json` / `.csv` (with an `environment` column)
* per environment: `outputs/by_environment/<id>/eval_<sweep>.json` / `.csv`

## Figures

All figures are **600 dpi** PNG with large-size axis / title / legend text.
Layout rule: **4 panels -> 2x2**, **5 or 6 panels -> 3x2**, **8 trajectories -> 4x2**.

One trajectory, comparison, extended-metrics, and HRL-H ablation figure **per
environment**. Scenario comparison includes all four environments.

| Figure | Contents |
|---|---|
| `figures/stress_env_snapshot_2x2.png` | 2x2 environment plot (open / layered / open+NFZ / layered+NFZ) |
| `figures/stress_env_snapshot_<id>.png` | Individual environment snapshots |
| `figures/stress_trajectories_<id>.png` | 4x2 trajectory grids (all 8 approaches), one env each |
| `figures/stress_comparison_summary_<id>.png` | Base-config bars, one env each |
| `figures/stress_extended_metrics_<id>.png` | Extended metrics (3x2), one env each |
| `figures/stress_hrlh_ablation_comparison_<id>.png` | HRL-H vs HRL-H-NAV vs HRL-H-QMIX, one env each |
| `figures/stress_scenario_comparison.png` | All 4 environments x coverage / stranded / reward |

Environment layouts are also stored as JSON in `outputs/env_snapshots/` so the
2x2 figure can be re-plotted later without re-running the simulator.

Rebuild figures and CSV tables from already-saved JSON (no training / eval):

```
python code/env_3d/replot_from_outputs.py
```

## Comparison tables

`outputs/comparison_table.csv`, `outputs/comparison_table.json`, and
`outputs/comparison_table.md` list every metric for every policy in every
environment (mean +/- std). Per-environment Markdown copies live under
`outputs/by_environment/<id>/comparison_table.md`.

## Evaluation metrics

Coverage (%), stranded UAVs (%), total reward, mean final SOC, episode length, and the
extended set (energy/task, 3D distance/task, min SOC, time below floor, contention,
information age, completion latency, blocked moves). Identical across the four
environments so they are directly comparable.

## Notes

* 3D observation space: own = x,y,z,soc (4); neighbor = rel_x,rel_y,rel_z,soc,
  staleness,valid (6); per-task = x,y,z,active (4). Padded to `N_TASKS_MAX = 32`,
  giving obs dimension 162 and action dimension 34, so trained networks generalize to
  any task count <= 32.
* MAPPO / MADDPG / QMIX use a **behavior-cloning warm-start** (Greedy-Nearest demos).
  The trained checkpoints ship in `outputs/`. Model implementations are unchanged;
  only evaluation is repeated in the four environments.
* Set `USE_STRESS_CFG = False` in the config cell to run the easier default environment.
  `ENV_CFGS` holds the four evaluation environments; `ENV_CFG` remains the training config.
