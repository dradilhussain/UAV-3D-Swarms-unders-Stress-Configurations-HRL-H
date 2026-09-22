# HRL-H figure package (matched to manuscript draft v5)

All tables of the draft are in Section 5 (there is no appendix), and every figure file is named after the draft table
(`TableNN_...`) or draft figure (`FigNN_...`) it illustrates. All figures are **600 dpi PNG** in `figures/`. Each figure is
produced by a notebook in `notebooks/`, which reads the stored results in `data/` through `common.py`.

## Run
```
pip install numpy matplotlib jupyter
cd notebooks
jupyter notebook            # open a notebook, then Kernel > Restart & Run All
```
Fonts: Times New Roman, falling back to Liberation Serif or DejaVu Serif.

## Figure map (draft v5 numbering)
| Notebook | Figure file | Draft table / figure | Style |
|---|---|---|---|
| 01 | `Table09_hrlh_radar.png` | Table 9 HRL-H in the four environments | radar |
| 01 | `Table10_base_matrix.png` | Table 10 base comparison (8 methods x 7 metrics x 4 environments) | annotated matrix |
| 01 | `Table10_base_radar.png` | Table 10 (overall profile) | radar |
| 01 | `Table11-12_extended_matrix.png` | Tables 11 and 12 extended metrics | annotated matrix |
| 02 | `Table13_feasible_matrix.png` | Table 13 NFZ feasible seeds | annotated matrix |
| 02 | `Table14_perseed_dots.png` | Table 14 per-seed HRL-H vs DMPC | paired dot plot |
| 02 | `Table15_ablation_matrix.png`, `Table15_ablation_radar.png` | Table 15 ablation | matrix, radar |
| 03 | `Table16_comm_matrix.png` | Table 16 packet-loss / range energy | sensitivity matrix |
| 03 | `Table17_packet_loss.png`, `Table18_comm_range.png` | Tables 17, 18 | annotated rank matrix |
| 03 | `Table19_task_density.png`, `Table20_uav_count.png` | Tables 19, 20 (draft Figures 9, 10) | annotated rank matrix |
| 03 | `Table21_dropout.png` | Table 21 dropout summary | annotated rank matrix (coverage, energy, stranded) |
| 03 | `Table22_dropout.png` | Table 22 UAV-dropout sweep | annotated rank matrix |
| 03 | `Table23_deadline.png`, `Table24_coupled.png`, `Table25_obstacles.png` | Tables 23, 24, 25 | annotated rank matrix |
| 03 | `Robustness_matrix_Tables16-25.png` | summary of Tables 16-25 | balloon fingerprint, all metrics, one panel per environment |
| 04 | `Fig01_architecture.png` | Figure 1 | diagram |
| 04 | `Fig03_traj_HRLH_2x2.png` | Figure 3 (environment-figure design) | 3D trajectories |
| 04 | `Fig05_...open`, `Fig06_...layered`, `Fig07_...open_nfz`, `Fig08_...layered_nfz` | Figures 5-8 | 3D trajectories |

Tables 1-8 (methods, configuration, parameters) are not results and have no figure. In the draft, Figure 4 (bar overview) is
superseded by `Table10_base_matrix.png`, and Figures 9 and 10 carry the same information as `Table19_...` and `Table20_...`.

## Reading the figures
* **Matrices**: colour = rank of the method within the column of that environment (green best, red worst; grey = not directional
  such as information age, or all values equal). The printed number is the mean over 10 seeds. Table 10 includes Random; sweep
  matrices (Tables 16–25) omit it so the colour scale separates the seven non-trivial methods. Table 16 is coloured by change
  from the base level rather than by rank.
* **Radars**: 7 axes scored relative to the best method (1 = best): coverage/best, 1 - stranded, best steps/steps, best
  energy/energy, best distance/distance, SOC/best, 1 - contention.
* **Robustness fingerprint**: one panel per environment; each circle is one metric at the hardest stressor level (mean
  over the stressors that apply). Colour and area = rank (green/large = best). The bar R is the mean rank across metrics.
* **NFZ environments**: in seed 8 of both NFZ environments a task is generated inside a no-fly zone and cannot be reached, and in
  layered NFZ seed 3 one UAV starts inside a zone. This inflates the all-seed means (see Tables 13 and 14).

## Data
`data/eval_*.json` are the project's evaluation outputs. `data/perseed_base.json` and `data/episodes_seed0.pkl` were regenerated
with the project code (`support_scripts/`); their aggregates reproduce the project's `eval_scenarios.json` exactly.
