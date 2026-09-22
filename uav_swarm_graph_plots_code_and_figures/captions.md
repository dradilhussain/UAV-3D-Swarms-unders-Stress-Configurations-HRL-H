# Suggested captions (edit the figure numbers to your final order)

- **Fig01_architecture** - Architecture of HRL-H: the learned high level produces normalised task preferences at every step, and the heuristic executor performs battery-safety filtering, commit-on-approach and a global energy-first auction before issuing 3D navigation actions.
- **Fig03_traj_HRLH_2x2** - HRL-H trajectories (seed 0) in the four environments. Triangles mark numbered UAV start positions, circles mark end positions, stars mark tasks, and trail colour shows final SOC (green = full, red = depleted).
- **Fig05-Fig08** - Seed-0 trajectories of all eight methods in the open volume / layered / open volume with NFZs / layered with NFZs. Panel titles give completed tasks and episode length.
- **Table09_hrlh_radar** - Profile of HRL-H in the four environments under the base configuration. Each axis is scored relative to the best value of that metric over all methods and environments (1 = best).
- **Table10_base_matrix** - Base-configuration results of all methods in the four environments (mean over 10 seeds). Colour shows the rank of a method for each metric within an environment (green = best, red = worst).
- **Table10_base_radar** - Performance profile of all methods in each environment; each axis is scored relative to the best method (1 = best).
- **Table11-12_extended_matrix** - Extended base-configuration metrics: team reward, minimum SOC, time below the SOC floor, information age, completion latency and blocked moves (mean over 10 seeds).
- **Table13_feasible_matrix** - NFZ environments on the nine feasible seeds (seed 8 excluded, left of the line) and over all ten seeds (right of the line).
- **Table14_perseed_dots** - Per-seed energy per task and episode length of HRL-H and DMPC in the NFZ environments. The shaded seed contains an unreachable task; two HRL-H seeds in the layered NFZ environment are slowed by a stalled UAV.
- **Table15_ablation_matrix / _radar** - Ablation of HRL-H: full method, QMIX as high-level learner, and executor removed (= MAPPO).
- **Table16_comm_matrix** - Energy per task under packet-loss and communication-range sweeps; colour shows the change relative to the base level (20% loss, 12 m).
- **Table17 / 18 / 19 / 20 / 23 / 24 / 25** - Energy per completed task and coverage at each sweep level (packet loss / communication range / number of tasks / number of UAVs / deadline / coupled-task share / number of NFZs). Colour is the rank of the method within each column (green = best, red = worst).
- **Table21_dropout** - Coverage, energy per completed task and stranded UAVs under UAV dropout (0%, 25%, 50%). Colour is the rank of the method within each column (green = best, red = worst).
- **Table22_dropout** - Energy per completed task and coverage versus the share of UAVs that fail during the mission; same rank-matrix layout as Tables 17–20 and 23–25.
- **Robustness_matrix_Tables16-25** - Robustness fingerprint in each environment: every recorded metric at the hardest level of each stressor (mean over the stressors that apply). Circle colour and area are the rank among the seven methods (green/large = best). The bar R is the mean rank across metrics.
