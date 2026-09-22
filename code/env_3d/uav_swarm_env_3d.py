"""
3D extension of the UAV swarm environment.

Adds a z (altitude) dimension to positions, tasks, and the base, on top of
the same SOC dynamics and communication-constraint model as the 2D version
(uav_swarm_env.py). Two scenario types are supported for how tasks are
distributed in the vertical dimension:

  "layered" -- tasks are assigned to a small number of discrete altitude
      bands (e.g. building floors, delivery drop heights, inspection
      levels), each a tight cluster in z. Realistic for urban/infrastructure
      inspection missions.

  "volume"  -- tasks are scattered uniformly through the full altitude
      range, independent of x,y. Realistic for open-airspace search/
      surveillance missions with no vertical structure.

Research extensions (all disabled by default, so existing behaviour/results
are bit-for-bit unchanged when they are off):

  * time-critical tasks  -- task_mode="time_critical". Tasks can spawn over a
      window (task_spawn_window, as a fraction of max_steps), carry a hard
      deadline (task_deadline_steps after spawn; tasks that miss it are
      marked expired and can never be completed), and their value can decay
      per step after spawn (task_value_decay_per_step).

  * coupled tasks (multi-agent) -- task_requirements=() means every task is
      single-agent (classic behaviour: one UAV reaching the task completes
      it). Supplying a per-task tuple R[k] makes task k need R[k] *distinct
      agents* to contribute (each agent present at the task adds +1 progress
      per step; the task completes when progress >= R[k]). Single-agent
      tasks (R=1) complete exactly as before.

  * obstacles / no-fly zones -- obstacles_enabled=True spawns
      n_obstacles cuboid no-fly zones (axis-aligned boxes spanning
      obstacle_z_min..obstacle_z_max, with half-width obstacle_half in x/y).
      Agents cannot enter them: the movement model slides along the blocking
      face, and any fully-blocked move costs only hover power. Task / agent /
      base placements avoid the boxes. Exposed as env.obstacles
      (m, 6) [x0,y0,z0,x1,y1,z1] and env.n_blocked_moves.

NOTE on reward shaping (see README.md for the full investigation): the
team-coverage bonus is paid ONCE per increment in coverage, not every
timestep. An earlier version paid it every timestep proportional to the
current (cumulative) coverage fraction, which silently rewarded taking
longer to finish a mission -- this made greedy-nearest appear to beat CBBA
by a small but statistically real margin, even though CBBA's task
allocation was never actually worse (identical task-completion counts,
equal-or-better SOC margins in every tested seed). Fixed here; see
diagnose_greedy_vs_cbba.py to reproduce the before/after comparison.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal, Tuple


class Box:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low, self.high, self.shape, self.dtype = low, high, shape, dtype


class Discrete:
    def __init__(self, n):
        self.n = n

    def sample(self, rng: np.random.Generator):
        return int(rng.integers(0, self.n))


class spaces:
    Box = Box
    Discrete = Discrete


@dataclass
class EnvConfig3D:
    n_agents: int = 6
    n_tasks: int = 12
    area_size: float = 100.0          # x,y footprint (square), meters
    altitude_min: float = 5.0         # minimum flight altitude, meters (ground-safety margin)
    altitude_max: float = 60.0        # maximum flight altitude, meters
    scenario: Literal["layered", "volume"] = "layered"
    n_layers: int = 3                 # number of altitude bands, used only if scenario="layered"
    layer_band_thickness: float = 4.0 # +/- spread of tasks within a layer, meters (layered only)
    base_altitude: float = 0.0        # base/depot sits at ground level (launch/land point)
    max_steps: int = 200
    dt: float = 1.0

    # --- SOC / energy model ---
    soc_min_safe: float = 0.15
    battery_capacity_wh: float = 90.0
    hover_power_w: float = 120.0
    move_power_coeff: float = 3.0     # W per (m/s)^2 of horizontal+vertical speed
    climb_power_coeff: float = 1.5    # additional W per (m/s) of *upward* climb rate (fighting gravity)
    max_speed: float = 8.0            # m/s, 3D speed cap

    # --- communication constraints (now 3D range) ---
    comm_range: float = 25.0
    packet_loss_prob: float = 0.2
    max_staleness_steps: int = 10
    neighbor_cap: int = 5

    # --- reward shaping ---
    task_reward: float = 10.0
    soc_violation_penalty_coeff: float = 50.0
    stranded_penalty: float = 200.0
    team_coverage_bonus_weight: float = 0.3

    # --- research extensions (defaults preserve original behaviour) -------
    task_mode: Literal["static", "time_critical"] = "static"
    task_spawn_window: float = 0.0    # fraction of max_steps over which tasks spawn (0 = all at t=0)
    task_deadline_steps: int = 0      # hard deadline after spawn, in steps (0 = none)
    task_value_decay_per_step: float = 0.0  # reward multiplier decays per step after spawn (0 = none)
    task_requirements: Tuple[int, ...] = ()  # per-task agents required; () = all single-agent (R=1)

    obstacles_enabled: bool = False
    n_obstacles: int = 0
    obstacle_half: float = 12.0       # half-width of each cuboid in x and y, meters
    obstacle_z_min: float = 0.0
    obstacle_z_max: float = 60.0

    seed: Optional[int] = None


def point_in_obstacle(p: np.ndarray, obstacles) -> bool:
    """True if point p (x, y, z) lies strictly inside any cuboid obstacle."""
    if obstacles is None or len(obstacles) == 0:
        return False
    for o in obstacles:
        if (o[0] <= p[0] <= o[1] and o[2] <= p[1] <= o[3]
                and o[4] <= p[2] <= o[5]):
            return True
    return False


def segment_hits_obstacle(p0: np.ndarray, p1: np.ndarray, obstacles) -> bool:
    """True if the line segment p0->p1 intersects any cuboid obstacle
    (Liang-Barsky clip against each axis-aligned box)."""
    if obstacles is None or len(obstacles) == 0:
        return False
    for o in obstacles:
        t0, t1 = 0.0, 1.0
        clipped = True
        for axis in range(3):
            d = float(p1[axis] - p0[axis])
            lo, hi = float(o[axis]), float(o[axis + 3])
            if abs(d) < 1e-12:
                if p0[axis] < lo or p0[axis] > hi:
                    clipped = False
                    break
            else:
                ta, tb = (lo - p0[axis]) / d, (hi - p0[axis]) / d
                if ta > tb:
                    ta, tb = tb, ta
                t0 = max(t0, ta)
                t1 = min(t1, tb)
                if t0 > t1:
                    clipped = False
                    break
        if clipped and t0 <= t1:
            return True
    return False


def move_towards(pos: np.ndarray, target: np.ndarray, step_dist: float,
                 obstacles=None) -> np.ndarray:
    """One step of the movement model: move `step_dist` toward `target`.
    If the straight-line step would enter an obstacle, slide along the
    blocking faces (axis-aligned moves that reduce distance to the target);
    if no sliding move is free, stay put. With no obstacles this is exactly
    the plain straight-line move. Returns the new position (a fresh array)."""
    delta = target - pos
    dist = np.linalg.norm(delta)
    if dist < 1e-6:
        return pos.copy()
    direction = delta / dist
    new = pos + direction * step_dist
    if not segment_hits_obstacle(pos, new, obstacles):
        return new
    # slide: try each axis individually, keep the one closest to the target
    best, best_d = None, dist
    for axis in range(3):
        cand = pos.copy()
        cand[axis] += direction[axis] * step_dist
        if segment_hits_obstacle(pos, cand, obstacles):
            continue
        d = np.linalg.norm(cand - target)
        if d < best_d:
            best_d, best = d, cand
    if best is not None:
        return best
    return pos.copy()


class UAVSwarmEnv3D:
    """3D parallel multi-agent env. Same action/observation API shape as the
    2D UAVSwarmEnv, but every position is now (x, y, z)."""

    metadata = {"name": "uav_swarm_soc_comm_3d_v1"}

    def __init__(self, config: EnvConfig3D = EnvConfig3D()):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

        self.n = config.n_agents
        self.n_tasks = config.n_tasks
        self.agents = [f"uav_{i}" for i in range(self.n)]

        self.RETURN_BASE = self.n_tasks
        self.HOLD = self.n_tasks + 1
        self.action_space_size = self.n_tasks + 2

        # own(4: x,y,z,soc) + neighbors(K*6: rel_x,rel_y,rel_z,soc_belief,staleness,valid)
        # + tasks(n_tasks*4: x,y,z,active)
        obs_dim = 4 + self.cfg.neighbor_cap * 6 + self.n_tasks * 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_space_size)

        self.base_pos = np.array([
            config.area_size / 2, config.area_size / 2, config.base_altitude
        ])

        self._t = 0
        self._reset_state()

    # ---------------------------------------------------------------- reset
    def _sample_task_altitudes(self) -> np.ndarray:
        cfg = self.cfg
        if cfg.scenario == "layered":
            layer_centers = np.linspace(
                cfg.altitude_min + cfg.layer_band_thickness,
                cfg.altitude_max - cfg.layer_band_thickness,
                cfg.n_layers,
            )
            layer_assignment = self.rng.integers(0, cfg.n_layers, size=self.n_tasks)
            centers = layer_centers[layer_assignment]
            jitter = self.rng.uniform(-cfg.layer_band_thickness, cfg.layer_band_thickness, size=self.n_tasks)
            z = np.clip(centers + jitter, cfg.altitude_min, cfg.altitude_max)
        elif cfg.scenario == "volume":
            z = self.rng.uniform(cfg.altitude_min, cfg.altitude_max, size=self.n_tasks)
        else:
            raise ValueError(f"unknown scenario: {cfg.scenario}")
        return z.astype(np.float32)

    def _sample_obstacles(self) -> np.ndarray:
        """Sample axis-aligned cuboid no-fly zones (m, 6) avoiding the base.

        Uses a dedicated RNG stream seeded from the episode seed, so turning
        obstacles on/off never changes the agent/task layout for a given seed
        (differences across obstacle levels are then pure obstacle effects)."""
        cfg = self.cfg
        if not cfg.obstacles_enabled:
            return np.zeros((0, 6), dtype=np.float32)
        rng = np.random.default_rng(cfg.seed + 777777)
        half = cfg.obstacle_half
        margin = half * 2.5
        boxes = []
        tries = 0
        while len(boxes) < cfg.n_obstacles and tries < 200:
            tries += 1
            cx = rng.uniform(margin, cfg.area_size - margin)
            cy = rng.uniform(margin, cfg.area_size - margin)
            x0, x1 = cx - half, cx + half
            y0, y1 = cy - half, cy + half
            z0, z1 = cfg.obstacle_z_min, cfg.obstacle_z_max
            o = np.array([x0, y0, z0, x1, y1, z1], dtype=np.float32)
            if x0 <= self.base_pos[0] <= x1 and y0 <= self.base_pos[1] <= y1:
                continue  # never cover the depot
            boxes.append(o)
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 6)

    def _free_xy(self, size=(1,), n_attempts=30) -> np.ndarray:
        """Sample points that do not lie inside any obstacle cuboid.
        When no obstacles are configured this is exactly a raw uniform draw
        (no extra RNG consumption), so default behaviour is seed-identical."""
        cfg = self.cfg
        n = int(np.prod(size))
        if len(self.obstacles) == 0:
            return self.rng.uniform(0, cfg.area_size, size=(n, 2))
        out = np.empty((0, 2), dtype=np.float32)
        for _ in range(n_attempts):
            cand = self.rng.uniform(0, cfg.area_size, size=(n, 2))
            z = self.rng.uniform(cfg.altitude_min, cfg.altitude_max, size=(n, 1))
            ok = np.array([not point_in_obstacle(
                np.array([cand[i, 0], cand[i, 1], z[i, 0]], dtype=np.float32),
                self.obstacles) for i in range(n)])
            out = np.vstack([out, cand[ok]])
            if len(out) >= n:
                return out[:n]
        # fall back to raw samples (rare, heavily-cluttered case)
        return self.rng.uniform(0, cfg.area_size, size=(n, 2))

    def _reset_state(self):
        cfg = self.cfg
        self.obstacles = self._sample_obstacles()

        xy = self._free_xy(size=(self.n,))
        z0 = self.rng.uniform(cfg.altitude_min, cfg.altitude_max, size=(self.n, 1))
        self.pos = np.concatenate([xy, z0], axis=1).astype(np.float32)  # (n, 3)

        self.soc = np.ones(self.n, dtype=np.float32)
        self.stranded = np.zeros(self.n, dtype=bool)

        task_xy = self._free_xy(size=(self.n_tasks,))
        task_z = self._sample_task_altitudes()
        self.task_pos = np.concatenate([task_xy, task_z[:, None]], axis=1).astype(np.float32)

        # ---- task lifecycle state ----
        req = np.asarray(cfg.task_requirements, dtype=int)
        if req.size == 0:
            req = np.ones(self.n_tasks, dtype=int)
        elif req.size != self.n_tasks:
            raise ValueError(
                f"task_requirements length {req.size} != n_tasks {self.n_tasks}")
        self.task_required = req

        spawn_frac = float(np.clip(cfg.task_spawn_window, 0.0, 1.0))
        self.task_spawn_time = np.floor(
            self.rng.uniform(0, spawn_frac * cfg.max_steps, size=self.n_tasks)
        ).astype(int) if spawn_frac > 0 else np.zeros(self.n_tasks, dtype=int)
        self.task_deadline = np.where(
            cfg.task_deadline_steps > 0,
            self.task_spawn_time + cfg.task_deadline_steps,
            -1).astype(int)
        self.task_progress = np.zeros(self.n_tasks, dtype=np.float32)
        self.task_completed = np.zeros(self.n_tasks, dtype=bool)
        self.task_expired = np.zeros(self.n_tasks, dtype=bool)
        self.task_active = np.ones(self.n_tasks, dtype=bool)
        self._refresh_task_active()

        self.current_action = np.full(self.n, self.HOLD, dtype=int)
        self.n_blocked_moves = 0

        self.belief_pos = np.tile(self.pos[None, :, :], (self.n, 1, 1)).astype(np.float32)
        self.belief_soc = np.tile(self.soc[None, :], (self.n, 1)).astype(np.float32)
        self.staleness = np.zeros((self.n, self.n), dtype=np.int32)

    def _refresh_task_active(self):
        spawned = self._t >= self.task_spawn_time
        self.task_active = (spawned & ~self.task_completed & ~self.task_expired)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._t = 0
        self._reset_state()
        return self._get_all_obs(), self._get_infos()

    # ----------------------------------------------------------------- step
    def step(self, actions: dict):
        self._t += 1
        act = np.array([actions[a] for a in self.agents], dtype=int)
        self.current_action = act

        rewards = np.zeros(self.n, dtype=np.float32)
        newly_stranded = np.zeros(self.n, dtype=bool)
        completed_before = self.task_completed.sum()

        # ---- task lifecycle: spawns, deadlines, value decay ----
        self._refresh_task_active()
        if self.cfg.task_deadline_steps > 0:
            overdue = self.task_active & (self._t > self.task_deadline)
            self.task_expired |= overdue
            self._refresh_task_active()

        for i in range(self.n):
            if self.stranded[i]:
                continue

            target = self._resolve_target(i, act[i])
            old_pos = self.pos[i].copy()
            step_dist = min(np.linalg.norm(target - self.pos[i]),
                            self.cfg.max_speed * self.cfg.dt)
            new_pos = move_towards(self.pos[i], target, step_dist, self.obstacles)
            moved = float(np.linalg.norm(new_pos - old_pos))
            # tolerance sized for float32 arithmetic (~1e-5 on 8 m steps);
            # a genuinely blocked move leaves the agent short by O(step_dist)
            if moved < step_dist - 1e-3 and step_dist > 1e-3:
                self.n_blocked_moves += 1
            self.pos[i] = new_pos

            # intended full-step speed (matches the base-config dynamics exactly;
            # a blocked agent still throttles toward its target)
            speed = step_dist / self.cfg.dt
            delta = target - old_pos
            dist = np.linalg.norm(delta)
            if dist > 1e-6:
                direction = delta / dist
                climb_rate = max(0.0, direction[2] * speed)
            else:
                climb_rate = 0.0

            power_w = (self.cfg.hover_power_w
                      + self.cfg.move_power_coeff * speed ** 2
                      + self.cfg.climb_power_coeff * climb_rate)
            energy_wh = power_w * (self.cfg.dt / 3600.0)
            self.soc[i] -= energy_wh / self.cfg.battery_capacity_wh
            self.soc[i] = max(self.soc[i], 0.0)

            k = act[i]
            if k < self.n_tasks and self.task_active[k]:
                if np.linalg.norm(self.pos[i] - self.task_pos[k]) < 1.5:
                    # one agent present contributes +1 progress; a single-agent
                    # task (R=1) completes on first contact exactly as before.
                    self.task_progress[k] += 1.0
                    if self.task_progress[k] >= self.task_required[k]:
                        self.task_completed[k] = True
                        val = self.task_value(k)
                        rewards[i] += self.cfg.task_reward * val
                        self._refresh_task_active()

            if self.soc[i] < self.cfg.soc_min_safe:
                deficit = self.cfg.soc_min_safe - self.soc[i]
                rewards[i] -= self.cfg.soc_violation_penalty_coeff * (deficit ** 2)

            if self.soc[i] <= 0.0:
                self.stranded[i] = True
                newly_stranded[i] = True
                rewards[i] -= self.cfg.stranded_penalty

        # FIXED reward shaping: pay the team bonus once per increment in
        # coverage this step, not every step proportional to cumulative
        # coverage (see module docstring / README.md for why this matters).
        completed_after = self.task_completed.sum()
        coverage_increment = (completed_after - completed_before) / max(self.n_tasks, 1)
        team_bonus = self.cfg.team_coverage_bonus_weight * coverage_increment * self.cfg.task_reward
        active_mask = ~self.stranded
        rewards[active_mask] += team_bonus

        self._update_comm_beliefs()

        all_settled = (self.task_completed | self.task_expired).all()
        terminated = bool(all_settled) or bool(self.stranded.all())
        truncated = self._t >= self.cfg.max_steps

        terminations = {a: terminated for a in self.agents}
        truncations = {a: truncated for a in self.agents}
        reward_dict = {a: float(rewards[i]) for i, a in enumerate(self.agents)}
        infos = self._get_infos()
        for i, a in enumerate(self.agents):
            infos[a]["newly_stranded"] = bool(newly_stranded[i])

        return self._get_all_obs(), reward_dict, terminations, truncations, infos

    def task_value(self, k: int) -> float:
        """Current reward multiplier for task k (value decays after spawn)."""
        if self.cfg.task_value_decay_per_step <= 0.0:
            return 1.0
        age = max(0, self._t - int(self.task_spawn_time[k]))
        return float(np.clip(1.0 - self.cfg.task_value_decay_per_step * age, 0.0, 1.0))

    def _resolve_target(self, i: int, action: int) -> np.ndarray:
        if action == self.RETURN_BASE:
            return self.base_pos
        if action == self.HOLD:
            return self.pos[i]
        if action < self.n_tasks and self.task_active[action]:
            return self.task_pos[action]
        return self.pos[i]

    # --------------------------------------------------------- comm model
    def _update_comm_beliefs(self):
        dists = np.linalg.norm(self.pos[:, None, :] - self.pos[None, :, :], axis=-1)
        in_range = dists <= self.cfg.comm_range
        received = in_range & (self.rng.random((self.n, self.n)) > self.cfg.packet_loss_prob)
        np.fill_diagonal(received, True)

        for i in range(self.n):
            for j in range(self.n):
                if received[i, j]:
                    self.belief_pos[i, j] = self.pos[j]
                    self.belief_soc[i, j] = self.soc[j]
                    self.staleness[i, j] = 0
                else:
                    self.staleness[i, j] = min(self.staleness[i, j] + 1, self.cfg.max_staleness_steps)

    # --------------------------------------------------------- observation
    def _get_all_obs(self):
        return {a: self._get_obs(i) for i, a in enumerate(self.agents)}

    def _get_obs(self, i: int) -> np.ndarray:
        own = np.array([*self.pos[i], self.soc[i]], dtype=np.float32)

        dists = np.linalg.norm(self.belief_pos[i] - self.pos[i], axis=-1)
        order = np.argsort(self.staleness[i] * 1000.0 + dists)
        order = [j for j in order if j != i]

        neigh_feats = np.zeros((self.cfg.neighbor_cap, 6), dtype=np.float32)
        for slot, j in enumerate(order[: self.cfg.neighbor_cap]):
            valid = self.staleness[i, j] < self.cfg.max_staleness_steps
            rel = self.belief_pos[i, j] - self.pos[i]
            neigh_feats[slot] = [
                rel[0], rel[1], rel[2],
                self.belief_soc[i, j],
                self.staleness[i, j] / self.cfg.max_staleness_steps,
                float(valid),
            ]

        task_feats = np.concatenate(
            [self.task_pos, self.task_active.astype(np.float32)[:, None]], axis=-1
        ).astype(np.float32)

        return np.concatenate([own, neigh_feats.flatten(), task_feats.flatten()])

    def _get_infos(self):
        return {
            a: {"soc": float(self.soc[i]), "stranded": bool(self.stranded[i])}
            for i, a in enumerate(self.agents)
        }

    # -------------------------------------------------------------- utils
    def action_mask(self, i: int) -> np.ndarray:
        mask = np.zeros(self.action_space_size, dtype=np.float32)
        mask[: self.n_tasks] = self.task_active.astype(np.float32)
        mask[self.RETURN_BASE] = 1.0
        mask[self.HOLD] = 1.0
        return mask


if __name__ == "__main__":
    for scenario in ["layered", "volume"]:
        cfg = EnvConfig3D(n_agents=5, n_tasks=10, scenario=scenario, seed=0)
        env = UAVSwarmEnv3D(cfg)
        obs, infos = env.reset()
        print(f"[{scenario}] obs_dim={env.observation_space.shape}, action_dim={env.action_space.n}")
        print(f"[{scenario}] task altitudes: {sorted(env.task_pos[:, 2].round(1))}")

    # --- smoke-test the research extensions ---
    cfg = EnvConfig3D(n_agents=4, n_tasks=6, scenario="layered", seed=0,
                      task_mode="time_critical", task_spawn_window=0.3,
                      task_deadline_steps=80, task_value_decay_per_step=0.005,
                      task_requirements=(1, 2, 2, 1, 3, 1),
                      obstacles_enabled=True, n_obstacles=3)
    env = UAVSwarmEnv3D(cfg)
    env.reset()
    print(f"extensions: required={env.task_required.tolist()} "
          f"obstacles={env.obstacles.tolist()} spawns={env.task_spawn_time.tolist()} "
          f"deadlines={env.task_deadline.tolist()}")
