"""
Distributed Model-Predictive Control (DMPC) baseline -- **3D port**.

Classical, model-based, zero-variance optimisation baseline that sets the
"efficiency ceiling" the learned policies must beat. At every step each UAV
solves a short-horizon optimal control problem using the *known* 3D dynamics
model (the same move-and-drain model the environment runs, including the
altitude/climb-power energy term):

  for each currently-active task k, simulate the UAV flying toward k for
  `horizon` steps under its own dynamics and accumulate the predicted cost:

      cost(k) = w_e * (energy to reach k + energy to return to base)
              + w_t * (steps to reach k) / max_steps
              + safety_penalty if predicted SOC ever drops below soc_min_safe
              + contention_penalty if another in-range UAV prefers the same k

  choose argmin cost(k); if no task is reachable with a safe return, go home.

Agents share their *preferred task* with in-range neighbours (subject to the
same comm model as every other baseline: range + packet loss + staleness), and
contention is resolved by the lower-cost bidder winning -- the CBBA-style
consensus step wrapped around a true model-predictive cost.

Extensions (all passive when the corresponding env features are off):

  * obstacle-aware predictions: the MPC simulation uses the env's exact
    slide-along-obstacle movement model, so a task behind a no-fly zone is
    either reached later (higher energy/time) or predicted unreachable, and
    the controller plans around it or goes home instead of flying into a wall.

  * deadline-aware cost: in task_mode="time_critical", tasks whose predicted
    arrival time exceeds their deadline get an extra cost term, so DMPC
    prioritizes tasks that are about to expire.

  * coupled-task capacity: a task requiring R[k] agents keeps the R[k]
    lowest-cost bidders instead of a single winner, so multi-agent tasks
    accumulate enough agents to complete.

This baseline is fully deterministic for a fixed seed: it only uses the env's
public physics, so it has zero sampling variance.
"""

import numpy as np
from uav_swarm_env_3d import move_towards


def _sim_step(pos, target, cfg, obstacles=None):
    """One step of the env's movement + energy model (identical to env.step),
    including the climb-power term and obstacle slide-around. Returns
    (new_pos, d_energy)."""
    step_dist = min(np.linalg.norm(target - pos), cfg.max_speed * cfg.dt)
    new_pos = move_towards(pos, target, step_dist, obstacles)
    moved = float(np.linalg.norm(new_pos - pos))
    speed = moved / cfg.dt
    delta = target - pos
    dist = np.linalg.norm(delta)
    if dist > 1e-6:
        direction = delta / dist
        climb_rate = max(0.0, direction[2] * speed)
    else:
        climb_rate = 0.0
    power_w = (cfg.hover_power_w + cfg.move_power_coeff * speed ** 2
               + cfg.climb_power_coeff * climb_rate)
    energy_wh = power_w * (cfg.dt / 3600.0)
    d_energy = energy_wh / cfg.battery_capacity_wh
    return new_pos, d_energy


def _deadline_penalty(env, task_k, steps_to_reach, w=2.0):
    """Extra MPC cost for time-critical tasks whose predicted arrival exceeds
    their deadline. Returns 0 when the env has no deadlines."""
    if env.cfg.task_deadline_steps <= 0 or not env.task_active[task_k]:
        return 0.0
    remaining = int(env.task_deadline[task_k]) - env._t
    if steps_to_reach <= remaining:
        return 0.0
    return w * (steps_to_reach - remaining) / max(env.cfg.max_steps, 1)


def _leg_analytic(p0, p1, cfg):
    """Closed-form energy and step count for a straight-line leg at the env's
    speed cap (exact when there are no obstacles). The last partial step uses
    the remaining distance, matching env.step's min(dist, max_speed*dt)."""
    delta = np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return 0.0, 0
    step_cap = cfg.max_speed * cfg.dt
    n_full = int(dist // step_cap)
    rem = dist - n_full * step_cap

    def energy_of(speed, n_steps):
        if n_steps <= 0:
            return 0.0
        climb = max(0.0, float(delta[2]) / dist * speed)
        power = (cfg.hover_power_w + cfg.move_power_coeff * speed ** 2
                 + cfg.climb_power_coeff * climb)
        return (power * n_steps * cfg.dt / 3600.0) / cfg.battery_capacity_wh

    energy = energy_of(cfg.max_speed, n_full)
    steps = n_full
    if rem > 1e-6:
        energy += energy_of(rem / cfg.dt, 1)
        steps += 1
    return energy, steps


def _predict_task(env, i, task_k, horizon=None):
    """Simulate agent i flying to active task k (full trip) and return
    (energy, steps_to_reach, safe). `horizon` is an upper bound on the
    simulation length (default: the episode horizon). With no obstacles the
    straight-line closed form is used (identical physics, much faster)."""
    cfg = env.cfg
    obstacles = getattr(env, "obstacles", None)
    if obstacles is None or len(obstacles) == 0:
        e1, s1 = _leg_analytic(env.pos[i], env.task_pos[task_k], cfg)
        e2, _ = _leg_analytic(env.task_pos[task_k], env.base_pos, cfg)
        energy = e1 + e2
        final_soc = float(env.soc[i]) - energy
        safe = final_soc >= cfg.soc_min_safe and energy < float(env.soc[i])
        return energy, s1, safe

    cap = cfg.max_steps if horizon is None else horizon
    # 400 m / 8 m/s = 50 steps to cross the area; 120 covers long NFZ detours.
    cap = min(int(cap), 120)
    pos = env.pos[i].copy()
    soc = float(env.soc[i])
    target = env.task_pos[task_k]
    energy = 0.0
    steps = 0
    reached = False
    for _ in range(cap):
        pos, de = _sim_step(pos, target, cfg, obstacles)
        soc -= de
        energy += de
        steps += 1
        if np.linalg.norm(pos - target) < 1.5:
            reached = True
            break
    # energy to return to base from the reached task position afterwards
    if reached:
        rpos = pos.copy()
        r_energy = 0.0
        for _ in range(cap):
            rpos, de = _sim_step(rpos, env.base_pos, cfg, obstacles)
            r_energy += de
            if np.linalg.norm(rpos - env.base_pos) < 1.0:
                break
        energy += r_energy
        final_soc = soc - r_energy
    else:
        final_soc = soc
    safe = final_soc >= cfg.soc_min_safe and energy < float(env.soc[i])
    return energy, steps if reached else cap, safe


def reachable_safe(env, i, task_k, horizon=None):
    """True if agent i can reach task k and still return to base above the safe
    floor (used by HRL-H's battery-safety low-level layer and DMPC)."""
    _, _, safe = _predict_task(env, i, task_k, horizon)
    return safe


def _task_cost(env, i, k, horizon, w_energy, w_time, w_safety):
    """Full MPC cost for agent i flying to task k (used by the consensus)."""
    energy, steps, safe = _predict_task(env, i, k, horizon)
    cost = (w_energy * energy + w_time * steps / max(env.cfg.max_steps, 1)
            + (0.0 if safe else w_safety)
            + _deadline_penalty(env, k, steps))
    return cost, energy, steps, safe


def dmpc_policy(env, horizon=None, w_energy=1.0, w_time=0.5, w_safety=5.0,
                n_consensus_rounds=3):
    """Distributed MPC action dict for the current env state.

    Pure function of the public env state + physics; deterministic given seed.
    Tasks requiring R[k] agents keep their R[k] lowest-cost bidders, so
    coupled (multi-agent) tasks receive enough agents to complete."""
    cfg = env.cfg
    n, n_tasks = env.n, env.n_tasks

    # ---- per-agent preferred task via short-horizon predictive cost ----
    pref_task = np.full(n, -1, dtype=int)
    pref_cost = np.full(n, np.inf, dtype=np.float64)
    for i in range(n):
        if env.stranded[i] or env.soc[i] < cfg.soc_min_safe:
            continue
        best, best_cost = -1, np.inf
        for k in range(n_tasks):
            if not env.task_active[k]:
                continue
            cost, _, _, _ = _task_cost(env, i, k, horizon, w_energy, w_time, w_safety)
            if cost < best_cost:
                best, best_cost = k, cost
        pref_task[i], pref_cost[i] = best, best_cost

    # ---- CBBA-style consensus with per-task capacity R[k] ----
    # winners[k] = list of (cost, agent) that won task k, capped at R[k].
    winners = [[] for _ in range(n_tasks)]
    for _ in range(n_consensus_rounds):
        dists = np.linalg.norm(env.pos[:, None, :] - env.pos[None, :, :], axis=-1)
        in_range = dists <= cfg.comm_range
        for i in range(n):
            k = pref_task[i]
            if k >= 0:
                winners[k].append((pref_cost[i], i))
            for j in range(n):
                if i == j or not in_range[i, j]:
                    continue
                if env.rng.random() < cfg.packet_loss_prob:
                    continue
                kj = pref_task[j]
                if kj >= 0:
                    winners[kj].append((pref_cost[j], j))
        # keep the R[k] lowest-cost bidders per task
        for k in range(n_tasks):
            if winners[k]:
                winners[k] = sorted(set(winners[k]))[: int(env.task_required[k])]

        # losers re-pick their next best task among those with free capacity
        for i in range(n):
            if pref_task[i] >= 0 and \
                    not any(a == i for _, a in winners[pref_task[i]]):
                pref_task[i], pref_cost[i] = -1, np.inf
                for k in range(n_tasks):
                    if not env.task_active[k]:
                        continue
                    if len(winners[k]) >= int(env.task_required[k]):
                        continue
                    cost, _, _, _ = _task_cost(env, i, k, horizon, w_energy, w_time, w_safety)
                    if cost < pref_cost[i]:
                        pref_task[i], pref_cost[i] = k, cost
        # merge newly preferred tasks into the winner lists for next round
        for i in range(n):
            k = pref_task[i]
            if k >= 0:
                winners[k].append((pref_cost[i], i))
        for k in range(n_tasks):
            if winners[k]:
                winners[k] = sorted(set(winners[k]))[: int(env.task_required[k])]

    actions = {}
    for i, a in enumerate(env.agents):
        if env.stranded[i]:
            actions[a] = env.HOLD
        elif env.soc[i] < cfg.soc_min_safe:
            actions[a] = env.RETURN_BASE
        elif pref_task[i] >= 0:
            actions[a] = int(pref_task[i])
        else:
            actions[a] = env.HOLD
    return actions
