import numpy as np
from uav_swarm_env_3d import UAVSwarmEnv3D, EnvConfig3D


def random_policy(env: UAVSwarmEnv3D) -> dict:
    actions = {}
    for i, a in enumerate(env.agents):
        mask = env.action_mask(i)
        valid = np.where(mask > 0)[0]
        actions[a] = int(env.rng.choice(valid))
    return actions


def greedy_nearest_policy(env: UAVSwarmEnv3D) -> dict:
    actions = {}
    for i, a in enumerate(env.agents):
        if env.stranded[i]:
            actions[a] = env.HOLD
            continue
        if not env.task_active.any():
            actions[a] = env.HOLD
            continue
        dists = np.linalg.norm(env.task_pos - env.pos[i], axis=-1)
        dists[~env.task_active] = np.inf
        actions[a] = int(np.argmin(dists))
    return actions


def estimate_energy_cost(pos: np.ndarray, target: np.ndarray, cfg: EnvConfig3D) -> float:
    delta = target - pos
    dist = np.linalg.norm(delta)
    time_s = dist / cfg.max_speed
    climb = max(0.0, delta[2]) / dist * cfg.max_speed if dist > 1e-6 else 0.0
    power_w = cfg.hover_power_w + cfg.move_power_coeff * cfg.max_speed ** 2 + cfg.climb_power_coeff * climb
    energy_wh = power_w * (time_s / 3600.0)
    return energy_wh / cfg.battery_capacity_wh


def cbba_policy(env: UAVSwarmEnv3D, n_consensus_rounds: int = 3) -> dict:
    """CBBA adapted to 3D with two optional extensions (passive when the
    corresponding env features are off):

      * per-task capacity: a task requiring R[k] agents keeps its R[k]
        highest-bidding agents instead of a single winner, so coupled
        (multi-agent) tasks accumulate enough agents to complete;
      * deadline awareness: in task_mode="time_critical", bidding adds an
        urgency bonus for tasks close to their deadline.
    """
    cfg = env.cfg
    n, n_tasks = env.n, env.n_tasks
    required = np.asarray(env.task_required, dtype=int)

    my_bid_task = np.full(n, -1, dtype=int)
    my_bid_value = np.full(n, -np.inf, dtype=np.float32)
    # winning bids known locally by each agent, per task: (bidder_id, value)
    known_winner = np.full((n, n_tasks), -1, dtype=int)
    known_value = np.full((n, n_tasks), -np.inf, dtype=np.float32)
    # capacity bookkeeping: per (agent, task) dict of distinct known bidders
    # (only read when some required[k] > 1; no RNG impact, R=1 matches base)
    known_bids = [[{} for _ in range(n_tasks)] for _ in range(n)]

    for i in range(n):
        if env.stranded[i] or env.soc[i] < cfg.soc_min_safe:
            continue  # returning-to-base logic handled outside bidding
        best_task, best_val = -1, -np.inf
        for k in range(n_tasks):
            if not env.task_active[k]:
                continue
            cost = estimate_energy_cost(env.pos[i], env.task_pos[k], cfg)
            if env.soc[i] - cost < cfg.soc_min_safe:
                continue  # can't safely reach + still return -- don't bid
            value = cfg.task_reward - cost * 100.0  # weight energy cost in bid
            # deadline urgency bonus (0 when the env has no deadlines)
            if cfg.task_deadline_steps > 0:
                remaining = int(env.task_deadline[k]) - env._t
                urgency = np.clip(
                    1.0 - remaining / max(cfg.task_deadline_steps, 1), 0.0, 1.0)
                value += 2.0 * cfg.task_reward * urgency
            if value > best_val:
                best_task, best_val = k, value
        my_bid_task[i], my_bid_value[i] = best_task, best_val

    # consensus rounds: propagate bids over the (lossy) comm graph and
    # resolve conflicts by highest value wins (capacity-aware when R[k] > 1)
    for _ in range(n_consensus_rounds):
        dists = np.linalg.norm(env.pos[:, None, :] - env.pos[None, :, :], axis=-1)
        in_range = dists <= cfg.comm_range
        for i in range(n):
            if my_bid_task[i] >= 0:
                k = my_bid_task[i]
                if my_bid_value[i] > known_value[i, k]:
                    known_value[i, k] = my_bid_value[i]
                    known_winner[i, k] = i
            for j in range(n):
                if i == j or not in_range[i, j]:
                    continue
                if env.rng.random() < cfg.packet_loss_prob:
                    continue  # dropped
                if my_bid_task[j] >= 0:
                    k = my_bid_task[j]
                    if my_bid_value[j] > known_value[i, k]:
                        known_value[i, k] = my_bid_value[j]
                        known_winner[i, k] = j
                    if known_bids[i][k].get(j, -np.inf) < my_bid_value[j]:
                        known_bids[i][k][j] = my_bid_value[j]

        # conflict resolution: drop a task when the agent knows R[k] (or, for
        # the single-agent default, at least one) distinct bidders with a
        # strictly better value.  For R=1 this is the original rule exactly.
        for i in range(n):
            k = my_bid_task[i]
            if k < 0 or not known_bids[i][k]:
                continue
            better = sum(1 for v in known_bids[i][k].values()
                         if v > my_bid_value[i])
            if better >= int(required[k]):
                my_bid_task[i], my_bid_value[i] = -1, -np.inf

    actions = {}
    for i, a in enumerate(env.agents):
        if env.stranded[i]:
            actions[a] = env.HOLD
        elif env.soc[i] < cfg.soc_min_safe:
            actions[a] = env.RETURN_BASE
        elif my_bid_task[i] >= 0:
            actions[a] = int(my_bid_task[i])
        else:
            actions[a] = env.HOLD
    return actions


def run_episode(policy_fn, env_cfg: EnvConfig3D, seed: int) -> dict:
    cfg = EnvConfig3D(**{**env_cfg.__dict__, "seed": seed})
    env = UAVSwarmEnv3D(cfg)
    env.reset()
    total_reward = 0.0
    for t in range(cfg.max_steps):
        actions = policy_fn(env)
        _, rewards, term, trunc, _ = env.step(actions)
        total_reward += sum(rewards.values())
        if all(term.values()) or all(trunc.values()):
            break
    tasks_completed = int((~env.task_active).sum())
    return {
        "total_reward": total_reward,
        "coverage_frac": tasks_completed / cfg.n_tasks,
        "stranded_frac": float(env.stranded.mean()),
        "steps_taken": t + 1,
    }


if __name__ == "__main__":
    policies = {"random": random_policy, "greedy_nearest": greedy_nearest_policy, "cbba": cbba_policy}
    n_seeds = 10

    for scenario in ["layered", "volume"]:
        print(f"\n=== Scenario: {scenario} ===")
        cfg = EnvConfig3D(n_agents=6, n_tasks=12, scenario=scenario, max_steps=200)
        for name, fn in policies.items():
            runs = [run_episode(fn, cfg, seed=s) for s in range(n_seeds)]
            cov = np.mean([r["coverage_frac"] for r in runs]) * 100
            strand = np.mean([r["stranded_frac"] for r in runs]) * 100
            rew = np.mean([r["total_reward"] for r in runs])
            steps = np.mean([r["steps_taken"] for r in runs])
            print(f"{name:<16} coverage={cov:6.1f}%  stranded={strand:6.1f}%  reward={rew:9.1f}  steps={steps:6.1f}")
