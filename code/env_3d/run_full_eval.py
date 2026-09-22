"""Train (if needed) and run the full 4-environment evaluation.

Saves JSON only (no CSV). Writes publication figures for snapshots,
base-config trajectories (all 4 environments), comparison / extended
metrics / HRL-H ablations per environment, and the 4-env scenario plot.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

NB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in [os.path.join(NB_DIR, "code"),
           os.path.join(NB_DIR, "code", "env_3d"),
           os.path.join(NB_DIR, "code", "mappo_3d")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from uav_swarm_env_3d import UAVSwarmEnv3D, EnvConfig3D
from baselines_3d import random_policy, greedy_nearest_policy, cbba_policy
from model import Actor, Critic, softmax
import train_3d as mt
from marl_algorithms.mlp import normalize_padded_obs
from marl_algorithms.maddpg import MADDPG, ReplayBuffer, update_from_buffer, \
    bc_warmstart_actor
from marl_algorithms.qmix import QMIX, build_state, qmix_update, qmix_bc_warmstart
from model_based.dmpc import dmpc_policy
from hrl_h.hrl_h import HRLH
from visualize_3d import (
    plot_env_3d, plot_environments_2x2, collect_env_snapshot, save_env_snapshot,
)
from eval_io import (
    apply_pub_style, panel_shape, figsize_for, write_json, dump_sweep_json,
    dump_base_json, dump_tables_json, hide_unused_axes,
    FONT_TITLE, FONT_LABEL, FONT_TICK, FONT_LEGEND, FONT_SUPTITLE, FONT_ANNOT,
)


# ---------------------------------------------------------------------------
# Config (stress preset, matches the notebook)
# ---------------------------------------------------------------------------
USE_STRESS_CFG = True
SCENARIO = "layered"
N_SEEDS = 10
WARM_START = True
BC_EPISODES = 400
BC_ITERS = 4000
ACTOR_HIDDEN = 64
N_TRAIN_EPISODES = 600
PPO_EPISODES_PER_UPDATE = 40
TRAIN_TASK_COUNTS = [8, 16, 32]
TARGET_PACKET_LOSS = 0.3
CURRICULUM_EPISODES = 300
SEED = 0
NFZ_N_OBSTACLES = 4
NFZ_OBSTACLE_HALF = 20.0
SHAPING_BETA = 0.2
MARL_DEMO_EPISODES = 120
MARL_TASK_COUNTS = [4, 8, 16]
MARL_EPS_DEMO = 0.1
MARL_BC_ITERS = 4000
MARL_RL_ITERS = 2000
MADDPG_LR = 3e-4
QMIX_LR = 1e-4
QMIX_BC_WEIGHT = 5.0
QMIX_Q_LIM = 50.0
QMIX_TARGET_EVERY = 500
GAMMA_BC = 0.99
FIG_DPI = 600

if USE_STRESS_CFG:
    _base_kw = dict(
        n_agents=8, n_tasks=8, area_size=400.0, altitude_min=5.0, altitude_max=60.0,
        n_layers=3, comm_range=12.0, battery_capacity_wh=18.0, soc_min_safe=0.25,
        max_steps=500,
    )
    CFG_NAME = "stress"
else:
    _base_kw = dict(n_agents=6, n_tasks=12, max_steps=200)
    CFG_NAME = "default"

ENV_CFG = EnvConfig3D(**_base_kw, scenario=SCENARIO)
SHAPING_DMAX = float((ENV_CFG.area_size ** 2 * 3) ** 0.5)
N_TASKS_MAX = max(32, ENV_CFG.n_tasks)
STRIDE = mt.TASK_FEAT_STRIDE

ENV_SPECS = [
    ("open", "Open volume", "volume", False, 0),
    ("layered", "Layered altitude bands", "layered", False, 0),
    ("open_nfz", "Open volume + no-fly zones", "volume", True, NFZ_N_OBSTACLES),
    ("layered_nfz", "Layered + no-fly zones", "layered", True, NFZ_N_OBSTACLES),
]
ENV_ORDER = [s[0] for s in ENV_SPECS]
ENV_LABELS = {s[0]: s[1] for s in ENV_SPECS}
ENV_CFGS = {
    eid: EnvConfig3D(**_base_kw, scenario=sc,
                     obstacles_enabled=obs, n_obstacles=nobs,
                     obstacle_half=NFZ_OBSTACLE_HALF)
    for eid, _, sc, obs, nobs in ENV_SPECS
}

OUT_DIR = os.path.join(NB_DIR, "outputs")
FIG_DIR = os.path.join(NB_DIR, "figures")
SNAP_DIR = os.path.join(OUT_DIR, "env_snapshots")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)

POLICY_ORDER = ["random", "greedy", "cbba", "mappo", "maddpg", "qmix",
                "dmpc", "hrlh"]
POLICY_LABELS = {"random": "Random", "greedy": "Greedy-Nearest",
                 "cbba": "CBBA (SOC-aware)", "mappo": "MAPPO",
                 "maddpg": "MADDPG", "qmix": "QMIX", "dmpc": "DMPC",
                 "hrlh": "HRL-H (proposed)"}
POLICY_COLORS = {"random": "#d62728", "greedy": "#1f77b4",
                 "cbba": "#2ca02c", "mappo": "#9467bd",
                 "maddpg": "#ff7f0e", "qmix": "#17becf",
                 "dmpc": "#7f7f7f", "hrlh": "#e377c2"}

apply_pub_style()


def save_fig(fig, name):
    path = os.path.join(FIG_DIR, f"{CFG_NAME}_{name}.png")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print("saved", path, flush=True)


def clear_caches(o, _seen=None):
    _seen = _seen if _seen is not None else set()
    if o is None or isinstance(o, (str, bytes, int, float, bool)) or id(o) in _seen:
        return
    _seen.add(id(o))
    if hasattr(o, "__dict__"):
        for name, val in list(vars(o).items()):
            if name == "_cache":
                vars(o)["_cache"] = None
            else:
                clear_caches(val, _seen)
    elif isinstance(o, dict):
        for val in o.values():
            clear_caches(val, _seen)
    elif isinstance(o, (list, tuple)):
        for val in o:
            clear_caches(val, _seen)


# ---------------------------------------------------------------------------
# Episode metrics
# ---------------------------------------------------------------------------
def run_episode(policy_fn, env_cfg, seed, fail_frac=0.0):
    cfg = EnvConfig3D(**{**env_cfg.__dict__, "seed": seed})
    env = UAVSwarmEnv3D(cfg)
    env.reset()

    inject_step = None
    fail_idx = None
    if fail_frac > 0:
        probe_cfg = EnvConfig3D(**{**env_cfg.__dict__, "seed": seed + 100000})
        probe_env = UAVSwarmEnv3D(probe_cfg)
        probe_env.reset()
        for _t in range(probe_cfg.max_steps):
            probe_actions = policy_fn(probe_env)
            _, _, _pt, _ptr, _ = probe_env.step(probe_actions)
            if all(_pt.values()) or all(_ptr.values()):
                break
        mission_len = _t + 1
        lo, hi = max(1, int(mission_len * 0.2)), max(2, int(mission_len * 0.7))
        inject_step = int(env.rng.integers(lo, hi))
        n_fail = max(1, int(round(cfg.n_agents * fail_frac)))
        fail_idx = env.rng.choice(cfg.n_agents, size=n_fail, replace=False)

    total_reward = 0.0
    steps = 0
    soc_log, pos_log, staleness_log = [], [], []
    contended_steps = 0
    completion_steps = []
    task_prev = env.task_completed.copy()
    for t in range(cfg.max_steps):
        if inject_step is not None and t == inject_step:
            env.stranded[fail_idx] = True
        actions = policy_fn(env)
        act = np.array([actions[a] for a in env.agents], dtype=int)
        engaged = act[(~env.stranded) & (act < env.n_tasks)]
        if len(engaged) > 1:
            engaged = engaged[env.task_active[engaged]]
            if len(np.unique(engaged)) < len(engaged):
                contended_steps += 1
        _, rewards, term, trunc, _ = env.step(actions)
        total_reward += sum(rewards.values())
        steps = t + 1
        soc_log.append(env.soc.copy())
        pos_log.append(env.pos.copy())
        staleness_log.append(float(env.staleness[~np.eye(env.n, dtype=bool)].mean()))
        newly_done = env.task_completed & ~task_prev
        if newly_done.any():
            completion_steps.extend([steps] * int(newly_done.sum()))
        task_prev = env.task_completed.copy()
        if all(term.values()) or all(trunc.values()):
            break

    completed = int(env.task_completed.sum())
    expired = int(env.task_expired.sum())
    alive = ~env.stranded
    soc_arr = np.array(soc_log) if soc_log else np.zeros((1, env.n))
    pos_arr = np.array(pos_log) if pos_log else np.zeros((1, env.n, 3))
    dist_total = float(np.linalg.norm(np.diff(pos_arr, axis=0), axis=-1).sum())
    energy_wh = float((cfg.battery_capacity_wh * (1.0 - env.soc)).sum())
    return {
        "total_reward": float(total_reward),
        "tasks_completed": completed,
        "tasks_total": cfg.n_tasks,
        "tasks_expired": expired,
        "coverage_frac": completed / cfg.n_tasks,
        "expired_frac": expired / cfg.n_tasks,
        "n_stranded": int(env.stranded.sum()),
        "stranded_frac": float(env.stranded.mean()),
        "mean_final_soc": float(env.soc[alive].mean()) if alive.any() else 0.0,
        "soc_std": float(env.soc.std()),
        "steps_taken": steps,
        "energy_wh_total": energy_wh,
        "energy_wh_per_task": energy_wh / max(completed, 1),
        "dist_total_m": dist_total,
        "dist_per_task_m": dist_total / max(completed, 1),
        "min_soc": float(soc_arr.min()),
        "below_floor_pct": float((soc_arr < cfg.soc_min_safe).mean() * 100),
        "contention_pct": float(contended_steps / steps * 100) if steps else 0.0,
        "mean_info_age": float(np.mean(staleness_log)) if staleness_log else 0.0,
        "latency_mean": float(np.mean(completion_steps)) if completion_steps else float("nan"),
        "latency_median": float(np.median(completion_steps)) if completion_steps else float("nan"),
        "completion_steps": list(completion_steps),
        "blocked_moves": int(getattr(env, "n_blocked_moves", 0)),
    }


def summarize_runs(runs, level_value):
    c = np.array([r["coverage_frac"] for r in runs]) * 100
    s = np.array([r["stranded_frac"] for r in runs]) * 100
    r = np.array([r["total_reward"] for r in runs])
    ex = np.array([r.get("expired_frac", 0.0) for r in runs]) * 100
    m = np.array([r["mean_final_soc"] for r in runs])
    st = np.array([r["steps_taken"] for r in runs])
    e = np.array([r["energy_wh_per_task"] for r in runs])
    d = np.array([r["dist_per_task_m"] for r in runs])
    mn = np.array([r["min_soc"] for r in runs])
    bf = np.array([r["below_floor_pct"] for r in runs])
    ct = np.array([r["contention_pct"] for r in runs])
    ia = np.array([r["mean_info_age"] for r in runs])
    lt = np.array([r["latency_mean"] for r in runs], dtype=float)
    lm = np.array([r["latency_median"] for r in runs], dtype=float)
    bm = np.array([r.get("blocked_moves", 0) for r in runs], dtype=float)

    def _ms(x):
        x = x[np.isfinite(x)]
        return (float(np.mean(x)), float(np.std(x))) if x.size else (0.0, 0.0)

    e_m, e_s = _ms(e); d_m, d_s = _ms(d); mn_m, mn_s = _ms(mn)
    bf_m, bf_s = _ms(bf); ct_m, ct_s = _ms(ct); ia_m, ia_s = _ms(ia)
    lt_m, lt_s = _ms(lt); lm_m, lm_s = _ms(lm); bm_m, bm_s = _ms(bm)
    return {
        "value": level_value,
        "coverage_mean": float(c.mean()), "coverage_std": float(c.std()),
        "stranded_mean": float(s.mean()), "stranded_std": float(s.std()),
        "reward_mean": float(r.mean()), "reward_std": float(r.std()),
        "expired_mean": float(ex.mean()), "expired_std": float(ex.std()),
        "soc_mean": float(m.mean()), "soc_std": float(m.std()),
        "steps_mean": float(st.mean()), "steps_std": float(st.std()),
        "energy_wh_per_task_mean": e_m, "energy_wh_per_task_std": e_s,
        "dist_per_task_m_mean": d_m, "dist_per_task_m_std": d_s,
        "min_soc_mean": mn_m, "min_soc_std": mn_s,
        "below_floor_pct_mean": bf_m, "below_floor_pct_std": bf_s,
        "contention_pct_mean": ct_m, "contention_pct_std": ct_s,
        "info_age_mean": ia_m, "info_age_std": ia_s,
        "latency_mean_mean": lt_m, "latency_mean_std": lt_s,
        "latency_median_mean": lm_m, "latency_median_std": lm_s,
        "blocked_moves_mean": bm_m, "blocked_moves_std": bm_s,
    }


def run_sweep(policy_fns, env_cfg, levels, level_field, n_seeds):
    results = {}
    for name, fn in policy_fns.items():
        results[name] = []
        for level in levels:
            cfg = EnvConfig3D(**{**env_cfg.__dict__, level_field: level})
            runs = [run_episode(fn, cfg, s) for s in range(n_seeds)]
            results[name].append(summarize_runs(runs, level))
            print(f"    {name} {level_field}={level} "
                  f"cov={results[name][-1]['coverage_mean']:.1f}%", flush=True)
    return results


def run_sweep_all_envs(policy_fns, env_cfgs, levels, level_field, n_seeds):
    out = {}
    for eid, cfg in env_cfgs.items():
        print(f"  sweep {level_field} @ {eid} ...", flush=True)
        out[eid] = run_sweep(policy_fns, cfg, levels, level_field, n_seeds)
    return out


def run_fault_sweep(policy_fns, env_cfg, levels, n_seeds):
    results = {}
    for name, fn in policy_fns.items():
        results[name] = []
        for frac in levels:
            runs = [run_episode(fn, env_cfg, s, fail_frac=frac) for s in range(n_seeds)]
            results[name].append(summarize_runs(runs, frac))
            print(f"    {name} fail={frac} "
                  f"cov={results[name][-1]['coverage_mean']:.1f}%", flush=True)
    return results


def run_fault_sweep_all_envs(policy_fns, env_cfgs, levels, n_seeds):
    out = {}
    for eid, cfg in env_cfgs.items():
        print(f"  fault sweep @ {eid} ...", flush=True)
        out[eid] = run_fault_sweep(policy_fns, cfg, levels, n_seeds)
    return out


def run_base_all_envs(policy_fns, env_cfgs, n_seeds):
    results, runs_by_env = {}, {}
    for eid, cfg in env_cfgs.items():
        print(f"  base config @ {eid} ...", flush=True)
        runs_by_env[eid] = {}
        results[eid] = {}
        for name, fn in policy_fns.items():
            rs = [run_episode(fn, cfg, s) for s in range(n_seeds)]
            runs_by_env[eid][name] = rs
            results[eid][name] = summarize_runs(rs, 0)
            rec = results[eid][name]
            print(f"    {name:<10} cov={rec['coverage_mean']:5.1f}%  "
                  f"strand={rec['stranded_mean']:5.1f}%  "
                  f"E={rec['energy_wh_per_task_mean']:.2f} Wh/task  "
                  f"steps={rec['steps_mean']:.1f}", flush=True)
    return results, runs_by_env


# ---------------------------------------------------------------------------
# MAPPO padding / training helpers
# ---------------------------------------------------------------------------
def build_padded_obs_mask(env):
    obs_dict = env._get_all_obs()
    obs_batch = np.stack([obs_dict[a] for a in env.agents])
    mask_batch = np.stack([env.action_mask(i) for i in range(env.n)])
    nt = env.n_tasks
    own_dim = mt.OWN_DIM
    ncap, ndim = env.cfg.neighbor_cap, mt.NEIGHBOR_DIM
    own = obs_batch[:, :own_dim]
    neigh = obs_batch[:, own_dim:own_dim + ncap * ndim]
    tasks = obs_batch[:, own_dim + ncap * ndim:]
    pad = (N_TASKS_MAX - nt) * STRIDE
    tasks_p = np.concatenate([tasks, np.zeros((obs_batch.shape[0], pad))], axis=1)
    obs_p = np.concatenate([own, neigh, tasks_p], axis=1)
    extra = N_TASKS_MAX - nt
    mask_p = np.concatenate(
        [mask_batch[:, :nt], np.zeros((obs_batch.shape[0], extra)),
         mask_batch[:, nt:]], axis=1)
    return obs_p, mask_p


def nearest_task_dist(env):
    tpos = env.task_pos[env.task_active]
    if len(tpos) == 0:
        return np.full(env.n, SHAPING_DMAX)
    d = np.linalg.norm(env.pos[:, None, :] - tpos[None, :, :], axis=-1)
    return np.min(d, axis=1).clip(None, SHAPING_DMAX)


def collect_rollout_vartask(env, actor, critic, n_episodes, rng, task_counts):
    out = {k: [] for k in ["own", "neigh_mask", "task", "actions",
                           "logprobs", "values", "rewards", "dones", "global"]}
    area = env.cfg.area_size
    alt = env.cfg.altitude_max
    for _ in range(n_episodes):
        env.cfg.n_tasks = int(rng.choice(task_counts))
        env.n_tasks = env.cfg.n_tasks
        env.action_space_size = env.cfg.n_tasks + 2
        env.RETURN_BASE = env.cfg.n_tasks
        env.HOLD = env.cfg.n_tasks + 1
        env.reset()
        done = False
        while not done:
            obs_p, mask_p = build_padded_obs_mask(env)
            own, neigh, valid_mask, tasks = mt.split_obs(
                obs_p, mt.OWN_DIM, env.cfg.neighbor_cap, mt.NEIGHBOR_DIM,
                area, alt)
            logits = actor.forward(own, neigh, valid_mask, tasks, mask_p)
            probs = softmax(logits)
            actions = mt.categorical_sample(probs, rng)
            logp = np.log(np.clip(probs[np.arange(env.n), actions], 1e-12, 1.0))
            gs = np.tile(mt.build_global_state(env), (env.n, 1))
            values = critic.forward(gs)
            action_dict = {a: int(actions[i]) for i, a in enumerate(env.agents)}
            _, reward_dict, term, trunc, _ = env.step(action_dict)
            rewards = np.array([reward_dict[a] for a in env.agents], dtype=np.float64)
            rewards = rewards + SHAPING_BETA * (1.0 - nearest_task_dist(env) / SHAPING_DMAX)
            done = all(term.values()) or all(trunc.values())
            out["own"].append(own); out["neigh_mask"].append((neigh, valid_mask))
            out["task"].append(tasks); out["actions"].append(actions)
            out["logprobs"].append(logp); out["values"].append(values)
            out["rewards"].append(rewards)
            out["dones"].append(np.full(env.n, float(done))); out["global"].append(gs)
    return out


def collect_greedy_demos(env, n_episodes, rng, task_counts):
    obs_all, act_all, mask_all, gs_all, ret_all = [], [], [], [], []
    for _ in range(n_episodes):
        env.cfg.n_tasks = int(rng.choice(task_counts))
        env.n_tasks = env.cfg.n_tasks
        env.action_space_size = env.cfg.n_tasks + 2
        env.RETURN_BASE = env.cfg.n_tasks
        env.HOLD = env.cfg.n_tasks + 1
        env.reset()
        done = False
        steps = []
        while not done:
            obs_p, mask_p = build_padded_obs_mask(env)
            gs = np.tile(mt.build_global_state(env), (env.n, 1))
            act_dict = greedy_nearest_policy(env)
            acts = np.array([act_dict[a] for a in env.agents], dtype=int)
            _, reward_dict, term, trunc, _ = env.step(act_dict)
            rewards = np.array([reward_dict[a] for a in env.agents], dtype=np.float64)
            steps.append((obs_p, acts, mask_p, gs, rewards))
            done = all(term.values()) or all(trunc.values())
        G = np.zeros(env.n)
        for obs_p, acts, mask_p, gs, rewards in reversed(steps):
            G = rewards + GAMMA_BC * G
            obs_all.append(obs_p); act_all.append(acts); mask_all.append(mask_p)
            gs_all.append(gs); ret_all.append(np.tile(G, (env.n, 1)))
    return (np.concatenate(obs_all), np.concatenate(act_all), np.concatenate(mask_all),
            np.concatenate(gs_all), np.concatenate(ret_all))


def bc_pretrain(actor, critic, actor_opt, critic_opt, data, iters, rng, batch=128):
    obs_all, act_all, mask_all, gs_all, ret_all = data
    n = obs_all.shape[0]
    losses = []
    for it in range(iters):
        idx = rng.integers(0, n, size=batch)
        own, neigh, valid_mask, tasks = mt.split_obs(
            obs_all[idx], mt.OWN_DIM, ENV_CFG.neighbor_cap, mt.NEIGHBOR_DIM,
            ENV_CFG.area_size, ENV_CFG.altitude_max)
        logits = actor.forward(own, neigh, valid_mask, tasks, mask_all[idx])
        probs = softmax(logits)
        losses.append(-float(np.mean(np.log(np.clip(
            probs[np.arange(batch), act_all[idx]], 1e-12, 1.0)))))
        onehot = np.zeros_like(logits)
        onehot[np.arange(batch), act_all[idx]] = 1.0
        actor_grads = actor.backward((probs - onehot) / batch)
        actor_param_grads = []
        for _name, _layer in actor.all_layers().items():
            _g = actor_grads[_name]
            actor_param_grads.append(_g["W"]); actor_param_grads.append(_g["b"])
        actor_opt.step(actor_param_grads)
        cidx = rng.integers(0, n, size=batch)
        dvalue = 2.0 * (critic.forward(gs_all[cidx]) - ret_all[cidx][:, 0]) / batch
        critic_grads = critic.backward(dvalue)
        critic_param_grads = []
        for _name, _layer in critic.all_layers().items():
            _g = critic_grads[1][_name]
            critic_param_grads.append(_g["W"]); critic_param_grads.append(_g["b"])
        critic_opt.step(critic_param_grads)
        if (it + 1) % 1000 == 0:
            print(f"  BC iter {it+1} loss={losses[-1]:.3f}", flush=True)
    return losses


def collect_mar_demos(n_episodes, task_counts, rng, eps=0.0):
    buf = ReplayBuffer(120000)
    _env = UAVSwarmEnv3D(EnvConfig3D(**{**ENV_CFG.__dict__, "seed": SEED}))
    for _ in range(n_episodes):
        _env.cfg.n_tasks = int(rng.choice(task_counts))
        _env.n_tasks = _env.cfg.n_tasks
        _env.action_space_size = _env.cfg.n_tasks + 2
        _env.RETURN_BASE = _env.cfg.n_tasks
        _env.HOLD = _env.cfg.n_tasks + 1
        _env.reset()
        done = False
        while not done:
            obs_p, mask_p = build_padded_obs_mask(_env)
            obs_p = normalize_padded_obs(obs_p, ENV_CFG.area_size,
                                         ENV_CFG.altitude_max,
                                         ENV_CFG.neighbor_cap, N_TASKS_MAX)
            state = build_state(_env)
            g = np.array([greedy_nearest_policy(_env)[a] for a in _env.agents],
                         dtype=int)
            act = g.copy()
            if eps > 0:
                for i in range(_env.n):
                    if rng.random() < eps:
                        valid = np.where(mask_p[i] > 0.5)[0]
                        act[i] = int(rng.choice(valid))
            _, reward_dict, term, trunc, _ = _env.step(
                {a: int(act[i]) for i, a in enumerate(_env.agents)})
            next_obs_p, next_mask_p = build_padded_obs_mask(_env)
            next_obs_p = normalize_padded_obs(next_obs_p, ENV_CFG.area_size,
                                              ENV_CFG.altitude_max,
                                              ENV_CFG.neighbor_cap, N_TASKS_MAX)
            next_state = build_state(_env)
            rew = np.array([reward_dict[a] for a in _env.agents], dtype=np.float64)
            done = all(term.values()) or all(trunc.values())
            buf.add(obs_p, act, rew, next_obs_p, mask_p, next_mask_p, done,
                    state, next_state)
    return buf


def mar_demo_stack(buf):
    obs, act, mask = [], [], []
    for b in buf.buf:
        obs.append(b[0]); act.append(b[1]); mask.append(b[4])
    return np.concatenate(obs), np.concatenate(act), np.concatenate(mask)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_scenario_comparison(results_by_env, env_ids, fname):
    n_env = len(env_ids)
    fig, axes = plt.subplots(n_env, 3, figsize=(24, 5.4 * n_env))
    metrics = [("coverage_mean", "Coverage (%)", "coverage_std"),
               ("stranded_mean", "Stranded UAVs (%)", "stranded_std"),
               ("reward_mean", "Total reward", "reward_std")]
    x = np.arange(len(POLICY_ORDER)); width = 0.62
    for r, eid in enumerate(env_ids):
        for c, (mk, label, sk) in enumerate(metrics):
            ax = axes[r, c]
            vals = [results_by_env[eid][name][mk] for name in POLICY_ORDER]
            errs = [results_by_env[eid][name][sk] for name in POLICY_ORDER]
            bars = ax.bar(x, vals, width, yerr=errs, capsize=5,
                          color=[POLICY_COLORS[n] for n in POLICY_ORDER],
                          edgecolor="k", linewidth=0.5)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=FONT_ANNOT)
            ax.set_xticks(x)
            ax.set_xticklabels([POLICY_LABELS[n] for n in POLICY_ORDER],
                               fontsize=FONT_TICK - 1, rotation=22, ha="right")
            ax.set_ylabel(label, fontsize=FONT_LABEL)
            ax.set_title(f"{ENV_LABELS.get(eid, eid)} -- {label}",
                         fontsize=FONT_TITLE)
            ax.tick_params(labelsize=FONT_TICK)
    fig.suptitle(f"3D environment comparison -- all 8 approaches ({CFG_NAME})",
                 fontsize=FONT_SUPTITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, fname)


def plot_summary_bars(base_results, fname, title_tag=""):
    metrics = [("coverage_mean", "Coverage (%)", "coverage_std"),
               ("stranded_mean", "Stranded UAVs (%)", "stranded_std"),
               ("reward_mean", "Total reward", "reward_std"),
               ("soc_mean", "Mean final SOC", "soc_std"),
               ("steps_mean", "Episode length (steps)", "steps_std")]
    nrows, ncols = panel_shape(len(metrics))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize_for(nrows, ncols, cell=(7.5, 5.6)))
    x = np.arange(len(POLICY_ORDER)); width = 0.68
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (mk, label, sk) in zip(axes_flat, metrics):
        vals = [base_results[name][mk] for name in POLICY_ORDER]
        errs = [base_results[name][sk] for name in POLICY_ORDER]
        bars = ax.bar(x, vals, width, yerr=errs, capsize=5,
                      color=[POLICY_COLORS[n] for n in POLICY_ORDER],
                      edgecolor="k", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=FONT_ANNOT)
        ax.set_xticks(x)
        ax.set_xticklabels([POLICY_LABELS[n] for n in POLICY_ORDER],
                           fontsize=FONT_TICK - 1, rotation=22, ha="right")
        ax.set_ylabel(label, fontsize=FONT_LABEL)
        ax.set_title(label, fontsize=FONT_TITLE)
        ax.tick_params(labelsize=FONT_TICK)
    hide_unused_axes(axes_flat, len(metrics))
    fig.suptitle(f"3D base-config comparison -- all 8 approaches ({CFG_NAME})"
                 + (f" -- {title_tag}" if title_tag else ""),
                 fontsize=FONT_SUPTITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, fname)


def plot_extended_metrics(base_results, fname, title_tag=""):
    metrics = [
        ("energy_wh_per_task_mean", "Energy per task (Wh)", "energy_wh_per_task_std"),
        ("dist_per_task_m_mean", "Distance per task (m)", "dist_per_task_m_std"),
        ("min_soc_mean", "Min SOC reached", "min_soc_std"),
        ("below_floor_pct_mean", "Time below safe floor (%)", "below_floor_pct_std"),
        ("contention_pct_mean", "Assignment contention (% steps)", "contention_pct_std"),
        ("info_age_mean", "Mean info age (steps)", "info_age_std"),
    ]
    nrows, ncols = panel_shape(len(metrics))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize_for(nrows, ncols))
    x = np.arange(len(POLICY_ORDER)); width = 0.68
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (mk, label, sk) in zip(axes_flat, metrics):
        vals = [base_results[name][mk] for name in POLICY_ORDER]
        errs = [base_results[name][sk] for name in POLICY_ORDER]
        bars = ax.bar(x, vals, width, yerr=errs, capsize=5,
                      color=[POLICY_COLORS[n] for n in POLICY_ORDER],
                      edgecolor="k", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=FONT_ANNOT)
        ax.set_xticks(x)
        ax.set_xticklabels([POLICY_LABELS[n] for n in POLICY_ORDER],
                           fontsize=FONT_TICK - 1, rotation=22, ha="right")
        ax.set_ylabel(label, fontsize=FONT_LABEL)
        ax.set_title(label, fontsize=FONT_TITLE)
        ax.tick_params(labelsize=FONT_TICK)
    hide_unused_axes(axes_flat, len(metrics))
    fig.suptitle(f"3D extended operational metrics -- all 8 approaches ({CFG_NAME})"
                 + (f" -- {title_tag}" if title_tag else ""),
                 fontsize=FONT_SUPTITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, fname)


def record_episode(policy_fn, env_cfg, seed):
    cfg = EnvConfig3D(**{**env_cfg.__dict__, "seed": seed})
    env = UAVSwarmEnv3D(cfg)
    env.reset()
    traj = [env.pos.copy()]
    soc_hist = [env.soc.copy()]
    for t in range(cfg.max_steps):
        actions = policy_fn(env)
        _, _, term, trunc, _ = env.step(actions)
        traj.append(env.pos.copy())
        soc_hist.append(env.soc.copy())
        if all(term.values()) or all(trunc.values()):
            break
    return env, np.stack(traj), np.stack(soc_hist)


def plot_trajectories_3d(records, fname, title_tag):
    fig = plt.figure(figsize=(26, 40))
    for idx, name in enumerate(POLICY_ORDER, start=1):
        ax = fig.add_subplot(4, 2, idx, projection="3d")
        env, traj, soc_hist = records[name]
        ax.scatter(*env.base_pos, marker="s", s=140, c="black",
                   label="Base / depot", depthshade=False)
        active = env.task_active
        if active.any():
            ax.scatter(env.task_pos[active, 0], env.task_pos[active, 1],
                       env.task_pos[active, 2], marker="*", s=130, c="orange",
                       edgecolors="k", label="Active task", depthshade=False)
        if (~active).any():
            ax.scatter(env.task_pos[~active, 0], env.task_pos[~active, 1],
                       env.task_pos[~active, 2], marker="x", s=50, c="lightgray",
                       label="Completed task", depthshade=False)
        colors = plt.cm.RdYlGn(np.clip(soc_hist[-1], 0, 1))
        for i in range(env.n):
            ax.plot(traj[:, i, 0], traj[:, i, 1], traj[:, i, 2], "-",
                    color=colors[i], alpha=0.85, linewidth=1.4)
            ax.scatter(*traj[0, i], marker="o", s=35, c="gray", depthshade=False)
            marker_end = "X" if env.stranded[i] else "^"
            ax.scatter(*traj[-1, i], marker=marker_end, s=100, c=[colors[i]],
                       edgecolors="k", depthshade=False)
        obs = getattr(env, "obstacles", None)
        if obs is not None and len(obs):
            for o in obs:
                x0, y0, z0, x1, y1, z1 = o
                corners = [
                    [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
                    [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
                ]
                for x in (x0, x1):
                    corners.append([(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)])
                for y in (y0, y1):
                    corners.append([(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)])
                ax.add_collection3d(Poly3DCollection(
                    corners, alpha=0.15, facecolor="#b30000", edgecolor="#7a0000",
                    linewidths=0.5))
        L = env.cfg.area_size
        xx, yy = np.meshgrid([0, L], [0, L])
        ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.05, color="gray")
        ax.set_xlabel("x (m)", fontsize=FONT_LABEL)
        ax.set_ylabel("y (m)", fontsize=FONT_LABEL)
        ax.set_zlabel("altitude z (m)", fontsize=FONT_LABEL)
        ax.tick_params(labelsize=FONT_TICK)
        ax.set_zlim(0, env.cfg.altitude_max * 1.1)
        ax.set_title(POLICY_LABELS[name], fontsize=FONT_TITLE)
        ax.legend(loc="upper left", fontsize=FONT_LEGEND)
    fig.suptitle(f"3D trajectories {title_tag} -- all 8 approaches ({CFG_NAME})\n"
                 "trail color = final SOC (red = nearly stranded, green = healthy)",
                 fontsize=FONT_SUPTITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, fname)


def plot_ablation_one_env(ablation_results, fname, title_tag):
    keys = ["hrlh", "hrlh_nav", "hrlh_qmix"]
    labels = {"hrlh": "HRL-H (MAPPO)",
              "hrlh_nav": "HRL-H-NAV (no heuristic)",
              "hrlh_qmix": "HRL-H-QMIX (QMIX high level)"}
    colors = {"hrlh": POLICY_COLORS["hrlh"], "hrlh_nav": "#8c564b",
              "hrlh_qmix": "#2ca02c"}
    metrics = [("coverage_mean", "Coverage (%)", "coverage_std"),
               ("stranded_mean", "Stranded UAVs (%)", "stranded_std"),
               ("energy_wh_per_task_mean", "Energy per task (Wh)",
                "energy_wh_per_task_std"),
               ("steps_mean", "Episode length (steps)", "steps_std")]
    fig, axes = plt.subplots(2, 2, figsize=figsize_for(2, 2))
    x = np.arange(len(keys)); width = 0.55
    for ax, (mk, label, sk) in zip(axes.ravel(), metrics):
        vals = [ablation_results[n][mk] for n in keys]
        errs = [ablation_results[n][sk] for n in keys]
        ax.bar(x, vals, width, yerr=errs, capsize=5,
               color=[colors[n] for n in keys], edgecolor="k", linewidth=0.5)
        for b, v in zip(ax.patches, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=FONT_ANNOT)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[n] for n in keys],
                           fontsize=FONT_TICK, rotation=12, ha="right")
        ax.set_ylabel(label, fontsize=FONT_LABEL)
        ax.set_title(label, fontsize=FONT_TITLE)
        ax.tick_params(labelsize=FONT_TICK)
    fig.suptitle(f"HRL-H ablation -- {title_tag} ({CFG_NAME})",
                 fontsize=FONT_SUPTITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, fname)


# ---------------------------------------------------------------------------
# Train / load
# ---------------------------------------------------------------------------
def train_or_load():
    CKPT = os.path.join(OUT_DIR, f"mappo_{CFG_NAME}_{SCENARIO}_{N_TRAIN_EPISODES}ep.pkl")
    CKPT_MADDPG = os.path.join(OUT_DIR, f"maddpg_{CFG_NAME}.pkl")
    CKPT_QMIX = os.path.join(OUT_DIR, f"qmix_{CFG_NAME}.pkl")
    train_env = UAVSwarmEnv3D(EnvConfig3D(**{**ENV_CFG.__dict__, "seed": SEED}))

    if os.path.exists(CKPT):
        with open(CKPT, "rb") as f:
            d = pickle.load(f)
        actor, critic, log = d["actor"], d["critic"], d["log"]
        trained_episodes = d["n_episodes"]
        print(f"Loaded MAPPO checkpoint ({trained_episodes} ep) -> {CKPT}", flush=True)
    else:
        rng = np.random.default_rng(SEED)
        actor = Actor(mt.OWN_DIM, mt.NEIGHBOR_DIM, ENV_CFG.neighbor_cap,
                      N_TASKS_MAX * STRIDE, N_TASKS_MAX + 2,
                      hidden_dim=ACTOR_HIDDEN, rng=rng)
        critic = Critic(5 * ENV_CFG.n_agents, hidden_dim=64, rng=rng)
        actor_opt, critic_opt = mt.make_actor_critic_optims(actor, critic, lr=1e-3)
        log = []
        t0 = time.time()
        if WARM_START:
            print(f"Collecting {BC_EPISODES} Greedy-Nearest demo episodes...", flush=True)
            demo = collect_greedy_demos(train_env, BC_EPISODES, rng, TRAIN_TASK_COUNTS)
            print(f"  -> {demo[0].shape[0]} transitions; BC {BC_ITERS} iters...", flush=True)
            bc_losses = bc_pretrain(actor, critic, actor_opt, critic_opt, demo,
                                    BC_ITERS, rng)
            log.append({"stage": "bc", "episodes": BC_EPISODES, "iters": BC_ITERS,
                        "final_bc_loss": float(bc_losses[-1])})
            actor_opt, critic_opt = mt.make_actor_critic_optims(actor, critic, lr=3e-4)
        episode = 0
        while episode < N_TRAIN_EPISODES:
            train_env.cfg.packet_loss_prob = TARGET_PACKET_LOSS * min(
                1.0, episode / max(CURRICULUM_EPISODES, 1))
            buf = collect_rollout_vartask(train_env, actor, critic,
                                          PPO_EPISODES_PER_UPDATE, rng, TRAIN_TASK_COUNTS)
            stats = mt.ppo_update(actor, critic, actor_opt, critic_opt, buf, ent_coef=0.02)
            episode += PPO_EPISODES_PER_UPDATE
            log.append({"episode": episode,
                        "packet_loss": float(train_env.cfg.packet_loss_prob),
                        "mean_reward": stats["mean_reward"],
                        "entropy": stats["entropy"],
                        "value_loss": stats["value_loss"]})
            print(f"[ep {episode:5d}] p_loss={train_env.cfg.packet_loss_prob:.2f} "
                  f"mean_reward={stats['mean_reward']:7.3f} entropy={stats['entropy']:.3f} "
                  f"value_loss={stats['value_loss']:.4f}  ({time.time()-t0:.0f}s)",
                  flush=True)
        trained_episodes = episode
        clear_caches(actor); clear_caches(critic)
        with open(CKPT, "wb") as f:
            pickle.dump({"actor": actor, "critic": critic, "log": log,
                         "cfg": ENV_CFG, "n_episodes": episode}, f)
        print(f"MAPPO done in {time.time()-t0:.0f}s -> {CKPT}", flush=True)

    OBS_DIM = 4 + ENV_CFG.neighbor_cap * 6 + N_TASKS_MAX * STRIDE
    N_ACTIONS = N_TASKS_MAX + 2
    STATE_DIM = ENV_CFG.n_agents * 5
    rng_mar = np.random.default_rng(SEED + 1)

    if os.path.exists(CKPT_MADDPG) and os.path.exists(CKPT_QMIX):
        with open(CKPT_MADDPG, "rb") as f:
            maddpg = pickle.load(f)
        with open(CKPT_QMIX, "rb") as f:
            qmix = pickle.load(f)
        print(f"Loaded MADDPG/QMIX -> {CKPT_MADDPG}, {CKPT_QMIX}", flush=True)
    else:
        t0 = time.time()
        print(f"Collecting {MARL_DEMO_EPISODES} MARL demo episodes...", flush=True)
        mar_buf = collect_mar_demos(MARL_DEMO_EPISODES, MARL_TASK_COUNTS,
                                    rng_mar, eps=MARL_EPS_DEMO)
        print(f"  -> {len(mar_buf)} transitions in {time.time()-t0:.0f}s", flush=True)

        t0 = time.time()
        maddpg = MADDPG(OBS_DIM, N_ACTIONS, ENV_CFG.n_agents, hidden=32,
                        lr=MADDPG_LR, seed=SEED)
        obs_all, act_all, mask_all = mar_demo_stack(mar_buf)
        losses = bc_warmstart_actor(maddpg, obs_all, act_all, mask_all,
                                    MARL_BC_ITERS, np.random.default_rng(SEED + 2),
                                    batch=128)
        print(f"MADDPG BC done loss={losses[-1]:.4f} in {time.time()-t0:.0f}s", flush=True)
        t0 = time.time()
        for it in range(MARL_RL_ITERS):
            st = update_from_buffer(maddpg, mar_buf, 128,
                                    np.random.default_rng(100 + it))
            if it % 500 == 0:
                print(f"  MADDPG it {it}: cL={st['critic_loss']:.3f} "
                      f"pL={st['policy_loss']:.3f}", flush=True)
        clear_caches(maddpg)
        with open(CKPT_MADDPG, "wb") as f:
            pickle.dump(maddpg, f)
        print(f"MADDPG done in {time.time()-t0:.0f}s -> {CKPT_MADDPG}", flush=True)

        maxabs = max(abs(b[2]).max() for b in mar_buf.buf) or 1.0
        qbuf = ReplayBuffer(len(mar_buf.buf))
        for b in mar_buf.buf:
            o, a, r, no, m, nm, d, s, ns = b
            qbuf.buf.append((o, a, r / maxabs, no, m, nm, d, s, ns))
            qbuf.idx = len(qbuf.buf) % qbuf.cap
        t0 = time.time()
        qmix = QMIX(OBS_DIM, N_ACTIONS, ENV_CFG.n_agents, STATE_DIM, hidden=32,
                    mix_hidden=32, lr=QMIX_LR, gamma=0.99, seed=SEED,
                    target_update_every=QMIX_TARGET_EVERY)
        obs_all, act_all, mask_all = mar_demo_stack(qbuf)
        losses = qmix_bc_warmstart(qmix, obs_all, act_all, mask_all,
                                   MARL_BC_ITERS, np.random.default_rng(SEED + 3),
                                   batch=128)
        print(f"QMIX BC done loss={losses[-1]:.4f} in {time.time()-t0:.0f}s", flush=True)
        t0 = time.time()
        for it in range(MARL_RL_ITERS):
            sample = qbuf.sample(128, np.random.default_rng(200 + it))
            st = qmix_update(qmix, sample, 128, np.random.default_rng(300 + it),
                             q_lim=QMIX_Q_LIM, bc_weight=QMIX_BC_WEIGHT)
            if it % 500 == 0:
                print(f"  QMIX it {it}: td={st['td_loss']:.1e} "
                      f"qtot={st['mean_q_tot']:.2f}", flush=True)
        clear_caches(qmix)
        with open(CKPT_QMIX, "wb") as f:
            pickle.dump(qmix, f)
        print(f"QMIX done in {time.time()-t0:.0f}s -> {CKPT_QMIX}", flush=True)

    return actor, critic, log, trained_episodes, maddpg, qmix


def build_policies(actor, maddpg, qmix):
    def mappo_policy(env):
        obs_p, mask_p = build_padded_obs_mask(env)
        own, neigh, valid_mask, tasks = mt.split_obs(
            obs_p, mt.OWN_DIM, env.cfg.neighbor_cap, mt.NEIGHBOR_DIM,
            env.cfg.area_size, env.cfg.altitude_max)
        probs = softmax(actor.forward(own, neigh, valid_mask, tasks, mask_p))
        acts = np.argmax(probs, axis=1)
        return {a: int(acts[i]) for i, a in enumerate(env.agents)}

    def maddpg_policy(env):
        o, m = build_padded_obs_mask(env)
        acts, _ = maddpg.act(normalize_padded_obs(
            o, ENV_CFG.area_size, ENV_CFG.altitude_max,
            ENV_CFG.neighbor_cap, N_TASKS_MAX), m, rng=None, greedy=True)
        return {env.agents[i]: int(acts[i]) for i in range(env.n)}

    def qmix_policy(env):
        o, m = build_padded_obs_mask(env)
        acts = qmix.act(normalize_padded_obs(
            o, ENV_CFG.area_size, ENV_CFG.altitude_max,
            ENV_CFG.neighbor_cap, N_TASKS_MAX), m, greedy=True)
        return {env.agents[i]: int(acts[i]) for i in range(env.n)}

    def mappo_prefer_fn(env):
        obs_p, mask_p = build_padded_obs_mask(env)
        own, neigh, valid_mask, tasks = mt.split_obs(
            obs_p, mt.OWN_DIM, env.cfg.neighbor_cap, mt.NEIGHBOR_DIM,
            env.cfg.area_size, env.cfg.altitude_max)
        probs = softmax(actor.forward(own, neigh, valid_mask, tasks, mask_p))
        return probs[:, :N_TASKS_MAX]

    def qmix_prefer_fn(env):
        obs_p, mask_p = build_padded_obs_mask(env)
        q = qmix.qnet.forward(normalize_padded_obs(
            obs_p, ENV_CFG.area_size, ENV_CFG.altitude_max,
            ENV_CFG.neighbor_cap, N_TASKS_MAX))
        q = q + np.where(mask_p > 0.5, 0.0, -1e9)
        scores = np.full((env.n, N_TASKS_MAX), -np.inf)
        scores[:, :env.n_tasks] = q[:, :env.n_tasks]
        return scores

    hrlh = HRLH(mappo_prefer_fn, with_navigation=True)
    hrlh_nav = HRLH(mappo_prefer_fn, with_navigation=False,
                    raw_policy_fn=mappo_policy)
    hrlh_q = HRLH(qmix_prefer_fn, with_navigation=True)
    policies = {
        "random": random_policy,
        "greedy": greedy_nearest_policy,
        "cbba": cbba_policy,
        "mappo": mappo_policy,
        "maddpg": maddpg_policy,
        "qmix": qmix_policy,
        "dmpc": dmpc_policy,
        "hrlh": hrlh,
    }
    ablations = {"hrlh": hrlh, "hrlh_nav": hrlh_nav, "hrlh_qmix": hrlh_q}
    return policies, ablations


def save_env_snapshots():
    env_snapshots = []
    for eid, label, scenario, obs_on, n_obs in ENV_SPECS:
        cfg = EnvConfig3D(**{**ENV_CFGS[eid].__dict__, "seed": 3})
        env0 = UAVSwarmEnv3D(cfg)
        env0.reset()
        snap = collect_env_snapshot(env0, eid, label)
        env_snapshots.append(snap)
        save_env_snapshot(snap, os.path.join(SNAP_DIR, f"{eid}.json"))
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")
        plot_env_3d(env0, f"{label} ({CFG_NAME})", ax=ax)
        fig.tight_layout()
        save_fig(fig, f"env_snapshot_{eid}")
    write_json({"config": CFG_NAME, "seed": 3, "environments": ENV_ORDER,
                "snapshots": {s["env_id"]: s for s in env_snapshots}},
               os.path.join(SNAP_DIR, "all_environments.json"))
    fig = plot_environments_2x2(
        env_snapshots,
        title=f"3D evaluation environments -- 2x2 ({CFG_NAME} config)",
        figsize=(18, 16),
    )
    save_fig(fig, "env_snapshot_2x2")


def save_base_trajectories(policies):
    all_traj = {}
    for eid in ENV_ORDER:
        print(f"trajectories (base) @ {eid} ...", flush=True)
        records, rows = {}, []
        for name, fn in policies.items():
            env, traj, soc = record_episode(fn, ENV_CFGS[eid], seed=SEED)
            records[name] = (env, traj, soc)
            rows.append({
                "policy": name,
                "environment": eid,
                "scenario": env.cfg.scenario,
                "obstacles_enabled": bool(env.cfg.obstacles_enabled),
                "n_obstacles": int(env.cfg.n_obstacles),
                "tasks_completed": int(env.task_completed.sum()),
                "tasks_total": int(env.n_tasks),
                "coverage_pct": float(env.task_completed.sum()) / env.n_tasks * 100.0,
                "n_stranded": int(env.stranded.sum()),
                "mean_final_soc": float(env.soc.mean()),
                "steps_taken": int(traj.shape[0] - 1),
                "blocked_moves": int(getattr(env, "n_blocked_moves", 0)),
                "trajectory": traj.tolist(),
                "final_soc": env.soc.tolist(),
            })
        plot_trajectories_3d(records, f"trajectories_{eid}",
                             f"({ENV_LABELS.get(eid, eid)})")
        payload = {"config": CFG_NAME, "environment": eid, "seed": SEED,
                   "label": ENV_LABELS.get(eid, eid),
                   "policies": {r["policy"]: r for r in rows}}
        write_json(payload, os.path.join(OUT_DIR, f"trajectories_{eid}.json"))
        write_json(payload, os.path.join(OUT_DIR, "by_environment", eid,
                                         "trajectories.json"))
        all_traj[eid] = {r["policy"]: {k: v for k, v in r.items() if k != "trajectory"}
                         for r in rows}
    write_json({"config": CFG_NAME, "seed": SEED, "environments": ENV_ORDER,
                "conditions": all_traj},
               os.path.join(OUT_DIR, "trajectories.json"))


def main():
    t_all = time.time()
    print("configuration:", CFG_NAME, "envs:", ENV_ORDER, flush=True)
    actor, critic, log, trained_episodes, maddpg, qmix = train_or_load()
    policies, ablations = build_policies(actor, maddpg, qmix)

    print("sanity one episode @ layered ...", flush=True)
    for name, fn in policies.items():
        r = run_episode(fn, ENV_CFG, seed=SEED)
        print(f"  {name:<10} reward={r['total_reward']:8.1f}  "
              f"coverage={r['coverage_frac'] * 100:5.1f}%  "
              f"stranded={r['n_stranded']}  steps={r['steps_taken']}", flush=True)

    save_env_snapshots()

    meta = {"config": CFG_NAME, "n_seeds": N_SEEDS}

    print("packet-loss sweep", flush=True)
    pl = run_sweep_all_envs(policies, ENV_CFGS, [0.0, 0.2, 0.4, 0.6],
                            "packet_loss_prob", N_SEEDS)
    dump_sweep_json(pl, OUT_DIR, "eval_packet_loss",
                    {**meta, "levels": [0.0, 0.2, 0.4, 0.6], "sweep": "packet_loss"})

    print("comm-range sweep", flush=True)
    cr = run_sweep_all_envs(policies, ENV_CFGS, [6.0, 12.0, 25.0, 50.0],
                            "comm_range", N_SEEDS)
    dump_sweep_json(cr, OUT_DIR, "eval_comm_range",
                    {**meta, "levels": [6.0, 12.0, 25.0, 50.0], "sweep": "comm_range"})

    print("task-density sweep", flush=True)
    td = run_sweep_all_envs(policies, ENV_CFGS, [4, 8, 16, 32],
                            "n_tasks", N_SEEDS)
    dump_sweep_json(td, OUT_DIR, "eval_task_density",
                    {**meta, "levels": [4, 8, 16, 32], "sweep": "task_density"})

    print("UAV-dropout sweep", flush=True)
    ft = run_fault_sweep_all_envs(policies, ENV_CFGS, [0.0, 0.25, 0.5], N_SEEDS)
    dump_sweep_json(ft, OUT_DIR, "eval_fault_tolerance",
                    {**meta, "levels": [0.0, 0.25, 0.5], "sweep": "fault_tolerance"})

    print("UAV-count sweep", flush=True)
    sc_results = {}
    for eid, env_cfg in ENV_CFGS.items():
        print(f"  scalability @ {eid} ...", flush=True)
        sc_results[eid] = {}
        for name, fn in policies.items():
            sc_results[eid][name] = []
            for n in [4, 8, 16]:
                cfg = EnvConfig3D(**{**env_cfg.__dict__, "n_agents": n, "n_tasks": n})
                runs = [run_episode(fn, cfg, s) for s in range(N_SEEDS)]
                sc_results[eid][name].append(summarize_runs(runs, n))
                print(f"    {name} n={n} cov={sc_results[eid][name][-1]['coverage_mean']:.1f}%",
                      flush=True)
    dump_sweep_json(sc_results, OUT_DIR, "eval_scalability",
                    {**meta, "levels": [4, 8, 16], "sweep": "scalability"})

    print("base-config evaluation", flush=True)
    scenario_results, scenario_runs = run_base_all_envs(policies, ENV_CFGS, N_SEEDS)
    plot_scenario_comparison(scenario_results, ENV_ORDER, "scenario_comparison")
    dump_base_json(scenario_results, OUT_DIR, "eval_scenarios",
                   {**meta, "environments": ENV_ORDER, "env_labels": ENV_LABELS})
    dump_base_json(scenario_results, OUT_DIR, "eval_base",
                   {**meta, "n_agents": ENV_CFG.n_agents, "n_tasks": ENV_CFG.n_tasks})
    dump_tables_json(scenario_results, OUT_DIR, POLICY_ORDER, POLICY_LABELS,
                     title=f"Base-config comparison ({CFG_NAME}, 8 UAVs / 8 tasks)")

    save_base_trajectories(policies)

    print("HRL-H ablations", flush=True)
    ablation_by_env = {}
    for eid, cfg in ENV_CFGS.items():
        print(f"  HRL-H ablations @ {eid} ...", flush=True)
        ablation_by_env[eid] = {
            name: summarize_runs(
                [run_episode(fn, cfg, s) for s in range(N_SEEDS)], 0)
            for name, fn in ablations.items()
        }
        plot_ablation_one_env(ablation_by_env[eid],
                              f"hrlh_ablation_comparison_{eid}", ENV_LABELS[eid])
    dump_base_json(ablation_by_env, OUT_DIR, "eval_hrlh",
                   {**meta, "variants": list(ablations.keys())})

    print("deadline sweep", flush=True)
    deadline = {}
    for eid, cfg in ENV_CFGS.items():
        print(f"  deadline @ {eid} ...", flush=True)
        deadline[eid] = {}
        for name, fn in policies.items():
            deadline[eid][name] = []
            for d in [0, 200, 100]:
                c = EnvConfig3D(**{**cfg.__dict__, "task_mode": "time_critical",
                                   "task_deadline_steps": d,
                                   "task_value_decay_per_step": 0.005 if d > 0 else 0.0})
                runs = [run_episode(fn, c, s) for s in range(N_SEEDS)]
                deadline[eid][name].append(summarize_runs(runs, d))
    dump_sweep_json(deadline, OUT_DIR, "eval_deadline",
                    {**meta, "levels": [0, 200, 100], "value_decay": 0.005})

    print("coupled-task sweep", flush=True)
    coupled = {}
    profiles = [
        ("all single-agent", 1, ()),
        ("half need 2", 1.5, tuple(2 if k % 2 == 0 else 1
                                   for k in range(ENV_CFG.n_tasks))),
        ("all need 2", 2.0, tuple(2 for _ in range(ENV_CFG.n_tasks))),
    ]
    for eid, cfg in ENV_CFGS.items():
        print(f"  coupled @ {eid} ...", flush=True)
        coupled[eid] = {}
        for name, fn in policies.items():
            coupled[eid][name] = []
            for label, value, req in profiles:
                c = EnvConfig3D(**{**cfg.__dict__, "task_requirements": req})
                runs = [run_episode(fn, c, s) for s in range(N_SEEDS)]
                coupled[eid][name].append(summarize_runs(runs, value))
    dump_sweep_json(coupled, OUT_DIR, "eval_coupled",
                    {**meta, "profiles": [(p[0], p[1], list(p[2])) for p in profiles]})

    print("obstacle-density sweep", flush=True)
    obstacle = {}
    for eid in ["open", "layered"]:
        print(f"  obstacles @ {eid} ...", flush=True)
        obstacle[eid] = {}
        for name, fn in policies.items():
            obstacle[eid][name] = []
            for nb in [0, 2, 4]:
                c = EnvConfig3D(**{**ENV_CFGS[eid].__dict__,
                                   "obstacles_enabled": nb > 0, "n_obstacles": nb,
                                   "obstacle_half": NFZ_OBSTACLE_HALF})
                runs = [run_episode(fn, c, s) for s in range(N_SEEDS)]
                obstacle[eid][name].append(summarize_runs(runs, nb))
    dump_sweep_json(obstacle, OUT_DIR, "eval_obstacles",
                    {**meta, "levels": [0, 2, 4], "obstacle_half": NFZ_OBSTACLE_HALF})

    for eid in ENV_ORDER:
        plot_summary_bars(scenario_results[eid], f"comparison_summary_{eid}",
                          ENV_LABELS[eid])
        plot_extended_metrics(scenario_results[eid], f"extended_metrics_{eid}",
                              ENV_LABELS[eid])

    write_json({
        "project": "uav_swarm_3d",
        "config": CFG_NAME,
        "environments": ENV_ORDER,
        "n_seeds": N_SEEDS,
        "mappo_training_episodes": trained_episodes,
        "elapsed_s": time.time() - t_all,
        "base": scenario_results,
    }, os.path.join(OUT_DIR, "eval_summary.json"))
    print(f"ALL DONE in {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
