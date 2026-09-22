"""
PPO training loop for the NumPy MAPPO reference implementation ported to the
3D UAV swarm environment (x, y, z positions with altitude/climb-cost energy
model). Mirrors the 2D `numpy_mappo/train.py` exactly, with the dimension
changes the 3D observation space requires:

  - own state        : [x, y, z, soc]              -> own_dim = 4
  - neighbor feats   : [rel_x, rel_y, rel_z, soc_belief, staleness, valid]
                                                     -> neighbor_dim = 6
  - per-task feats   : [x, y, z, active]           -> task_feat_stride = 4
  - global state     : [x, y, z, soc, stranded] per agent -> 5*n_agents

Position-scale features are normalized by area_size for x/y and altitude_max
for z, keeping network inputs O(1). Everything else (GAE, clipped objective,
entropy bonus, value loss, comm-degradation curriculum) is unchanged from the
2D version.

This copy is used by the 3D notebook (uav_swarm_marl_3d.ipynb). The module
path handling below makes the imports work regardless of where the notebook
lives, as long as the bundled `code/` directory is on sys.path.
"""

import sys
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)          # .../code
_ENV = os.path.join(_PKG, "env_3d")
for _p in (_HERE, _ENV):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from uav_swarm_env_3d import UAVSwarmEnv3D, EnvConfig3D
from model import Actor, Critic, softmax, categorical_sample
from nn_blocks import Adam


OWN_DIM = 4
NEIGHBOR_DIM = 6
TASK_FEAT_STRIDE = 4  # per-task features: x, y, z, active
ACTIVE_IDX = TASK_FEAT_STRIDE - 1  # column holding the task's active flag


def build_global_state(env):
    per_agent = np.stack(
        [env.pos[:, 0], env.pos[:, 1], env.pos[:, 2], env.soc,
         env.stranded.astype(np.float64)], axis=-1
    )
    return per_agent.flatten()


def split_obs(obs_batch, own_dim, neighbor_cap, neighbor_dim, area_size,
              altitude_max=60.0):
    """Split + normalize the raw 3D observation into (own, neighbors,
    valid_mask, tasks). x/y are normalized by area_size, z by altitude_max."""
    obs_batch = obs_batch.copy()
    own = obs_batch[:, :own_dim].copy()
    own[:, 0] /= area_size
    own[:, 1] /= area_size
    own[:, 2] /= altitude_max

    neigh_flat = obs_batch[:, own_dim: own_dim + neighbor_cap * neighbor_dim].copy()
    neighbors = neigh_flat.reshape(-1, neighbor_cap, neighbor_dim)
    neighbors[:, :, 0] /= area_size  # rel_x
    neighbors[:, :, 1] /= area_size  # rel_y
    neighbors[:, :, 2] /= altitude_max  # rel_z

    tasks = obs_batch[:, own_dim + neighbor_cap * neighbor_dim:].copy()
    n_tasks = tasks.shape[1] // TASK_FEAT_STRIDE
    tasks_r = tasks.reshape(-1, n_tasks, TASK_FEAT_STRIDE)
    tasks_r[:, :, 0] /= area_size
    tasks_r[:, :, 1] /= area_size
    tasks_r[:, :, 2] /= altitude_max
    tasks = tasks_r.reshape(-1, n_tasks * TASK_FEAT_STRIDE)

    valid_mask = neighbors[:, :, -1]
    return own, neighbors, valid_mask, tasks


def collect_rollout(env, actor, critic, n_episodes, rng, deterministic=False,
                    area_size=None, altitude_max=None):
    area_size = area_size or env.cfg.area_size
    altitude_max = altitude_max or env.cfg.altitude_max
    obs_l, neigh_mask_l, task_l, own_l = [], [], [], []
    actions_l, logprobs_l, values_l, rewards_l, dones_l, global_l = [], [], [], [], [], []

    for _ in range(n_episodes):
        obs_dict, _ = env.reset()
        done = False
        while not done:
            obs_batch = np.stack([obs_dict[a] for a in env.agents])
            mask_batch = np.stack([env.action_mask(i) for i in range(env.n)])
            own, neigh, valid_mask, tasks = split_obs(
                obs_batch, OWN_DIM, env.cfg.neighbor_cap, NEIGHBOR_DIM,
                area_size, altitude_max)

            logits = actor.forward(own, neigh, valid_mask, tasks, mask_batch)
            probs = softmax(logits)
            if deterministic:
                actions = np.argmax(probs, axis=1)
            else:
                actions = categorical_sample(probs, rng)
            logp = np.log(np.clip(probs[np.arange(env.n), actions], 1e-12, 1.0))

            global_state = build_global_state(env)
            gs_rep = np.tile(global_state, (env.n, 1))
            values = critic.forward(gs_rep)

            action_dict = {a: int(actions[i]) for i, a in enumerate(env.agents)}
            next_obs_dict, reward_dict, term_dict, trunc_dict, infos = env.step(action_dict)
            rewards = np.array([reward_dict[a] for a in env.agents], dtype=np.float64)
            done = all(term_dict.values()) or all(trunc_dict.values())

            obs_l.append(obs_batch); own_l.append(own); neigh_mask_l.append((neigh, valid_mask))
            task_l.append(tasks); actions_l.append(actions); logprobs_l.append(logp)
            values_l.append(values); rewards_l.append(rewards)
            dones_l.append(np.full(env.n, float(done))); global_l.append(gs_rep)

            obs_dict = next_obs_dict

    return {
        "obs": obs_l, "own": own_l, "neigh_mask": neigh_mask_l, "task": task_l,
        "actions": actions_l, "logprobs": logprobs_l, "values": values_l,
        "rewards": rewards_l, "dones": dones_l, "global": global_l,
    }


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = [None] * T
    last_gae = np.zeros_like(rewards[0])
    next_value = np.zeros_like(values[0])
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae.copy()
        next_value = values[t]
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def ppo_update(actor, critic, actor_opt, critic_opt, buf,
                clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    advantages, returns = compute_gae(buf["rewards"], buf["values"], buf["dones"])
    adv_flat = np.concatenate(advantages)
    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
    ret_flat = np.concatenate(returns)
    old_logp_flat = np.concatenate(buf["logprobs"])
    actions_flat = np.concatenate(buf["actions"])

    own_flat = np.concatenate(buf["own"])
    task_flat = np.concatenate(buf["task"])
    neigh_flat = np.concatenate([nm[0] for nm in buf["neigh_mask"]])
    mask_flat = np.concatenate([nm[1] for nm in buf["neigh_mask"]])
    global_flat = np.concatenate(buf["global"])

    # re-derive the exact action masks used at collection time from the
    # stored task-activity features (matches env.action_mask()'s own logic).
    # Works with fixed-size OR zero-padded (variable task count) buffers.
    n_tasks = (task_flat.shape[1]) // TASK_FEAT_STRIDE
    task_active = task_flat.reshape(-1, n_tasks, TASK_FEAT_STRIDE)[:, :, ACTIVE_IDX]
    n_actions = n_tasks + 2
    action_mask_flat = np.zeros((task_flat.shape[0], n_actions), dtype=np.float64)
    action_mask_flat[:, :n_tasks] = task_active
    action_mask_flat[:, n_tasks] = 1.0    # return-to-base always valid
    action_mask_flat[:, n_tasks + 1] = 1.0  # hold always valid

    # ---- forward pass (current policy) ----
    logits = actor.forward(own_flat, neigh_flat, mask_flat, task_flat, action_mask_flat)
    probs = softmax(logits)
    B = probs.shape[0]
    new_logp = np.log(np.clip(probs[np.arange(B), actions_flat], 1e-12, 1.0))
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)

    ratio = np.exp(new_logp - old_logp_flat)
    surr1 = ratio * adv_flat
    surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv_flat
    policy_obj = np.minimum(surr1, surr2)  # (B,) -- this is what we MAXIMIZE

    # ---- policy gradient w.r.t. logits ----
    use_unclipped = surr1 <= surr2
    dratio_dlogp = ratio  # d(exp(new_logp - old_logp))/d(new_logp) = ratio
    dobj_dlogp = np.where(use_unclipped, dratio_dlogp * adv_flat, 0.0)
    dloss_dlogp = -dobj_dlogp / B

    onehot = np.zeros_like(probs)
    onehot[np.arange(B), actions_flat] = 1.0
    dlogp_dlogits = onehot - probs  # (B, n_actions)

    dlogits_from_policy = dloss_dlogp[:, None] * dlogp_dlogits

    logp_all = np.log(np.clip(probs, 1e-12, 1.0))
    ent_grad_logits = -probs * (logp_all - (probs * logp_all).sum(axis=1, keepdims=True))
    dloss_dlogits_ent = -ent_coef * ent_grad_logits / B

    dlogits = dlogits_from_policy + dloss_dlogits_ent
    dlogits = dlogits * action_mask_flat  # zero gradient at masked (invalid) actions

    actor_grads = actor.backward(dlogits)

    # ---- critic update: MSE toward returns ----
    values_pred = critic.forward(global_flat)
    dvalue = vf_coef * (values_pred - ret_flat) / B
    critic_grads = critic.backward(dvalue)

    # ---- apply Adam updates ----
    actor_param_grads = []
    for name, layer in actor.all_layers().items():
        g = actor_grads[name]
        actor_param_grads.append(g["W"])
        actor_param_grads.append(g["b"])
    actor_opt.step(actor_param_grads)

    critic_param_grads = []
    for name, layer in critic.all_layers().items():
        g = critic_grads[1][name]
        critic_param_grads.append(g["W"])
        critic_param_grads.append(g["b"])
    critic_opt.step(critic_param_grads)

    mean_reward = np.mean([r.mean() for r in buf["rewards"]])
    return {
        "policy_obj": float(policy_obj.mean()),
        "entropy": float(entropy.mean()),
        "value_loss": float(np.mean((values_pred - ret_flat) ** 2)),
        "mean_reward": float(mean_reward),
    }


def make_actor_critic_optims(actor, critic, lr=3e-4):
    actor_params = []
    for name, layer in actor.all_layers().items():
        actor_params.append((layer.W, f"{name}.W"))
        actor_params.append((layer.b, f"{name}.b"))
    actor_opt = Adam(actor_params, lr=lr)

    critic_params = []
    for name, layer in critic.all_layers().items():
        critic_params.append((layer.W, f"{name}.W"))
        critic_params.append((layer.b, f"{name}.b"))
    critic_opt = Adam(critic_params, lr=lr)
    return actor_opt, critic_opt


def train(env_cfg, total_episodes=3000, episodes_per_update=10,
          target_packet_loss=0.3, curriculum_episodes=1200,
          seed=0, log_every=200, hidden_dim=32, lr=3e-4, ent_coef=0.01):
    rng = np.random.default_rng(seed)
    env = UAVSwarmEnv3D(env_cfg)

    task_dim = env_cfg.n_tasks * TASK_FEAT_STRIDE
    n_actions = env.action_space_size

    actor = Actor(OWN_DIM, NEIGHBOR_DIM, env_cfg.neighbor_cap, task_dim, n_actions,
                  hidden_dim=hidden_dim, rng=rng)
    critic = Critic(5 * env_cfg.n_agents, hidden_dim=hidden_dim * 2, rng=rng)
    actor_opt, critic_opt = make_actor_critic_optims(actor, critic, lr=lr)

    log = []
    episode = 0
    while episode < total_episodes:
        env.cfg.packet_loss_prob = target_packet_loss * min(1.0, episode / max(curriculum_episodes, 1))
        buf = collect_rollout(env, actor, critic, episodes_per_update, rng,
                              area_size=env_cfg.area_size, altitude_max=env_cfg.altitude_max)
        stats = ppo_update(actor, critic, actor_opt, critic_opt, buf, ent_coef=ent_coef)
        episode += episodes_per_update
        if episode % log_every < episodes_per_update:
            log.append((episode, env.cfg.packet_loss_prob, stats["mean_reward"], stats["entropy"]))
            print(f"[ep {episode:5d}] p_loss={env.cfg.packet_loss_prob:.2f} "
                  f"mean_reward={stats['mean_reward']:.3f} entropy={stats['entropy']:.3f} "
                  f"value_loss={stats['value_loss']:.4f}")

    return actor, critic, log
