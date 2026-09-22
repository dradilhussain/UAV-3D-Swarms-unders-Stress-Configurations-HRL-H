"""
MADDPG baseline (off-policy MARL, centralized critic / decentralized actors).

The environment's action space is discrete (choose a task, return, or hold), so
this is a **discrete-action adaptation** of MADDPG: the per-agent actors are
softmax (stochastic) policies and the centralized critic conditions on the
concatenated observations AND the joint (one-hot) action of the whole swarm.

The actor update uses the REINFORCE-style policy gradient with the centralized
critic value as baseline (a recognised discrete-action MADDPG variant -- the
original paper assumes continuous actions and a deterministic actor / DPG
update). Everything else follows MADDPG: a replay buffer, centralized critics,
and slowly-tracking target networks for both actor and critic.

Training-only: at execution the critic is dropped and each agent acts from its
own observation (CTDE), so the deployed policy is fully decentralized.
"""

import numpy as np
from .mlp import (MLP, Adam, softmax, log_softmax, one_hot,
                  categorical_sample, masked_logits)


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.cap = capacity
        self.buf = []
        self.idx = 0

    def add(self, obs, act, rew, next_obs, mask, next_mask, done,
            state=None, next_state=None):
        if len(self.buf) < self.cap:
            self.buf.append(None)
        self.buf[self.idx] = (obs, act, rew, next_obs, mask, next_mask, done,
                              state, next_state)
        self.idx = (self.idx + 1) % self.cap

    def sample(self, batch_size, rng):
        idxs = rng.integers(0, len(self.buf), size=batch_size)
        batch = [self.buf[i] for i in idxs]
        obs = np.stack([b[0] for b in batch])
        act = np.stack([b[1] for b in batch])
        rew = np.stack([b[2] for b in batch])
        next_obs = np.stack([b[3] for b in batch])
        mask = np.stack([b[4] for b in batch])
        next_mask = np.stack([b[5] for b in batch])
        done = np.array([b[6] for b in batch])
        state = np.stack([b[7] for b in batch])
        next_state = np.stack([b[8] for b in batch])
        return obs, act, rew, next_obs, mask, next_mask, done, state, next_state

    def __len__(self):
        return len(self.buf)


class MADDPG:
    def __init__(self, obs_dim, n_actions, n_agents, hidden=64, lr=3e-4,
                 gamma=0.99, tau=0.005, ent_coef=0.01, rng=None, seed=0):
        rng = rng or np.random.default_rng(seed)
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.gamma, self.tau, self.ent_coef = gamma, tau, ent_coef

        # decentralized actor: the same MLP serves every agent (CTDE sharing)
        self.actor = MLP([obs_dim, hidden, hidden, n_actions], rng)
        self.actor_target = MLP([obs_dim, hidden, hidden, n_actions], rng)
        self._copy_params(self.actor_target, self.actor)

        # centralized critic: (all obs, all one-hot actions) -> Q per agent
        critic_in = n_agents * (obs_dim + n_actions)
        self.critic = MLP([critic_in, hidden, hidden, n_agents], rng)
        self.critic_target = MLP([critic_in, hidden, hidden, n_agents], rng)
        self._copy_params(self.critic_target, self.critic)

        self.actor_opt = Adam(self.actor.param_refs(), lr=lr)
        self.critic_opt = Adam(self.critic.param_refs(), lr=lr)

    @staticmethod
    def _copy_params(dst, src):
        for (d, _), (s, _) in zip(dst.param_refs(), src.param_refs()):
            d[...] = s

    def _soft_update(self, dst, src):
        for (d, _), (s, _) in zip(dst.param_refs(), src.param_refs()):
            d[...] = self.tau * s + (1 - self.tau) * d

    # ------------------------------------------------------------------ act
    def act(self, obs, mask, rng=None, greedy=True):
        """obs: (n_agents, obs_dim), mask: (n_agents, n_actions).
        Returns (actions, probs)."""
        logits = masked_logits(self.actor.forward(obs), mask)
        probs = softmax(logits)
        if greedy:
            return np.argmax(probs, axis=1), probs
        return categorical_sample(probs, rng), probs

    def target_greedy(self, obs, mask):
        logits = masked_logits(self.actor_target.forward(obs), mask)
        return np.argmax(softmax(logits), axis=1)


def update_from_buffer(maddpg, buffer, batch_size, rng):
    """One off-policy MADDPG update from a shared ReplayBuffer.

    Returns a stats dict (critic_loss, policy_loss, entropy, mean_q)."""
    obs, act, rew, next_obs, mask, next_mask, done, _, _ = \
        buffer.sample(batch_size, rng)
    B, n, d = obs.shape
    A = maddpg.n_actions
    obs_flat = obs.reshape(B, n * d)
    next_obs_flat = next_obs.reshape(B, n * d)
    act_oh = one_hot(act.reshape(B * n), A).reshape(B, n * A)

    # ---- critic TD target: target nets + target-greedy next joint action
    next_act = np.stack([
        maddpg.target_greedy(next_obs[:, i], next_mask[:, i]) for i in range(n)
    ], axis=1)
    next_act_oh = one_hot(next_act.reshape(B * n), A).reshape(B, n * A)
    target_q = maddpg.critic_target.forward(
        np.hstack([next_obs_flat, next_act_oh]))
    y = rew + maddpg.gamma * (1.0 - done)[:, None] * target_q

    critic_in = np.hstack([obs_flat, act_oh])
    q = maddpg.critic.forward(critic_in)
    critic_loss = float(np.mean((q - y) ** 2))
    dcritic = 2.0 * (q - y) / B
    _, critic_grads = maddpg.critic.backward(dcritic)
    cgrads = []
    for name in [nm for _, nm in maddpg.critic.param_refs()]:
        g = critic_grads[name[:-2]]
        cgrads.append(g["W"] if name.endswith("W") else g["b"])
    maddpg.critic_opt.step(cgrads)

    # ---- actor: REINFORCE-style PG with centralized critic as baseline
    actor_in = obs.reshape(B * n, d)
    logits = masked_logits(maddpg.actor.forward(actor_in),
                           mask.reshape(B * n, A))
    probs = softmax(logits)
    a_sample = categorical_sample(probs, rng)
    a_oh = one_hot(a_sample, A).reshape(B, n * A)
    q_sample = maddpg.critic.forward(np.hstack([obs_flat, a_oh]))
    adv = (q_sample - q).reshape(-1)                      # baseline = Q(behaviour)
    logp = log_softmax(logits)[np.arange(B * n), a_sample]
    policy_loss = float(-np.mean(adv * logp))
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)

    dlogp_dlogits = one_hot(a_sample, A) - probs
    dlogits = -((adv[:, None] * dlogp_dlogits) / (B * n))
    # entropy bonus: d(-H)/dz = p * (log p + H); subtracting ent_coef * H from
    # the loss keeps exploration alive
    dlogits += maddpg.ent_coef * (
        probs * (log_softmax(logits) + entropy[:, None])) / (B * n)
    dlogits = dlogits * mask.reshape(B * n, A)
    _, actor_grads = maddpg.actor.backward(dlogits)
    agrads = []
    for name in [nm for _, nm in maddpg.actor.param_refs()]:
        g = actor_grads[name[:-2]]
        agrads.append(g["W"] if name.endswith("W") else g["b"])
    maddpg.actor_opt.step(agrads)

    maddpg._soft_update(maddpg.critic_target, maddpg.critic)
    maddpg._soft_update(maddpg.actor_target, maddpg.actor)

    return {"critic_loss": critic_loss, "policy_loss": policy_loss,
            "entropy": float(entropy.mean()), "mean_q": float(q.mean())}


def bc_warmstart_actor(maddpg, obs, act, mask, iters, rng, batch=128):
    """Behavior-clone the actor onto demonstrated (greedy-style) actions so the
    off-policy RL starts task-competent instead of cold. Same idea as the MAPPO
    warm-start used for the other learned policy in this project.
    obs may be (T, n, obs_dim) transitions -- they are flattened to per-agent
    rows before training."""
    if obs.ndim == 3:
        obs = obs.reshape(-1, obs.shape[-1])
        act = act.reshape(-1)
        mask = mask.reshape(-1, mask.shape[-1])
    n = obs.shape[0]
    losses = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=batch)
        logits = masked_logits(maddpg.actor.forward(obs[idx]), mask[idx])
        probs = softmax(logits)
        logp = np.log(np.clip(probs[np.arange(batch), act[idx]], 1e-12, 1.0))
        losses.append(-float(np.mean(logp)))
        onehot = np.zeros_like(logits)
        onehot[np.arange(batch), act[idx]] = 1.0
        dlogits = (probs - onehot) / batch
        dlogits = dlogits * mask[idx]
        _, grads = maddpg.actor.backward(dlogits)
        g = []
        for name in [nm for _, nm in maddpg.actor.param_refs()]:
            gg = grads[name[:-2]]
            g.append(gg["W"] if name.endswith("W") else gg["b"])
        maddpg.actor_opt.step(g)
    return losses
