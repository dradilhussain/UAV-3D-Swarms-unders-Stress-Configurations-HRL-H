"""
NumPy reference implementation of the actor/critic architecture specified
in Section 4.2 of the manuscript. Mirrors mappo_model.py's design (own-state
encoder, neighbor attention, task encoder, fusion head, action masking;
centralized critic over true global state) but with single-head attention
and manual backprop, since this sandbox has no torch/network access.
"""

import numpy as np
from nn_blocks import Linear, ReLULayer, MLP2, SingleHeadAttention, relu, relu_grad


class Actor:
    def __init__(self, own_dim, neighbor_dim, neighbor_cap, task_dim, n_actions,
                 hidden_dim=32, rng=None):
        rng = rng or np.random.default_rng(0)
        self.own_dim, self.neighbor_dim = own_dim, neighbor_dim
        self.neighbor_cap, self.task_dim = neighbor_cap, task_dim
        self.hidden_dim = hidden_dim

        self.own_enc = Linear(own_dim, hidden_dim, rng)
        self.own_act = ReLULayer()

        self.neigh_proj = Linear(neighbor_dim, hidden_dim, rng)  # project raw neighbor feats to d_model before attention
        self.neigh_proj_act = ReLULayer()
        self.attn = SingleHeadAttention(hidden_dim, rng)

        self.task_enc = Linear(task_dim, hidden_dim, rng)
        self.task_act = ReLULayer()

        self.fuse = MLP2(hidden_dim * 3, hidden_dim, n_actions, rng)

        self._cache = None

    def all_layers(self):
        return {
            "own_enc": self.own_enc, "neigh_proj": self.neigh_proj,
            "attn_Wq": self.attn.Wq, "attn_Wk": self.attn.Wk, "attn_Wv": self.attn.Wv,
            "task_enc": self.task_enc, "fuse_l1": self.fuse.l1, "fuse_l2": self.fuse.l2,
        }

    def forward(self, own, neighbors, valid_mask, task_feat, action_mask):
        """
        own: (B, own_dim), neighbors: (B, K, neighbor_dim), valid_mask: (B, K)
        task_feat: (B, task_dim), action_mask: (B, n_actions) 1=valid
        returns: logits (B, n_actions), plus everything needed for backward
        """
        B, K, _ = neighbors.shape

        own_h = self.own_act.forward(self.own_enc.forward(own))                # (B, H)

        neigh_flat = neighbors.reshape(B * K, self.neighbor_dim)
        neigh_proj = self.neigh_proj_act.forward(self.neigh_proj.forward(neigh_flat)).reshape(B, K, self.hidden_dim)

        ctx, attn_w = self.attn.forward(own_h, neigh_proj, valid_mask)          # (B, H)

        task_h = self.task_act.forward(self.task_enc.forward(task_feat))       # (B, H)

        fused_in = np.concatenate([own_h, ctx, task_h], axis=1)                 # (B, 3H)
        logits = self.fuse.forward(fused_in)                                    # (B, n_actions)

        masked_logits = np.where(action_mask > 0.5, logits, -1e9)

        self._cache = (own, neighbors, valid_mask, task_feat, action_mask,
                        own_h, neigh_proj, ctx, attn_w, task_h, fused_in, B, K)
        return masked_logits

    def backward(self, dlogits):
        """dlogits: gradient w.r.t. the masked_logits output (masked entries
        should already carry zero gradient, since they never affect the loss
        for a sampled action that was itself never chosen there)."""
        (own, neighbors, valid_mask, task_feat, action_mask,
         own_h, neigh_proj, ctx, attn_w, task_h, fused_in, B, K) = self._cache

        dfused_in, g_fuse = self.fuse.backward(dlogits)
        d_own_h1 = dfused_in[:, :self.hidden_dim]
        d_ctx = dfused_in[:, self.hidden_dim:2 * self.hidden_dim]
        d_task_h = dfused_in[:, 2 * self.hidden_dim:]

        d_task_pre = self.task_act.backward(d_task_h)
        _, g_task = self.task_enc.backward(d_task_pre)

        d_own_h2, d_neigh_proj, g_attn = self.attn.backward(d_ctx)

        d_own_h = d_own_h1 + d_own_h2
        d_own_pre = self.own_act.backward(d_own_h)
        _, g_own = self.own_enc.backward(d_own_pre)

        d_neigh_proj_flat = d_neigh_proj.reshape(B * K, self.hidden_dim)
        d_neigh_pre = self.neigh_proj_act.backward(d_neigh_proj_flat)
        _, g_neigh_proj = self.neigh_proj.backward(d_neigh_pre)

        return {
            "own_enc": g_own, "neigh_proj": g_neigh_proj,
            "attn_Wq": g_attn["Wq"], "attn_Wk": g_attn["Wk"], "attn_Wv": g_attn["Wv"],
            "task_enc": g_task, "fuse_l1": g_fuse["l1"], "fuse_l2": g_fuse["l2"],
        }


class Critic:
    def __init__(self, global_state_dim, hidden_dim=64, rng=None):
        rng = rng or np.random.default_rng(1)
        self.net = MLP2(global_state_dim, hidden_dim, 1, rng)

    def forward(self, global_state):
        return self.net.forward(global_state).squeeze(-1)  # (B,)

    def backward(self, dvalue):
        return self.net.backward(dvalue[:, None])

    def all_layers(self):
        return {"l1": self.net.l1, "l2": self.net.l2}


def softmax(logits):
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


def categorical_sample(probs, rng):
    B, A = probs.shape
    u = rng.random(B)
    cdf = np.cumsum(probs, axis=1)
    actions = (u[:, None] > cdf).sum(axis=1)
    return np.clip(actions, 0, A - 1)
