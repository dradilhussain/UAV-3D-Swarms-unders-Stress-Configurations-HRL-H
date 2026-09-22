"""
Minimal NumPy autograd-free neural network toolkit: just enough to build
and train the MAPPO actor/critic architecture of the manuscript (Section
4.2) without a torch dependency. Every layer implements .forward() and
.backward() by hand, with analytic gradients (verified against numerical
gradients in test_gradients.py). This is deliberately small and specific
to this architecture -- it is not a general autograd system.
"""

import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(x.dtype)


class Linear:
    """y = x @ W + b, with x of shape (B, in_dim), y of shape (B, out_dim)."""

    def __init__(self, in_dim, out_dim, rng):
        limit = np.sqrt(6.0 / (in_dim + out_dim))  # Xavier/Glorot uniform
        self.W = rng.uniform(-limit, limit, size=(in_dim, out_dim)).astype(np.float64)
        self.b = np.zeros(out_dim, dtype=np.float64)
        self._cache = None

    def forward(self, x):
        self._cache = x
        return x @ self.W + self.b

    def backward(self, dy):
        x = self._cache
        dW = x.T @ dy
        db = dy.sum(axis=0)
        dx = dy @ self.W.T
        return dx, {"W": dW, "b": db}

    def params(self):
        return {"W": self.W, "b": self.b}


class ReLULayer:
    def __init__(self):
        self._cache = None

    def forward(self, x):
        self._cache = x
        return relu(x)

    def backward(self, dy):
        return dy * relu_grad(self._cache)


class MLP2:
    """Linear -> ReLU -> Linear, the standard encoder/head block used
    throughout the actor and critic."""

    def __init__(self, in_dim, hidden_dim, out_dim, rng):
        self.l1 = Linear(in_dim, hidden_dim, rng)
        self.act = ReLULayer()
        self.l2 = Linear(hidden_dim, out_dim, rng)

    def forward(self, x):
        h = self.act.forward(self.l1.forward(x))
        return self.l2.forward(h)

    def backward(self, dy):
        dh, g2 = self.l2.backward(dy)
        dh = self.act.backward(dh)
        dx, g1 = self.l1.backward(dh)
        return dx, {"l1": g1, "l2": g2}

    def layers(self):
        return {"l1": self.l1, "l2": self.l2}


class SingleHeadAttention:
    """
    Single-head scaled dot-product attention, matching Eq. 11 of the
    manuscript (with n_heads=1 for tractability of hand-written backprop
    in this NumPy reference implementation -- the torch version in
    mappo_model.py uses full multi-head attention).

    query:  (B, d_model)          -- one query per batch row (own-state)
    keys:   (B, K, d_model)       -- K neighbor slots per batch row
    values: (B, K, d_model)
    mask:   (B, K)                -- 1 = valid neighbor, 0 = masked out
    """

    def __init__(self, d_model, rng):
        self.d_model = d_model
        self.Wq = Linear(d_model, d_model, rng)
        self.Wk = Linear(d_model, d_model, rng)
        self.Wv = Linear(d_model, d_model, rng)
        self._cache = None

    def forward(self, own_feat, neighbor_feat, mask):
        B, K, D = neighbor_feat.shape
        Q = self.Wq.forward(own_feat)                       # (B, D)
        K_ = self.Wk.forward(neighbor_feat.reshape(B * K, D)).reshape(B, K, D)
        V = self.Wv.forward(neighbor_feat.reshape(B * K, D)).reshape(B, K, D)

        scores = np.einsum("bd,bkd->bk", Q, K_) / np.sqrt(D)  # (B, K)
        neg_inf = np.full_like(scores, -1e9)
        masked_scores = np.where(mask > 0.5, scores, neg_inf)

        # numerically stable softmax, with an all-masked-row guard
        m = masked_scores.max(axis=1, keepdims=True)
        exp = np.exp(masked_scores - m) * (mask > 0.5)
        denom = exp.sum(axis=1, keepdims=True)
        denom_safe = np.where(denom == 0, 1.0, denom)
        attn = exp / denom_safe                                # (B, K)

        context = np.einsum("bk,bkd->bd", attn, V)             # (B, D)

        self._cache = (Q, K_, V, attn, mask, own_feat, neighbor_feat, B, K, D)
        return context, attn

    def backward(self, dcontext):
        Q, K_, V, attn, mask, own_feat, neighbor_feat, B, K, D = self._cache

        # context = sum_k attn[b,k] * V[b,k,:]
        dattn = np.einsum("bd,bkd->bk", dcontext, V)             # (B, K)
        dV = np.einsum("bd,bk->bkd", dcontext, attn)             # (B, K, D)

        # softmax backward: ds_k = a_k * (da_k - sum_j a_j da_j)
        weighted = (attn * dattn).sum(axis=1, keepdims=True)     # (B, 1)
        dscores = attn * (dattn - weighted)                       # (B, K)
        dscores = dscores * (mask > 0.5)                          # masked slots contribute no gradient

        dscores = dscores / np.sqrt(D)
        dQ = np.einsum("bk,bkd->bd", dscores, K_)                # (B, D)
        dK_ = np.einsum("bk,bd->bkd", dscores, Q)                # (B, K, D)

        d_own, gq = self.Wq.backward(dQ)
        d_neighbor_k, gk = self.Wk.backward(dK_.reshape(B * K, D))
        d_neighbor_v, gv = self.Wv.backward(dV.reshape(B * K, D))
        d_neighbor = (d_neighbor_k + d_neighbor_v).reshape(B, K, D)

        return d_own, d_neighbor, {"Wq": gq, "Wk": gk, "Wv": gv}

    def layers(self):
        return {"Wq": self.Wq, "Wk": self.Wk, "Wv": self.Wv}


class Adam:
    """Standard Adam optimizer operating directly on the numpy arrays
    referenced inside each Linear layer (in-place updates)."""

    def __init__(self, param_refs, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.param_refs = param_refs  # list of (array, name) -- updated in place via index assignment
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = [np.zeros_like(p) for p, _ in param_refs]
        self.v = [np.zeros_like(p) for p, _ in param_refs]
        self.t = 0

    def step(self, grads):
        """grads: list of gradient arrays, same order/shape as param_refs."""
        self.t += 1
        for i, ((p, _), g) in enumerate(zip(self.param_refs, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
