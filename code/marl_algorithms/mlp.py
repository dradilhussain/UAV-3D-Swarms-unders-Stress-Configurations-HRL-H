"""
Tiny self-contained NumPy neural-network building blocks shared by the
additional MARL baselines (MADDPG, QMIX) and the HRL-H proposed model.

Deliberately mirrors the spirit of `code/numpy_mappo/nn_blocks.py` (analytic
forward/backward, no torch) so the whole project stays dependency-free:
numpy + matplotlib only. Layers are batched: inputs are (B, in_dim) and the
SAME weights are applied to every row, which is exactly what we need for the
"shared" networks used in CTDE (one actor / one Q-network / one critic serving
all agents via a batch dimension).

This is the **3D port**: the observation layout follows the 3D environment
(own [x,y,z,soc] = 4, neighbors [rel_x,rel_y,rel_z,soc,staleness,valid] = 6,
tasks [x,y,z,active] = 4) and position-scale features are normalised by
area_size (x/y) and altitude_max (z).
"""

import numpy as np


def relu(x):
    return np.maximum(0.0, x)


def relu_grad(x):
    return (x > 0.0).astype(np.float64)


class Linear:
    """y = x @ W + b with x (B, in_dim) -> y (B, out_dim)."""

    def __init__(self, in_dim, out_dim, rng):
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.W = rng.uniform(-limit, limit, size=(in_dim, out_dim)).astype(np.float64)
        self.b = np.zeros(out_dim, dtype=np.float64)
        self._cache = None

    def forward(self, x):
        self._cache = x
        return x @ self.W + self.b

    def backward(self, dy):
        x = self._cache
        return (dy @ self.W.T,
                {"W": x.T @ dy, "b": dy.sum(axis=0)})

    def params(self):
        return {"W": self.W, "b": self.b}


class ReLU:
    def __init__(self):
        self._cache = None

    def forward(self, x):
        self._cache = x
        return relu(x)

    def backward(self, dy):
        return dy * relu_grad(self._cache)


class MLP:
    """Linear -> ReLU -> Linear -> ... -> Linear. `sizes` are the layer sizes
    including input and output, e.g. MLP([in_dim, 64, out_dim], rng)."""

    def __init__(self, sizes, rng):
        self.layers = []  # list of Linear / ReLU alternating, Linear first
        self.param_names = []  # ordered (layer_name, W|b) for Adam
        for i in range(len(sizes) - 1):
            lin = Linear(sizes[i], sizes[i + 1], rng)
            self.layers.append(("lin", lin))
            self.param_names.append((f"l{i}", "W"))
            self.param_names.append((f"l{i}", "b"))
            if i < len(sizes) - 2:
                self.layers.append(("relu", ReLU()))
        self._cache = None

    def forward(self, x):
        self._cache = []
        for kind, layer in self.layers:
            self._cache.append(x)
            x = layer.forward(x)
        return x

    def backward(self, dy):
        grads = {name: None for name, _ in self.param_names}
        for (kind, layer), x_in in reversed(list(zip(self.layers, self._cache))):
            if kind == "relu":
                dy = layer.backward(dy)
            else:
                idx = None
                for k, (ln, _) in enumerate(self.layers):
                    if ln == "lin" and layer is self.layers[k][1]:
                        idx = k // 2  # layer index i
                        break
                dx, g = layer.backward(dy)
                grads[f"l{idx}"] = {"W": g["W"], "b": g["b"]}
                dy = dx
        return dy, grads

    def all_layers(self):
        return {f"l{i}": lin for i, (kind, lin) in enumerate(self.layers) if kind == "lin"}

    def param_refs(self):
        refs = []
        li = 0
        for kind, lin in self.layers:
            if kind == "lin":
                refs.append((lin.W, f"l{li}.W"))
                refs.append((lin.b, f"l{li}.b"))
                li += 1
        return refs


class Adam:
    """Adam over a flat list of (array, name) references, updated in place."""

    def __init__(self, param_refs, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.refs = param_refs
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p, _ in param_refs]
        self.v = [np.zeros_like(p) for p, _ in param_refs]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, ((p, _), g) in enumerate(zip(self.refs, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def softmax(logits):
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


def log_softmax(logits):
    p = softmax(logits)
    return np.log(np.clip(p, 1e-12, 1.0))


def one_hot(actions, n_actions):
    B = actions.shape[0]
    oh = np.zeros((B, n_actions), dtype=np.float64)
    oh[np.arange(B), actions] = 1.0
    return oh


def categorical_sample(probs, rng):
    B, A = probs.shape
    u = rng.random(B)
    cdf = np.cumsum(probs, axis=1)
    actions = (u[:, None] > cdf).sum(axis=1)
    return np.clip(actions, 0, A - 1)


def masked_logits(logits, mask):
    """logits: (B, A), mask: (B, A) with 1 = valid. Invalid -> -1e9."""
    return np.where(mask > 0.5, logits, -1e9)


def normalize_padded_obs(obs_p, area_size, altitude_max, neighbor_cap,
                         n_tasks_max):
    """Normalize the position-scale features of a padded observation batch in
    place (same normalization the MAPPO policy applies via split_obs), so the
    MLP inputs are O(1) regardless of area/altitude scale. Layout of a padded
    row (3D):
      [own x,y,z,soc] + [neighbor_cap x (rel_x,rel_y,rel_z,soc,staleness,valid)]
      + [n_tasks_max x (task_x, task_y, task_z, active)].
    x/y are divided by area_size, z by altitude_max. Padded slots are already
    zero, so dividing keeps them zero."""
    o = obs_p.copy()
    o[:, 0] /= area_size   # own x
    o[:, 1] /= area_size   # own y
    o[:, 2] /= altitude_max  # own z
    off = 4
    if neighbor_cap > 0:
        neigh = o[:, off:off + neighbor_cap * 6].reshape(-1, neighbor_cap, 6)
        neigh[:, :, 0] /= area_size
        neigh[:, :, 1] /= area_size
        neigh[:, :, 2] /= altitude_max
        o[:, off:off + neighbor_cap * 6] = neigh.reshape(-1, neighbor_cap * 6)
        off += neighbor_cap * 6
    tasks = o[:, off:].reshape(-1, n_tasks_max, 4)
    tasks[:, :, 0] /= area_size
    tasks[:, :, 1] /= area_size
    tasks[:, :, 2] /= altitude_max
    o[:, off:] = tasks.reshape(-1, n_tasks_max * 4)
    return o
