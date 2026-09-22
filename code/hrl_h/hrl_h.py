"""
HRL-H -- Hierarchical RL + Heuristic Navigation (the proposed model).

Two-level architecture that decouples *what to do* from *how to get there*:

  High level (RL)   : a learned policy (MAPPO or QMIX) produces a per-agent
                      preference over active tasks (a discrete allocation).
  Low level (heuristic) : a deterministic, model-based executor converts that
                      allocation into the actual per-step env action:

                      * commit-and-fly: an agent keeps flying to its assigned
                        task until completion -- it never re-decides every step,
                        which removes the wandering of a raw neural policy;
                      * battery safety: if an agent can't reach its task and
                        return above the safe floor (checked with the known
                        dynamics model), it is re-allocated or sent home before
                        it strands;
                      * deconfliction: if two agents prefer the same task, only
                        the one with the smallest predicted energy cost keeps it
                        and the other re-prefers -- CBBA-style consensus on top
                        of the RL scores.

The RL policy is queried *only at decision points* (agent free / just finished a
task), so the neural part solves the easy "which task" problem while the
heuristic guarantees energy-efficient, short, safe trajectories.

Ablations supported by `with_navigation`:
  * with_navigation=True  -> full HRL-H (proposed model)
  * with_navigation=False -> HRL-H-NAV: the same high-level RL is queried every
                             step and its raw argmax is executed directly
                             (no commit-and-fly, no battery safety, no
                             deconfliction). Isolates the heuristic layer.
The high-level *algorithm* is swapped by passing a different `prefer_fn`
(MAPPO actor for HRL-H, QMIX q-network for HRL-H-QMIX).
"""

import numpy as np
from model_based.dmpc import _predict_task


class HRLHExecutor:
    def __init__(self, prefer_fn, horizon=None, min_soc_margin=0.0):
        """prefer_fn(env) -> (n_agents, n_tasks_max) preference scores, higher =
        better. n_tasks_max may exceed env.n_tasks (zero-padded); only the first
        env.n_tasks columns correspond to real active tasks."""
        self.prefer_fn = prefer_fn
        self.horizon = horizon
        self.min_soc_margin = min_soc_margin
        # Energy-first allocation: model-based cost dominates, RL is a mild
        # tie-break so the high-level policy still participates.
        self.w_energy = 6.0
        self.w_time = 1.0
        self.w_rl = 0.03
        self.w_deadline = 0.8
        self.assign = np.full(0, -1, dtype=int)   # per-agent assigned task
        self.returning = np.zeros(0, dtype=bool)

    # ------------------------------------------------------------------- act
    def __call__(self, env):
        # episode-boundary detection: fresh episode -> clear per-episode state
        if env._t == 0:
            self.assign = np.full(env.n, -1, dtype=int)
            self.returning = np.zeros(env.n, dtype=bool)

        cfg = env.cfg
        n = env.n
        actions = np.zeros(n, dtype=int)
        if len(self.assign) != n:   # agent count changed mid-sweep
            self.assign = np.full(n, -1, dtype=int)
            self.returning = np.zeros(n, dtype=bool)

        active = ~env.stranded
        n_active = int(env.task_active.sum())
        scores = self.prefer_fn(env) if active.any() and n_active > 0 else None

        need = np.zeros(n, dtype=bool)   # agents that need a (new) decision
        for i in range(n):
            if not active[i]:
                actions[i] = env.HOLD
                self.assign[i] = -1
                self.returning[i] = False
                continue

            # 1) low battery: never risk stranding -- go home
            if env.soc[i] < cfg.soc_min_safe - self.min_soc_margin:
                actions[i] = env.RETURN_BASE
                self.returning[i] = True
                self.assign[i] = -1
                continue

            # 2) returning: keep returning until home, then re-allocate
            if self.returning[i]:
                if np.linalg.norm(env.pos[i] - env.base_pos) < 1.0:
                    self.returning[i] = False
                    self.assign[i] = -1
                    need[i] = True
                else:
                    actions[i] = env.RETURN_BASE
                continue

            # 3) assigned task still valid -> commit-and-fly (env completes
            #    the task when the agent acts on it within 1.0 m)
            k = self.assign[i]
            # Commit only on final approach. Mid-mission re-planning lets
            # agents switch off a contended task the way DMPC does.
            if (k >= 0 and k < env.n_tasks and env.task_active[k]
                    and np.linalg.norm(env.pos[i] - env.task_pos[k]) < 12.0):
                actions[i] = int(k)
                continue
            # task completed/removed (or nothing assigned) -> re-decide
            self.assign[i] = -1
            need[i] = True

        # 4) batch decision for every agent that needs a task: greedy auction
        #    over the high-level RL preference, filtered by battery feasibility
        #    (safe round trip, checked with the known dynamics model) and by
        #    deconfliction with per-task capacity (a task requiring R[k] agents
        #    keeps up to R[k] agents, so coupled tasks get enough contributors).
        #    Global greedy matching on raw energy (+ mild RL tie-break) assigns
        #    the best remaining (agent, task) pair until every free UAV is set.
        if need.any() and scores is not None:
            idx = np.where(need & active)[0]
            est = np.full((n, env.n_tasks), np.inf)
            nsteps = np.full((n, env.n_tasks), cfg.max_steps, dtype=np.float64)
            safe = np.zeros((n, env.n_tasks), dtype=bool)
            for i in idx:
                for k in range(env.n_tasks):
                    if not env.task_active[k]:
                        continue
                    e, st, s = _predict_task(env, i, k, self.horizon)
                    est[i, k] = e
                    nsteps[i, k] = st
                    safe[i, k] = s
            taken = np.zeros(env.n_tasks, dtype=int)   # agents currently assigned per task
            required = np.array(env.task_required, dtype=int)
            for i in range(n):
                if self.assign[i] >= 0:
                    taken[self.assign[i]] += 1
            # Raw (not per-agent-normalized) costs so scores are comparable
            # across UAVs. RL is a bounded tie-break; energy still dominates.
            combined = np.full((n, env.n_tasks), -np.inf)
            max_steps = max(cfg.max_steps, 1)
            for i in idx:
                finite = env.task_active & np.isfinite(est[i])
                if not finite.any():
                    continue
                rl = np.asarray(scores[i, :env.n_tasks], dtype=np.float64)
                rl_n = np.zeros(env.n_tasks, dtype=np.float64)
                if np.isfinite(rl[finite]).any():
                    rmin = float(np.nanmin(rl[finite]))
                    rmax = float(np.nanmax(rl[finite]))
                    rl_n[finite] = (rl[finite] - rmin) / (rmax - rmin + 1e-12)
                urgent = np.zeros(env.n_tasks, dtype=np.float64)
                if cfg.task_deadline_steps > 0:
                    remaining_dl = cfg.task_deadline_steps - (
                        env._t - env.task_spawn_time[:env.n_tasks])
                    urgent = np.clip(
                        1.0 - remaining_dl / max(cfg.task_deadline_steps, 1),
                        0.0, 1.0)
                combined[i, finite] = (
                    -self.w_energy * est[i, finite]
                    - self.w_time * nsteps[i, finite] / max_steps
                    + self.w_rl * rl_n[finite]
                    + self.w_deadline * urgent[finite])
            # Global auction (same structure as DMPC consensus, but without
            # comm-range / packet-loss). Re-run every step except final
            # approach so assignments can change after completions.
            remaining = set(int(i) for i in idx)
            while remaining:
                best_i, best_k, best_s = -1, -1, -np.inf
                for i in remaining:
                    avail = env.task_active & (taken < required) & np.isfinite(est[i])
                    if (avail & safe[i]).any():
                        avail = avail & safe[i]
                    if not avail.any():
                        continue
                    ks = np.where(avail)[0]
                    k = int(ks[np.argmax(combined[i, ks])])
                    s = float(combined[i, k])
                    if s > best_s:
                        best_s, best_i, best_k = s, i, k
                if best_i < 0:
                    break
                self.assign[best_i] = best_k
                actions[best_i] = int(best_k)
                taken[best_k] += 1
                remaining.remove(best_i)
            for i in remaining:
                actions[i] = (env.RETURN_BASE
                              if env.soc[i] < cfg.soc_min_safe else env.HOLD)
        return {env.agents[i]: int(actions[i]) for i in range(n)}


class HRLH:
    """Callable policy facade: prefer_fn + navigation flag -> action dict.

    with_navigation=True  -> full HRL-H: the heuristic executor turns the RL
                             allocation into energy-safe, deconflicted,
                             commit-and-fly trajectories (proposed model).
    with_navigation=False -> HRL-H-NAV ablation: the heuristic navigation layer
                             is removed entirely and the raw high-level RL
                             policy acts directly (argmax over its full action
                             space, including return/hold) every step.
    The high-level *algorithm* is swapped by passing a different `prefer_fn`
    (MAPPO actor for HRL-H, QMIX q-network for HRL-H-QMIX)."""

    def __init__(self, prefer_fn, with_navigation=True, horizon=None,
                 raw_policy_fn=None):
        self.prefer_fn = prefer_fn
        self.with_navigation = with_navigation
        self.horizon = horizon
        self._exec = HRLHExecutor(prefer_fn, horizon=horizon) if with_navigation else None
        self._raw = raw_policy_fn

    def __call__(self, env):
        if self.with_navigation:
            return self._exec(env)
        # ablation: the raw RL policy acts directly, no heuristic layer
        if self._raw is not None:
            return self._raw(env)
        # fallback: raw argmax over the prefer_fn scores only (task indices)
        n = env.n
        actions = {}
        if env.task_active.sum() == 0:
            for i, a in enumerate(env.agents):
                actions[a] = env.HOLD
            return actions
        scores = self.prefer_fn(env)
        for i, a in enumerate(env.agents):
            if env.stranded[i]:
                actions[a] = env.HOLD
                continue
            s = scores[i, :env.n_tasks].copy()
            s[~env.task_active] = -np.inf
            k = int(np.argmax(s))
            actions[a] = int(k) if env.task_active[k] else env.HOLD
        return actions
