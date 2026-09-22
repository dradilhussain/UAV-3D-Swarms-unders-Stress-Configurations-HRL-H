import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from uav_swarm_env_3d import UAVSwarmEnv3D, EnvConfig3D


FONT_TITLE = 18
FONT_LABEL = 16
FONT_TICK = 13
FONT_LEGEND = 13
FONT_ANNOT = 12
FONT_SUPTITLE = 20


def _as_array(x, dtype=float):
    return np.asarray(x, dtype=dtype)


class EnvSnapshot:
    """Lightweight stand-in for UAVSwarmEnv3D used only for (re)plotting."""

    def __init__(self, data):
        cfg = data.get("cfg", {})
        if isinstance(cfg, EnvConfig3D):
            self.cfg = cfg
        else:
            self.cfg = EnvConfig3D(**{k: v for k, v in cfg.items()
                                      if k in EnvConfig3D.__dataclass_fields__})
        self.base_pos = _as_array(data["base_pos"])
        self.pos = _as_array(data["pos"])
        self.soc = _as_array(data["soc"])
        self.stranded = _as_array(data["stranded"], dtype=bool)
        self.task_pos = _as_array(data["task_pos"])
        self.task_active = _as_array(data["task_active"], dtype=bool)
        self.task_completed = _as_array(data.get("task_completed", ~self.task_active), dtype=bool)
        obs = data.get("obstacles", [])
        self.obstacles = _as_array(obs) if len(obs) else np.zeros((0, 6), dtype=np.float32)
        self.n = int(self.pos.shape[0])
        self.n_tasks = int(self.task_pos.shape[0])
        self.env_id = data.get("env_id", "")
        self.label = data.get("label", self.env_id)


def collect_env_snapshot(env, env_id="", label=""):
    cfg = env.cfg
    return {
        "env_id": env_id,
        "label": label or env_id,
        "scenario": cfg.scenario,
        "obstacles_enabled": bool(cfg.obstacles_enabled),
        "n_obstacles": int(cfg.n_obstacles),
        "n_agents": int(cfg.n_agents),
        "n_tasks": int(cfg.n_tasks),
        "area_size": float(cfg.area_size),
        "altitude_min": float(cfg.altitude_min),
        "altitude_max": float(cfg.altitude_max),
        "comm_range": float(cfg.comm_range),
        "seed": cfg.seed,
        "cfg": {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in cfg.__dict__.items()},
        "base_pos": env.base_pos.tolist(),
        "pos": env.pos.tolist(),
        "soc": env.soc.tolist(),
        "stranded": env.stranded.astype(bool).tolist(),
        "task_pos": env.task_pos.tolist(),
        "task_active": env.task_active.astype(bool).tolist(),
        "task_completed": env.task_completed.astype(bool).tolist(),
        "obstacles": np.asarray(getattr(env, "obstacles", [])).tolist(),
    }


def save_env_snapshot(snapshot, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=1)


def load_env_snapshot(path):
    with open(path) as f:
        return json.load(f)


def _draw_obstacles(ax, obstacles, alpha=0.18):
    if obstacles is None or len(obstacles) == 0:
        return
    for i, o in enumerate(obstacles):
        x0, y0, z0, x1, y1, z1 = [float(v) for v in o]
        corners = [
            [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
            [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        ]
        for x in (x0, x1):
            corners.append([(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)])
        for y in (y0, y1):
            corners.append([(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)])
        ax.add_collection3d(Poly3DCollection(
            corners, alpha=alpha, facecolor="#b30000", edgecolor="#7a0000",
            linewidths=0.6))


def _legend_handles(has_obstacles=False, has_completed=True, has_stranded=True):
    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="black",
               markeredgecolor="k", markersize=12, linestyle="None",
               label="Base / depot"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#4caf50",
               markeredgecolor="k", markersize=12, linestyle="None",
               label="UAV (color = SOC, green=full)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="orange",
               markeredgecolor="k", markersize=16, linestyle="None",
               label="Active task"),
    ]
    if has_completed:
        handles.append(Line2D([0], [0], marker="x", color="gray",
                              markersize=10, linestyle="None",
                              label="Completed / inactive task"))
    if has_stranded:
        handles.append(Line2D([0], [0], marker="X", color="w",
                              markerfacecolor="#c62828", markeredgecolor="k",
                              markersize=12, linestyle="None",
                              label="Stranded UAV"))
    if has_obstacles:
        handles.append(Patch(facecolor="#b30000", edgecolor="#7a0000", alpha=0.35,
                             label="No-fly zone (cuboid)"))
    return handles


def style_3d_ax(ax, env, title):
    ax.set_xlabel("x (m)", fontsize=FONT_LABEL, labelpad=8)
    ax.set_ylabel("y (m)", fontsize=FONT_LABEL, labelpad=8)
    ax.set_zlabel("altitude z (m)", fontsize=FONT_LABEL, labelpad=8)
    ax.tick_params(labelsize=FONT_TICK)
    ax.set_zlim(0, env.cfg.altitude_max * 1.1)
    L = env.cfg.area_size
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_title(title, fontsize=FONT_TITLE, pad=10)
    try:
        ax.xaxis.set_pane_color((1, 1, 1, 0.0))
        ax.yaxis.set_pane_color((1, 1, 1, 0.0))
        ax.zaxis.set_pane_color((1, 1, 1, 0.05))
    except Exception:
        pass


def plot_env_3d(env, title, ax=None, show_legend=True, agent_labels=True):
    own_ax = ax is None
    if own_ax:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")

    ax.scatter(*env.base_pos, marker="s", s=180, c="black",
               label="Base / depot", depthshade=False)

    active = np.asarray(env.task_active, dtype=bool)
    if active.any():
        ax.scatter(env.task_pos[active, 0], env.task_pos[active, 1], env.task_pos[active, 2],
                   marker="*", s=180, c="orange", edgecolors="k",
                   label="Active task", depthshade=False)
    if (~active).any():
        ax.scatter(env.task_pos[~active, 0], env.task_pos[~active, 1], env.task_pos[~active, 2],
                   marker="x", s=80, c="lightgray",
                   label="Completed / inactive task", depthshade=False)

    colors = plt.cm.RdYlGn(np.clip(env.soc, 0, 1))
    n = int(env.pos.shape[0])
    for i in range(n):
        marker = "X" if env.stranded[i] else "^"
        ax.scatter(*env.pos[i], marker=marker, s=150, c=[colors[i]],
                   edgecolors="k", depthshade=False)
        if agent_labels:
            ax.text(env.pos[i, 0], env.pos[i, 1], env.pos[i, 2],
                    f"  {i}", fontsize=FONT_ANNOT)

    L = env.cfg.area_size
    xx, yy = np.meshgrid([0, L], [0, L])
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.06, color="gray")

    obs = getattr(env, "obstacles", None)
    _draw_obstacles(ax, obs)

    style_3d_ax(ax, env, title)
    if show_legend:
        has_obs = obs is not None and len(obs) > 0
        ax.legend(handles=_legend_handles(has_obstacles=has_obs,
                                          has_completed=(~active).any(),
                                          has_stranded=bool(np.any(env.stranded))),
                  loc="upper left", fontsize=FONT_LEGEND, framealpha=0.92)
    return ax


def plot_environments_2x2(snapshots, title="3D environments (2x2)",
                          figsize=(18, 16)):
    """Plot four environment snapshots in a 2x2 grid with a shared legend."""
    fig = plt.figure(figsize=figsize)
    axes = []
    any_obs = False
    any_completed = False
    any_stranded = False
    for i, snap in enumerate(snapshots):
        env = snap if isinstance(snap, EnvSnapshot) else EnvSnapshot(snap)
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        label = getattr(env, "label", None) or (
            snap.get("label", "") if isinstance(snap, dict) else "")
        nfz = "yes" if len(env.obstacles) else "no"
        panel = (f"{label}\n"
                 f"tasks={env.n_tasks}, UAVs={env.n}, "
                 f"scenario={env.cfg.scenario}, NFZ={nfz}")
        plot_env_3d(env, panel, ax=ax, show_legend=False, agent_labels=True)
        axes.append(ax)
        any_obs = any_obs or len(env.obstacles) > 0
        any_completed = any_completed or (not env.task_active.all())
        any_stranded = any_stranded or bool(np.any(env.stranded))

    handles = _legend_handles(has_obstacles=any_obs,
                              has_completed=any_completed,
                              has_stranded=any_stranded)
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)),
               fontsize=FONT_LEGEND, frameon=True, framealpha=0.95,
               bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(title, fontsize=FONT_SUPTITLE, y=0.98)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    return fig


def plot_env_from_snapshot(snapshot, title=None, ax=None, show_legend=True):
    env = snapshot if isinstance(snapshot, EnvSnapshot) else EnvSnapshot(snapshot)
    return plot_env_3d(env, title or env.label, ax=ax, show_legend=show_legend)


def replot_environments_2x2_from_dir(snap_dir, title=None, figsize=(18, 16)):
    """Rebuild the 2x2 environment figure from saved snapshot JSON files."""
    order = ["open", "layered", "open_nfz", "layered_nfz"]
    combined = os.path.join(snap_dir, "all_environments.json")
    snaps = []
    if os.path.isfile(combined):
        payload = load_env_snapshot(combined)
        by_id = payload.get("snapshots", payload)
        for eid in order:
            if eid in by_id:
                snaps.append(by_id[eid])
        if not snaps:
            snaps = list(by_id.values()) if isinstance(by_id, dict) else []
    else:
        for eid in order:
            p = os.path.join(snap_dir, f"{eid}.json")
            if os.path.isfile(p):
                snaps.append(load_env_snapshot(p))
    if not snaps:
        raise FileNotFoundError(f"no environment snapshots in {snap_dir}")
    return plot_environments_2x2(snaps, title=title or "3D environments (2x2)",
                                 figsize=figsize)


if __name__ == "__main__":
    specs = [
        ("open", "Open volume", "volume", False, 0),
        ("layered", "Layered altitude bands", "layered", False, 0),
        ("open_nfz", "Open volume + no-fly zones", "volume", True, 4),
        ("layered_nfz", "Layered + no-fly zones", "layered", True, 4),
    ]
    snaps = []
    for eid, label, scenario, obs_on, n_obs in specs:
        cfg = EnvConfig3D(n_agents=8, n_tasks=8, area_size=400.0,
                          altitude_min=5.0, altitude_max=60.0,
                          scenario=scenario, n_layers=3, comm_range=12.0,
                          battery_capacity_wh=18.0, soc_min_safe=0.25,
                          max_steps=500, seed=3,
                          obstacles_enabled=obs_on, n_obstacles=n_obs,
                          obstacle_half=20.0)
        env = UAVSwarmEnv3D(cfg)
        env.reset()
        snaps.append(collect_env_snapshot(env, eid, label))
    fig = plot_environments_2x2(snaps)
    fig.savefig("env_3d_2x2.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved env_3d_2x2.png")
