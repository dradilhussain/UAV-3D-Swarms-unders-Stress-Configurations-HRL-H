"""Rebuild figures and CSV tables from saved JSON (no model re-run).

Usage (from repo root):
    python code/env_3d/replot_from_outputs.py

Sweep results stay as JSON/CSV tables. Figures: env snapshots, per-env
trajectories / comparison / extended / HRL-H, and one 4-env scenario comparison.
"""
from __future__ import annotations

import json
import os
import sys

NB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(NB_DIR, "code", "env_3d"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from visualize_3d import (
    plot_env_3d, plot_environments_2x2, load_env_snapshot, EnvSnapshot,
    replot_environments_2x2_from_dir, _draw_obstacles,
)
from eval_io import (
    dump_tables, write_csv, write_json, apply_pub_style, panel_shape,
    figsize_for, hide_unused_axes, FONT_TITLE, FONT_LABEL, FONT_TICK,
    FONT_LEGEND, FONT_SUPTITLE, FONT_ANNOT,
)


OUT_DIR = os.path.join(NB_DIR, "outputs")
FIG_DIR = os.path.join(NB_DIR, "figures")
SNAP_DIR = os.path.join(OUT_DIR, "env_snapshots")
FIG_DPI = 200

ENV_ORDER = ["open", "layered", "open_nfz", "layered_nfz"]
ENV_LABELS = {
    "open": "Open volume",
    "layered": "Layered altitude bands",
    "open_nfz": "Open volume + no-fly zones",
    "layered_nfz": "Layered + no-fly zones",
    "volume": "Open volume",
}

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
POLICY_MARKERS = {"random": "o", "greedy": "s", "cbba": "^", "mappo": "D",
                  "maddpg": "*", "qmix": "P", "dmpc": "x", "hrlh": "h"}

ABLATION_KEYS = ["hrlh", "hrlh_nav", "hrlh_qmix"]
ABLATION_LABELS = {"hrlh": "HRL-H (MAPPO)",
                   "hrlh_nav": "HRL-H-NAV (no heuristic)",
                   "hrlh_qmix": "HRL-H-QMIX (QMIX high level)"}
ABLATION_COLORS = {"hrlh": POLICY_COLORS["hrlh"], "hrlh_nav": "#8c564b",
                   "hrlh_qmix": "#2ca02c"}


def _save(fig, name, dpi=FIG_DPI):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def _map_env_id(k):
    return "open" if k == "volume" else k


def load_mapped_base(path):
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        payload = json.load(f)
    results = payload.get("results", payload.get("ablation_results", {}))
    if not results:
        return {}
    sample = next(iter(results.values()))
    # env -> policy -> metrics
    if isinstance(sample, dict) and sample and "coverage_mean" not in sample:
        return {_map_env_id(k): v for k, v in results.items()}
    # policy -> metrics (legacy single env)
    env_id = _map_env_id(payload.get("environment") or payload.get("scenario")
                         or "layered")
    return {env_id: results}


def export_legacy_jsons():
    """Turn already-saved eval JSON (single-env or 2-scenario) into CSV copies."""
    sweeps = [
        "eval_packet_loss", "eval_comm_range", "eval_task_density",
        "eval_fault_tolerance", "eval_scalability", "eval_deadline",
        "eval_coupled", "eval_obstacles",
    ]
    for name in sweeps:
        path = os.path.join(OUT_DIR, f"{name}.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            payload = json.load(f)
        results = payload.get("results", {})
        if not results:
            continue
        sample = next(iter(results.values()))
        if isinstance(sample, dict) and sample and isinstance(
                next(iter(sample.values())), list):
            from eval_io import sweep_to_rows
            extra = {k: v for k, v in payload.items()
                     if not isinstance(v, (list, dict))}
            all_rows = []
            for eid, res in results.items():
                eid = _map_env_id(eid)
                rows = sweep_to_rows(res, {**extra, "environment": eid})
                all_rows.extend(rows)
                sub = os.path.join(OUT_DIR, "by_environment", eid)
                write_json({**{k: v for k, v in payload.items() if k != "results"},
                            "environment": eid, "results": res},
                           os.path.join(sub, f"{name}.json"))
                write_csv(rows, os.path.join(sub, f"{name}.csv"))
            write_csv(all_rows, os.path.join(OUT_DIR, f"{name}.csv"))
            print("csv (by env)", name)
            continue
        if isinstance(sample, list):
            env_id = _map_env_id(payload.get("scenario", "layered"))
            extra = {k: v for k, v in payload.items()
                     if not isinstance(v, (list, dict))}
            extra["environment"] = env_id
            from eval_io import sweep_to_rows
            rows = sweep_to_rows(results, extra)
            write_csv(rows, os.path.join(OUT_DIR, f"{name}.csv"))
            sub = os.path.join(OUT_DIR, "by_environment", env_id)
            write_json(payload, os.path.join(sub, f"{name}.json"))
            write_csv(rows, os.path.join(sub, f"{name}.csv"))
            print("csv (legacy)", name, "as", env_id)
            continue
        print("skip", name, type(sample))

    mapped = load_mapped_base(os.path.join(OUT_DIR, "eval_scenarios.json"))
    if not mapped:
        mapped = load_mapped_base(os.path.join(OUT_DIR, "eval_base.json"))
    if mapped:
        from eval_io import base_to_rows
        extra = {"config": "stress"}
        all_rows = []
        for eid, res in mapped.items():
            rows = base_to_rows(res, {**extra, "environment": eid})
            all_rows.extend(rows)
            sub = os.path.join(OUT_DIR, "by_environment", eid)
            write_json({"environment": eid, "results": res, **extra},
                       os.path.join(sub, "eval_scenarios.json"))
            write_csv(rows, os.path.join(sub, "eval_scenarios.csv"))
            write_json({"environment": eid, "results": res, **extra},
                       os.path.join(sub, "eval_base.json"))
            write_csv(rows, os.path.join(sub, "eval_base.csv"))
        write_csv(all_rows, os.path.join(OUT_DIR, "eval_scenarios.csv"))
        write_csv(all_rows, os.path.join(OUT_DIR, "eval_base.csv"))
        dump_tables(mapped, OUT_DIR, POLICY_ORDER, POLICY_LABELS,
                    title="Base-config comparison (saved results)")
        print("tables from eval_scenarios / eval_base")

    hrlh_mapped = load_mapped_base(os.path.join(OUT_DIR, "eval_hrlh.json"))
    if hrlh_mapped:
        from eval_io import base_to_rows
        all_rows = []
        for eid, abr in hrlh_mapped.items():
            rows = base_to_rows(abr, {"environment": eid})
            all_rows.extend(rows)
            sub = os.path.join(OUT_DIR, "by_environment", eid)
            write_csv(rows, os.path.join(sub, "eval_hrlh.csv"))
            write_json({"environment": eid, "results": abr},
                       os.path.join(sub, "eval_hrlh.json"))
        write_csv(all_rows, os.path.join(OUT_DIR, "eval_hrlh.csv"))
        print("csv eval_hrlh")


def plot_env_snapshots(dpi=FIG_DPI):
    if not os.path.isdir(SNAP_DIR):
        print("no snapshots at", SNAP_DIR)
        return
    fig = replot_environments_2x2_from_dir(
        SNAP_DIR, title="3D evaluation environments -- 2x2 (stress config)")
    _save(fig, "stress_env_snapshot_2x2.png", dpi=dpi)
    combined = os.path.join(SNAP_DIR, "all_environments.json")
    if os.path.isfile(combined):
        payload = load_env_snapshot(combined)
        snaps = payload.get("snapshots", {})
        for eid, snap in snaps.items():
            env = EnvSnapshot(snap)
            f = plt.figure(figsize=(9, 8))
            ax = f.add_subplot(111, projection="3d")
            plot_env_3d(env, f"{ENV_LABELS.get(eid, eid)} (stress)", ax=ax)
            f.tight_layout()
            _save(f, f"stress_env_snapshot_{eid}.png", dpi=dpi)


def plot_scenario_comparison(mapped, dpi=FIG_DPI):
    env_ids = [e for e in ENV_ORDER if e in mapped] or list(mapped)
    if not env_ids:
        return
    n_env = len(env_ids)
    fig, axes = plt.subplots(n_env, 3, figsize=(24, 5.4 * n_env))
    if n_env == 1:
        axes = np.array([axes])
    metrics = [("coverage_mean", "Coverage (%)", "coverage_std"),
               ("stranded_mean", "Stranded UAVs (%)", "stranded_std"),
               ("reward_mean", "Total reward", "reward_std")]
    width = 0.62
    for r, eid in enumerate(env_ids):
        recs = mapped[eid]
        names = [n for n in POLICY_ORDER if n in recs]
        x = np.arange(len(names))
        for c, (mk, label, sk) in enumerate(metrics):
            ax = axes[r, c]
            vals = [recs[name][mk] for name in names]
            errs = [recs[name][sk] for name in names]
            bars = ax.bar(x, vals, width, yerr=errs, capsize=5,
                          color=[POLICY_COLORS[n] for n in names],
                          edgecolor="k", linewidth=0.5)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=FONT_ANNOT)
            ax.set_xticks(x)
            ax.set_xticklabels([POLICY_LABELS[n] for n in names],
                               fontsize=FONT_TICK - 1, rotation=22, ha="right")
            ax.set_ylabel(label, fontsize=FONT_LABEL)
            ax.set_title(f"{ENV_LABELS.get(eid, eid)} -- {label}",
                         fontsize=FONT_TITLE)
            ax.tick_params(labelsize=FONT_TICK)
    fig.suptitle("3D environment comparison -- all 8 approaches (stress)",
                 fontsize=FONT_SUPTITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "stress_scenario_comparison.png", dpi=dpi)


def plot_bars_one_env(recs, metrics, fname, title, dpi=FIG_DPI):
    names = [n for n in POLICY_ORDER if n in recs]
    if not names:
        return
    nrows, ncols = panel_shape(len(metrics))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize_for(nrows, ncols))
    axes_flat = np.atleast_1d(axes).ravel()
    x = np.arange(len(names)); width = 0.68
    for ax, (mk, label, sk) in zip(axes_flat, metrics):
        vals = [recs[n].get(mk, 0.0) for n in names]
        errs = [recs[n].get(sk, 0.0) for n in names]
        ax.bar(x, vals, width, yerr=errs, capsize=5,
               color=[POLICY_COLORS[n] for n in names],
               edgecolor="k", linewidth=0.5)
        for b, v in zip(ax.patches, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=FONT_ANNOT)
        ax.set_xticks(x)
        ax.set_xticklabels([POLICY_LABELS[n] for n in names],
                           fontsize=FONT_TICK - 1, rotation=22, ha="right")
        ax.set_ylabel(label, fontsize=FONT_LABEL)
        ax.set_title(label, fontsize=FONT_TITLE)
        ax.tick_params(labelsize=FONT_TICK)
    hide_unused_axes(axes_flat, len(metrics))
    fig.suptitle(title, fontsize=FONT_SUPTITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, fname, dpi=dpi)


def plot_comparison_and_extended(mapped, dpi=FIG_DPI):
    core = [("coverage_mean", "Coverage (%)", "coverage_std"),
            ("stranded_mean", "Stranded UAVs (%)", "stranded_std"),
            ("reward_mean", "Total reward", "reward_std"),
            ("soc_mean", "Mean final SOC", "soc_std"),
            ("steps_mean", "Episode length (steps)", "steps_std")]
    ext = [
        ("energy_wh_per_task_mean", "Energy per task (Wh)", "energy_wh_per_task_std"),
        ("dist_per_task_m_mean", "Distance per task (m)", "dist_per_task_m_std"),
        ("min_soc_mean", "Min SOC reached", "min_soc_std"),
        ("below_floor_pct_mean", "Time below safe floor (%)", "below_floor_pct_std"),
        ("contention_pct_mean", "Assignment contention (% steps)", "contention_pct_std"),
        ("info_age_mean", "Mean info age (steps)", "info_age_std"),
    ]
    for eid, recs in mapped.items():
        tag = ENV_LABELS.get(eid, eid)
        plot_bars_one_env(recs, core, f"stress_comparison_summary_{eid}.png",
                          f"3D base-config comparison -- {tag} (stress)", dpi=dpi)
        plot_bars_one_env(recs, ext, f"stress_extended_metrics_{eid}.png",
                          f"3D extended operational metrics -- {tag} (stress)",
                          dpi=dpi)


def plot_hrlh_one_env(abr, eid, dpi=FIG_DPI):
    keys = [k for k in ABLATION_KEYS if k in abr]
    if not keys:
        return
    metrics = [("coverage_mean", "Coverage (%)", "coverage_std"),
               ("stranded_mean", "Stranded UAVs (%)", "stranded_std"),
               ("energy_wh_per_task_mean", "Energy per task (Wh)",
                "energy_wh_per_task_std"),
               ("steps_mean", "Episode length (steps)", "steps_std")]
    fig, axes = plt.subplots(2, 2, figsize=figsize_for(2, 2))
    x = np.arange(len(keys)); width = 0.55
    for ax, (mk, label, sk) in zip(axes.ravel(), metrics):
        vals = [abr[n].get(mk, 0.0) for n in keys]
        errs = [abr[n].get(sk, 0.0) for n in keys]
        ax.bar(x, vals, width, yerr=errs, capsize=5,
               color=[ABLATION_COLORS[n] for n in keys],
               edgecolor="k", linewidth=0.5)
        for b, v in zip(ax.patches, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=FONT_ANNOT)
        ax.set_xticks(x)
        ax.set_xticklabels([ABLATION_LABELS[n] for n in keys],
                           fontsize=FONT_TICK, rotation=12, ha="right")
        ax.set_ylabel(label, fontsize=FONT_LABEL)
        ax.set_title(label, fontsize=FONT_TITLE)
        ax.tick_params(labelsize=FONT_TICK)
    tag = ENV_LABELS.get(eid, eid)
    fig.suptitle(f"HRL-H ablation -- {tag} (stress)", fontsize=FONT_SUPTITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, f"stress_hrlh_ablation_comparison_{eid}.png", dpi=dpi)


def plot_hrlh_ablations(dpi=FIG_DPI):
    mapped = load_mapped_base(os.path.join(OUT_DIR, "eval_hrlh.json"))
    for eid, abr in mapped.items():
        plot_hrlh_one_env(abr, eid, dpi=dpi)


def plot_trajectories_from_json(dpi=FIG_DPI):
    """Replot 4x2 trajectory grids when per-env JSON includes path arrays."""
    candidates = []
    combined = os.path.join(OUT_DIR, "trajectories.json")
    if os.path.isfile(combined):
        with open(combined) as f:
            payload = json.load(f)
        cond = payload.get("conditions")
        if isinstance(cond, dict):
            for eid, pols in cond.items():
                candidates.append((_map_env_id(eid), pols,
                                   payload.get("seed", 0)))
        elif "policies" in payload:
            eid = _map_env_id(payload.get("environment")
                              or payload.get("scenario") or "layered")
            candidates.append((eid, payload["policies"],
                               payload.get("seed", 0)))
    for eid in ENV_ORDER:
        p = os.path.join(OUT_DIR, f"trajectories_{eid}.json")
        if os.path.isfile(p):
            with open(p) as f:
                payload = json.load(f)
            pols = payload.get("policies", {})
            if pols:
                candidates.append((eid, pols, payload.get("seed", 0)))
        p = os.path.join(OUT_DIR, "by_environment", eid, "trajectories.json")
        if os.path.isfile(p):
            with open(p) as f:
                payload = json.load(f)
            pols = payload.get("policies", {})
            if pols:
                candidates.append((eid, pols, payload.get("seed", 0)))

    seen = set()
    snaps = {}
    combined_snap = os.path.join(SNAP_DIR, "all_environments.json")
    if os.path.isfile(combined_snap):
        payload = load_env_snapshot(combined_snap)
        snaps = payload.get("snapshots", {})

    for eid, pols, _seed in candidates:
        if eid in seen:
            continue
        sample = next(iter(pols.values())) if pols else {}
        if not isinstance(sample, dict) or "trajectory" not in sample:
            continue
        seen.add(eid)
        snap = snaps.get(eid, {})
        fig = plt.figure(figsize=(26, 40))
        for idx, name in enumerate(POLICY_ORDER, start=1):
            if name not in pols:
                continue
            ax = fig.add_subplot(4, 2, idx, projection="3d")
            rec = pols[name]
            traj = np.asarray(rec["trajectory"])
            soc = np.asarray(rec.get("final_soc", rec.get("soc_history", [[1.0] * traj.shape[1]])[-1]))
            stranded = np.asarray(rec.get("stranded", [False] * traj.shape[1]), dtype=bool)
            if snap:
                env = EnvSnapshot(snap)
                ax.scatter(*env.base_pos, marker="s", s=140, c="black",
                           label="Base / depot", depthshade=False)
                active = env.task_active
                if active.any():
                    ax.scatter(env.task_pos[active, 0], env.task_pos[active, 1],
                               env.task_pos[active, 2], marker="*", s=130,
                               c="orange", edgecolors="k", label="Active task",
                               depthshade=False)
                if (~active).any():
                    ax.scatter(env.task_pos[~active, 0], env.task_pos[~active, 1],
                               env.task_pos[~active, 2], marker="x", s=50,
                               c="lightgray", label="Completed task",
                               depthshade=False)
                _draw_obstacles(ax, env.obstacles)
                L = env.cfg.area_size
                ax.set_zlim(0, env.cfg.altitude_max * 1.1)
            else:
                L = float(np.nanmax(traj[:, :, :2])) if traj.size else 400.0
            xx, yy = np.meshgrid([0, L], [0, L])
            ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.05, color="gray")
            colors = plt.cm.RdYlGn(np.clip(soc, 0, 1))
            n = traj.shape[1]
            for i in range(n):
                ax.plot(traj[:, i, 0], traj[:, i, 1], traj[:, i, 2], "-",
                        color=colors[i], alpha=0.85, linewidth=1.4)
                ax.scatter(*traj[0, i], marker="o", s=35, c="gray",
                           depthshade=False)
                marker_end = "X" if (i < len(stranded) and stranded[i]) else "^"
                ax.scatter(*traj[-1, i], marker=marker_end, s=100,
                           c=[colors[i]], edgecolors="k", depthshade=False)
            ax.set_xlabel("x (m)", fontsize=FONT_LABEL)
            ax.set_ylabel("y (m)", fontsize=FONT_LABEL)
            ax.set_zlabel("altitude z (m)", fontsize=FONT_LABEL)
            ax.tick_params(labelsize=FONT_TICK)
            ax.set_title(POLICY_LABELS.get(name, name), fontsize=FONT_TITLE)
            ax.legend(loc="upper left", fontsize=FONT_LEGEND)
        fig.suptitle(
            f"3D trajectories ({ENV_LABELS.get(eid, eid)}) -- all 8 approaches (stress)\n"
            "trail color = final SOC (red = nearly stranded, green = healthy)",
            fontsize=FONT_SUPTITLE)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        _save(fig, f"stress_trajectories_{eid}.png", dpi=dpi)


def main():
    apply_pub_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    export_legacy_jsons()
    plot_env_snapshots(dpi=FIG_DPI)
    mapped = load_mapped_base(os.path.join(OUT_DIR, "eval_scenarios.json"))
    if not mapped:
        mapped = load_mapped_base(os.path.join(OUT_DIR, "eval_base.json"))
    if mapped:
        plot_scenario_comparison(mapped, dpi=FIG_DPI)
        plot_comparison_and_extended(mapped, dpi=FIG_DPI)
    plot_hrlh_ablations(dpi=FIG_DPI)
    plot_trajectories_from_json(dpi=FIG_DPI)
    print("done (no sweep plots)")


if __name__ == "__main__":
    main()
