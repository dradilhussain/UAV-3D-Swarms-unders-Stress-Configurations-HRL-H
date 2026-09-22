"""Export manuscript-ready tables, sweep figures, and Results/Annex markdown."""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(NB_DIR, "code", "env_3d"))
from eval_io import (
    apply_pub_style, FONT_TITLE, FONT_LABEL, FONT_TICK, FONT_LEGEND, FONT_SUPTITLE,
)

OUT_DIR = os.path.join(NB_DIR, "outputs")
FIG_DIR = os.path.join(NB_DIR, "figures")
DOC_DIR = os.path.join(NB_DIR, "docs")

ENV_ORDER = ["open", "layered", "open_nfz", "layered_nfz"]
ENV_LABELS = {
    "open": "Open volume",
    "layered": "Layered altitude bands",
    "open_nfz": "Open volume + no-fly zones",
    "layered_nfz": "Layered + no-fly zones",
}
POLS = ["random", "greedy", "cbba", "mappo", "maddpg", "qmix", "dmpc", "hrlh"]
POLS_PLOT = ["greedy", "cbba", "mappo", "maddpg", "qmix", "dmpc", "hrlh"]
LABELS = {
    "random": "Random",
    "greedy": "Greedy-Nearest",
    "cbba": "CBBA (SOC-aware)",
    "mappo": "MAPPO",
    "maddpg": "MADDPG",
    "qmix": "QMIX",
    "dmpc": "DMPC",
    "hrlh": "HRL-H (proposed)",
}
COLORS = {
    "random": "#d62728", "greedy": "#1f77b4", "cbba": "#2ca02c",
    "mappo": "#9467bd", "maddpg": "#ff7f0e", "qmix": "#17becf",
    "dmpc": "#7f7f7f", "hrlh": "#e377c2",
}
MARKERS = {
    "greedy": "s", "cbba": "^", "mappo": "D", "maddpg": "*",
    "qmix": "P", "dmpc": "x", "hrlh": "h",
}

BASE_METRICS = [
    ("coverage_mean", "coverage_std", "Coverage (%)", True),
    ("stranded_mean", "stranded_std", "Stranded (%)", False),
    ("reward_mean", "reward_std", "Reward", True),
    ("expired_mean", "expired_std", "Expired (%)", False),
    ("soc_mean", "soc_std", "Final SOC", True),
    ("steps_mean", "steps_std", "Steps", False),
    ("energy_wh_per_task_mean", "energy_wh_per_task_std", "Energy (Wh/task)", False),
    ("dist_per_task_m_mean", "dist_per_task_m_std", "Distance (m/task)", False),
    ("min_soc_mean", "min_soc_std", "Min SOC", True),
    ("below_floor_pct_mean", "below_floor_pct_std", "Below floor (%)", False),
    ("contention_pct_mean", "contention_pct_std", "Contention (%)", False),
    ("info_age_mean", "info_age_std", "Info age (steps)", None),
    ("latency_mean_mean", "latency_mean_std", "Latency mean", False),
    ("latency_median_mean", "latency_median_std", "Latency median", False),
    ("blocked_moves_mean", "blocked_moves_std", "Blocked moves", False),
]

ANNEX_METRICS = [
    ("coverage_mean", "coverage_std", "Coverage (%)", True),
    ("energy_wh_per_task_mean", "energy_wh_per_task_std", "Energy (Wh/task)", False),
    ("steps_mean", "steps_std", "Steps", False),
    ("contention_pct_mean", "contention_pct_std", "Contention (%)", False),
    ("soc_mean", "soc_std", "Final SOC", True),
    ("stranded_mean", "stranded_std", "Stranded (%)", False),
    ("latency_mean_mean", "latency_mean_std", "Latency", False),
    ("expired_mean", "expired_std", "Expired (%)", False),
    ("blocked_moves_mean", "blocked_moves_std", "Blocked", False),
]

SWEEPS = [
    ("eval_packet_loss.json", "Packet loss", "packet_loss",
     "Packet-loss probability", None),
    ("eval_comm_range.json", "Communication range", "comm_range",
     "Communication range (m)", None),
    ("eval_task_density.json", "Task density", "task_density",
     "Number of tasks", None),
    ("eval_fault_tolerance.json", "UAV dropout", "fault_tolerance",
     "Dropout fraction", None),
    ("eval_scalability.json", "Scalability", "scalability",
     "Number of UAVs", None),
    ("eval_deadline.json", "Task deadline", "deadline",
     "Deadline (steps)", "categorical"),
    ("eval_coupled.json", "Coupled tasks", "coupled",
     "Mean agents per task", None),
    ("eval_obstacles.json", "Obstacle density", "obstacles",
     "Number of no-fly zones", None),
]


def load(name):
    with open(os.path.join(OUT_DIR, name)) as f:
        return json.load(f)


def pm(mean, std):
    if mean is None:
        return "--"
    am = abs(mean)
    if am >= 100:
        nd = 1
    elif am >= 10:
        nd = 1
    elif am >= 1:
        nd = 2
    else:
        nd = 3
    if std is None:
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} +/- {std:.{nd}f}"


def best_mask(rows, key, higher, skip=("random",)):
    vals = []
    for p, rec in rows:
        if p in skip:
            vals.append(None)
        else:
            vals.append(rec.get(key))
    finite = [v for v in vals if v is not None]
    if not finite or higher is None:
        return [False] * len(rows)
    target = max(finite) if higher else min(finite)
    return [v is not None and abs(v - target) <= 1e-9 for v in vals]


def table_methods_as_rows(records_by_pol, metrics, pols=None):
    pols = pols or POLS
    rows = [(p, records_by_pol[p]) for p in pols]
    header = ["Method"] + [lab for _, _, lab, _ in metrics]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    bold = [best_mask(rows, mk, hib) for mk, _, _, hib in metrics]
    for i, (p, rec) in enumerate(rows):
        cells = [LABELS[p]]
        for j, (mk, sk, _, _) in enumerate(metrics):
            txt = pm(rec.get(mk), rec.get(sk))
            if bold[j][i]:
                txt = f"**{txt}**"
            cells.append(txt)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt_level(sweep, value):
    if sweep == "packet_loss":
        return f"{int(round(value * 100))}%"
    if sweep == "comm_range":
        return f"{int(value)} m"
    if sweep == "task_density":
        return f"{int(value)} tasks"
    if sweep == "fault_tolerance":
        return f"{int(round(value * 100))}% dropout"
    if sweep == "scalability":
        return f"{int(value)} UAVs"
    if sweep == "deadline":
        return "none" if int(value) == 0 else f"{int(value)} steps"
    if sweep == "coupled":
        return f"{value:g} agents/task"
    if sweep == "obstacles":
        return f"{int(value)} obstacles"
    return str(value)


def collect_sweep_rows(eid):
    out = []
    for fname, title, key, _xlabel, _style in SWEEPS:
        payload = load(fname)
        results = payload["results"]
        if eid not in results:
            continue
        nlev = len(results[eid][POLS_PLOT[0]])
        for i in range(nlev):
            val = results[eid][POLS_PLOT[0]][i]["value"]
            lab = fmt_level(key, val)
            for p in POLS:
                rec = results[eid][p][i]
                out.append((p, title, lab, rec))
    return out



ENV_LAYOUT = {
    "open": "open 3D volume (`scenario=volume`); tasks uniform in x, y, and altitude 5-60 m; 0 no-fly zones",
    "layered": "three discrete altitude bands (`scenario=layered`, `n_layers=3`) in a 400 m square; 0 no-fly zones",
    "open_nfz": "open 3D volume plus 4 cubic no-fly zones (`n_obstacles=4`, `obstacle_half=20` m)",
    "layered_nfz": "three altitude bands plus 4 cubic no-fly zones (`n_obstacles=4`, `obstacle_half=20` m)",
}

STRESS_CFG = (
    "stress preset: 8 UAVs, 8 single-agent tasks, 400 m area, altitude 5-60 m, "
    "comm range 12 m, packet loss 0.2, battery 18 Wh, SOC floor 0.25, "
    "`max_steps=500`, 10 random seeds"
)

SWEEP_VARY = {
    "packet_loss": "This table varies packet-loss probability (0%, 20%, 40%, 60%). Other knobs stay at the stress defaults.",
    "comm_range": "This table varies communication range (6, 12, 25, 50 m). Other knobs stay at the stress defaults.",
    "task_density": "This table varies task count (4, 8, 16, 32). Fleet size stays at 8 UAVs.",
    "fault_tolerance": "This table varies UAV dropout (0%, 25%, 50%). Remaining agents keep the stress radio and task count.",
    "scalability": "This table varies fleet size (4, 8, 16 UAVs) with a matching task count.",
    "deadline": "This table varies task deadline (none, 200 steps, 100 steps).",
    "coupled": "This table varies mean agents per task (1, 1.5, 2).",
    "obstacles": "This table varies no-fly-zone count (0, 2, 4 cuboids, half-width 20 m).",
}


def base_table_note(eid):
    return (
        f"**Environment** `{eid}`: {ENV_LAYOUT[eid]}.\n\n"
        f"**Config**: {STRESS_CFG}. All sweep knobs at defaults (packet loss 0.2, "
        "comm range 12 m, 1 agent/task, no deadline). MAPPO / MADDPG / QMIX are "
        "trained on layered obstacle-free flight and evaluated zero-shot. "
        "Rows are methods; columns are metrics. Values are mean +/- std."
    )


def annex_sweep_note(eid, key):
    return (
        f"**Environment** `{eid}`: {ENV_LAYOUT[eid]}.\n\n"
        f"**Config**: {STRESS_CFG}. {SWEEP_VARY[key]} "
        "Rows are methods, repeated per setting; columns are metrics. "
        "Boldface is the best competitive method within each setting (Random excluded)."
    )


def annex_table_one_sweep(eid, key, results_eid):
    nlev = len(results_eid[POLS_PLOT[0]])
    header = ["Method", "Setting"] + [lab for _, _, lab, _ in ANNEX_METRICS]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for i in range(nlev):
        val = results_eid[POLS_PLOT[0]][i]["value"]
        lab = fmt_level(key, val)
        items = [(p, results_eid[p][i]) for p in POLS]
        bold = [best_mask(items, mk, hib) for mk, _, _, hib in ANNEX_METRICS]
        for j, (p, rec) in enumerate(items):
            cells = [LABELS[p], lab]
            for k, (mk, sk, _, _) in enumerate(ANNEX_METRICS):
                txt = pm(rec.get(mk), rec.get(sk))
                if bold[k][j]:
                    txt = f"**{txt}**"
                cells.append(txt)
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def annex_table(eid):
    rows_raw = collect_sweep_rows(eid)
    header = ["Method", "Sweep", "Setting"] + [lab for _, _, lab, _ in ANNEX_METRICS]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    groups = []
    cur = None
    buf = []
    for item in rows_raw:
        key = (item[1], item[2])
        if cur is None:
            cur = key
        if key != cur:
            groups.append(buf)
            buf = []
            cur = key
        buf.append(item)
    if buf:
        groups.append(buf)
    for items in groups:
        pair = [(p, rec) for p, _, _, rec in items]
        bold = [best_mask(pair, mk, hib) for mk, _, _, hib in ANNEX_METRICS]
        for i, (p, sw, setting, rec) in enumerate(items):
            cells = [LABELS[p], sw, setting]
            for j, (mk, sk, _, _) in enumerate(ANNEX_METRICS):
                txt = pm(rec.get(mk), rec.get(sk))
                if bold[j][i]:
                    txt = f"**{txt}**"
                cells.append(txt)
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_fig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def plot_sweep(fname, title, key, xlabel, style, extra_metric=False):
    payload = load(fname)
    results = payload["results"]
    envs = [e for e in ENV_ORDER if e in results]
    metrics = [("energy_wh_per_task_mean", "energy_wh_per_task_std", "Energy (Wh/task)")]
    if extra_metric:
        metrics.append(("coverage_mean", "coverage_std", "Coverage (%)"))
    nrows = len(metrics)
    ncols = len(envs)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows), squeeze=False)
    for c, eid in enumerate(envs):
        sample = results[eid][POLS_PLOT[0]]
        raw_x = [r["value"] for r in sample]
        if style == "categorical":
            xs = list(range(len(raw_x)))
            xticklabels = [fmt_level(key, v) for v in raw_x]
        else:
            xs = raw_x
            xticklabels = None
        for r, (mk, sk, ylab) in enumerate(metrics):
            ax = axes[r][c]
            for p in POLS_PLOT:
                ys = [rec[mk] for rec in results[eid][p]]
                ye = [rec[sk] for rec in results[eid][p]]
                ax.errorbar(
                    xs, ys, yerr=ye, marker=MARKERS[p], color=COLORS[p],
                    label=LABELS[p], capsize=3, linewidth=1.8, markersize=7)
            ax.set_title(ENV_LABELS[eid] if r == 0 else "", fontsize=FONT_TITLE)
            ax.set_ylabel(ylab, fontsize=FONT_LABEL)
            if r == nrows - 1:
                ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
            if xticklabels is not None:
                ax.set_xticks(xs)
                ax.set_xticklabels(xticklabels, fontsize=FONT_TICK - 1)
            ax.tick_params(labelsize=FONT_TICK)
            if mk.startswith("coverage"):
                ax.set_ylim(0, 105)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=4, frameon=True,
        fontsize=FONT_LEGEND, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, fontsize=FONT_SUPTITLE, y=1.08 if nrows == 1 else 1.04)
    fig.tight_layout()
    save_fig(fig, f"stress_sweep_{key}.png")


def write_docs():
    os.makedirs(DOC_DIR, exist_ok=True)
    base = load("eval_base.json")["results"]
    hrlh = load("eval_hrlh.json")["results"]

    res = ["# Results and Analysis", ""]
    res.append(
        "Stress preset: 8 UAVs, 8 tasks, area 400 m, communication range 12 m, "
        "10 random seeds. Methods: Random, Greedy-Nearest, CBBA (SOC-aware), "
        "MAPPO, MADDPG, QMIX, DMPC, and HRL-H (proposed). Boldface marks the "
        "best competitive method in each column (Random excluded). Values are "
        "mean +/- std.")
    res += ["", "## Experimental Setup", ""]
    res.append(
        "Four layouts share the same fleet and task count. Open volume is free "
        "3D flight. Layered altitude bands restrict vertical motion. The two "
        "NFZ layouts add four cubic no-fly zones (half-width 20 m). MAPPO is "
        "trained for 600 episodes on layered obstacle-free flight and "
        "transferred zero-shot. HRL-H uses energy-first global matching "
        "(w_energy=6, w_rl=0.03) and re-plans every step until the last "
        "12 m of approach.")
    res += ["", "## Base-Config Comparison", ""]
    res.append(
        "HRL-H records the lowest energy on open (1.69 Wh/task), layered "
        "(1.66), and open_nfz (3.46), with 0% assignment contention on all "
        "four layouts. Coverage is 100% on the two obstacle-free layouts and "
        "98.8% on both NFZ layouts, tying DMPC. DMPC is the energy and "
        "makespan leader on layered_nfz (3.65 Wh/task, 74.8 steps vs HRL-H "
        "4.61 / 99.5), where two NFZ seeds send HRL-H on long detours. Greedy "
        "and CBBA finish the mission with ~100% contention and 1.4-1.8x the "
        "energy of HRL-H. MARL methods keep high coverage on easy layouts and "
        "pay 4-10x energy plus near-100% contention; QMIX drops to 95% "
        "coverage on open and MAPPO to 93.8% on layered.")
    res += ["",
            "Use Fig. 1 for the four layouts, Fig. 2 for example trajectories, "
            "Figs. 3-4 for per-environment bars, and Fig. 5 for the "
            "four-environment overview.", ""]
    for i, eid in enumerate(ENV_ORDER, start=1):
        res.append(f"### Table {i}. {ENV_LABELS[eid]} (base config)")
        res.append("")
        res.append(base_table_note(eid))
        res.append("")
        res.append(table_methods_as_rows(base[eid], BASE_METRICS))
        res.append("")
    res += ["## Robustness Sweeps", ""]
    res.append(
        "Across 102 sweep cells, HRL-H is the lowest-energy method with "
        "coverage >= 95% in 64 cells; DMPC wins the remaining 37, concentrated "
        "on layered_nfz, tight NFZ deadlines, and a few high-density NFZ "
        "settings.")
    res += ["",
            "Packet loss (0-60%) and communication range (6-50 m) leave HRL-H "
            "and DMPC unchanged because assignment is model-based. CBBA energy "
            "improves slightly as range grows (open: 2.52 to 2.27 Wh/task).",
            "",
            "Task density (4-32 tasks): HRL-H leads energy on open and layered "
            "at almost every density. MARL coverage collapses at 16-32 tasks "
            "(MAPPO ~20%). On NFZ layouts DMPC is slightly cheaper at 16-32 "
            "tasks while both keep ~96% coverage.",
            "",
            "UAV dropout (0/25/50%): HRL-H keeps 100% coverage on open and "
            "layered and 97.5-98.8% on NFZ. DMPC is cheaper on NFZ dropout.",
            "",
            "Scalability (4/8/16 UAVs with matched tasks): HRL-H is cheapest "
            "on 16-UAV open (1.27 Wh/task, 0% contention). MAPPO coverage "
            "falls to ~53-57% at 16 agents.",
            "",
            "Deadlines (none / 200 / 100 steps): HRL-H and DMPC finish open "
            "and layered in ~28 steps, so coverage stays 100% at the 100-step "
            "deadline. MARL expires a large fraction of tasks at 100 steps.",
            "",
            "Coupled tasks (1 / 1.5 / 2 agents per task): HRL-H still leads "
            "energy on open, layered, and open_nfz. Contention rises to ~100% "
            "at 2 agents/task because co-presence is required.",
            "",
            "Obstacles (0/2/4, open and layered): HRL-H leads at 0 and 2 "
            "obstacles. Four obstacles on layered is the layered_nfz layout, "
            "the DMPC energy win.",
            "", "## HRL-H Ablation", ""]
    res.append(
        "Replacing the MAPPO high-level tie-break with QMIX is a wash "
        "(open 1.68 vs 1.69 Wh/task). Removing the heuristic (HRL-H-NAV) "
        "reverts to MAPPO-level energy and 100% contention.")
    res += ["", "### Table 5. HRL-H ablation (all four environments)", ""]
    res.append(
        "**Environment**: all four layouts at the same stress base config as "
        "Tables 1-4.\n\n"
        "**Config**: 8 UAVs, 8 tasks, comm range 12 m, packet loss 0.2, 10 seeds. "
        "HRL-H uses MAPPO high-level scores; HRL-H-QMIX swaps in QMIX; "
        "HRL-H-NAV drops the heuristic executor and runs the raw RL argmax "
        "every step. Cells are energy (Wh/task) / coverage (%).")
    res += ["",
            "| Method | open E / Cov | layered E / Cov | open_nfz E / Cov | layered_nfz E / Cov |",
            "|---|---|---|---|---|"]
    alabels = {
        "hrlh": "HRL-H (MAPPO)",
        "hrlh_qmix": "HRL-H-QMIX",
        "hrlh_nav": "HRL-H-NAV",
    }
    for k in ["hrlh", "hrlh_qmix", "hrlh_nav"]:
        cells = [alabels[k]]
        for eid in ENV_ORDER:
            r = hrlh[eid][k]
            cells.append(
                f"{r['energy_wh_per_task_mean']:.2f} / {r['coverage_mean']:.1f}")
        res.append("| " + " | ".join(cells) + " |")
    res += ["", "## Where DMPC Still Leads", ""]
    res.append(
        "Layered + four no-fly zones is the hard case: seeds 3 and 5 send "
        "HRL-H on long detours (episode length 99.5 vs 74.8). Coverage remains "
        "tied at 98.8%, and HRL-H still has 0% contention vs DMPC 2.6%.")
    res += ["", "## Figure Order (Results)", ""]
    res.append("Place figures in this order. Files live under figures/.")
    res += [
        "",
        "| Fig. | File | Caption |",
        "|---|---|---|",
        "| 1 | stress_env_snapshot_2x2.png | Four evaluation layouts. |",
        "| 2a | stress_trajectories_open.png | Trajectories, open volume. |",
        "| 2b | stress_trajectories_layered.png | Trajectories, layered. |",
        "| 2c | stress_trajectories_open_nfz.png | Trajectories, open + NFZ. |",
        "| 2d | stress_trajectories_layered_nfz.png | Trajectories, layered + NFZ. |",
        "| 3a-d | stress_comparison_summary_{env}.png | Base bars: coverage, stranded, reward, SOC, steps. |",
        "| 4a-d | stress_extended_metrics_{env}.png | Base bars: energy, distance, min SOC, floor, contention, info age. |",
        "| 5 | stress_scenario_comparison.png | Four-environment overview. |",
        "| 6a-d | stress_hrlh_ablation_comparison_{env}.png | HRL-H vs NAV vs QMIX. |",
        "| 7 | stress_sweep_packet_loss.png | Energy vs packet loss. |",
        "| 8 | stress_sweep_comm_range.png | Energy vs communication range. |",
        "| 9 | stress_sweep_task_density.png | Energy and coverage vs task count. |",
        "| 10 | stress_sweep_fault_tolerance.png | Energy vs UAV dropout. |",
        "| 11 | stress_sweep_scalability.png | Energy and coverage vs fleet size. |",
        "| 12 | stress_sweep_deadline.png | Energy vs task deadline. |",
        "| 13 | stress_sweep_coupled.png | Energy vs mean agents per task. |",
        "| 14 | stress_sweep_obstacles.png | Energy vs number of no-fly zones. |",
        "",
        "## Figure Order (Annex)",
        "",
        "| Fig. | File | Caption |",
        "|---|---|---|",
        "| A1 | stress_env_snapshot_open.png | Open-volume snapshot. |",
        "| A2 | stress_env_snapshot_layered.png | Layered snapshot. |",
        "| A3 | stress_env_snapshot_open_nfz.png | Open + NFZ snapshot. |",
        "| A4 | stress_env_snapshot_layered_nfz.png | Layered + NFZ snapshot. |",
        "",
        "Annex tables (one table per environment and sweep) are in docs/manuscript_annex.md.",
        "",
    ]
    path = os.path.join(DOC_DIR, "manuscript_results.md")
    with open(path, "w") as f:
        f.write("\n".join(res) + "\n")
    print("wrote", path)

    anx = [
        "# Annex: Full Sweep Tables",
        "",
        "Each table is one environment and one sweep. The heading names the "
        "layout and the knob. The paragraph under the heading states the "
        "environment and the config. Rows are methods, repeated per setting; "
        "columns are metrics. Boldface is the best competitive method within "
        "each setting (Random excluded). Values are mean +/- std over 10 seeds. "
        "The obstacle-density sweep applies to open and layered.",
        "",
    ]
    tnum = 1
    for eid in ENV_ORDER:
        anx.append(f"## {ENV_LABELS[eid]}")
        anx.append("")
        for fname, title, key, _xlabel, _style in SWEEPS:
            payload = load(fname)
            results = payload["results"]
            if eid not in results:
                continue
            anx.append(f"### Table A{tnum}. {ENV_LABELS[eid]}: {title}")
            anx.append("")
            anx.append(annex_sweep_note(eid, key))
            anx.append("")
            anx.append(annex_table_one_sweep(eid, key, results[eid]))
            anx.append("")
            tnum += 1
    path = os.path.join(DOC_DIR, "manuscript_annex.md")
    with open(path, "w") as f:
        f.write("\n".join(anx) + "\n")
    print("wrote", path)


def main():
    apply_pub_style()
    extra = {"task_density", "scalability"}
    for fname, title, key, xlabel, style in SWEEPS:
        plot_sweep(
            fname, title, key, xlabel, style, extra_metric=(key in extra))
    write_docs()


if __name__ == "__main__":
    main()
