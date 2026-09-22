"""2x2 sweep figures (4 environments, 8 models) and merged 4-env tables.

Writes docs/decide_tables.md and figures/stress_sweep_{sweep}_{metric}.png
plus energy aliases figures/stress_sweep_{sweep}.png.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(NB_DIR, "code", "env_3d"))

import export_manuscript as em

FTITLE = 22
FLABEL = 20
FTICK = 16
FLEGEND = 14
FSUP = 24

MARKERS = dict(em.MARKERS)
MARKERS["random"] = "o"

SWEEP_PLOT_METRICS = [
    ("coverage_mean", "coverage_std", "Coverage (%)", "coverage", True, False),
    ("stranded_mean", "stranded_std", "Stranded (%)", "stranded", False, False),
    ("reward_mean", "reward_std", "Reward", "reward", True, False),
    ("expired_mean", "expired_std", "Expired (%)", "expired", False, False),
    ("soc_mean", "soc_std", "Final SOC", "soc", True, False),
    ("steps_mean", "steps_std", "Steps", "steps", False, False),
    ("energy_wh_per_task_mean", "energy_wh_per_task_std", "Energy (Wh/task)", "energy", False, True),
    ("dist_per_task_m_mean", "dist_per_task_m_std", "Distance (m/task)", "distance", False, True),
    ("min_soc_mean", "min_soc_std", "Min SOC", "min_soc", True, False),
    ("below_floor_pct_mean", "below_floor_pct_std", "Below floor (%)", "below_floor", False, False),
    ("contention_pct_mean", "contention_pct_std", "Contention (%)", "contention", False, False),
    ("info_age_mean", "info_age_std", "Info age (steps)", "info_age", None, False),
    ("latency_mean_mean", "latency_mean_std", "Latency mean", "latency", False, False),
    ("latency_median_mean", "latency_median_std", "Latency median", "latency_median", False, False),
    ("blocked_moves_mean", "blocked_moves_std", "Blocked moves", "blocked", False, False),
]


def axes_grid(n):
    if n == 4:
        return 2, 2
    if n == 2:
        return 1, 2
    if n == 3:
        return 1, 3
    return 1, max(n, 1)


def suggest_place(sweep_key, slug):
    if slug == "energy":
        return "Results"
    if slug == "coverage" and sweep_key in {
            "task_density", "scalability", "deadline", "fault_tolerance"}:
        return "Results"
    if slug == "contention" and sweep_key == "coupled":
        return "Results"
    if slug == "expired" and sweep_key == "deadline":
        return "Results"
    if slug == "steps" and sweep_key in {"obstacles", "deadline"}:
        return "Optional"
    if slug in {"coverage", "steps", "contention", "soc"}:
        return "Optional"
    return "Annex"


def plot_sweep_metric_2x2(results, title, key, xlabel, style, mk, sk, ylab, slug, logy):
    envs = [e for e in em.ENV_ORDER if e in results]
    nrows, ncols = axes_grid(len(envs))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.4 * ncols, 6.2 * nrows), squeeze=False)
    for i, eid in enumerate(envs):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        sample = results[eid][em.POLS[0]]
        raw_x = [rec["value"] for rec in sample]
        if style == "categorical":
            xs = list(range(len(raw_x)))
            xticklabels = [em.fmt_level(key, v) for v in raw_x]
        else:
            xs = raw_x
            xticklabels = None
        for p in em.POLS:
            ys = [rec[mk] for rec in results[eid][p]]
            ye = [rec[sk] for rec in results[eid][p]]
            ax.errorbar(
                xs, ys, yerr=ye, marker=MARKERS[p], color=em.COLORS[p],
                label=em.LABELS[p], capsize=3, linewidth=2.0, markersize=8)
        ax.set_title(em.ENV_LABELS[eid], fontsize=FTITLE, pad=8)
        ax.set_xlabel(xlabel, fontsize=FLABEL)
        ax.set_ylabel(ylab, fontsize=FLABEL)
        ax.tick_params(labelsize=FTICK)
        if xticklabels is not None:
            ax.set_xticks(xs)
            ax.set_xticklabels(xticklabels, fontsize=FTICK)
        if mk.startswith(("coverage", "contention", "stranded", "expired", "below_floor")):
            ax.set_ylim(0, 105)
        if logy:
            ax.set_yscale("log")
    for j in range(len(envs), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=4, frameon=True,
        fontsize=FLEGEND, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{title}: {ylab}", fontsize=FSUP, y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    em.save_fig(fig, f"stress_sweep_{key}_{slug}.png")
    if slug == "energy":
        import shutil
        dest = os.path.join(em.FIG_DIR, f"stress_sweep_{key}.png")
        shutil.copyfile(
            os.path.join(em.FIG_DIR, f"stress_sweep_{key}_{slug}.png"), dest)
        print("copied", dest)


def merged_table(env_dict, mk, sk, higher, sweep_key):
    envs = [e for e in em.ENV_ORDER if e in env_dict]
    sample = env_dict[envs[0]][em.POLS[0]]
    header = ["Method", "Setting"] + [em.ENV_LABELS[e] for e in envs]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    nlev = len(sample)
    for i in range(nlev):
        lab = em.fmt_level(sweep_key, sample[i]["value"])
        recs = {e: {p: env_dict[e][p][i] for p in em.POLS} for e in envs}
        best = {}
        for e in envs:
            vals = [recs[e][p][mk] for p in em.POLS
                    if p != "random" and recs[e][p].get(mk) is not None]
            if higher is None or not vals:
                best[e] = None
            else:
                best[e] = max(vals) if higher else min(vals)
        for p in em.POLS:
            cells = [em.LABELS[p], lab]
            for e in envs:
                txt = em.pm(recs[e][p].get(mk), recs[e][p].get(sk))
                val = recs[e][p].get(mk)
                if (best[e] is not None and val is not None and p != "random"
                        and abs(val - best[e]) <= 1e-9):
                    txt = f"**{txt}**"
                cells.append(txt)
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def merged_base_table(base, mk, sk, higher):
    header = ["Method"] + [em.ENV_LABELS[e] for e in em.ENV_ORDER]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    best = {}
    for e in em.ENV_ORDER:
        vals = [base[e][p][mk] for p in em.POLS
                if p != "random" and base[e][p].get(mk) is not None]
        if higher is None or not vals:
            best[e] = None
        else:
            best[e] = max(vals) if higher else min(vals)
    for p in em.POLS:
        cells = [em.LABELS[p]]
        for e in em.ENV_ORDER:
            rec = base[e][p]
            txt = em.pm(rec.get(mk), rec.get(sk))
            val = rec.get(mk)
            if (best[e] is not None and val is not None and p != "random"
                    and abs(val - best[e]) <= 1e-9):
                txt = f"**{txt}**"
            cells.append(txt)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_decide_tables():
    os.makedirs(em.DOC_DIR, exist_ok=True)
    base = em.load("eval_base.json")["results"]
    lines = []
    a = lines.append
    a("# Decide tables and 2x2 sweep figures")
    a("")
    a("Candidate tables and figures for the manuscript. Each item is independent: copy it into Results, put it in the Annex, or skip it.")
    a("")
    a("Figure recipe: **2x2 subfigures** (one environment each), **8 models** on every panel, **shared legend at top center**, large axis / tick / title type. Random is included. Energy and distance use a log y-axis so HRL-H / DMPC / CBBA remain readable.")
    a("")
    a("Merged tables put the **four environments in columns** and the **eight methods in rows**. Values are mean +/- std over 10 seeds. Boldface is the best competitive method in that environment column (Random excluded).")
    a("")
    a("## Shortlist")
    a("")
    a("| ID | Item | Placement |")
    a("|---|---|---|")
    a("| S0 | Base energy, four environments (merged table) | Results |")
    a("| S1 | Base coverage, four environments (merged table) | Results |")
    a("| S2 | Base contention, four environments (merged table) | Results |")
    a("| S3 | Base steps, four environments (merged table) | Optional |")
    a("| F-energy | 2x2 energy figure for every sweep | Results |")
    a("| F-cov-density | 2x2 coverage, task density | Results |")
    a("| F-cov-scale | 2x2 coverage, scalability | Results |")
    a("| F-cov-deadline | 2x2 coverage, deadline | Results |")
    a("| F-cont-coupled | 2x2 contention, coupled tasks | Results |")
    a("| F-obs-energy | 2x2 energy, obstacle density (open + layered) | Results |")
    a("")
    a("The 15-metric 2x2 set is listed under each sweep. Use the energy 2x2 in Results and park the rest in the Annex, or skip a metric that a table already covers.")
    a("")
    a("## Base config, four environments merged")
    a("")
    a("**Environment**: all four layouts.")
    a("")
    a(f"**Config**: {em.STRESS_CFG}. Sweep knobs at defaults.")
    a("")
    for mk, sk, ylab, slug, higher, _logy in SWEEP_PLOT_METRICS:
        place = "Results" if slug in {"energy", "coverage", "contention"} else (
            "Optional" if slug in {"steps", "soc", "distance", "latency"} else "Annex")
        a(f"### Table B-{slug}. {ylab}")
        a("")
        a(f"**Placement**: {place}.")
        a("")
        a("**Environment**: `open`, `layered`, `open_nfz`, `layered_nfz`.")
        a("")
        a(f"**Config**: {em.STRESS_CFG}.")
        a("")
        a(merged_base_table(base, mk, sk, higher))
        a("")

    a("## Sweep figures (2x2) and merged tables")
    a("")
    a("Each sweep has one 2x2 PNG per metric (8 models, legend top center) and one merged 4-environment table per metric.")
    a("")
    a("| Sweep | Metric | Figure file | Placement |")
    a("|---|---|---|---|")
    fig_rows = []
    for fname, title, key, xlabel, style in em.SWEEPS:
        payload = em.load(fname)
        results = payload["results"]
        n_env = sum(1 for e in em.ENV_ORDER if e in results)
        layout = "2x2" if n_env == 4 else "1x2"
        for mk, sk, ylab, slug, higher, logy in SWEEP_PLOT_METRICS:
            figname = f"stress_sweep_{key}_{slug}.png"
            place = suggest_place(key, slug)
            a(f"| {title} | {ylab} | {figname} ({layout}) | {place} |")
            fig_rows.append((fname, title, key, xlabel, style, mk, sk, ylab, slug, higher, logy, place))
    a("")

    for fname, title, key, xlabel, style in em.SWEEPS:
        payload = em.load(fname)
        results = payload["results"]
        envs = [e for e in em.ENV_ORDER if e in results]
        env_ids = ", ".join(f"`{e}`" for e in envs)
        a(f"## {title}")
        a("")
        a(f"**Environments**: {env_ids}.")
        a("")
        a(f"**Config**: {em.STRESS_CFG}. {em.SWEEP_VARY[key]}")
        a("")
        a(f"**Figures**: 2x2 (or 1x2 if the sweep has two layouts), 8 models, shared legend at top center, large type. Files `figures/stress_sweep_{key}_<metric>.png`.")
        a("")
        for mk, sk, ylab, slug, higher, _logy in SWEEP_PLOT_METRICS:
            place = suggest_place(key, slug)
            a(f"### Table {key}-{slug}. {title}: {ylab}")
            a("")
            a(f"**Placement**: {place}.")
            a("")
            a(f"**Environment**: {env_ids}.")
            a("")
            a(f"**Config**: {em.STRESS_CFG}. {em.SWEEP_VARY[key]} Columns are the four layouts. Rows are the eight methods, repeated per setting.")
            a("")
            a(merged_table(results, mk, sk, higher, key))
            a("")

    path = os.path.join(em.DOC_DIR, "decide_tables.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)


def plot_all_figures():
    em.apply_pub_style()
    plt.rcParams.update({
        "axes.titlesize": FTITLE,
        "axes.labelsize": FLABEL,
        "xtick.labelsize": FTICK,
        "ytick.labelsize": FTICK,
        "legend.fontsize": FLEGEND,
        "figure.titlesize": FSUP,
        "font.size": FTICK,
    })
    for fname, title, key, xlabel, style in em.SWEEPS:
        payload = em.load(fname)
        results = payload["results"]
        for mk, sk, ylab, slug, _higher, logy in SWEEP_PLOT_METRICS:
            plot_sweep_metric_2x2(
                results, title, key, xlabel, style, mk, sk, ylab, slug, logy)


def main():
    write_decide_tables()
    plot_all_figures()


if __name__ == "__main__":
    main()
