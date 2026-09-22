"""Shared helpers for the HRL-H figure notebooks (data loading, style, scoring, saving)."""
import json, pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
DATA, FIG = ROOT / "data", ROOT / "figures"
FIG.mkdir(exist_ok=True)
DPI = 600                                            # all figures are saved at 600 dpi

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.6, "font.size": 8,
})

ENVS = ["open", "layered", "open_nfz", "layered_nfz"]
ENVN = {"open": "Open volume", "layered": "Layered", "open_nfz": "Open + NFZ", "layered_nfz": "Layered + NFZ"}
METH = ["random", "greedy", "cbba", "mappo", "maddpg", "qmix", "dmpc", "hrlh"]
MN = {"random": "Random", "greedy": "Greedy-Nearest", "cbba": "CBBA", "mappo": "MAPPO", "maddpg": "MADDPG",
      "qmix": "QMIX", "dmpc": "DMPC", "hrlh": "HRL-H (proposed)"}
LAB = {"random": "Random", "greedy": "Greedy", "cbba": "CBBA", "mappo": "MAPPO", "maddpg": "MADDPG",
       "qmix": "QMIX", "dmpc": "DMPC", "hrlh": "HRL-H"}
COL = {"random": "#7f7f7f", "greedy": "#2ca02c", "cbba": "#ff7f0e", "mappo": "#9467bd",
       "maddpg": "#8c564b", "qmix": "#e377c2", "dmpc": "#1f77b4", "hrlh": "#d62728"}

SWEEP_METH = [m for m in METH if m != "random"]   # seven non-trivial methods; HRL-H last


def _load(name):
    with open(DATA / name) as f:
        return json.load(f)


BASE = _load("eval_scenarios.json")["results"]          # base configuration, 8 methods x 4 environments
HRLH = _load("eval_hrlh.json")["results"]               # ablation variants x 4 environments
SW = {}                                                 # stress sweeps: SW[key]['levels'], SW[key]['res'][env][method][level]
for key, fname in [("packet_loss", "eval_packet_loss.json"), ("comm_range", "eval_comm_range.json"),
                   ("task_density", "eval_task_density.json"), ("scalability", "eval_scalability.json"),
                   ("fault", "eval_fault_tolerance.json"), ("deadline", "eval_deadline.json"),
                   ("coupled", "eval_coupled.json"), ("obstacles", "eval_obstacles.json")]:
    d = _load(fname)
    SW[key] = {"levels": d.get("levels") or d.get("profiles"), "res": d["results"]}
PS = _load("perseed_base.json")                         # per-seed base runs (10 seeds), PS[env][method][seed]


def bm(e, m, k, src=None):
    """mean of metric k for environment e / method m (default: base configuration)."""
    return (src or BASE)[e][m][k + "_mean"]


def bs(e, m, k, src=None):
    """standard deviation (over seeds) of metric k."""
    return (src or BASE)[e][m][k + "_std"]


def rank_score(values, higher_is_better):
    """Rank-based goodness in [0,1] (1 = best). Ties share the average rank; NaN stays NaN.
    A column in which all values are equal returns NaN (drawn neutral)."""
    v = np.asarray(values, float)
    out = np.full(len(v), np.nan)
    ok = ~np.isnan(v)
    if ok.sum() < 2 or np.ptp(v[ok]) == 0:
        return out
    s = np.round(v[ok] if higher_is_better else -v[ok], 8)
    order = np.argsort(np.argsort(s))
    sc = np.zeros(len(s))
    for u in np.unique(s):
        idx = np.where(s == u)[0]
        sc[idx] = order[idx].mean()
    out[ok] = sc / (len(s) - 1)
    return out


def annotated_matrix(ax, V, S, fmts, row_labels, col_labels, hl_row=None, cmap="RdYlGn", vmin=0, vmax=1, fs=6.0):
    """Draw a values-annotated heat matrix. V: values, S: colour scores (NaN -> light grey)."""
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#e6e6e6")
    ax.imshow(np.ma.masked_invalid(S), cmap=cm, vmin=vmin, vmax=vmax, aspect="auto", alpha=0.88)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            v = V[i, j]
            t = "–" if np.isnan(v) else fmts[j](v)
            ax.text(j, i, t, ha="center", va="center", fontsize=fs, fontweight="bold" if i == hl_row else "normal")
    ax.set_xticks(range(V.shape[1])); ax.set_xticklabels(col_labels, fontsize=fs - 0.4, linespacing=1.0)
    ax.xaxis.tick_top()
    ax.set_yticks(range(V.shape[0])); ax.set_yticklabels(row_labels, fontsize=fs + 0.4)
    if hl_row is not None:
        ax.get_yticklabels()[hl_row].set_fontweight("bold")
    ax.set_xticks(np.arange(-.5, V.shape[1], 1), minor=True); ax.set_yticks(np.arange(-.5, V.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8); ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if hl_row is not None:
        ax.add_patch(Rectangle((-.5, hl_row - .5), V.shape[1], 1, fill=False, ec="k", lw=1.3))


def sweep_matrix(key, xlabels, name, metrics=None, fs=5.6, figsize=None):
    """Rank-coloured sweep matrix, one panel per environment, methods as rows.

    Columns are grouped by metric (energy, then coverage, then optional stranded).
    Colour = rank within each column (green best, red worst). Printed values are
    means; Random is omitted so the scale separates the seven non-trivial methods.
    """
    metrics = metrics or [
        ("energy_wh_per_task_mean", "Energy (Wh/task) ↓", False, lambda v: f"{v:.2f}"),
        ("coverage_mean", "Coverage (%) ↑", True, lambda v: f"{v:.0f}"),
    ]
    res = SW[key]["res"]
    envs = [e for e in ENVS if e in res]
    n_env, n_lev, n_met = len(envs), len(xlabels), len(metrics)
    n_col = n_lev * n_met
    nrows_fig, ncols_fig = (1, 2) if n_env == 2 else (2, 2)
    if figsize is None:
        figsize = (6.9, 3.7) if n_env == 2 else (6.9, 5.35)
    fig, axs = plt.subplots(nrows_fig, ncols_fig, figsize=figsize, squeeze=False)
    col_labels = list(xlabels) * n_met
    fmts = [fmt for *_, fmt in metrics for _ in range(n_lev)]
    hibs = [hib for _, _, hib, _ in metrics for _ in range(n_lev)]
    cols = [(met[0], i) for met in metrics for i in range(n_lev)]
    for k, e in enumerate(envs):
        ax = axs[k // ncols_fig, k % ncols_fig]
        V = np.array([[res[e][m][i][mk] for mk, i in cols] for m in SWEEP_METH], float)
        S = np.column_stack([rank_score(V[:, j], hibs[j]) for j in range(n_col)])
        annotated_matrix(ax, V, S, fmts, [LAB[m] for m in SWEEP_METH], col_labels,
                         hl_row=len(SWEEP_METH) - 1, fs=fs)
        for g in range(1, n_met):
            ax.axvline(g * n_lev - 0.5, color="k", lw=1.0)
        n0 = 0
        for _, glabel, *__ in metrics:
            cx = n0 + (n_lev - 1) / 2.0
            ax.annotate(glabel, xy=(cx, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(0, 13), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.3, annotation_clip=False)
            n0 += n_lev
        ax.set_title(f"({'abcd'[k]}) {ENVN[e]}", fontsize=8, pad=28)
    if n_env == 2:
        fig.subplots_adjust(left=0.085, right=0.995, top=0.78, bottom=0.16, wspace=0.24, hspace=0.45)
        cax = fig.add_axes([0.28, 0.055, 0.44, 0.028])
    else:
        fig.subplots_adjust(left=0.085, right=0.995, top=0.90, bottom=0.09, wspace=0.22, hspace=0.62)
        cax = fig.add_axes([0.28, 0.032, 0.44, 0.016])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, 1)),
                      cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["worst rank", "best rank"])
    cb.ax.tick_params(labelsize=6.4, length=0)
    cb.outline.set_visible(False)
    save(fig, name)


# Table 21: coverage, energy, stranded (7 methods)
DROPOUT_SUMMARY_METRICS = [
    ("coverage_mean", "Coverage (%) ↑", True, lambda v: f"{v:.0f}"),
    ("energy_wh_per_task_mean", "Energy (Wh) ↓", False, lambda v: f"{v:.2f}"),
    ("stranded_mean", "Stranded (%) ↓", False, lambda v: f"{v:.0f}"),
]


ROB_METRICS = [
    ("coverage_mean", "Cov", True, lambda v: f"{v:.0f}"),
    ("stranded_mean", "Str", False, lambda v: f"{v:.0f}"),
    ("expired_mean", "Exp", False, lambda v: f"{v:.0f}"),
    ("energy_wh_per_task_mean", "En", False, lambda v: f"{v:.1f}"),
    ("dist_per_task_m_mean", "Dist", False, lambda v: f"{v / 1000:.1f}k" if v >= 1000 else f"{v:.0f}"),
    ("steps_mean", "Stp", False, lambda v: f"{v:.0f}"),
    ("latency_mean_mean", "Lavg", False, lambda v: f"{v:.0f}"),
    ("latency_median_mean", "Lmed", False, lambda v: f"{v:.0f}"),
    ("soc_mean", "SOC", True, lambda v: f"{v:.2f}"),
    ("min_soc_mean", "Min", True, lambda v: f"{v:.2f}"),
    ("below_floor_pct_mean", "Flr", False, lambda v: f"{v:.0f}"),
    ("contention_pct_mean", "Cnt", False, lambda v: f"{v:.0f}"),
    ("blocked_moves_mean", "Blk", False, lambda v: f"{v:.0f}" if v < 1000 else f"{v / 1000:.1f}k"),
    ("reward_mean", "Rew", True, lambda v: f"{v:.0f}" if abs(v) < 1000 else f"{v / 1000:.1f}k"),
    ("info_age_mean", "Age", None, lambda v: f"{v:.1f}"),
]
ROB_GROUPS = [(0, 3, "Success"), (3, 8, "Efficiency"), (8, 11, "Battery"), (11, 15, "Team")]
ROB_STRESS = [
    ("packet_loss", 3, "Loss 60%"), ("comm_range", 0, "Range 6 m"),
    ("task_density", 3, "32 tasks"), ("scalability", 2, "16 UAVs"),
    ("fault", 2, "Drop 50%"), ("deadline", 2, "DL 100"),
    ("coupled", 2, "All:2"), ("obstacles", 2, "4 NFZs"),
]


def robustness_fingerprint(name="Robustness_matrix_Tables16-25"):
    """One balloon-fingerprint panel per environment, all recorded metrics.

    Each cell is the mean at the hardest level of every stressor that includes
    that environment. Colour and area = rank within the column (green/large =
    best). The R bar is the mean rank across directional metrics.
    """
    n_m, n_k = len(SWEEP_METH), len(ROB_METRICS)
    hl = n_m - 1
    cmap = plt.get_cmap("RdYlGn")
    fig = plt.figure(figsize=(7.25, 7.05))
    outer = fig.add_gridspec(2, 2, left=0.075, right=0.985, top=0.91, bottom=0.09,
                             wspace=0.28, hspace=0.52)
    for p, e in enumerate(ENVS):
        inner = outer[p // 2, p % 2].subgridspec(1, 2, width_ratios=[1.0, 0.20], wspace=0.08)
        ax = fig.add_subplot(inner[0])
        axb = fig.add_subplot(inner[1])
        V = np.zeros((n_m, n_k), float)
        for i, m in enumerate(SWEEP_METH):
            for j, (mk, *_) in enumerate(ROB_METRICS):
                vals = [SW[key]["res"][e][m][li][mk]
                        for key, li, _ in ROB_STRESS if e in SW[key]["res"]]
                V[i, j] = np.mean(vals)
        S = np.column_stack([
            np.full(n_m, np.nan) if hib is None else rank_score(V[:, j], hib)
            for j, (*_, hib, _) in enumerate(ROB_METRICS)
        ])
        ax.set_xlim(-0.55, n_k - 0.45)
        ax.set_ylim(n_m - 0.55, -0.55)
        ax.set_xticks(range(n_k))
        ax.set_xticklabels([lab for _, lab, *_ in ROB_METRICS], fontsize=5.4, rotation=90)
        ax.xaxis.tick_top()
        ax.set_yticks(range(n_m))
        ax.set_yticklabels([LAB[m] for m in SWEEP_METH], fontsize=6.6)
        ax.get_yticklabels()[hl].set_fontweight("bold")
        ax.tick_params(length=0, pad=3)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_facecolor("#f7f7f7")
        for _, end, _ in ROB_GROUPS[:-1]:
            ax.axvline(end - 0.5, color="0.15", lw=0.8, zorder=1)
        for j in range(n_k):
            for i in range(n_m):
                s = S[i, j]
                if np.isnan(s):
                    c, sz = "#d5d5d5", 52
                else:
                    c, sz = cmap(s), 64 + 150 * float(s)
                ax.scatter(j, i, s=sz, c=[c], zorder=3, linewidths=0.25,
                           edgecolors="0.2", clip_on=False)
                tc = "white" if (not np.isnan(s) and s < 0.30) else "0.08"
                ax.text(j, i, ROB_METRICS[j][3](V[i, j]), ha="center", va="center",
                        fontsize=4.6, fontweight="bold" if i == hl else "normal",
                        zorder=4, color=tc)
        ax.add_patch(Rectangle((-0.5, hl - 0.5), n_k, 1, fill=False, ec="k", lw=1.2, zorder=5))
        for a, b, glabel in ROB_GROUPS:
            ax.annotate(glabel, xy=((a + b - 1) / 2, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(0, 22), textcoords="offset points", ha="center", va="bottom",
                        fontsize=6.0, annotation_clip=False)
        ax.set_title(f"({'abcd'[p]}) {ENVN[e]}", fontsize=8, pad=36)
        R = np.nanmean(S, axis=1)
        axb.set_ylim(n_m - 0.55, -0.55)
        axb.set_xlim(0, 1.08)
        axb.barh(np.arange(n_m), np.nan_to_num(R), height=0.58,
                 color=[cmap(0.5 if np.isnan(r) else r) for r in R],
                 edgecolor=["k" if i == hl else "0.25" for i in range(n_m)],
                 linewidth=[1.2 if i == hl else 0.35 for i in range(n_m)], zorder=3)
        for i, r in enumerate(R):
            inside = r >= 0.78
            axb.text(r - 0.04 if inside else r + 0.03, i, f"{r:.2f}", va="center",
                     ha="right" if inside else "left", fontsize=5.3,
                     fontweight="bold" if i == hl else "normal", color="0.08")
        axb.set_xticks([])
        axb.set_yticks([])
        for sp in axb.spines.values():
            sp.set_visible(False)
        axb.set_xlabel("R", fontsize=7, labelpad=2)
        axb.xaxis.set_label_position("top")
    cax = fig.add_axes([0.30, 0.035, 0.40, 0.016])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, 1)),
                      cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["worst rank", "best rank"])
    cb.ax.tick_params(labelsize=6.4, length=0)
    cb.outline.set_visible(False)
    save(fig, name)


RADAR_AXES = ["Coverage", "No stranding", "Speed", "Energy\neff.", "Path\neff.", "Final\nSOC", "No\ncontention"]


def profile_scores(records, ref=None):
    """7 radar scores in [0,1] (1 = best) for a list of metric dicts.
    coverage/best, 1-stranded, best_steps/steps, best_energy/energy, best_dist/dist, soc/best, 1-contention.
    `ref` = optional dict of best values; default: best within `records`."""
    g = lambda k: np.array([r[k] for r in records], float)
    cov, st, stp, en, ds, so, co = (g(k) for k in ["coverage", "stranded", "steps", "energy", "dist", "soc", "contention"])
    ref = ref or dict(cov=cov.max(), stp=stp.min(), en=en.min(), ds=ds.min(), so=so.max())
    return np.vstack([cov / ref["cov"], 1 - st / 100, ref["stp"] / stp, ref["en"] / en, ref["ds"] / ds,
                      so / max(ref["so"], 1e-9), 1 - co / 100]).T


def rec(src, e, m):
    """collect the 7 base metrics of (env, method) from a results dict into a record."""
    return dict(coverage=bm(e, m, "coverage", src), stranded=bm(e, m, "stranded", src), steps=bm(e, m, "steps", src),
                energy=bm(e, m, "energy_wh_per_task", src), dist=bm(e, m, "dist_per_task_m", src),
                soc=bm(e, m, "soc", src), contention=bm(e, m, "contention_pct", src))


def radar_axes_setup(ax, labels=RADAR_AXES, fs=6):
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    ax.set_xticks(ang); ax.set_xticklabels(labels, fontsize=fs)
    ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1]); ax.set_yticklabels(["", "0.5", "", "1"], fontsize=5.5)
    ax.tick_params(pad=3); ax.grid(lw=0.3)
    return np.r_[ang, ang[0]]


def save(fig, name, dpi=DPI):
    path = FIG / f"{name}.png"
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print("saved", path.name)
