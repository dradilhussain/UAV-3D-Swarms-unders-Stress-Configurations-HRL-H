"""CSV / JSON export and publication-figure helpers for the 3D notebook.

Does not touch model implementations. The notebook uses these to dump every
sweep (per environment and combined) and to pick 2x2 / 3x2 panel layouts.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


FONT_TITLE = 18
FONT_LABEL = 16
FONT_TICK = 13
FONT_LEGEND = 12
FONT_SUPTITLE = 20
FONT_ANNOT = 11

SUMMARY_FIELDS = [
    "value",
    "coverage_mean", "coverage_std",
    "stranded_mean", "stranded_std",
    "reward_mean", "reward_std",
    "expired_mean", "expired_std",
    "soc_mean", "soc_std",
    "steps_mean", "steps_std",
    "energy_wh_per_task_mean", "energy_wh_per_task_std",
    "dist_per_task_m_mean", "dist_per_task_m_std",
    "min_soc_mean", "min_soc_std",
    "below_floor_pct_mean", "below_floor_pct_std",
    "contention_pct_mean", "contention_pct_std",
    "info_age_mean", "info_age_std",
    "latency_mean_mean", "latency_mean_std",
    "latency_median_mean", "latency_median_std",
    "blocked_moves_mean", "blocked_moves_std",
]

TABLE_METRICS = [
    ("coverage_mean", "coverage_std", "Coverage (%)"),
    ("stranded_mean", "stranded_std", "Stranded UAVs (%)"),
    ("reward_mean", "reward_std", "Total reward"),
    ("soc_mean", "soc_std", "Mean final SOC"),
    ("steps_mean", "steps_std", "Episode length (steps)"),
    ("energy_wh_per_task_mean", "energy_wh_per_task_std", "Energy per task (Wh)"),
    ("dist_per_task_m_mean", "dist_per_task_m_std", "Distance per task (m)"),
    ("min_soc_mean", "min_soc_std", "Min SOC"),
    ("below_floor_pct_mean", "below_floor_pct_std", "Time below floor (%)"),
    ("contention_pct_mean", "contention_pct_std", "Contention (% steps)"),
    ("info_age_mean", "info_age_std", "Mean info age (steps)"),
    ("latency_mean_mean", "latency_mean_std", "Mean completion latency"),
    ("latency_median_mean", "latency_median_std", "Median completion latency"),
    ("blocked_moves_mean", "blocked_moves_std", "Blocked moves"),
]


def apply_pub_style():
    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 100,
        "axes.titlesize": FONT_TITLE,
        "axes.labelsize": FONT_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "figure.titlesize": FONT_SUPTITLE,
        "font.size": FONT_TICK,
    })


def panel_shape(n_panels: int) -> Tuple[int, int]:
    """Choose a grid: 4 -> 2x2, 5/6 -> 3x2, 2 -> 1x2, 3 -> 1x3, 8 -> 4x2."""
    n = max(int(n_panels), 1)
    if n == 1:
        return 1, 1
    if n == 2:
        return 1, 2
    if n == 3:
        return 1, 3
    if n == 4:
        return 2, 2
    if n <= 6:
        return 3, 2
    if n <= 8:
        return 4, 2
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def figsize_for(nrows: int, ncols: int, cell=(7.2, 5.6)) -> Tuple[float, float]:
    return (ncols * cell[0], nrows * cell[1])


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_json(obj: Any, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if not np.isfinite(v):
            return ""
        return f"{v:.8g}"
    return str(v)


def write_csv(rows: Sequence[Dict[str, Any]], path: str,
              fieldnames: Optional[Sequence[str]] = None):
    ensure_dir(os.path.dirname(path) or ".")
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    if fieldnames is None:
        keys = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(r.get(k)) for k in fieldnames})


def flatten_summary(row: Dict[str, Any], extra: Optional[Dict[str, Any]] = None
                    ) -> Dict[str, Any]:
    out = dict(extra or {})
    for k in SUMMARY_FIELDS:
        if k in row:
            out[k] = row[k]
    for k, v in row.items():
        if k not in out and not isinstance(v, (list, dict)):
            out[k] = v
    return out


def sweep_to_rows(results: Dict[str, List[Dict[str, Any]]],
                  extra: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    rows = []
    for policy, levels in results.items():
        for rec in levels:
            r = flatten_summary(rec, extra)
            r["policy"] = policy
            r["level"] = rec.get("value")
            rows.append(r)
    return rows


def base_to_rows(results: Dict[str, Dict[str, Any]],
                 extra: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    rows = []
    for policy, rec in results.items():
        r = flatten_summary(rec, extra)
        r["policy"] = policy
        rows.append(r)
    return rows


def dump_sweep(results, out_dir, name, meta, env_id=None):
    """Write JSON + CSV for one sweep. If env_id is set, also write a per-env copy."""
    payload = dict(meta)
    payload["results"] = results
    write_json(payload, os.path.join(out_dir, f"{name}.json"))
    extra = {k: v for k, v in meta.items()
             if not isinstance(v, (list, dict))}
    if env_id is not None:
        extra["environment"] = env_id
    rows = sweep_to_rows(results, extra)
    write_csv(rows, os.path.join(out_dir, f"{name}.csv"))
    if env_id:
        sub = os.path.join(out_dir, "by_environment", env_id)
        write_json(payload, os.path.join(sub, f"{name}.json"))
        write_csv(rows, os.path.join(sub, f"{name}.csv"))
    return rows


def dump_base(results, out_dir, name, meta, env_id=None):
    payload = dict(meta)
    payload["results"] = results
    write_json(payload, os.path.join(out_dir, f"{name}.json"))
    extra = {k: v for k, v in meta.items()
             if not isinstance(v, (list, dict))}
    if env_id is not None:
        extra["environment"] = env_id
    rows = base_to_rows(results, extra)
    write_csv(rows, os.path.join(out_dir, f"{name}.csv"))
    if env_id:
        sub = os.path.join(out_dir, "by_environment", env_id)
        write_json(payload, os.path.join(sub, f"{name}.json"))
        write_csv(rows, os.path.join(sub, f"{name}.csv"))
    return rows


def comparison_table_rows(results_by_env: Dict[str, Dict[str, Dict[str, Any]]],
                          policy_order: Sequence[str],
                          policy_labels: Dict[str, str]
                          ) -> List[Dict[str, Any]]:
    """One row per (environment, metric); columns are policies mean+/-std."""
    rows = []
    for env_id, pol_map in results_by_env.items():
        for mean_k, std_k, label in TABLE_METRICS:
            row = {"environment": env_id, "metric": label,
                   "metric_key": mean_k}
            for p in policy_order:
                rec = pol_map.get(p, {})
                m, s = rec.get(mean_k), rec.get(std_k)
                row[p] = m
                row[f"{p}_std"] = s
                if m is None:
                    row[f"{p}_pm"] = ""
                elif s is None:
                    row[f"{p}_pm"] = f"{m:.4g}"
                else:
                    row[f"{p}_pm"] = f"{m:.4g} +/- {s:.4g}"
                row[policy_labels.get(p, p)] = row[f"{p}_pm"]
            rows.append(row)
    return rows


def write_markdown_table(results: Dict[str, Dict[str, Any]],
                         policy_order: Sequence[str],
                         policy_labels: Dict[str, str],
                         path: str, title: str = ""):
    ensure_dir(os.path.dirname(path) or ".")
    headers = ["Metric"] + [policy_labels.get(p, p) for p in policy_order]
    lines = []
    if title:
        lines.append(f"# {title}")
        lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for mean_k, std_k, label in TABLE_METRICS:
        cells = [label]
        for p in policy_order:
            rec = results.get(p, {})
            m, s = rec.get(mean_k), rec.get(std_k)
            if m is None:
                cells.append("")
            elif s is None:
                cells.append(f"{m:.3g}")
            else:
                cells.append(f"{m:.3g} +/- {s:.3g}")
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def hide_unused_axes(axes, n_used: int):
    flat = np.atleast_1d(axes).ravel()
    for ax in flat[n_used:]:
        ax.set_visible(False)


def _meta_flat(meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = meta or {}
    return {k: v for k, v in meta.items() if not isinstance(v, (list, dict))}


def dump_sweep_by_env(results_by_env, out_dir, name, meta=None):
    """Write combined + per-environment JSON/CSV for a sweep keyed by env id."""
    meta = meta or {}
    write_json({**meta, "environments": list(results_by_env.keys()),
                "results": results_by_env},
               os.path.join(out_dir, f"{name}.json"))
    rows = []
    extra_base = _meta_flat(meta)
    for eid, res in results_by_env.items():
        extra = {**extra_base, "environment": eid}
        sub_rows = sweep_to_rows(res, extra)
        rows.extend(sub_rows)
        sub = os.path.join(out_dir, "by_environment", eid)
        write_json({**meta, "environment": eid, "results": res},
                   os.path.join(sub, f"{name}.json"))
        write_csv(sub_rows, os.path.join(sub, f"{name}.csv"))
    write_csv(rows, os.path.join(out_dir, f"{name}.csv"))
    return rows


def dump_base_by_env(results_by_env, out_dir, name, meta=None):
    """Write combined + per-environment JSON/CSV for base-config results."""
    meta = meta or {}
    write_json({**meta, "environments": list(results_by_env.keys()),
                "results": results_by_env},
               os.path.join(out_dir, f"{name}.json"))
    rows = []
    extra_base = _meta_flat(meta)
    for eid, res in results_by_env.items():
        extra = {**extra_base, "environment": eid}
        sub_rows = base_to_rows(res, extra)
        rows.extend(sub_rows)
        sub = os.path.join(out_dir, "by_environment", eid)
        write_json({**meta, "environment": eid, "results": res},
                   os.path.join(sub, f"{name}.json"))
        write_csv(sub_rows, os.path.join(sub, f"{name}.csv"))
    write_csv(rows, os.path.join(out_dir, f"{name}.csv"))
    return rows


def dump_sweep_json(results_by_env, out_dir, name, meta=None):
    """JSON-only sweep dump (combined + per-environment)."""
    meta = meta or {}
    write_json({**meta, "environments": list(results_by_env.keys()),
                "results": results_by_env},
               os.path.join(out_dir, f"{name}.json"))
    for eid, res in results_by_env.items():
        sub = os.path.join(out_dir, "by_environment", eid)
        write_json({**meta, "environment": eid, "results": res},
                   os.path.join(sub, f"{name}.json"))


def dump_base_json(results_by_env, out_dir, name, meta=None):
    """JSON-only base-config dump (combined + per-environment)."""
    meta = meta or {}
    write_json({**meta, "environments": list(results_by_env.keys()),
                "results": results_by_env},
               os.path.join(out_dir, f"{name}.json"))
    for eid, res in results_by_env.items():
        sub = os.path.join(out_dir, "by_environment", eid)
        write_json({**meta, "environment": eid, "results": res},
                   os.path.join(sub, f"{name}.json"))


def dump_tables_json(results_by_env, out_dir, policy_order, policy_labels, title=""):
    """JSON-only comparison tables (combined + per environment)."""
    rows = comparison_table_rows(results_by_env, policy_order, policy_labels)
    write_json({"title": title, "rows": rows,
                "environments": list(results_by_env.keys()),
                "policies": list(policy_order)},
               os.path.join(out_dir, "comparison_table.json"))
    for eid, res in results_by_env.items():
        sub = os.path.join(out_dir, "by_environment", eid)
        write_json({"title": f"{title} -- {eid}".strip(" --"),
                    "environment": eid, "results": res},
                   os.path.join(sub, "comparison_table.json"))
    return rows


def dump_tables(results_by_env, out_dir, policy_order, policy_labels, title=""):
    """CSV + Markdown comparison tables (all models x all environments)."""
    rows = comparison_table_rows(results_by_env, policy_order, policy_labels)
    write_csv(rows, os.path.join(out_dir, "comparison_table.csv"))
    write_json({"title": title, "rows": rows,
                "environments": list(results_by_env.keys()),
                "policies": list(policy_order)},
               os.path.join(out_dir, "comparison_table.json"))
    for eid, res in results_by_env.items():
        sub = os.path.join(out_dir, "by_environment", eid)
        write_markdown_table(res, policy_order, policy_labels,
                             os.path.join(sub, "comparison_table.md"),
                             title=f"{title} -- {eid}".strip(" --"))
        write_csv(base_to_rows(res, {"environment": eid}),
                  os.path.join(sub, "comparison_table.csv"))
    # combined markdown: one section per environment
    lines = [f"# {title or 'Comparison table'}", ""]
    for eid, res in results_by_env.items():
        headers = ["Metric"] + [policy_labels.get(p, p) for p in policy_order]
        lines.append(f"## {eid}")
        lines.append("")
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for mean_k, std_k, label in TABLE_METRICS:
            cells = [label]
            for p in policy_order:
                rec = res.get(p, {})
                m, s = rec.get(mean_k), rec.get(std_k)
                if m is None:
                    cells.append("")
                elif s is None:
                    cells.append(f"{m:.3g}")
                else:
                    cells.append(f"{m:.3g} +/- {s:.3g}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "comparison_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return rows
