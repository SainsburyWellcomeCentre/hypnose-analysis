"""SLEAP-trace visualizations for the single-rewarded sequence task.

For one subject, plots the movement trace of every trial from the moment it
leaves the odor cue port to its outcome endpoint, one figure per outcome
category (Hit / Miss / False Alarm / Correct Rejection):

- Hit and False Alarm trials end at the reward-port poke (they went to a port).
- Miss and Correct Rejection trials end at the next cue-port poke (next trial
  initiation) — they withheld, so we follow them until the next trial starts.

Trials are categorised with the same ``_classify_trial`` used by the singrew
metrics, so categories/subcategories match exactly. With ``plot_subcategories``
each category is split into its subcategories (one figure each). Traces are
colored by A/B port group (A=red, B=blue), matching the other movement plots.
Multiple dates are overlaid in one figure per category; ``show_average`` adds a
thick mean trace per group (one for A, one for B) across all overlaid sessions.
Intended to be called with a single subject.

Reuses ``_load_tracking_and_behavior`` (tracking + behavior loader) and
``_classify_trial`` (categorisation); the small trace helpers are local.
"""

from __future__ import annotations

import json
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from hypnose.io.save import save_figure
from hypnose.metric_analysis.sing_rew_metrics import _classify_trial
from hypnose.visualization.movement_analysis_utils import _load_tracking_and_behavior
from hypnose.visualization.pred_seq_utils import _collect_sessions


CATEGORY_ORDER = ["hit", "miss", "false_alarm", "correct_rejection"]
CATEGORY_TITLES = {
    "hit": "Hit",
    "miss": "Miss",
    "false_alarm": "False Alarm",
    "correct_rejection": "Correct Rejection",
}
# Subcategories per category, matching _classify_trial's second return value.
SUBCATEGORY_ORDER = {
    "hit": ["rewarded_hit", "port_error_hit", "anticipatory_hit", "anticipatory_port_error"],
    "miss": ["omission_miss", "forfeit_miss"],
    "false_alarm": ["completed_fa", "aborted_fa"],
    "correct_rejection": ["passive_cr", "active_cr"],
}
SUBCATEGORY_TITLES = {
    "rewarded_hit": "Rewarded Hit",
    "port_error_hit": "Port-Error Hit",
    "anticipatory_hit": "Anticipatory Hit",
    "anticipatory_port_error": "Anticipatory Port-Error Hit",
    "omission_miss": "Omission Miss",
    "forfeit_miss": "Forfeit Miss",
    "completed_fa": "Completed FA",
    "aborted_fa": "Aborted FA",
    "passive_cr": "Passive CR",
    "active_cr": "Active CR",
}

# Port/last-odor grouping colors, matching the A=red / B=blue scheme used by the
# other movement_analysis plots. Traces (and per-group means) are colored by the
# trial's A/B side.
GROUP_ORDER = ["A", "B", "other"]
GROUP_COLORS = {"A": "red", "B": "blue", "other": "lightgray"}
GROUP_MEAN_COLORS = {"A": "darkred", "B": "darkblue", "other": "dimgray"}
GROUP_LABELS = {"A": "A", "B": "B", "other": "Other/Unknown"}


def _port_letter(value):
    """Map a reward-port id (1 -> A, 2 -> B) to its letter, else None."""
    try:
        return {1: "A", 2: "B"}.get(int(value))
    except (TypeError, ValueError):
        return None


def _odor_letter(value):
    """Map an odor / identity (OdorA/A/1 -> A, OdorB/B/2 -> B) to A/B, else None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    s = str(value).strip().upper()
    if s in ("A", "ODORA", "1"):
        return "A"
    if s in ("B", "ODORB", "2"):
        return "B"
    return None


def _trial_port_group(row, main):
    """A/B group for a trial (else None -> "other").

    Hit / False Alarm: the port the animal poked. Miss / Correct Rejection: what
    the sequence would have ended in (``determined_final_odor``), falling back to
    the last odor of the sequence.
    """
    if main == "hit":
        if bool(row.get("is_aborted")):  # anticipatory
            return _port_letter(row.get("fa_port"))
        if str(row.get("response_time_category")) == "rewarded":
            return _port_letter(row.get("first_supply_port")) or _odor_letter(row.get("first_supply_odor_identity"))
        return _port_letter(row.get("first_reward_poke_port")) or _odor_letter(row.get("first_reward_poke_odor_identity"))
    if main == "false_alarm":
        if bool(row.get("is_aborted")):
            return _port_letter(row.get("fa_port"))
        return _port_letter(row.get("fr_port")) or _odor_letter(row.get("fr_odor_identity"))
    if main in ("miss", "correct_rejection"):
        return (_odor_letter(row.get("determined_final_odor"))
                or _odor_letter(row.get("last_odor"))
                or _odor_letter(row.get("last_odor_name")))
    return None


def _naive_dt(value):
    """Parse to tz-naive datetime(s) (scalar Timestamp/NaT or Series), so
    comparisons between tracking time and trial times never fail on a tz mismatch."""
    ts = pd.to_datetime(value, errors="coerce")
    if hasattr(ts, "dt"):  # Series
        try:
            if ts.dt.tz is not None:
                ts = ts.dt.tz_localize(None)
        except (AttributeError, TypeError):
            pass
        return ts
    try:  # scalar Timestamp / NaT
        if getattr(ts, "tz", None) is not None:
            ts = ts.tz_localize(None)
    except (AttributeError, TypeError):
        pass
    return ts


def _last_poke_out(row):
    """Time the animal leaves the odor cue port = poke_odor_end of the last
    presented position (parsed from position_poke_times), with fallbacks."""
    pts = row.get("position_poke_times")
    if isinstance(pts, str):
        try:
            pts = json.loads(pts)
        except Exception:
            pts = None
    if isinstance(pts, dict) and pts:
        vals = list(pts.values())
        if all(isinstance(v, dict) and "position" in v for v in vals):
            vals = sorted(vals, key=lambda v: v.get("position", 0))
        last = vals[-1]
        if isinstance(last, dict):
            return _naive_dt(last.get("poke_odor_end"))
    if isinstance(pts, list) and pts and isinstance(pts[-1], dict):
        return _naive_dt(pts[-1].get("poke_odor_end"))
    for cand in ("poke_odor_end", "last_poke_out_time", "last_poke_time", "sequence_end"):
        if cand in row and pd.notna(row.get(cand)):
            return _naive_dt(row.get(cand))
    return pd.NaT


def _segment_end(row, main, next_init):
    """Endpoint time for a trial's trace, by outcome category.

    Hit/FA end at the port poke (reward, wrong-port, or false-alarm poke depending
    on subcategory); Miss/CR end at the next trial's initiation (``next_init``).
    """
    is_aborted = bool(row.get("is_aborted"))
    if main == "hit":
        if is_aborted:  # anticipatory hit: went to a port during an abort
            return _naive_dt(row.get("fa_time"))
        if str(row.get("response_time_category")) == "rewarded":
            return _naive_dt(row.get("first_supply_time"))
        end = _naive_dt(row.get("first_reward_poke_time"))  # port-error (unrewarded) poke
        if pd.isna(end):
            end = _naive_dt(row.get("first_supply_time"))
        return end
    if main == "false_alarm":
        if is_aborted:  # aborted false alarm
            return _naive_dt(row.get("fa_time"))
        return _naive_dt(row.get("fr_time"))  # completed no-go false response
    if main in ("miss", "correct_rejection"):
        return next_init
    return pd.NaT


def _smooth_tracking(tracking, smooth_window):
    """Rolling-mean smooth of X/Y over the whole session (centered)."""
    if smooth_window is None or smooth_window <= 1:
        return tracking
    df = tracking.copy()
    df["X"] = pd.Series(df["X"]).rolling(window=smooth_window, center=True, min_periods=1).mean()
    df["Y"] = pd.Series(df["Y"]).rolling(window=smooth_window, center=True, min_periods=1).mean()
    return df


def _extract_segment(tracking, start, end):
    """(X, Y) arrays of tracking frames within [start, end], or None."""
    if pd.isna(start) or pd.isna(end) or end <= start:
        return None
    mask = (tracking["time"] >= start) & (tracking["time"] <= end)
    if not mask.any():
        return None
    seg = tracking.loc[mask, ["X", "Y"]].dropna()
    if len(seg) < 2:
        return None
    return seg["X"].to_numpy(dtype=float), seg["Y"].to_numpy(dtype=float)


def _resample_trace(x, y, n_points):
    """Resample a trajectory onto a normalized arc-length grid (for averaging)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    seg_len = np.hypot(np.diff(x), np.diff(y))
    cumlen = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = cumlen[-1]
    if total <= 0:
        return None
    s = cumlen / total
    s_new = np.linspace(0.0, 1.0, n_points)
    return np.interp(s_new, s, x), np.interp(s_new, s, y)


def _plot_category(segments, title, *,
                   show_average, linewidth, alpha, n_average_points, figsize, invert_y):
    """Draw one category/subcategory figure (all its trials overlaid), colored by
    A/B port group. With ``show_average``, a thick mean trace is added per group
    (one for A, one for B, arc-length resampled across all overlaid sessions)."""
    fig, ax = plt.subplots(figsize=figsize)
    for seg in segments:
        ax.plot(seg["x"], seg["y"], color=GROUP_COLORS.get(seg["group"], "lightgray"),
                alpha=alpha, linewidth=linewidth, zorder=2)

    if show_average:
        for group in ("A", "B"):
            grp = [s for s in segments if s["group"] == group]
            resampled = [r for r in (_resample_trace(s["x"], s["y"], n_average_points) for s in grp)
                         if r is not None]
            if resampled:
                ax.plot(np.nanmean([r[0] for r in resampled], axis=0),
                        np.nanmean([r[1] for r in resampled], axis=0),
                        color=GROUP_MEAN_COLORS[group], linewidth=3.0, zorder=5)

    ax.set_title(f"{title} (n={len(segments)})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="datalim")
    if invert_y:
        ax.invert_yaxis()

    groups_present = [g for g in GROUP_ORDER if any(s["group"] == g for s in segments)]
    handles = [Line2D([], [], color=GROUP_COLORS[g], label=GROUP_LABELS[g]) for g in groups_present]
    if show_average:
        for group in ("A", "B"):
            if any(s["group"] == group for s in segments):
                handles.append(Line2D([], [], color=GROUP_MEAN_COLORS[group], linewidth=3.0,
                                      label=f"Mean {group}"))
    if handles:
        ax.legend(handles=handles, title="Port", fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_category_traces(
    subjid: int,
    dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
    *,
    show_average: bool = False,
    plot_subcategories: bool = False,
    smooth_window: int = 10,
    linewidth: float = 1.0,
    alpha: float = 0.5,
    n_average_points: int = 200,
    figsize=(8, 8),
    invert_y: bool = True,
    save: bool = False,
):
    """Per-category SLEAP movement traces for one subject.

    One figure per outcome category (Hit / Miss / False Alarm / Correct
    Rejection), each overlaying every matching trial's trace from leaving the
    odor cue port to its endpoint (reward-port poke for Hit/FA, next trial
    initiation for Miss/CR). With ``plot_subcategories=True`` each category is
    split into its subcategories (one figure each). Traces are colored by A/B
    port group (A=red, B=blue): the poked port for Hit/FA, and the would-be
    reward odor (``determined_final_odor``, else last odor) for Miss/CR. Multiple
    ``dates`` are overlaid in the same figure; ``show_average`` adds a thick mean
    trace per group (one for A, one for B) across all overlaid trials.

    Parameters mirror the other movement plots (``smooth_window`` etc.). Call
    with a single subject.

    Returns a list of matplotlib figures.
    """
    # Resolve the subject's sessions/dates via the shared iterator.
    collected = list(_collect_sessions([subjid], dates))
    if not collected:
        print(f"[plot_category_traces] No sessions found for subject {subjid}.")
        return []
    _sid, date_vals, _dirs = collected[0]

    # category -> subcategory -> list of {x, y, date}
    store = {cat: {sub: [] for sub in SUBCATEGORY_ORDER[cat]} for cat in CATEGORY_ORDER}

    for date in date_vals:
        try:
            tracking, behavior = _load_tracking_and_behavior(subjid, date)
        except Exception as exc:
            print(f"[plot_category_traces] Skipping date {date}: {exc}")
            continue
        trial_data = behavior.get("trial_data", pd.DataFrame())
        if trial_data is None or trial_data.empty or "global_trial_id" not in trial_data.columns:
            continue

        tracking = tracking.copy()
        tracking["time"] = _naive_dt(tracking["time"])
        tracking = _smooth_tracking(tracking, smooth_window)

        td = trial_data.sort_values("global_trial_id").reset_index(drop=True)
        next_init = _naive_dt(td["sequence_start"]).shift(-1) if "sequence_start" in td.columns \
            else pd.Series([pd.NaT] * len(td))

        for i, row in td.iterrows():
            main, sub = _classify_trial(row)
            if main not in CATEGORY_ORDER or sub is None:
                continue
            start = _last_poke_out(row)
            end = _segment_end(row, main, next_init.iloc[i])
            seg = _extract_segment(tracking, start, end)
            if seg is None:
                continue
            group = _trial_port_group(row, main) or "other"
            store[main][sub].append({"x": seg[0], "y": seg[1], "group": group})

    figs = []
    if plot_subcategories:
        plan = [(sub, SUBCATEGORY_TITLES[sub], store[cat][sub])
                for cat in CATEGORY_ORDER for sub in SUBCATEGORY_ORDER[cat]]
    else:
        plan = [(cat, CATEGORY_TITLES[cat],
                 [seg for sub in SUBCATEGORY_ORDER[cat] for seg in store[cat][sub]])
                for cat in CATEGORY_ORDER]

    for key, title, segments in plan:
        fig = _plot_category(
            segments, title,
            show_average=show_average, linewidth=linewidth,
            alpha=alpha, n_average_points=n_average_points, figsize=figsize, invert_y=invert_y,
        )
        figs.append(fig)
        if save:
            save_figure(fig, f"sing_rew_traces_{key}", subjids=[subjid], dates=list(date_vals))

    return figs
