"""Visualizations for the single-rewarded sequence task.

In the single-reward protocol not every candidate sequence is rewarded at its
final position (see README, "Single-Reward Protocol / False Response
Information"). Completed non-rewarded ("no-go") sequences are scored via the
``false_response`` / ``fr_label`` / ``fr_latency_ms`` / ``fr_port`` columns,
analogous to the false-alarm columns used for aborted trials.

General plotting helpers (boxplots, rolling/daily summary lines, marker sizing,
session collection) are imported from ``pred_seq_utils`` and reused here.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from matplotlib.ticker import MaxNLocator

from hypnose.io.save import save_figure
from hypnose.visualization.pred_seq_utils import (
    _collect_sessions,
    _load_sorted_session,
    _parse_json_value,
    _last_position_entry,
    _plot_box_with_points,
    _plot_summary_rolling,
    _count_to_marker_size,
    _add_size_legend,
    _summary_save_suffix,
)


def _normalize_fr_types(fr_types):
    """Coerce the ``fr_types`` argument to a set of fr_label strings (or None).

    ``None`` means "do not filter" — count every ``false_response == True`` trial
    regardless of its ``fr_label``.
    """
    if fr_types is None:
        return None
    if isinstance(fr_types, str):
        return {fr_types}
    return set(fr_types)


def _port_label(value):
    """Map a supply/reward port id (1 → A, 2 → B) to its letter, else None."""
    try:
        port_val = int(value)
    except (TypeError, ValueError):
        return None
    if port_val == 1:
        return "A"
    if port_val == 2:
        return "B"
    return None


def _response_time_ms(row):
    """Response time for a completed trial, computed as in ``pred_seq_utils``:
    last odor poke_odor_end → first_supply_time, in milliseconds."""
    pos_dict = _parse_json_value(row.get("position_poke_times"))
    last_entry = _last_position_entry(pos_dict)
    if not last_entry:
        return None
    poke_end = last_entry.get("poke_odor_end")
    supply_time = row.get("first_supply_time")
    if poke_end is None or supply_time is None:
        return None
    poke_end_dt = pd.to_datetime(poke_end, errors="coerce")
    supply_dt = pd.to_datetime(supply_time, errors="coerce")
    if pd.isna(poke_end_dt) or pd.isna(supply_dt):
        return None
    return float((supply_dt - poke_end_dt).total_seconds() * 1000.0)


def _subject_color_map(labels):
    """Assign a stable color per subject label using the tab10 colormap."""
    cmap = plt.get_cmap("tab10")
    return {label: cmap(i % 10) for i, label in enumerate(labels)}


def _fr_mask(frame, fr_labels):
    """Boolean mask over ``frame`` selecting false-response trials.

    A trial is a false response iff ``false_response == True`` and (when
    ``fr_labels`` is not None) its ``fr_label`` is in ``fr_labels``. Sessions
    without the single-reward columns yield an all-False mask.
    """
    if "false_response" not in frame.columns:
        return pd.Series(False, index=frame.index)
    mask = frame["false_response"] == True  # noqa: E712 (element-wise, NaN-safe)
    if fr_labels is not None and "fr_label" in frame.columns:
        mask = mask & frame["fr_label"].isin(fr_labels)
    return mask


def _trim_leading_empty(series):
    """Drop leading ``None`` (no-data) entries from a chronological session series,
    keeping middle gaps intact. Returns ([] when there is no data at all)."""
    first = next((i for i, v in enumerate(series) if v is not None), None)
    if first is None:
        return []
    return series[first:]


def _plot_fr_ratio_daily(series_by_subject, color_map, subj_order, ylabel, title):
    """Per-session FR ratio, one line per subject.

    ``series_by_subject`` maps each subject label to a chronological list of
    ``(ratio, n_completed)`` tuples (or ``None`` for a mid-run session with no
    data). Each subject's series has already had its leading empty sessions
    trimmed, so every subject starts at session 1. Mid-run gaps are left blank
    (the line breaks there). X ticks are kept sparse for long runs.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    max_len = 0
    all_counts = []
    for label in subj_order:
        series = series_by_subject.get(label)
        if not series:
            continue
        max_len = max(max_len, len(series))
        xs = list(range(1, len(series) + 1))
        ys = [np.nan if v is None else v[0] for v in series]
        color = color_map.get(label)
        # NaNs break the connecting line at mid-run gaps.
        ax.plot(xs, ys, color=color, linewidth=2, label=label)
        valid = [(x, v[0], v[1]) for x, v in zip(xs, series) if v is not None]
        if valid:
            vx = [x for x, _, _ in valid]
            vy = [y for _, y, _ in valid]
            counts = [c for _, _, c in valid]
            all_counts.extend(counts)
            ax.scatter(vx, vy, s=[_count_to_marker_size(c) for c in counts], color=color, zorder=3)

    ax.set_xlabel("Session")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if max_len <= 12:
        ax.set_xticks(range(1, max_len + 1))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(left=0.5, right=max_len + 0.5)
    ax.legend(loc="best")
    _add_size_legend(ax, all_counts)
    fig.tight_layout()
    return fig


def FR_ratio(
    subjids: Optional[Iterable[int]] = None,
    dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
    fr_types: Optional[Iterable[str]] = "FR_time_in",
    save: bool = False,
    moving_avg: bool = False,
    window_size: int = 10,
    step_size: int = 1,
):
    """Ratio of false-response trials to all completed trials.

    For every completed trial (``is_aborted == False``) a binary value is
    assigned: 1 if it is a false response (``false_response == True``, further
    filtered to ``fr_types`` matched against ``fr_label``; default
    ``"FR_time_in"`` only), else 0. The mean of this value is the
    ``#false_response / #completed`` ratio.

    All subjects are drawn on one figure, with one colored series per subject so
    matched subjects are connected across sessions. Each subject's leading empty
    sessions (no single-reward data) are trimmed so every subject starts at
    session 1 with its first data session; mid-run sessions with no data are left
    blank (the line breaks there).

    - ``moving_avg=False``: one point per subject per session (whole-session
      ratio); X axis is the ordinal session number, with sparse ticks for long
      runs.
    - ``moving_avg=True``: a rolling ratio over the last ``window_size`` completed
      trials (step ``step_size``) within each session; X axis is the continuous
      (across-session) trial id. Reuses the rolling moving-average plotter.
    """
    fr_labels = _normalize_fr_types(fr_types)
    subjects = list(_collect_sessions(subjids, dates))
    if not subjects:
        return []

    subj_labels = []
    all_subjids = []
    all_dates = []
    n_sessions_total = 0
    n_sessions_with_fr = 0
    # Per subject: chronological list of per-session entry-lists ([(trial_idx,
    # binary), ...]; empty list = session with no usable data).
    per_subject_sessions = {}

    for subjid, date_vals, results_dirs in subjects:
        label = f"sub-{subjid}"
        subj_labels.append(label)
        all_subjids.append(subjid)
        all_dates.extend(date_vals)
        sess_entries = []
        for results_dir in results_dirs:
            n_sessions_total += 1
            df = _load_sorted_session(results_dir)
            entries = []
            if not df.empty:
                completed = df[df.get("is_aborted") == False]
                if not completed.empty and "false_response" in completed.columns:
                    n_sessions_with_fr += 1
                    fr_mask = _fr_mask(completed, fr_labels)
                    entries = [
                        (int(idx), 1.0 if is_fr else 0.0)
                        for idx, is_fr in zip(completed["_trial_idx"], fr_mask)
                    ]
            sess_entries.append(entries)
        per_subject_sessions[label] = sess_entries

    if n_sessions_with_fr == 0:
        print(
            f"[FR_ratio] No single-reward data found: none of the {n_sessions_total} "
            f"loaded session(s) have a 'false_response' column. These columns are only "
            f"written for single-reward protocol sessions (re-run trial classification "
            f"if you expected them). Nothing to plot."
        )
        return []

    color_map = _subject_color_map(subj_labels)

    # Trim each subject's leading empty sessions so everyone starts at session 1;
    # mid-run empty sessions are kept as gaps (None).
    trimmed = {}
    max_len = 0
    for label in subj_labels:
        series = [e if e else None for e in per_subject_sessions[label]]
        series = _trim_leading_empty(series)
        if series:
            trimmed[label] = series
            max_len = max(max_len, len(series))

    if moving_avg:
        # Build session_data aligned by trimmed session index (subjects as groups)
        # and reuse the rolling plotter.
        session_data = [{"n_trials": 0, "groups": {}} for _ in range(max_len)]
        for label, series in trimmed.items():
            for j, entries in enumerate(series):
                if entries:
                    session_data[j]["groups"][label] = entries
        fig = _plot_summary_rolling(
            session_data,
            color_map=color_map,
            group_order=subj_labels,
            ylabel="FR Ratio",
            title="False response ratio (rolling)",
            window_size=window_size,
            step_size=step_size,
            ylim_bottom=0,
        )
    else:
        # Per-subject series of (ratio, n_completed) for the daily plot.
        series_by_subject = {}
        for label, series in trimmed.items():
            ratios = []
            for entries in series:
                if entries:
                    vals = [v for _, v in entries]
                    ratios.append((float(np.mean(vals)), len(vals)))
                else:
                    ratios.append(None)
            series_by_subject[label] = ratios
        fig = _plot_fr_ratio_daily(
            series_by_subject,
            color_map,
            subj_labels,
            "FR Ratio",
            "False response ratio (session)",
        )

    figs = []
    if fig is not None:
        figs.append(fig)
        if save:
            suffix = _summary_save_suffix(moving_avg, window_size, step_size)
            save_figure(
                fig,
                f"FR_ratio_{suffix}",
                subjids=all_subjids,
                dates=all_dates,
            )
    return figs


def _plot_latency_box_AB(data, y_label, title):
    """Grouped boxplot (mean ± SEM with jittered points) split by reward port.

    ``data`` maps each group label ("False Response", "Reward") to
    ``{"A": [...], "B": [...]}``. Port A is drawn in red (left offset), port B in
    green (right offset), mirroring the A/B styling in ``fa_analysis``.
    """
    labels = list(data.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    mean_half_width = 0.14
    cap_half_width = 0.08
    mean_lw = 2.2
    sem_lw = 1.4
    point_offset = 0.18
    jitter_half_width = 0.12
    rng = np.random.default_rng(0)

    for i, label in enumerate(labels, start=1):
        for port_label, color, offset in (
            ("A", "red", -point_offset),
            ("B", "green", point_offset),
        ):
            values = data[label].get(port_label, [])
            if not values:
                continue
            x_pos = i + offset
            xs = x_pos + rng.uniform(-jitter_half_width, jitter_half_width, size=len(values))
            ax.scatter(xs, values, s=18, color=color, alpha=0.4, zorder=2)
            mean_val = float(np.mean(values))
            sem_val = (
                float(np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
            ax.hlines(
                mean_val,
                x_pos - mean_half_width,
                x_pos + mean_half_width,
                colors="black",
                linewidth=mean_lw,
                zorder=3,
            )
            ax.vlines(
                x_pos,
                mean_val - sem_val,
                mean_val + sem_val,
                colors="black",
                linewidth=sem_lw,
                zorder=3,
            )
            if sem_val > 0:
                ax.hlines(
                    mean_val - sem_val,
                    x_pos - cap_half_width,
                    x_pos + cap_half_width,
                    colors="black",
                    linewidth=sem_lw,
                    zorder=3,
                )
                ax.hlines(
                    mean_val + sem_val,
                    x_pos - cap_half_width,
                    x_pos + cap_half_width,
                    colors="black",
                    linewidth=sem_lw,
                    zorder=3,
                )

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(y_label)
    ax.set_xlabel("")
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.legend(
        handles=[
            Patch(facecolor="red", edgecolor="black", label="Port A"),
            Patch(facecolor="green", edgecolor="black", label="Port B"),
        ],
        loc="upper right",
    )
    fig.tight_layout()
    return fig


def _session_fr_latencies(completed, fr_labels):
    """False-response latencies for one session: (all_values, {"A": [...], "B": [...]}).

    ``all_values`` includes every FR latency (even when the port is unknown); the
    per-port dict only includes trials with a resolvable ``fr_port``.
    """
    if "false_response" not in completed.columns:
        return [], {"A": [], "B": []}
    fr_df = completed[_fr_mask(completed, fr_labels)]
    all_vals = []
    by_port = {"A": [], "B": []}
    for _, row in fr_df.iterrows():
        lat = row.get("fr_latency_ms")
        if lat is None or pd.isna(lat):
            continue
        lat = float(lat)
        all_vals.append(lat)
        port = _port_label(row.get("fr_port"))
        if port is not None:
            by_port[port].append(lat)
    return all_vals, by_port


def _session_reward_rts(completed):
    """Reward response times for one session: (all_values, {"A": [...], "B": [...]})."""
    rew_df = completed[completed.get("response_time_category") == "rewarded"]
    all_vals = []
    by_port = {"A": [], "B": []}
    for _, row in rew_df.iterrows():
        rt = _response_time_ms(row)
        if rt is None:
            continue
        all_vals.append(rt)
        port = _port_label(row.get("first_supply_port"))
        if port is not None:
            by_port[port].append(rt)
    return all_vals, by_port


def FR_latency(
    subjids: Optional[Iterable[int]] = None,
    dates: Optional[Union[Iterable[Union[int, str]], tuple]] = None,
    fr_types: Optional[Iterable[str]] = "FR_time_in",
    save: bool = False,
    split_AB: bool = False,
):
    """Joint boxplot of false-response latency vs. reward response time per subject.

    One figure per subject, pooling all selected days into a single boxplot with
    two groups on the X axis:

    - "False Response": ``fr_latency_ms`` of false-response trials
      (``false_response == True``, optionally filtered to ``fr_types`` via
      ``fr_label``; default ``"FR_time_in"`` only).
    - "Reward": response time of rewarded trials
      (``response_time_category == "rewarded"``), computed as in
      ``pred_seq_utils`` (last odor poke-out → first supply poke).

    Leading days are trimmed up to (and including) the first day on which FR data
    is available; days after that are all pooled, including mid-run days that
    happen to have no FR data (their reward data is still included).

    With ``split_AB=True`` each group is further split by port — false responses
    by ``fr_port`` and rewards by ``first_supply_port`` — into A (red) and B
    (green), still grouped by False Response vs. Reward.
    """
    fr_labels = _normalize_fr_types(fr_types)
    figs = []
    n_subjects_with_fr = 0
    for subjid, date_vals, results_dirs in _collect_sessions(subjids, dates):
        # Per-session extraction in chronological order.
        sessions = []
        for results_dir in results_dirs:
            df = _load_sorted_session(results_dir)
            if df.empty:
                sessions.append(None)
                continue
            completed = df[df.get("is_aborted") == False]
            if completed.empty:
                sessions.append(None)
                continue
            fr_vals, fr_port = _session_fr_latencies(completed, fr_labels)
            rew_vals, rew_port = _session_reward_rts(completed)
            sessions.append(
                {"fr_vals": fr_vals, "fr_port": fr_port, "rew_vals": rew_vals, "rew_port": rew_port}
            )

        # First day with FR data available; trim everything before it.
        start = next(
            (i for i, s in enumerate(sessions) if s is not None and s["fr_vals"]),
            None,
        )
        if start is None:
            continue
        n_subjects_with_fr += 1

        fr_all, rew_all = [], []
        fr_AB = {"A": [], "B": []}
        rew_AB = {"A": [], "B": []}
        for s in sessions[start:]:
            if s is None:
                continue
            fr_all += s["fr_vals"]
            rew_all += s["rew_vals"]
            for p in ("A", "B"):
                fr_AB[p] += s["fr_port"][p]
                rew_AB[p] += s["rew_port"][p]

        used_dates = list(date_vals[start:])

        if not split_AB:
            if not fr_all and not rew_all:
                continue
            groups = {"False Response": fr_all, "Reward": rew_all}
            fig, ax = plt.subplots(figsize=(7, 5))
            _plot_box_with_points(ax, groups, "Latency (ms)", "")
            ax.set_ylim(bottom=0)
            ax.set_title(f"Subjid {subjid} FR vs reward latency")
            fig.tight_layout()
            figs.append(fig)
            if save:
                save_figure(fig, "FR_latency", subjids=[subjid], dates=used_dates)
        else:
            data = {"False Response": fr_AB, "Reward": rew_AB}
            if not any(vals for group in data.values() for vals in group.values()):
                continue
            fig = _plot_latency_box_AB(
                data,
                "Latency (ms)",
                f"Subjid {subjid} FR vs reward latency (A/B)",
            )
            figs.append(fig)
            if save:
                save_figure(fig, "FR_latency_AB", subjids=[subjid], dates=used_dates)

    if n_subjects_with_fr == 0:
        print(
            "[FR_latency] No single-reward data found: none of the loaded session(s) "
            "have false-response data. These columns are only written for single-reward "
            "protocol sessions (re-run trial classification if you expected them). "
            "Nothing to plot."
        )
    return figs
