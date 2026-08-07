# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

import sys
import os
import json
from dotmap import DotMap
import pandas as pd
import numpy as np
import math
from glob import glob
import ast
from IPython.display import display
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from typing import Iterable, Optional, Union
from hypnose_behavior.utils.helpers import _filter_session_dirs, _iter_subject_dirs
from hypnose_behavior.io.paths import get_derivatives_root
from hypnose_behavior.io.layout import list_sessions
from hypnose_behavior.io.loaders import _load_trial_views, _odor_to_letter
# `parse_json_column` is defined in frames.py (it is frame construction, not a
# metric) and re-exported here so existing importers keep working.
from hypnose_behavior.metric_analysis.frames import (  # noqa: F401
    build_position_data,
    odor_letter,
    odor_sequence_tokens,
    parse_json_column,
    presented_positions,
    reached_counts as _reached_counts,
    sampled_positions,
    sequence_depth,
)
# Phase 4b is splitting this module into `metric_analysis/metrics/`, one module
# per behavioural construct. What is left here has not moved yet; these imports
# are what the remainder still calls, and they re-export the moved names so
# importers keep working until the split finishes and this file goes.
from hypnose_behavior.metric_analysis.metrics.common import (  # noqa: F401
    _aborted_mask,
    _flag,
    _initiated,
    _is_truthy,
    _latency_ms,
    _position_rows,
    _reduce_rate,
    _trial_position_frame,
    _trial_timestamp,
    _truthy,
    _tz_naive,
    reduce_rate,
)
from hypnose_behavior.metric_analysis.metrics.accuracy import (  # noqa: F401
    choice_timeout_rate,
    choice_timeout_rate_contributions,
    choice_timeout_rate_session,
    decision_accuracy,
    decision_accuracy_by_odor,
    decision_accuracy_by_odor_session,
    decision_accuracy_contributions,
    decision_accuracy_session,
    global_choice_accuracy,
    global_choice_accuracy_contributions,
    global_choice_accuracy_session,
    response_rate,
    response_rate_contributions,
    response_rate_session,
    rolling_reward_fraction,
)
from hypnose_behavior.metric_analysis.metrics.sequence import (  # noqa: F401
    abortion_rate_positionX,
    abortion_rate_positionX_session,
    odor_initiation_bias,
    odor_initiation_bias_session,
    odorx_abortion_rate,
    odorx_abortion_rate_session,
    presentation_counts_by_odor,
    sequence_completion_rate,
    sequence_completion_rate_contributions,
    sequence_completion_rate_session,
)
from hypnose_behavior.metric_analysis.metrics.false_alarm import (  # noqa: F401
    FA_avg_response_times,
    FA_avg_response_times_session,
    FA_odor_bias,
    FA_odor_bias_session,
    FA_position_bias,
    FA_position_bias_session,
    fa_abortion_stats,
    fa_abortion_stats_session,
    fa_latency_from_pokeout,
    fa_port_counts,
    fa_port_label,
    fa_port_ratio,
    fa_port_ratio_by_odor,
    fa_port_ratio_by_odor_session,
    fa_port_share_a,
    fa_rate_by_odor,
    fa_rate_by_position,
    false_response_ratio,
    false_response_ratio_contributions,
    get_fa_ratio_a_stats,
    global_FA_rate,
    global_FA_rate_contributions,
    global_FA_rate_session,
    premature_response_rate,
    premature_response_rate_contributions,
    premature_response_rate_session,
    response_contingent_FA_rate,
    response_contingent_FA_rate_contributions,
    response_contingent_FA_rate_session,
)
from hypnose_behavior.metric_analysis.metrics.hidden_rule import (  # noqa: F401
    hidden_rule_counts_by_odor,
    hidden_rule_counts_by_odor_session,
    hidden_rule_detection_rate,
    hidden_rule_detection_rate_contributions,
    hidden_rule_detection_rate_session,
    hidden_rule_mask,
    hidden_rule_performance,
    hidden_rule_performance_contributions,
    hidden_rule_performance_session,
    hr_abort_poke_gap,
    hr_odor_associations,
    rolling_hr_reward_fraction,
)


# ================== Metric cores: tier 2, grouped inside a position blob =========
#
# restructure_2 Phase 4a, decision D0 tier 2. These metrics group by odor or by
# position -- keys that used to live inside a per-trial JSON blob, so each one
# parsed a blob inline. They now read `position_data`, the long
# `trial x position` frame `load_session_results` derives (see
# `metric_analysis/frames.build_position_data`).
#
# Summation style is reproduced deliberately, not incidentally: the two pooled
# `avg_sampling_time_*` metrics accumulate `total += x` left to right, while
# `avg_sampling_time_odor_x` calls `np.mean` on a per-odor list, which is
# pairwise. The two disagree in the last ULP over a few hundred values -- enough
# to move the metrics md5, and invisible in any printed output.


def _sequential_mean(values):
    """Mean by left-to-right accumulation.

    Matches the `total = 0.0; total += x` loops this replaces. `np.mean` and
    `Series.mean` sum pairwise and differ in the last ULP over a few hundred
    values, which moves the metrics fingerprint.
    """
    total = 0.0
    n = 0
    for v in values:
        total += v
        n += 1
    return total / n if n > 0 else np.nan


def avg_sampling_time_odor_x(position_data):
    """Mean `poke_time_ms` per odor over completed trials, from `position_poke_times`."""
    rows = _position_rows(position_data, "in_poke_times", aborted=False)
    if rows is None or rows.empty:
        return pd.Series(dtype=float)
    rows = rows[rows["odor_name"].notna() & rows["poke_time_ms"].notna()]
    if rows.empty:
        return pd.Series(dtype=float)
    # `np.mean` on each group's values rather than `Series.mean()`: this
    # replaces a per-odor Python list passed to `np.mean`, and pandas' reduction
    # kernel can land a ULP away. `rename`s keep the unnamed Series shape the
    # single-dict construction produced.
    avg_times = (rows.groupby("odor_name")["poke_time_ms"]
                 .apply(lambda s: np.mean(s.to_numpy()))
                 .rename(None).rename_axis(None).sort_index())
    return avg_times


def avg_sampling_time_odor_x_session(results):
    avg_times = avg_sampling_time_odor_x(results.get("position_data"))
    for odor, avg_time in avg_times.items():
        print(f"{odor} Average Sampling Time: {avg_time:.2f} ms")
    return avg_times


def avg_sampling_time_completed_sequence(position_data):
    """Pooled mean `poke_time_ms` over completed trials' `position_poke_times`."""
    rows = _position_rows(position_data, "in_poke_times", aborted=False)
    if rows is None or rows.empty:
        return np.nan
    return _sequential_mean(rows.loc[rows["poke_time_ms"].notna(), "poke_time_ms"])


def avg_sampling_time_completed_sequence_session(results):
    if results.get("trial_data", pd.DataFrame()).empty:
        return np.nan
    avg = avg_sampling_time_completed_sequence(results.get("position_data"))
    print(f"Average Sampling Time (Completed Sequences): {avg:.2f} ms")
    return avg


def avg_sampling_time_aborted_sequence(position_data):
    """Pooled mean `poke_time_ms` over aborted trials' `presentations`.

    Excludes the abort event itself -- the entry whose `index_in_trial` equals
    the trial's `last_event_index`. A null `last_event_index` matches nothing, so
    that trial contributes every entry, as it does today.
    """
    rows = _position_rows(position_data, "in_presentations", aborted=True)
    if rows is None or rows.empty:
        return np.nan
    idx = rows["index_in_trial"]
    keep = (idx.notna() & (idx != rows["last_event_index"])
            & rows["poke_time_ms"].notna())
    return _sequential_mean(rows.loc[keep, "poke_time_ms"])


def avg_sampling_time_aborted_sequence_session(results):
    # Silent on an empty trial table and on a session with no aborted trials --
    # both bail before the print today, where a session that aborted but
    # recorded no usable presentation still prints "nan ms".
    trials = results.get("trial_data", pd.DataFrame())
    if trials.empty or not _aborted_mask(trials).any():
        return np.nan
    avg = avg_sampling_time_aborted_sequence(results.get("position_data"))
    print(f"Average Sampling Time (Aborted Sequences): {avg:.2f} ms")
    return avg


def avg_response_time(trials):
    """Mean `response_time_ms` by category, plus the pooled rewarded+unrewarded."""
    if (trials.empty or "response_time_category" not in trials.columns
            or "response_time_ms" not in trials.columns):
        return {}
    vals = pd.to_numeric(trials["response_time_ms"], errors="coerce")
    out = {}
    for label, key in [("Rewarded", "rewarded"), ("Unrewarded", "unrewarded"),
                       ("Reward Timeout", "timeout_delayed")]:
        s = vals[trials["response_time_category"] == key].dropna()
        out[label] = float(s.mean()) if not s.empty else np.nan
    both = vals[trials["response_time_category"].isin(["rewarded", "unrewarded"])].dropna()
    out["Average Response Time (Rewarded + Unrewarded)"] = float(both.mean()) if not both.empty else np.nan
    return out


def avg_response_time_session(results):
    df = results.get("trial_data", pd.DataFrame())
    if df.empty or "response_time_category" not in df.columns or "response_time_ms" not in df.columns:
        print("No response time data available.")
        return {}
    out = avg_response_time(df)
    vals = pd.to_numeric(df["response_time_ms"], errors="coerce")
    for label, key in [("Rewarded", "rewarded"), ("Unrewarded", "unrewarded"),
                       ("Reward Timeout", "timeout_delayed")]:
        avg, n = out[label], len(vals[df["response_time_category"] == key].dropna())
        print(f"{label}: {avg:.1f} ms (n={n})" if not np.isnan(avg) else f"{label}: nan (n={n})")
    key = "Average Response Time (Rewarded + Unrewarded)"
    avg_both = out[key]
    n_both = len(vals[df["response_time_category"].isin(["rewarded", "unrewarded"])].dropna())
    print(f"{key}: {avg_both:.1f} ms (n={n_both})" if not np.isnan(avg_both)
          else f"{key}: nan (n={n_both})")
    return out


def manual_vs_auto_stop_preference(position_data):
    """Valve durations on completed trials, split at 1000 ms.

    Reads `position_valve_times` only. That blob is a *superset* of the other
    two -- it records positions whose poke registered as ~0 ms -- so this is the
    one metric that would gain rows if the provenance filter were dropped.
    """
    rows = _position_rows(position_data, "in_valve_times", aborted=False)
    if rows is None or rows.empty:
        return {"short_valve": 0, "long_valve": 0, "ratio": np.nan}
    dur = rows.loc[rows["valve_duration_ms"].notna(), "valve_duration_ms"]
    # `if dur <= 1000 ... elif dur >= 1000`: exactly 1000 ms counts short only.
    short = int((dur <= 1000).sum())
    long = int((dur > 1000).sum())
    return {"short_valve": short, "long_valve": long,
            "ratio": short / long if long > 0 else float('nan')}


def manual_vs_auto_stop_preference_session(results):
    if results.get("trial_data", pd.DataFrame()).empty:
        return {"short_valve": 0, "long_valve": 0, "ratio": np.nan}
    out = manual_vs_auto_stop_preference(results.get("position_data"))
    print(f"Manual Stops: {out['short_valve']}")
    print(f"Auto Stops: {out['long_valve']}")
    print(f"Manual vs Auto Stop: {out['ratio']:.2f}")
    return out


# ================== NEW metrics -- no canonical version before Phase 4a =========
#
# restructure_2 Phase 4a. Each of these was computed inside a plotter in
# `visualization/` and existed nowhere in `metric_analysis` -- the audit's `NEW`
# class, i.e. the "lose no metric" checklist. The arithmetic reproduces the
# plotters'; what stays behind in `visualization/` is axis construction,
# cross-subject aggregation and styling, which are properties of a figure rather
# than of the data.
#
# None of these is reached by `run_all_metrics`, so none enters `metrics_*.json`
# or the regression fingerprint. Whether to save them is a 4b question (the
# registry decides), not a 4a one.


def poke_durations(position_data, *, aborted=False):
    """Per-position poke durations for one outcome class, as a tidy frame.

    Completed trials come from `position_poke_times`; aborted trials from
    `presentations` with the abort event excluded -- the same sources, and the
    same exclusion, the canonical `avg_sampling_time_*` metrics use.

    **No `poke_time_ms > 0` filter.** The four extractors in `visualization/`
    each carried one; measured across all 9 fixture sessions it drops nothing,
    because a ~0 ms position is currently omitted by the writer entirely. Once
    Phase 7b writes those positions the filter would start excluding exactly the
    rows that fix adds, so it is removed rather than relocated --
    `sampled_positions(only_true_pokes=True)` is its proper successor.
    """
    empty = pd.DataFrame(columns=["position", "odor_name", "poke_time_ms"])
    if aborted:
        rows = _position_rows(position_data, "in_presentations", aborted=True)
        if rows is None or rows.empty:
            return empty
        idx = rows["index_in_trial"]
        rows = rows[idx.notna() & (idx != rows["last_event_index"])]
    else:
        rows = _position_rows(position_data, "in_poke_times", aborted=False)
        if rows is None or rows.empty:
            return empty
    rows = rows[rows["poke_time_ms"].notna()]
    if rows.empty:
        return empty
    return rows.loc[:, ["position", "odor_name", "poke_time_ms"]].reset_index(drop=True)


def _mean_sd_by(frame, key):
    """Mean, population SD and count of `poke_time_ms` per `key`.

    `np.mean` / `np.std` on each group's array, deliberately **not** the pandas
    reductions. Both are the population SD, but the two sum in a different order
    and disagree in the last ULP -- measured, that moved 28 drawn values in
    `plot_sampling_times_analysis` alone. This is the same "summation style is
    part of the metric" trap the audit records for `avg_sampling_time_*`.
    """
    if frame.empty:
        return pd.DataFrame(columns=["mean", "sd", "n"])
    grouped = frame.dropna(subset=[key]).groupby(key, sort=True)["poke_time_ms"]
    stats = {}
    for name, values in grouped:
        arr = values.to_numpy(dtype=float)
        stats[name] = (float(np.mean(arr)), float(np.std(arr)), int(arr.size))
    out = pd.DataFrame.from_dict(stats, orient="index", columns=["mean", "sd", "n"])
    out.index.name = key
    return out


def poke_duration_by_position(position_data, *, aborted=False):
    """Mean and population SD of `poke_time_ms` per position. Checklist 3."""
    return _mean_sd_by(poke_durations(position_data, aborted=aborted), "position")


def poke_duration_by_odor(position_data, *, aborted=False):
    """Mean and population SD of `poke_time_ms` per odor.

    Checklist 4 in its `aborted=True` form: the canonical
    `avg_sampling_time_aborted_sequence` pools every aborted trial into one
    scalar, and no per-odor version existed. With `aborted=False` it is the
    per-odor completed-trial mean, i.e. `avg_sampling_time_odor_x` with an SD
    and a count alongside.
    """
    return _mean_sd_by(poke_durations(position_data, aborted=aborted), "odor_name")


def inter_trial_interval(trials):
    """Seconds from one trial ending to the next starting. Checklist 6.

    `sequence_start.shift(-1) - sequence_end`, so the last row is NaN. Pass a
    single session's trials: shifting across a session boundary would measure the
    gap between recordings, which is not an inter-trial interval.
    """
    if (trials.empty or "sequence_start" not in trials.columns
            or "sequence_end" not in trials.columns):
        return pd.Series(np.nan, index=trials.index, dtype=float)
    start = _tz_naive(trials["sequence_start"])
    end = _tz_naive(trials["sequence_end"])
    return (start.shift(-1) - end).dt.total_seconds()


# ---- trial-timing family (checklist 17-22) -------------------------------------
#
# All indexed by `global_trial_id`, so pass one session's frames: pooled frames
# repeat ids and the index alignment below would mis-pair trials.
#
# The 10x-group-mean outlier rule that `pred_seq_utils.response_time` and
# `fa_analysis` apply is deliberately **not** here. Judgement call 4 of the audit
# settles it: metrics raw, filtering is display -- so the rule stays in
# `visualization/`, where it can be seen and changed.


def _deepest_position_timestamp(position_data, blob, field):
    """`field` at each trial's deepest position, tz-naive, indexed by trial id."""
    rows = _trial_position_frame(position_data, blob)
    if rows is None:
        return None
    frame = pd.DataFrame({"gid": rows["global_trial_id"].to_numpy(),
                          "ts": _tz_naive(rows[field]).to_numpy()})
    return frame.groupby("gid", sort=True)["ts"].agg(lambda s: s.iloc[-1])


def trial_poke_span(position_data):
    """Wall-clock span of a trial's odor-sampling phase, in ms. Checklist 17.

    `poke_odor_end` at the deepest position minus `poke_odor_start` at position 1.
    Distinct from `trial_poke_total`: the span contains the travel between ports,
    the sum does not. Trials missing either timestamp are dropped.
    """
    rows = _trial_position_frame(position_data, "in_poke_times")
    if rows is None:
        return pd.Series(dtype=float)
    frame = pd.DataFrame({
        "gid": rows["global_trial_id"].to_numpy(),
        # Only position 1 contributes a start, so `max` picks it out.
        "start": _tz_naive(rows["poke_odor_start"]).where(rows["position"] == 1).to_numpy(),
        "end": _tz_naive(rows["poke_odor_end"]).to_numpy(),
    })
    grouped = frame.groupby("gid", sort=True)
    span = grouped["end"].agg(lambda s: s.iloc[-1]) - grouped["start"].max()
    return span.dropna().dt.total_seconds() * 1000.0


def trial_poke_total(position_data):
    """Sum of `poke_time_ms` across a trial's positions, in ms. Checklist 21.

    Related to `avg_sampling_time_completed_sequence` but per trial rather than a
    session mean.
    """
    rows = _position_rows(position_data, "in_poke_times")
    if rows is None or rows.empty or "global_trial_id" not in rows.columns:
        return pd.Series(dtype=float)
    usable = rows[rows["poke_time_ms"].notna()]
    if usable.empty:
        return pd.Series(dtype=float)
    return usable.groupby("global_trial_id")["poke_time_ms"].sum()


def reward_delivery_latency(trials, position_data):
    """`first_supply_time` minus the last odor poke-out, in ms. Checklist 18.

    **Not** `trial_data.response_time_ms`, which is measured from the reward-port
    poke rather than from leaving the odor port -- the audit's finding 11, two
    quantities sharing an everyday name and not a definition. Written twice
    today, as `pred_seq_utils.response_time` and `sing_rew._response_time_ms`.
    """
    return _latency_ms(_trial_timestamp(trials, "first_supply_time"),
                       _deepest_position_timestamp(position_data, "in_poke_times",
                                                   "poke_odor_end"))


def valve_to_reward_latency(trials, position_data):
    """`first_supply_time` minus the last position's `valve_start`, in ms.

    Checklist 20. Nothing canonical measures anything from a valve opening.
    """
    return _latency_ms(_trial_timestamp(trials, "first_supply_time"),
                       _deepest_position_timestamp(position_data, "in_valve_times",
                                                   "valve_start"))

