# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Latencies: how long between one thing happening and the next.

The trial-timing family is indexed by ``global_trial_id``, so pass one session's
frames -- pooled frames repeat ids and the index alignment mis-pairs trials.

Two quantities here share an everyday name with a ``trial_data`` column and are
**not** it (the audit's finding 11): ``reward_delivery_latency`` is measured from
leaving the odor port, where ``response_time_ms`` is measured from the reward-port
poke. The FA counterpart, ``fa_latency_from_pokeout``, lives in ``false_alarm.py``
-- grouped by what it measures, not by the fact that it returns a time.

The 10x-group-mean outlier rule that ``pred_seq_utils.response_time`` and
``fa_analysis`` apply is deliberately **not** here. Judgement call 4 of the audit
settles it: metrics raw, filtering is display -- so the rule stays in
``visualization/``, where it can be seen and changed.
"""

import numpy as np
import pandas as pd

from hypnose_behavior.metric_analysis.metrics.common import (
    _latency_ms,
    _trial_position_frame,
    _trial_timestamp,
    _tz_naive,
)

__all__ = [
    "avg_response_time", "avg_response_time_session",
    "inter_trial_interval",
    "reward_delivery_latency", "valve_to_reward_latency",
]


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


def _deepest_position_timestamp(position_data, blob, field):
    """`field` at each trial's deepest position, tz-naive, indexed by trial id."""
    rows = _trial_position_frame(position_data, blob)
    if rows is None:
        return None
    frame = pd.DataFrame({"gid": rows["global_trial_id"].to_numpy(),
                          "ts": _tz_naive(rows[field]).to_numpy()})
    return frame.groupby("gid", sort=True)["ts"].agg(lambda s: s.iloc[-1])


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
