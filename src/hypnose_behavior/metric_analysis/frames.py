# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""The frames metric cores consume, and the helpers that build them.

Phase 4a (restructure_2) gives every metric a pure ``f(frame) -> value`` core.
There are two frames:

``trials``
    ``trial_data`` -- one row per trial. Unchanged.

``position_data``
    long form, one row per ``trial x position``. Derived at *load* time by
    expanding the per-trial JSON blobs ``position_poke_times`` /
    ``presentations`` / ``position_valve_times``, so metrics never parse a blob
    and carry no legacy-session branch. Pairs with Phase 7b's ``position_data``
    side-table, which will make the expansion a read rather than a derivation.

This module also owns the two "how far did the sequence get" helpers that the
Phase 4a audit's Q5 resolution mandates. See ``sequence_depth`` for the one
place where today's behaviour and the eventual definition differ, and why.
"""

import json
from typing import Optional

import pandas as pd

__all__ = [
    "parse_json_column",
    "presented_positions",
    "sequence_depth",
    "sampled_positions",
    "reached_counts",
    "build_position_data",
]


def parse_json_column(val):
    """Best-effort decode of a JSON-blob cell.

    Cells arrive as ``str`` from CSV and as already-decoded objects from
    parquet, so non-strings pass straight through.
    """
    if isinstance(val, str):
        try:
            val_fixed = val.replace('""', '"')
            return json.loads(val_fixed)
        except Exception:
            return {} if val.strip().startswith("{") else []
    return val


def _is_aborted(trial) -> bool:
    # `== True` rather than `bool(...)`, to match the `df["is_aborted"] == True`
    # mask every caller builds: a missing column yields False, and a stringy
    # "True" is not silently promoted.
    return trial.get("is_aborted") == True  # noqa: E712


def _as_int(val) -> Optional[int]:
    try:
        return int(val) if pd.notnull(val) else None
    except (TypeError, ValueError):
        return None


def _last_position(trial) -> Optional[int]:
    for col in ("last_odor_position", "last_event_index"):
        if col in trial:
            return _as_int(trial.get(col))
    return None


def _max_poke_time_position(trial) -> Optional[int]:
    """Deepest position with a poke entry, or None.

    All-or-nothing on the key cast, matching the walk this replaced: one
    unparseable key discards the whole blob and defers to ``_last_position``.
    """
    ppt = parse_json_column(trial.get("position_poke_times", {}))
    if isinstance(ppt, dict) and ppt:
        try:
            return max(int(k) for k in ppt.keys())
        except Exception:
            return None
    return None


def sequence_depth(trial) -> Optional[int]:
    """How far the sequence got, as a contiguous depth ``1..n``. Never filtered.

    This is the denominator of every per-position rate ("of the trials that
    reached position *p*, how many aborted there"), so it must be **monotonic**:
    a trial that reached position 5 reached positions 1-4 as well. That is why
    it returns a depth to be filled contiguously rather than a set of positions
    -- see ``sampled_positions`` for the filterable, possibly-gappy counterpart,
    which answers a different question and must not be substituted here.

    The contiguous fill is doing real work today, not just tidying: a position
    whose poke registers as ~0 ms is currently omitted from
    ``position_poke_times``, ``presentations`` *and* ``num_odors`` even though
    the odor was presented and the sequence advanced through it. ``1..max``
    recovers it; plain membership would silently under-count.

    **On the source, and a deliberate difference from the audit's target
    definition.** ``docs/metric_audit.md`` (Q5) specifies the eventual form as
    ``1..max(position in presentations)`` for every trial. That is the right end
    state, but it is not what the canonical metrics compute today: for an
    aborted trial they walk ``1..last_odor_position``, and the two disagree
    whenever the ``PRE_ODOR_GRACE_MS`` path wrote a synthetic trailing position
    that the abort-detection logic never counted (``classification_utils``
    :1281-1293 vs :2986), or ``last_odor_position`` is null.

    Adopting the ``presentations`` form now would not be *more* correct -- it
    would bake that grace artifact into the denominators, since nothing yet
    distinguishes a genuine short poke from a synthesised one. Phase 7b adds
    ``poke_source`` and writes the 0 ms positions; only then can the two sources
    agree. So this reproduces **today's** rule exactly, and the aborted/completed
    split below then collapses into the single ``presentations`` form.

    Measured on the 9 fixture sessions: 10 of 1731 trials disagree, changing the
    reached counts on 3 of them (sub-057, sub-059, sub-048) and with them
    ``abortion_rate_positionX`` and ``fa_abortion_stats``.
    """
    if _is_aborted(trial):
        return _last_position(trial)
    max_pos = _max_poke_time_position(trial)
    return max_pos if max_pos is not None else _last_position(trial)


def presented_positions(trial) -> list[int]:
    """``sequence_depth`` as the contiguous position list ``[1..depth]``."""
    depth = sequence_depth(trial)
    return list(range(1, depth + 1)) if depth else []


def reached_counts(trials) -> dict[int, int]:
    """``{position: n trials that reached it}`` -- the per-position denominator.

    The single definition of "reached" for the whole package. Before Phase 4a
    this walk was written out four times: twice identically in ``metrics_utils``
    (``abortion_rate_positionX`` and ``fa_abortion_stats``) and twice more in
    ``visualization/`` under two *different* definitions.
    """
    reached: dict[int, int] = {}
    for _, trial in trials.iterrows():
        for pos in presented_positions(trial):
            reached[pos] = reached.get(pos, 0) + 1
    return reached


def sampled_positions(trial, *, only_true_pokes: bool = False) -> Optional[list[int]]:
    """Positions with a recorded poke on this trial. May be gappy; filterable.

    "Was this position sampled" -- the counterpart to ``sequence_depth``'s "how
    far did the sequence get". A gap is meaningless for the latter and perfectly
    natural here ("no sample at position 3 on this trial"), which is why they are
    two helpers rather than one with a flag.

    ``only_true_pokes=True`` keeps only entries marked ``poke_source == "poke"``,
    excluding grace-synthesised and 0 ms entries. That field does not exist yet
    and will never exist on already-saved sessions, so when it is **absent this
    returns None** -- callers must omit the filtered variant rather than fall
    back to the unfiltered value, which would make old and new sessions look
    comparable when they are not. Phase 7b adds the field.
    """
    ppt = parse_json_column(trial.get("position_poke_times", {}))
    entries = ppt if isinstance(ppt, dict) else {}
    if not entries:
        pres = parse_json_column(trial.get("presentations", []))
        if isinstance(pres, list):
            entries = {
                p.get("position"): p for p in pres
                if isinstance(p, dict) and p.get("position") is not None
            }

    if only_true_pokes:
        if not entries:
            # No positions recorded at all is an *empty* answer, not an
            # unanswerable one. Only an absent `poke_source` yields None, so a
            # caller testing `is None` to decide whether to emit the filtered
            # variant does not also silently skip position-less trials.
            return []
        if not any(isinstance(v, dict) and v.get("poke_source") is not None
                   for v in entries.values()):
            return None
        entries = {k: v for k, v in entries.items()
                   if isinstance(v, dict) and v.get("poke_source") == "poke"}

    out = [_as_int(k) for k in entries.keys()]
    return sorted(p for p in out if p is not None)


# ---------------------------------------------------------------- position_data

# Fields carried by each blob. `presentations` is the only one with
# index_in_trial / is_last_event; only `position_poke_times` has the poke
# start/end timestamps; only `position_valve_times` has valve_end.
_POKE_FIELDS = ("poke_time_ms", "poke_odor_start", "poke_odor_end", "poke_first_in")
_VALVE_FIELDS = ("valve_start", "valve_end", "valve_duration_ms")
_PRES_FIELDS = ("index_in_trial", "is_last_event", "poke_time_ms", "poke_first_in",
                "valve_start", "valve_end", "valve_duration_ms")

_ID_COLUMNS = ("trial_id", "global_trial_id", "subjid", "date", "session_num")


def _entries_by_position(val) -> dict[int, dict]:
    """Normalise a position blob (dict-of-dicts or list-of-dicts) to {position: entry}."""
    parsed = parse_json_column(val)
    out: dict[int, dict] = {}
    if isinstance(parsed, dict):
        items = parsed.items()
    elif isinstance(parsed, list):
        items = ((e.get("position") if isinstance(e, dict) else None, e) for e in parsed)
    else:
        return out
    for key, entry in items:
        if not isinstance(entry, dict):
            continue
        pos = _as_int(entry.get("position", key))
        if pos is None:
            pos = _as_int(key)
        if pos is not None:
            out[pos] = entry
    return out


def build_position_data(trials) -> pd.DataFrame:
    """Expand the per-trial position blobs into one row per ``trial x position``.

    The long frame every per-position and per-odor metric groups on. Built from
    the union of ``position_poke_times``, ``presentations`` and
    ``position_valve_times``, because **the three do not carry the same
    positions** and a metric that reads one must not silently pick up rows from
    another:

    - on a *completed* trial ``position_valve_times`` holds every position with a
      valve activation, including ones whose poke registered as ~0 ms -- which
      ``position_poke_times``, ``presentations`` and ``num_odors`` all drop
      (Q5 pattern 2, seen from the writer's side);
    - on an *aborted* trial all three are restricted to positions with a poke.

    So each row records **which blobs it came from** (``in_poke_times`` /
    ``in_presentations`` / ``in_valve_times``) and every metric filters on the
    provenance matching the blob it reads today. Without that,
    ``manual_vs_auto_stop_preference`` -- which counts valve durations -- would
    gain the 0 ms positions and change value.

    ``poke_source`` is deliberately **not** synthesised: Phase 7b adds it, and an
    absent column is how ``sampled_positions`` knows to omit the
    ``only_true_pokes`` variants rather than return the unfiltered value.
    """
    if trials is None or len(trials) == 0:
        return pd.DataFrame()

    id_cols = [c for c in _ID_COLUMNS if c in trials.columns]
    rows = []
    for _, trial in trials.iterrows():
        poke = _entries_by_position(trial.get("position_poke_times", {}))
        pres = _entries_by_position(trial.get("presentations", []))
        valve = _entries_by_position(trial.get("position_valve_times", {}))
        aborted = _is_aborted(trial)

        for pos in sorted(set(poke) | set(pres) | set(valve)):
            p_entry, r_entry, v_entry = poke.get(pos, {}), pres.get(pos, {}), valve.get(pos, {})
            row = {c: trial.get(c) for c in id_cols}
            row.update({
                "position": pos,
                "is_aborted": aborted,
                "in_poke_times": pos in poke,
                "in_presentations": pos in pres,
                "in_valve_times": pos in valve,
                # odor_name agrees across blobs; take whichever is present.
                "odor_name": (p_entry.get("odor_name") or v_entry.get("odor_name")
                              or r_entry.get("odor_name")),
                "required_min_sampling_time_ms": (
                    p_entry.get("required_min_sampling_time_ms")
                    or v_entry.get("required_min_sampling_time_ms")
                    or r_entry.get("required_min_sampling_time_ms")),
            })
            # Poke fields: position_poke_times wins, presentations fills in.
            for f in _POKE_FIELDS:
                row[f] = p_entry.get(f, r_entry.get(f))
            # Valve fields: position_valve_times wins, presentations fills in.
            for f in _VALVE_FIELDS:
                row[f] = v_entry.get(f, r_entry.get(f))
            for f in ("index_in_trial", "is_last_event"):
                row[f] = r_entry.get(f)
            rows.append(row)

    return pd.DataFrame(rows)
