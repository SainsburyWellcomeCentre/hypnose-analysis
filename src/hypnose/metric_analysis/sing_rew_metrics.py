"""Single-reward ("singrew") outcome-category metrics.

Sorts every trial of a single-reward session into one of the signal-detection
outcomes -- Hit, Miss, False Alarm, Correct Rejection -- plus two "premature"
categories for ambiguous aborts, using the trial_data columns produced by the
single-reward trial classification (``is_aborted``, ``response_time_category``,
``reward_determinacy``, ``determined_final_odor``, ``fa_label``, ``false_response``,
``fa_port``).

For each (sub)category we record the trial count and the list of ``global_trial_id``s
so the membership can be recovered downstream.

Classification (mutually exclusive)
-----------------------------------
Hit:
  - is_aborted=False & response_time_category in {rewarded, unrewarded}
  - is_aborted=True  & reward_determinacy=rewarded    & fa_label != nFA
Miss:
  - is_aborted=False & response_time_category = timeout_delayed
  - is_aborted=True  & reward_determinacy=rewarded    & fa_label = nFA
False Alarm:
  - false_response = True
  - is_aborted=True  & reward_determinacy=nonrewarded & fa_label != nFA
Correct Rejection:
  - false_response = False
  - is_aborted=True  & reward_determinacy=nonrewarded & fa_label = nFA
Premature (ambiguous aborts):
  - premature_port_entry: is_aborted=True & reward_determinacy=ambiguous & fa_label != nFA
  - premature_abort:      is_aborted=True & reward_determinacy=ambiguous & fa_label = nFA

Subcategories
-------------
Hit:  rewarded_hit (completed, rewarded) / port_error_hit (completed, unrewarded) /
      anticipatory_hit (aborted, determined_final_odor port == fa_port) /
      anticipatory_port_error (aborted, determined_final_odor port != fa_port)
Miss: omission_miss (completed timeout_delayed) / forfeit_miss (aborted, rewarded, nFA)
False Alarm: completed_fa (false_response=True) / aborted_fa (aborted, nonrewarded, FA)
Correct Rejection: passive_cr (false_response=False) /
      active_cr (aborted, nonrewarded, nFA)

Anything matching no rule (e.g. an aborted ``off_protocol`` trial, or a session
missing the singrew columns) is collected under ``uncategorized``. A ``validation``
block flags any ``global_trial_id`` that is in no category or in more than one
category, so the partition can be audited.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd


# main category -> ordered list of its subcategories
CATEGORY_SUBCATEGORIES = {
    "hit": ["rewarded_hit", "port_error_hit", "anticipatory_hit", "anticipatory_port_error"],
    "miss": ["omission_miss", "forfeit_miss"],
    "false_alarm": ["completed_fa", "aborted_fa"],
    "correct_rejection": ["passive_cr", "active_cr"],
}

# top-level categories that have no subcategories
PREMATURE_CATEGORIES = ["premature_port_entry", "premature_abort"]

# every mutually-exclusive top-level category (excludes "uncategorized")
MAIN_CATEGORIES = list(CATEGORY_SUBCATEGORIES.keys()) + PREMATURE_CATEGORIES


def _to_bool(value):
    """Coerce a possibly-NaN / numpy / string boolean to a Python bool, or None."""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "1"):
            return True
        if s in ("false", "0", ""):
            return False
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return bool(value)


def _odor_to_identity(odor):
    """Map a final-odor name to its reward identity letter (OdorA -> "A"), else None."""
    if odor is None:
        return None
    try:
        if pd.isna(odor):
            return None
    except (TypeError, ValueError):
        pass
    s = str(odor).strip()
    if s.startswith("Odor"):
        s = s[4:].strip()
    return s or None


def _port_to_identity(port):
    """Map a reward-port id (1 -> "A", 2 -> "B") to its identity letter, else None."""
    try:
        p = int(port)
    except (TypeError, ValueError):
        return None
    return {1: "A", 2: "B"}.get(p)


def _fa_is_false_alarm(fa_label):
    """True if fa_label denotes an actual false alarm (i.e. present and != 'nFA')."""
    if fa_label is None:
        return False
    try:
        if pd.isna(fa_label):
            return False
    except (TypeError, ValueError):
        pass
    return str(fa_label) != "nFA"


def _fa_is_nfa(fa_label):
    """True if fa_label is exactly 'nFA' (no false alarm)."""
    return str(fa_label) == "nFA"


def _classify_trial(row):
    """Return (main_category, subcategory) for one trial row.

    ``subcategory`` is None for the premature categories and for "uncategorized".
    """
    is_aborted = _to_bool(row.get("is_aborted")) or False
    rtc = row.get("response_time_category")
    rd = row.get("reward_determinacy")
    fa_label = row.get("fa_label")
    fr = _to_bool(row.get("false_response"))
    fa_is_fa = _fa_is_false_alarm(fa_label)
    fa_is_nfa = _fa_is_nfa(fa_label)

    # --- HIT ---
    if (not is_aborted) and rtc in ("rewarded", "unrewarded"):
        return "hit", ("rewarded_hit" if rtc == "rewarded" else "port_error_hit")
    if is_aborted and rd == "rewarded" and fa_is_fa:
        expected = _odor_to_identity(row.get("determined_final_odor"))
        actual = _port_to_identity(row.get("fa_port"))
        if expected is not None and actual is not None and expected == actual:
            return "hit", "anticipatory_hit"
        return "hit", "anticipatory_port_error"

    # --- MISS ---
    if (not is_aborted) and rtc == "timeout_delayed":
        return "miss", "omission_miss"
    if is_aborted and rd == "rewarded" and fa_is_nfa:
        return "miss", "forfeit_miss"

    # --- FALSE ALARM ---
    if fr is True:
        return "false_alarm", "completed_fa"
    if is_aborted and rd == "nonrewarded" and fa_is_fa:
        return "false_alarm", "aborted_fa"

    # --- CORRECT REJECTION ---
    if fr is False:
        return "correct_rejection", "passive_cr"
    if is_aborted and rd == "nonrewarded" and fa_is_nfa:
        return "correct_rejection", "active_cr"

    # --- PREMATURE (ambiguous aborts) ---
    if is_aborted and rd == "ambiguous" and fa_is_fa:
        return "premature_port_entry", None
    if is_aborted and rd == "ambiguous" and fa_is_nfa:
        return "premature_abort", None

    return "uncategorized", None


def _empty_bucket():
    return {"n": 0, "global_trial_ids": []}


def _empty_structure():
    out = {}
    for cat, subs in CATEGORY_SUBCATEGORIES.items():
        out[cat] = {"n": 0, "global_trial_ids": [], "subcategories": {s: _empty_bucket() for s in subs}}
    for cat in PREMATURE_CATEGORIES:
        out[cat] = _empty_bucket()
    out["uncategorized"] = _empty_bucket()
    out["total_trials"] = 0
    out["validation"] = {
        "n_total_trials": 0,
        "n_classified": 0,
        "not_in_any_category": {"n": 0, "global_trial_ids": []},
        "in_multiple_categories": {"n": 0, "global_trial_ids": {}},
        "n_trials_missing_global_trial_id": 0,
    }
    return out


def _stage_names(results):
    """Collect all stage_name strings from the session summary/manifest runs."""
    names = []
    for key in ("summary", "manifest"):
        container = results.get(key)
        if not isinstance(container, dict):
            continue
        runs = (container.get("session", {}) or {}).get("runs", []) or []
        for run in runs:
            if isinstance(run, dict):
                stage = run.get("stage", {}) or {}
                name = stage.get("stage_name")
                if name:
                    names.append(str(name))
    return names


def is_singrew_session(results) -> bool:
    """Whether the session uses the single-reward protocol.

    Prefers the reliable ``session.is_singrew`` flag written to manifest/summary by
    the trial classifier (derived from the schema's sequence reward info). Falls back
    to a stage_name "singrew" substring match for older sessions saved before the flag
    existed.
    """
    for key in ("summary", "manifest"):
        container = results.get(key)
        if isinstance(container, dict):
            flag = (container.get("session", {}) or {}).get("is_singrew")
            if flag is not None:
                return bool(flag)
    return any("singrew" in name.lower() for name in _stage_names(results))


def compute_sing_rew_metrics(results) -> dict:
    """Sort all trials into outcome (sub)categories with per-(sub)category trial counts
    and ``global_trial_id`` lists, plus a ``validation`` block (see module docstring)."""
    out = _empty_structure()
    df = results.get("trial_data")
    if df is None or len(df) == 0:
        return out

    out["total_trials"] = int(len(df))
    has_gid = "global_trial_id" in df.columns

    # Track each gid's set of top-level categories to flag not-in-any / in-multiple.
    gid_to_categories = defaultdict(set)
    n_missing_gid = 0

    for _, row in df.iterrows():
        cat, sub = _classify_trial(row)
        gid = row.get("global_trial_id") if has_gid else None
        try:
            gid = int(gid) if gid is not None and not pd.isna(gid) else None
        except (TypeError, ValueError):
            gid = None
        if gid is None:
            n_missing_gid += 1

        if cat == "uncategorized":
            out["uncategorized"]["n"] += 1
            if gid is not None:
                out["uncategorized"]["global_trial_ids"].append(gid)
                gid_to_categories[gid].add("uncategorized")
            continue

        out[cat]["n"] += 1
        if gid is not None:
            out[cat]["global_trial_ids"].append(gid)
            gid_to_categories[gid].add(cat)
        if sub is not None:
            bucket = out[cat]["subcategories"][sub]
            bucket["n"] += 1
            if gid is not None:
                bucket["global_trial_ids"].append(gid)

    # --- Validation: flag gids in no real category, or in more than one ---
    not_in_any = sorted(out["uncategorized"]["global_trial_ids"])
    in_multiple = {
        gid: sorted(cats) for gid, cats in gid_to_categories.items() if len(cats) > 1
    }
    n_classified = sum(out[cat]["n"] for cat in MAIN_CATEGORIES)
    out["validation"] = {
        "n_total_trials": out["total_trials"],
        "n_classified": n_classified,
        "not_in_any_category": {"n": len(not_in_any), "global_trial_ids": not_in_any},
        "in_multiple_categories": {
            "n": len(in_multiple),
            "global_trial_ids": {str(k): v for k, v in sorted(in_multiple.items())},
        },
        "n_trials_missing_global_trial_id": n_missing_gid,
    }
    return out
