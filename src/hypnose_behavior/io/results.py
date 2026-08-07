# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Reading a session's saved analysis results, and the merged-output conventions.

The read side of `io/save_results.py`: trial classification writes
`saved_analysis_results/`, this reads it back into the `results` dict every
metric wrapper consumes. Moved out of `metric_analysis/metrics_utils.py` in
restructure_2 Phase 4b, which separates plumbing from metric definitions.

`load_session_results` calls `metric_analysis.frames.build_position_data`. That
edge is deliberate and was checked, not assumed: `frames.py` is a leaf (standard
library and pandas only) and both package `__init__`s are docstring-only, so
`io -> metric_analysis.frames` is one-way with no cycle. See "Where
`build_position_data` lives" in `docs/metric_audit.md`. **Keep `frames.py` a
leaf** -- the day it imports anything else in the package this becomes a real
cycle.
"""

import json

import pandas as pd

from hypnose_behavior.io.layout import derivatives
from hypnose_behavior.io.paths import get_derivatives_root
from hypnose_behavior.metric_analysis.frames import build_position_data

__all__ = [
    "load_session_results",
    "merged_results_output_dir",
    "merged_metrics_filename",
]


def load_session_results(subjid, date):
    """
    Load saved analysis results for a given subject and date.
    Returns a dict with trial_data, non-initiated tables, and metadata.
    """
    # One resolver for the whole family (restructure_2 Phase 2b); it reports the
    # available sessions on a miss and raises rather than warning on an ambiguous
    # subject or date.
    session = derivatives.find_session(subjid, date=date)
    subject_dir = session.subject_dir
    session_dir = session.path

    results_dir = session_dir / "saved_analysis_results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Load manifest and summary
    manifest = json.load(open(results_dir / "manifest.json"))
    summary = json.load(open(results_dir / "summary.json"))

    results: dict = {}

    # Prefer the unified trial_data parquet; fall back to CSV if needed
    trial_parquet = results_dir / "trial_data.parquet"
    trial_csv = results_dir / "trial_data.csv"
    trial_df = pd.DataFrame()
    if trial_parquet.exists():
        try:
            trial_df = pd.read_parquet(trial_parquet)
        except Exception as e:
            print(f"Warning: failed to read {trial_parquet}: {e}")
    if trial_df.empty and trial_csv.exists():
        trial_df = pd.read_csv(trial_csv)
    results["trial_data"] = trial_df

    # Long per-position frame, derived here rather than written by the classifier,
    # so metrics never parse a JSON blob and legacy sessions need no
    # compatibility branch (D0, tier 2). Phase 7b's position_data side-table
    # turns this from a derivation into a read.
    results["position_data"] = build_position_data(trial_df)

    # The three `non_initiated_*` tables are deliberately not loaded. Phase 4a
    # step 6 dropped non-initiated trials from the metric set: they are not in
    # `trial_data`, so every metric over them needed its own frame and its own
    # shape, and integrating them properly is its own piece of work. Trial
    # classification still writes the tables; nothing in `metric_analysis` reads
    # them.

    # Attach manifest and summary
    results["manifest"] = manifest
    results["summary"] = summary
    results["results_dir"] = str(results_dir)

    return results


# The two helpers below have **no callers**: `batch_run_all_metrics_with_merge`
# builds its merged paths inline. They are moved here rather than deleted because
# 4b is a restructuring phase, and the plan's "all derivatives-path conventions in
# one place" is the right home for them if the batch driver is ever repointed at
# them. Candidates for deletion in Phase 9/10 otherwise.

def merged_results_output_dir(subjids, dates, protocol):
    """
    Determine the output directory for merged results based on subjids, dates, and protocol.
    """
    derivatives_dir = get_derivatives_root()
    subjids = sorted(set(str(s) for s in subjids))
    dates = sorted(set(str(d) for d in dates))
    if len(subjids) == 1:
        subj_dir = derivatives.subject_dir(subjids[0])
        merged_dir = subj_dir / "merged_results"
    else:
        merged_dir = derivatives_dir / "merged"
        merged_dir = merged_dir / ("protocol_merged" if protocol else "merged")
    merged_dir.mkdir(parents=True, exist_ok=True)
    return merged_dir


def merged_metrics_filename(subjids, dates, protocol):
    """
    Construct merged metrics filename based on subjids, dates, and protocol.
    """
    subjids = sorted(set(str(s) for s in subjids))
    dates = sorted(set(str(d) for d in dates))
    n_dates = len(dates)
    if len(subjids) == 1:
        proto = protocol if protocol else "all"
        fname = f"merged_{proto}_{n_dates}_dates"
    else:
        subj_str = "_".join(subjids)
        fname = f"merged_subjids_{subj_str}_{n_dates}_dates"
    return fname
