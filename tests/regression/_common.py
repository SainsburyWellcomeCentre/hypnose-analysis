"""Shared core for the restructuring regression harness.

The restructuring of this code-base is done as a series of *pure moves* (split /
rename / relocate, no logic changes). To prove each step preserves behaviour we
fingerprint the two pipeline outputs that must stay byte-for-byte identical:

  1. per-session ``trial_data`` (trial classification)
  2. the metrics dict (metric analysis)

Design choices that make the fingerprint trustworthy:

* Reads the real, read-only ``rawdata``; redirects ALL derivatives I/O to a
  throwaway temp dir via ``HYPNOSE_DERIVATIVES_ROOT`` so nothing touches the
  server and runs never collide.
* Fingerprints the *canonical CSV* of ``trial_data`` (sorted columns, reset
  index) -- NOT parquet bytes, whose pyarrow/version/compression metadata is
  non-deterministic -- and never the manifest/summary files (wall-clock
  timestamps live there).
* Fingerprints metrics from the *returned dict* (no file, no timestamps).
* Imports every pipeline entry point in ONE place (below). When modules move
  during the restructuring, only these import lines change -- the md5s must not.
"""
from __future__ import annotations

import os
import io
import json
import hashlib
import tempfile
import contextlib
from pathlib import Path

import pandas as pd

# --- single import surface -------------------------------------------------
# Update ONLY these lines as modules move during the restructuring. The md5
# fingerprints they produce must remain identical at every step.
import hypnose.io.paths as _paths
from hypnose.trial_classification.classification_utils import analyze_session_multi_run_by_id_date
from hypnose.metric_analysis.metrics_utils import load_session_results, run_all_metrics
# ---------------------------------------------------------------------------


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _canonical_trial_data(csv_path: Path) -> str:
    """Column-order- and index-independent CSV serialization of trial_data."""
    df = pd.read_csv(csv_path)
    df = df.reindex(sorted(df.columns), axis=1).reset_index(drop=True)
    return df.to_csv(index=False)


def _canonical_metrics(metrics: dict) -> str:
    """Deterministic, timestamp-free serialization of the metric values."""
    return json.dumps(metrics, sort_keys=True, default=str)


def _redirect_derivatives(tmp: Path) -> None:
    """Point all derivatives I/O at `tmp` and clear cached path lookups."""
    os.environ["HYPNOSE_DERIVATIVES_ROOT"] = str(tmp)
    for name in ("get_derivatives_root", "get_server_root", "get_rawdata_root"):
        fn = getattr(_paths, name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()


def fingerprint_session(subjid, date) -> dict:
    """Run classification + metrics for one session in an isolated temp
    derivatives dir and return ``{'trial_data': md5, 'metrics': md5}``.

    Raises on any failure so a broken session is never silently fingerprinted.
    """
    subjid = str(subjid)
    date = str(date)
    with tempfile.TemporaryDirectory(prefix="hyp_regress_") as tmp_str:
        tmp = Path(tmp_str)
        _redirect_derivatives(tmp)

        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            analyze_session_multi_run_by_id_date(
                subjid, date, verbose=False, save=True, print_summary=False
            )

        matches = list(tmp.glob(f"**/ses-*_date-{date}/saved_analysis_results/trial_data.csv"))
        if not matches:
            raise FileNotFoundError(
                f"trial_data.csv not found for subj={subjid} date={date} under {tmp}"
            )
        trial_data_md5 = _md5(_canonical_trial_data(matches[0]))

        with contextlib.redirect_stdout(sink):
            results = load_session_results(subjid, date)
            metrics = run_all_metrics(results, save_txt=False, save_json=False)
        metrics_md5 = _md5(_canonical_metrics(metrics))

    return {"trial_data": trial_data_md5, "metrics": metrics_md5}


def env_fingerprint() -> dict:
    """Versions that the md5s depend on; recorded with the fixtures."""
    import sys
    import numpy as np
    return {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
