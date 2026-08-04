from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Optional, Union

from hypnose_behavior.io.layout import filter_sessions, layout_for, list_sessions

CACHE = OrderedDict()
CACHE_MAX_ITEMS = 40


def vprint(verbose: bool, *args, **kwargs):
    """print(...) only when verbose is True."""
    if verbose:
        print(*args, **kwargs)


def read_tracking_table(path: Union[str, Path]):
    """Read a tracking table from .parquet or .csv.

    Parquet preserves dtypes (tz-aware datetimes, nullable ints) natively; CSV keeps
    the historical utf-8/latin1 fallback.
    """
    import pandas as pd

    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def find_tracking_file(results_dir: Path, stem_glob: str) -> Optional[Path]:
    """Find a tracking file matching ``stem_glob`` (a filename glob WITHOUT extension),
    preferring .parquet over .csv. Returns None if nothing matches.

    Example: find_tracking_file(results_dir, "*_combined_sleap_tracking_timestamps")
    """
    for ext in ("parquet", "csv"):
        matches = [f for f in sorted(results_dir.glob(f"{stem_glob}.{ext}"))
                   if not f.name.startswith("._")]
        if matches:
            return matches[0]
    return None


def _update_cache(subjid, dates, data, kind):
    """Update cache entries for a subject/date set and kind."""
    global CACHE
    for date in dates:
        key = (subjid, date, kind)
        if key in CACHE:
            del CACHE[key]
        CACHE[key] = {
            "kind": kind,
            "data": data[date],
        }
    while len(CACHE) > CACHE_MAX_ITEMS:
        CACHE.popitem(last=False)


def _get_from_cache(subjid, date, kind):
    """Retrieve cached data for (subjid, date, kind)."""
    key = (subjid, date, kind)
    if key in CACHE and CACHE[key]["kind"] == kind:
        return CACHE[key]["data"]
    return None


def clear_cache():
    """Clear all cached items."""
    CACHE.clear()


def _iter_subject_dirs(derivatives_dir: Optional[Path], subjids: Optional[Iterable[int]]):
    """Yield (subjid, subject_dir) tuples from derivatives.

    Thin wrapper over the shared layout walker (restructure_2 Phase 2b) so this repo's
    ~21 call sites keep working unchanged. Named subjects that do not exist are still
    skipped rather than raised on; two directories for one subject now raise instead of
    yielding both.
    """
    yield from layout_for(derivatives_dir).iter_subjects(subjids)


def _filter_sessions(subj_dir: Path,
                     dates: Optional[Union[Iterable[Union[int, str]], tuple]]) -> list:
    """`SessionRef`s for a subject directory, filtered by date list or (start, end) range.

    A 2-tuple is an inclusive range (either bound may be None); any other iterable is a
    membership test; None means every session. That is the convention the call sites in
    `visualization/` already use.

    Prefer this over `_filter_session_dirs` in new code: a `SessionRef` carries `ses`,
    `date` and `session_index` alongside the path, which saves the caller re-parsing the
    directory name -- the habit that produced 17 copies of this lookup.
    """
    sessions = list_sessions(subj_dir)
    if isinstance(dates, tuple) and len(dates) == 2:
        return filter_sessions(sessions, date_range=dates)
    return filter_sessions(sessions, date=dates)


def _filter_session_dirs(subj_dir: Path,
                         dates: Optional[Union[Iterable[Union[int, str]], tuple]]):
    """Filter session directories by date list or (start, end) range."""
    return [s.path for s in _filter_sessions(subj_dir, dates)]
