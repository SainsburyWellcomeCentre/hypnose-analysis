from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Optional, Union

from hypnose_analysis.paths import get_derivatives_root

CACHE = OrderedDict()
CACHE_MAX_ITEMS = 40


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
    """Yield (subjid, subject_dir) tuples from derivatives."""
    base_dir = derivatives_dir or get_derivatives_root()
    if subjids is None:
        for subj_dir in sorted(base_dir.glob("sub-*_id-*")):
            try:
                subjid = int(subj_dir.name.split("_")[0].replace("sub-", ""))
            except Exception:
                subjid = None
            yield subjid, subj_dir
    else:
        for subjid in subjids:
            sub_str = f"sub-{int(subjid):03d}"
            for subj_dir in sorted(base_dir.glob(f"{sub_str}_id-*")):
                yield int(subjid), subj_dir


def _filter_session_dirs(subj_dir: Path, dates: Optional[Union[Iterable[Union[int, str]], tuple]]):
    """Filter session directories by date list or (start, end) range."""
    ses_dirs = sorted(subj_dir.glob("ses-*_date-*"))
    if dates is None:
        return ses_dirs

    def norm_date(d):
        s = str(d)
        return int(s) if s.isdigit() else None

    if isinstance(dates, tuple) and len(dates) == 2:
        start = norm_date(dates[0])
        end = norm_date(dates[1])
        out = []
        for d in ses_dirs:
            try:
                ds = int(d.name.split("_date-")[-1])
                if (start is None or ds >= start) and (end is None or ds <= end):
                    out.append(d)
            except Exception:
                pass
        return out

    wanted = {norm_date(d) for d in dates}
    out = []
    for d in ses_dirs:
        try:
            ds = int(d.name.split("_date-")[-1])
            if ds in wanted:
                out.append(d)
        except Exception:
            pass
    return out
