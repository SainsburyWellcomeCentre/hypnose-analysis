"""Figure destinations for the behavioural derivatives tree, plus the shared
styling/saving re-exported from hypnose-helpers.

The styles, `save_figure` and the small figure utilities moved to
`hypnose_helpers.viz` during restructure_2 Phase 2a. What stays here is the part that
knows THIS dataset's layout: resolving `sub-{id:03d}_id-*` / `ses-*_date-*` directories.
"""
from __future__ import annotations

import os
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

from hypnose_helpers.viz.styles import (  # noqa: F401  (re-exported for existing callers)
    nature_style, poster_style, presentation_style, use_style, use_presentation_style,
    nice_x_locator, _presentation_active, _resolve_style,
)
from hypnose_helpers.viz.save import (  # noqa: F401
    set_size, strip_legends, _coerce_list, _unique_sorted, _format_span,
)
from hypnose_helpers.viz.save import save_figure as _save_figure


# Apply the default (nature) style globally so display and saved figures match.
# To switch styles for a notebook, call one of:
#     mpl.rcParams.update(poster_style())
#     use_presentation_style()          # presentation (also caps y-ticks)
#
# NOTE (restructure_2): helpers deliberately does NOT mutate rcParams at import. This
# line is kept here for now so figure appearance is unchanged; removing it is a separate,
# deliberate change -- the regression fingerprint covers trial_data + metrics, not
# figures, so it cannot catch a restyle.
mpl.rcParams.update(nature_style())

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def _resolve_subject_dir(deriv_root: Path, subjid: int) -> Path:
    candidates = list(deriv_root.glob(f"sub-{subjid:03d}_id-*"))
    if not candidates:
        raise FileNotFoundError(f"No subject directory found for sub-{subjid:03d} under {deriv_root}")
    return candidates[0]


def _resolve_session_dir(subj_dir: Path, date) -> Path:
    date_str = str(date)
    candidates = list(subj_dir.glob(f"ses-*_date-{date_str}"))
    if not candidates:
        raise FileNotFoundError(f"No session directory for date {date_str} under {subj_dir}")
    return candidates[0]


# Optional hook letting a *consuming* repo with a different derivatives layout
# reuse save_figure without wrapping it. Registered once at import; save_figure
# then resolves through it instead of resolve_figure_dir(). None = use the
# default hypnose-behavior-analysis layout below.
_FIGURE_DIR_RESOLVER = None


def set_figure_dir_resolver(fn) -> None:
    """Register a ``(subjids, dates) -> Path`` callable used by save_figure.

    Lets another project (e.g. hypnose-eeg-preprocessing, whose derivatives tree
    is laid out differently) reuse save_figure — and therefore the shared styles —
    with one registration call instead of a wrapper. Pass None to restore the
    default `resolve_figure_dir` behaviour.

    An explicit ``fig_dir=`` argument to save_figure still takes precedence.
    """
    global _FIGURE_DIR_RESOLVER
    _FIGURE_DIR_RESOLVER = fn


def resolve_figure_dir(subjids, dates=None) -> Path:
    """Determine where to save figures based on subject/session scope.

    Rules:
    - Multiple subjects: figures at derivatives_root / "figures".
    - Single subject, multiple sessions: figures at subject_dir / "figures".
    - Single subject, single session: figures at session_dir / "figures".
    """
    # Imported here rather than at module scope: paths.py requires Python 3.10+,
    # and consumers that supply their own figure directory (see
    # set_figure_dir_resolver) never reach this function.
    from hypnose_behavior.io.paths import get_derivatives_root

    deriv_root = Path(get_derivatives_root())
    subj_list = _coerce_list(subjids)
    date_list = _coerce_list(dates)

    if len(subj_list) == 0:
        raise ValueError("At least one subjid is required to resolve figure path")

    if len(subj_list) > 1:
        fig_dir = deriv_root / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        return fig_dir

    # Single subject
    subj_dir = _resolve_subject_dir(deriv_root, int(subj_list[0]))

    if len(date_list) <= 1 and len(date_list) == 1:
        ses_dir = _resolve_session_dir(subj_dir, date_list[0])
        fig_dir = ses_dir / "figures"
    else:
        fig_dir = subj_dir / "figures"

    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def save_figure(fig, save_name: str, *, subjids, dates=None, subdir=None,
                fig_dir=None, **kwargs):
    """Save a figure into the behavioural derivatives tree.

    Resolves the destination, then delegates to `hypnose_helpers.viz.save.save_figure`.
    Directory resolution, most specific first: an explicit `fig_dir` wins; then a resolver
    registered by a consuming repo; otherwise this repo's subject/session layout.
    """
    if fig_dir is None:
        if _FIGURE_DIR_RESOLVER is not None:
            fig_dir = _FIGURE_DIR_RESOLVER(subjids, dates)
        else:
            fig_dir = resolve_figure_dir(subjids, dates)
    return _save_figure(fig, save_name, fig_dir=fig_dir, subjids=subjids, dates=dates,
                        subdir=subdir, **kwargs)
