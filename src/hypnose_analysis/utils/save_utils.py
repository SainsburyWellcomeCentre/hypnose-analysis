import os
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from hypnose_analysis.paths import get_derivatives_root



# --------------------------------------
# Style Presets
# --------------------------------------

def nature_style() -> dict:
    """
    Return rcParams dict for 'nature-style' figures.
    """
    return {
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "regular",

        # Axes
        "axes.linewidth": 0.8,
        "axes.labelsize": 20,
        "axes.titlesize": 9,
        "axes.labelpad": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,          # ensure no grid
        "axes.facecolor": "white",
        "axes.formatter.useoffset": False,

        # Lines
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "lines.markeredgewidth": 0.8,

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,

        # Legend
        "legend.frameon": False,
        "legend.fontsize": 7,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4,

        # Color cycle
        "axes.prop_cycle": cycler(color=[
            "#E64B35", "#4DBBD5", "#00A087",
            "#3C5488", "#F39B7F", "#8491B4",
            "#91D1C2", "#DC0000", "#7E6148"
        ]),

        # Figure (keep display compact; saving uses explicit dpi)
        "figure.dpi": 110,
        "savefig.dpi": 600,
        "figure.facecolor": "white",

        # PDF/SVG
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        "image.composite_image": False,
    }


# Apply the style globally so display and saved figures match
mpl.rcParams.update(nature_style())

# --------------------------------------
# Size Presets
# --------------------------------------

def set_size(fig, width="single", aspect=0.75):

    if width == "single":
        w = 3.5
    elif width == "double":
        w = 7.2
    else:
        w = width  

    h = w * aspect
    fig.set_size_inches(w, h)


# --------------------------------------
# Save Utility
# --------------------------------------


def _coerce_list(val):
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        return list(val)
    return [val]


def _unique_sorted(items):
    try:
        return sorted(set(items))
    except Exception:
        return list(dict.fromkeys(items))


def _format_span(items, prefix: str) -> str:
    """Format subject/date identifiers into compact spans.

    - One item: prefix-<item>
    - Two items: prefix-<a>_<b>
    - Three or more: prefix-<first>-<last>
    """
    vals = _unique_sorted(items)
    if not vals:
        return ""
    if len(vals) == 1:
        return f"{prefix}-{vals[0]}"
    if len(vals) == 2:
        return f"{prefix}-{vals[0]}_{vals[1]}"
    return f"{prefix}-{vals[0]}-{vals[-1]}"


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


def resolve_figure_dir(subjids, dates=None) -> Path:
    """Determine where to save figures based on subject/session scope.

    Rules:
    - Multiple subjects: figures at derivatives_root / "figures".
    - Single subject, multiple sessions: figures at subject_dir / "figures".
    - Single subject, single session: figures at session_dir / "figures".
    """

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


def save_figure(
    fig: mpl.figure.Figure,
    save_name: str,
    *,
    subjids,
    dates=None,
    subdir: str | Path | None = None,
    dpi: int = 600,
    bbox_inches=None,
):
    """Save a matplotlib figure as PDF into a location derived from subject/session scope.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    save_name : str
        Base file name (without extension). Subject/date tags are appended automatically.
    subjids : int | list[int]
        Subject id(s) related to the figure.
    dates : int | list[int] | None
        Session date(s). Determines whether we save at session- or subject-level.
    subdir : str | Path | None
        Optional subdirectory inside the resolved figures directory. When provided,
        the folder is created automatically (e.g., "movement_figures").
    dpi : int
        Dots per inch passed to savefig (default 300).
    """

    if fig is None:
        raise ValueError("fig cannot be None")
    if not save_name:
        raise ValueError("save_name must be non-empty")
    
    subj_list = _coerce_list(subjids)
    date_list = _coerce_list(dates)

    subj_tag = _format_span([f"{int(s):03d}" for s in subj_list], "sub") if subj_list else "sub-unknown"
    date_tag = _format_span([int(d) if str(d).isdigit() else d for d in date_list], "date") if date_list else "date-unknown"

    filename = f"{save_name}_{subj_tag}_{date_tag}.pdf"

    fig_dir = Path(resolve_figure_dir(subjids, dates))
    if subdir:
        # Normalize to a relative path segment and avoid absolute traversal
        subdir_path = Path(str(subdir).strip()).as_posix().strip("./")
        if subdir_path:
            fig_dir = fig_dir / subdir_path
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_path = fig_dir / filename

    bbox = bbox_inches if bbox_inches is not None else "tight"
    fig.savefig(out_path, bbox_inches=bbox, dpi=dpi)

    return out_path

