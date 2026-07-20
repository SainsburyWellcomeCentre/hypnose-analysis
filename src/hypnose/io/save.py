import os
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from hypnose.io.paths import get_derivatives_root



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


def poster_style() -> dict:
    """
    Return rcParams dict for poster figures.

    Same typographic family and color cycle as nature_style(), but with:
    - titles hidden (axes.titlesize = 0)
    - larger fonts for axis labels, ticks, legend
    - thicker spines, ticks, and lines for visibility at viewing distance
    - 600 dpi for both display and save
    """
    return {
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 22,
        "font.weight": "bold",       # bold tick labels (no per-tick weight rcParam exists)
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "regular",

        # Axes
        "axes.linewidth": 3.5,
        "axes.labelsize": 36,
        "axes.labelweight": "bold",
        "axes.titlesize": 0,         # no per-axes titles on posters
        "axes.titleweight": "bold",
        "axes.titlepad": 0,
        "axes.labelpad": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.facecolor": "white",
        "axes.formatter.useoffset": False,

        # Figure-level title (suptitle) — also suppressed for posters
        "figure.titlesize": 0,
        "figure.titleweight": "bold",

        # Lines
        #"lines.linewidth": 2.5,
        #"lines.markersize": 8,
        #"lines.markeredgewidth": 1.5,

        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "lines.markeredgewidth": 0.8,

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "xtick.major.size": 12,
        "ytick.major.size": 12,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.labelsize": 28,
        "ytick.labelsize": 28,

        # Legend — off by default. Auto-added legends (e.g. from seaborn) become
        # invisible via fontsize=0; add legends manually with an explicit fontsize
        # kwarg, e.g. ax.legend(fontsize=18, frameon=False), when you want one.
        "legend.frameon": False,
        "legend.fontsize": 10,

        # Color cycle (matches nature_style for consistency across figure sets)
        "axes.prop_cycle": cycler(color=[
            "#E64B35", "#4DBBD5", "#00A087",
            "#3C5488", "#F39B7F", "#8491B4",
            "#91D1C2", "#DC0000", "#7E6148"
        ]),

        # Figure
        "figure.dpi": 110,
        "savefig.dpi": 600,
        "figure.facecolor": "white",

        # PDF/SVG
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        "image.composite_image": False,
    }


# Distinctive rcParam values that identify the presentation style at save time
# (used so the y-tick cap works no matter how the style was applied — via
# use_presentation_style() OR a bare mpl.rcParams.update(presentation_style())).
_PRES_AXES_LABELSIZE = 24
_PRES_TICK_LABELSIZE = 18
_PRES_AXES_LINEWIDTH = 2.0
# Boxplot-style figures get even bigger x-tick labels (the category positions
# are the most important thing to read). Applied by save_figure(boxplot=True).
_PRES_XTICK_BOXPLOT_LABELSIZE = 28

# Number of y-ticks the presentation style caps to (no rcParam exists for tick
# count, so save_figure enforces it per-axes). Configurable via
# use_presentation_style(max_yticks=...).
_PRESENTATION_MAX_YTICKS = 4
# Same idea for x-ticks: cap a numeric x-axis to a few nicely rounded values
# (5, 10, 15, ... rather than 7, 14, 21). Categorical x-axes (explicit string
# tick labels, e.g. boxplot/violin/position plots) are left untouched.
# Configurable via use_presentation_style(max_xticks=...).
_PRESENTATION_MAX_XTICKS = 5


def presentation_style() -> dict:
    """Return rcParams dict for 'presentation' figures (projector-friendly).

    Same as nature_style(), but tuned for readability on a big projector:
    - bigger, bold tick labels (x and y)
    - bigger, bold axis labels
    - thicker axis lines and ticks

    The y-axis tick count is also capped (default 4). That has no rcParam
    equivalent, so it is enforced per-axes by save_figure whenever this style is
    active — detected from the rcParams below — so both
    ``mpl.rcParams.update(presentation_style())`` and
    ``use_presentation_style()`` get the cap.
    """
    style = nature_style()
    style.update({
        # Bold fonts throughout. There is no per-tick weight rcParam, so bolding
        # the global font weight is what makes tick labels bold.
        "font.weight": "bold",

        # Axis labels: bigger + bold
        "axes.labelsize": _PRES_AXES_LABELSIZE,
        "axes.labelweight": "bold",

        # Tick labels: bigger (bold comes from font.weight above)
        "xtick.labelsize": _PRES_TICK_LABELSIZE,
        "ytick.labelsize": _PRES_TICK_LABELSIZE,

        # Thicker axis lines and ticks
        "axes.linewidth": _PRES_AXES_LINEWIDTH,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
    })
    return style


def _presentation_active() -> bool:
    """True when the presentation style is the active matplotlib style."""
    try:
        return (
            float(mpl.rcParams.get("axes.labelsize", 0)) == float(_PRES_AXES_LABELSIZE)
            and float(mpl.rcParams.get("xtick.labelsize", 0)) == float(_PRES_TICK_LABELSIZE)
            and float(mpl.rcParams.get("axes.linewidth", 0)) == float(_PRES_AXES_LINEWIDTH)
        )
    except (TypeError, ValueError):
        return False


# Registry of named styles so `use_style("nature")` (etc.) resolves to a builder.
_STYLE_BUILDERS = {
    "nature": nature_style,
    "poster": poster_style,
    "presentation": presentation_style,
}


def _resolve_style(style) -> dict:
    """Resolve a style spec to an rcParams dict.

    Accepts a style builder callable (e.g. ``nature_style``), a name string
    (``"nature"``, ``"poster"``, ``"presentation"``, with or without a
    ``_style`` suffix), or an rcParams dict.
    """
    if callable(style):
        return dict(style())
    if isinstance(style, dict):
        return dict(style)
    if isinstance(style, str):
        key = style.lower().removesuffix("_style")
        if key in _STYLE_BUILDERS:
            return dict(_STYLE_BUILDERS[key]())
        raise ValueError(
            f"Unknown style {style!r}; known styles: {sorted(set(_STYLE_BUILDERS))}"
        )
    raise TypeError(f"style must be a callable, dict, or name string, got {type(style)!r}")


def use_style(style="nature", max_yticks: int = 4, max_xticks: int = 5) -> None:
    """Activate a figure style globally and set the tick caps.

    Call once at the top of a notebook (``use_style()`` for the default nature style,
    ``use_style("presentation")`` or ``use_style(nature_style)`` for others), so every
    figure created afterwards — including in other projects that import this — picks up
    the style. ``style`` accepts a style builder callable, a name string, or an rcParams
    dict; add new named styles by registering them in ``_STYLE_BUILDERS``.

    The tick caps only take effect under the presentation style: save_figure detects it
    and caps y-ticks (to ``max_yticks``) and numeric x-ticks (to ``max_xticks``).
    """
    global _PRESENTATION_MAX_YTICKS, _PRESENTATION_MAX_XTICKS
    _PRESENTATION_MAX_YTICKS = max_yticks
    _PRESENTATION_MAX_XTICKS = max_xticks
    mpl.rcParams.update(_resolve_style(style))


def use_presentation_style(max_yticks: int = 4, max_xticks: int = 5) -> None:
    """Deprecated alias for ``use_style("presentation", ...)``; kept for existing callers."""
    use_style("presentation", max_yticks=max_yticks, max_xticks=max_xticks)


def nice_x_locator(max_ticks: int | None = None):
    """A locator giving a few nicely-rounded *integer* x-ticks (5, 10, 15, ...
    rather than 3, 6, 9, or fractional values for small ranges).

    Numeric x-axes that would otherwise set one tick per session/day should use
    this so the *displayed* figure already matches the presentation save-time
    x-tick cap (which uses the same settings). ``max_ticks`` defaults to the
    presentation x-tick cap.
    """
    from matplotlib.ticker import MaxNLocator
    n = max_ticks if max_ticks is not None else _PRESENTATION_MAX_XTICKS
    return MaxNLocator(nbins=n or 5, steps=[1, 2, 2.5, 5, 10], integer=True)


# Apply the default (nature) style globally so display and saved figures match.
# To switch styles for a notebook, call one of:
#     mpl.rcParams.update(poster_style())
#     use_presentation_style()          # presentation (also caps y-ticks)
mpl.rcParams.update(nature_style())

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

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


def strip_legends(fig_or_ax) -> int:
    """Remove every legend on the figure (or a single axes).

    Use this when you want a guaranteed legend-free figure regardless of what
    upstream plotting calls (seaborn, pandas .plot, etc.) auto-added.
    Call this just before saving or showing. Returns the number of legends removed.
    """
    if isinstance(fig_or_ax, mpl.figure.Figure):
        axes_iter = list(fig_or_ax.axes)
    elif isinstance(fig_or_ax, mpl.axes.Axes):
        axes_iter = [fig_or_ax]
    else:
        raise TypeError(f"Expected Figure or Axes, got {type(fig_or_ax).__name__}")
    removed = 0
    for ax in axes_iter:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
            removed += 1
    return removed


def save_figure(
    fig: mpl.figure.Figure,
    save_name: str,
    *,
    subjids,
    dates=None,
    subdir: str | Path | None = None,
    dpi: int = 600,
    bbox_inches=None,
    clear_legends: bool = False,
    boxplot: bool = False,
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
    boxplot : bool
        Mark this figure as a boxplot-style plot (categorical x positions). Under
        the presentation style, its x-tick labels are enlarged beyond the y-ticks
        since the positions are the most important thing to read.
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

    if clear_legends:
        strip_legends(fig)

    # presentation style: cap y-ticks (and numeric x-ticks) to a few round
    # values (no rcParam for this), and, for boxplot-style figures, enlarge the
    # x-tick labels.
    if _presentation_active():
        from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter
        for _ax in fig.axes:
            if _PRESENTATION_MAX_YTICKS:
                # nbins is a MAX; the "nice" steps prefer round values (0, 0.5,
                # 1.0 rather than 0, 0.3, 0.6, 0.9), so fewer ticks are fine.
                _ax.yaxis.set_major_locator(
                    MaxNLocator(nbins=_PRESENTATION_MAX_YTICKS, steps=[1, 2, 2.5, 5, 10])
                )
            if _PRESENTATION_MAX_XTICKS:
                # Only touch a *numeric* x-axis: skip categorical axes (explicit
                # string labels / fixed ticks, e.g. boxplot/violin/position
                # plots), which set a FixedFormatter/FixedLocator. integer=True
                # keeps ticks whole (5, 10, 15 — never 7.5).
                _x_categorical = (
                    isinstance(_ax.xaxis.get_major_formatter(), FixedFormatter)
                    or isinstance(_ax.xaxis.get_major_locator(), FixedLocator)
                )
                if not _x_categorical:
                    _ax.xaxis.set_major_locator(nice_x_locator())
            if boxplot:
                _ax.tick_params(axis="x", labelsize=_PRES_XTICK_BOXPLOT_LABELSIZE)
            _ax.figure.canvas.draw_idle()

    bbox = bbox_inches if bbox_inches is not None else "tight"
    fig.savefig(out_path, bbox_inches=bbox, dpi=dpi)

    return out_path

