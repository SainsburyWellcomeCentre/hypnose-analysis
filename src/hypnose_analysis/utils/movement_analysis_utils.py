import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from collections import defaultdict
from typing import Iterable, Optional, Union, Tuple
from hypnose_analysis.utils.metrics_utils import (
    load_session_results,
    run_all_metrics,
    parse_json_column,
)
from datetime import timedelta, datetime
from hypnose_analysis.utils.classification_utils import load_all_streams, load_experiment
from hypnose_analysis.paths import (
    get_data_root,
    get_rawdata_root,
    get_derivatives_root,
    get_server_root,
)
from hypnose_analysis.utils.visualization_utils import (
    _clean_graph, 
    _update_cache, 
    _get_from_cache,
    _load_table_with_trial_data,
    _load_trial_views,
    _extract_metric_value,
    _iter_subject_dirs,
    _filter_session_dirs,
    _load_protocol_from_summary,
    _ensure_metrics_json,
    load_tracking_with_behavior,
)
import re
import numpy as np
import json
from collections import OrderedDict
from scipy.stats import kruskal, mannwhitneyu


def _binned_speed(tracking_df, t_zero, t_end, pre_buffer_s, bin_s, mode):
    """Compute binned speed (mean or max) between start and end relative to t_zero.

    Returns mids (bin centers) and arr (speed per bin) or (None, None) on failure.
    """
    if pd.isna(t_zero) or pd.isna(t_end) or t_end <= t_zero:
        return None, None
    start_dt = t_zero - pd.Timedelta(seconds=pre_buffer_s)
    seg = tracking_df[(tracking_df["time"] >= start_dt) & (tracking_df["time"] <= t_end)].copy()
    if len(seg) < 2 or {"X", "Y", "time"} - set(seg.columns):
        return None, None
    t_rel = (seg["time"] - t_zero).dt.total_seconds().to_numpy()
    if not np.isfinite(t_rel).all() or np.ptp(t_rel) == 0:
        return None, None
    x = seg["X"].to_numpy()
    y = seg["Y"].to_numpy()
    vx = np.gradient(x, t_rel)
    vy = np.gradient(y, t_rel)
    speed = np.hypot(vx, vy)

    dur = t_rel.max() - t_rel.min()
    edges = np.arange(-pre_buffer_s, dur + bin_s + bin_s * 0.5, bin_s)
    if len(edges) < 2:
        return None, None

    seg_df = pd.DataFrame({"t_rel": t_rel, "speed": speed})
    seg_df["bin"] = pd.cut(seg_df["t_rel"], bins=edges, right=False, include_lowest=True)
    grouped = seg_df.groupby("bin", observed=False)["speed"]
    agg_series = grouped.max() if mode == "max" else grouped.mean()
    mids = edges[:-1] + (edges[1] - edges[0]) / 2
    arr = np.full_like(mids, np.nan, dtype=float)
    bin_to_idx = {b: i for i, b in enumerate(agg_series.index.categories)}
    for b, v in agg_series.items():
        idx = bin_to_idx.get(b)
        if idx is not None:
            arr[idx] = v
    return mids, arr


def _load_tracking_and_behavior(subjid, date, tracking_source='sleap'):
    """
    Load combined tracking CSV (SLEAP) and behavior results for a session.
    """
    # Try cache first
    cached = _get_from_cache(subjid, date, kind="sleap_session")
    if cached is not None:
        print(f"[CACHE HIT] SLEAP session for subjid={subjid}, date={date}")
        return cached["tracking"], cached["behavior"]

    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()

    sub_str = f"sub-{str(subjid).zfill(3)}"
    date_str = str(date)

    subject_dirs = list(derivatives_dir.glob(f"{sub_str}_id-*"))
    if not subject_dirs:
        raise FileNotFoundError(f"No subject directory found for {sub_str}")
    subject_dir = subject_dirs[0]

    session_dirs = list(subject_dir.glob(f"ses-*_date-{date_str}"))
    if not session_dirs:
        raise FileNotFoundError(f"No session found for date {date_str}")
    session_dir = session_dirs[0]

    results_dir = session_dir / "saved_analysis_results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Find tracking files (SLEAP only)
    sleap_files = [f for f in results_dir.glob("*_combined_sleap_tracking_timestamps.csv")
                   if not f.name.startswith("._")]

    if not sleap_files:
        raise FileNotFoundError(
            f"No SLEAP tracking file found in {results_dir}."
        )

    csv_path = sleap_files[0]
    source_used = 'sleap'

    try:
        tracking = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        tracking = pd.read_csv(csv_path, encoding='latin1')

    tracking['time'] = pd.to_datetime(tracking['time'], errors='coerce')

    # For SLEAP data: use 'centroid_x' and 'centroid_y' if available, else 'X'/'Y'
    if 'centroid_x' in tracking.columns and 'centroid_y' in tracking.columns:
        tracking['X'] = tracking['centroid_x']
        tracking['Y'] = tracking['centroid_y']
    elif 'X' not in tracking.columns:
        # Try to find any x/y columns
        x_cols = [c for c in tracking.columns if 'x' in c.lower() and 'score' not in c.lower()]
        y_cols = [c for c in tracking.columns if 'y' in c.lower() and 'score' not in c.lower()]
        if x_cols and y_cols:
            tracking['X'] = tracking[x_cols[0]]
            tracking['Y'] = tracking[y_cols[0]]

    behavior = load_session_results(subjid, date)

    # Fallback: populate key tables from trial_data when legacy tables are missing
    views = _load_trial_views(results_dir)
    td = views.get("trial_data", pd.DataFrame())
    if td is not None and not td.empty:
        # Completed sequences
        if behavior.get('completed_sequences', pd.DataFrame()).empty:
            comp = views.get("completed", pd.DataFrame()).copy()
            if not comp.empty:
                if 'last_odor' not in comp.columns and 'last_odor_name' in comp.columns:
                    comp = comp.rename(columns={'last_odor_name': 'last_odor'})
                behavior['completed_sequences'] = comp
        # Completed rewarded
        if behavior.get('completed_sequence_rewarded', pd.DataFrame()).empty:
            comp = views.get("completed", pd.DataFrame())
            if comp is not None and not comp.empty:
                rew = comp[comp.get("response_time_category", "") == "rewarded"].copy()
                if not rew.empty:
                    if 'last_odor' not in rew.columns and 'last_odor_name' in rew.columns:
                        rew = rew.rename(columns={'last_odor_name': 'last_odor'})
                    behavior['completed_sequence_rewarded'] = rew
        # Initiated sequences fallback (use completed as proxy)
        if behavior.get('initiated_sequences', pd.DataFrame()).empty:
            init_df = td.copy()
            if not init_df.empty:
                behavior['initiated_sequences'] = init_df

    
    print(f"Loaded {source_used.upper()} tracking: {len(tracking)} frames from {csv_path.name}")

    # Cache the processed session (tracking+behavior)
    session_data = {
        "tracking": tracking,
        "behavior": behavior,
    }
    _update_cache(subjid, [date], {date: session_data}, kind="sleap_session")

    return tracking, behavior


def plot_movement_trace(subjid, date, smooth_window=10, linewidth=1, alpha=0.5, figsize=(10, 10), 
                       xlim=None, ylim=None, invert_y=True, title=None, save_path=None):
    """
    Plot animal movement trace from ezTrack location tracking CSV.
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    date : int or str
        Session date (e.g., 20251017)
    smooth_window : int, optional
        Number of frames for moving average smoothing (default: 10)
        Set to 1 for no smoothing
    linewidth : float, optional
        Width of the trace line (default: 1)
    alpha : float, optional
        Transparency of the trace (0-1, default: 0.5)
    figsize : tuple, optional
        Figure size (width, height) (default: (10, 10))
    xlim : tuple, optional
        X-axis limits (min, max). If None, auto-scales to data
    ylim : tuple, optional
        Y-axis limits (min, max). If None, auto-scales to data
    invert_y : bool, optional
        Whether to invert Y-axis to match video coordinates (default: True)
    title : str, optional
        Plot title. If None, uses default
    save_path : str or Path, optional
        If provided, saves the plot to this path
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Build path to combined tracking CSV
    base_path = get_rawdata_root()
    server_root = get_server_root()
    derivatives_dir = get_derivatives_root()
    
    # Find subject directory
    sub_str = f"sub-{str(subjid).zfill(3)}"
    subject_dirs = list(derivatives_dir.glob(f"{sub_str}_id-*"))
    if not subject_dirs:
        raise FileNotFoundError(f"No subject directory found for {sub_str}")
    subject_dir = subject_dirs[0]
    
    # Find session directory
    date_str = str(date)
    session_dirs = list(subject_dir.glob(f"ses-*_date-{date_str}"))
    if not session_dirs:
        raise FileNotFoundError(f"No session found for date {date_str}")
    session_dir = session_dirs[0]
    
    # Find combined tracking CSV
    results_dir = session_dir / "saved_analysis_results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Look for combined tracking file (exclude macOS metadata files)
    tracking_files = [f for f in results_dir.glob(f"*_combined_tracking_with_timestamps.csv") 
                    if not f.name.startswith('._')]
    if not tracking_files:
        raise FileNotFoundError(
            f"No combined tracking file found in {results_dir}\n"
            f"Run add_timestamps_to_tracking({subjid}, {date}) first to create it."
        )

    csv_path = tracking_files[0]
    print(f"Loading tracking data from: {csv_path.name}")

    # Load the tracking data with encoding handling
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin1')
        print("Note: Used latin1 encoding")

    # Extract X and Y coordinates
    x = df['X'].values
    y = df['Y'].values
    
    # Apply moving average smoothing
    if smooth_window > 1:
        x_smooth = pd.Series(x).rolling(window=smooth_window, center=True, min_periods=1).mean().values
        y_smooth = pd.Series(y).rolling(window=smooth_window, center=True, min_periods=1).mean().values
    else:
        x_smooth = x
        y_smooth = y
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the trace
    ax.plot(x_smooth, y_smooth, color='black', linewidth=linewidth, alpha=alpha)
    
    # Set axis properties
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title(title if title else f'Animal Movement Trace - Subject {subjid}, {date_str}')
    
    # Set axis limits
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    # Invert y-axis to match video coordinates (origin at top-left)
    if invert_y:
        ax.invert_yaxis()
    
    # Equal aspect ratio for proper spatial representation
    ax.set_aspect('equal', adjustable='box')
    
    # White background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Add grid for reference
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return fig, ax



def plot_movement_by_trial_state(subjid, date, smooth_window=10, linewidth=1, alpha=0.6, 
                                 figsize=(10, 10), xlim=None, ylim=None, invert_y=True,
                                 in_trial_color='blue', out_trial_color='gray',
                                 title=None, save_path=None, show=True):
    """
    Plot animal movement trace colored by trial state (in-trial vs between-trials).
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    date : int or str
        Session date (e.g., 20251017)
    smooth_window : int, optional
        Number of frames for moving average smoothing (default: 10)
    linewidth : float, optional
        Width of the trace line (default: 1)
    alpha : float, optional
        Transparency of the trace (0-1, default: 0.6)
    figsize : tuple, optional
        Figure size (width, height) (default: (10, 10))
    xlim, ylim : tuple, optional
        Axis limits (min, max). If None, auto-scales
    invert_y : bool, optional
        Whether to invert Y-axis (default: True)
    in_trial_color : str, optional
        Color for in-trial segments (default: 'blue')
    out_trial_color : str, optional
        Color for between-trial segments (default: 'gray')
    title : str, optional
        Plot title
    save_path : str or Path, optional
        If provided, saves the plot
    show : bool, optional
        If True, displays the plot (default: True)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Load data
    data = load_tracking_with_behavior(subjid, date)
    df = data['tracking_labeled']
    
    # Apply smoothing
    if smooth_window > 1:
        df['X_smooth'] = df['X'].rolling(window=smooth_window, center=True, min_periods=1).mean()
        df['Y_smooth'] = df['Y'].rolling(window=smooth_window, center=True, min_periods=1).mean()
    else:
        df['X_smooth'] = df['X']
        df['Y_smooth'] = df['Y']
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot in segments based on trial state
    # This ensures continuous lines within each state
    current_state = None
    segment_x = []
    segment_y = []
    
    for idx, row in df.iterrows():
        if row['in_trial'] != current_state:
            # State changed, plot accumulated segment
            if segment_x:
                color = in_trial_color if current_state else out_trial_color
                label = 'In trial' if current_state else 'Between trials'
                # Only add label once per state
                if current_state is not None:
                    existing_labels = [t.get_label() for t in ax.lines]
                    if label in existing_labels:
                        label = None
                
                ax.plot(segment_x, segment_y, color=color, linewidth=linewidth, 
                       alpha=alpha, label=label)
            
            # Start new segment
            segment_x = [row['X_smooth']]
            segment_y = [row['Y_smooth']]
            current_state = row['in_trial']
        else:
            # Continue current segment
            segment_x.append(row['X_smooth'])
            segment_y.append(row['Y_smooth'])
    
    # Plot final segment
    if segment_x:
        color = in_trial_color if current_state else out_trial_color
        label = 'In trial' if current_state else 'Between trials'
        existing_labels = [t.get_label() for t in ax.lines]
        if label in existing_labels:
            label = None
        ax.plot(segment_x, segment_y, color=color, linewidth=linewidth, 
               alpha=alpha, label=label)
    
    # Set properties
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title(title if title else f'Movement by Trial State - Subject {subjid}, {date}')
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    if invert_y:
        ax.invert_yaxis()
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def plot_movement_with_behavior(
    subjid, date,
    mode="simple",                # "simple" | "trial_state" | "last_odor" | "time_windows" | "trial_windows"
    time_windows=None,            # list of ("HH:MM:SS","HH:MM:SS")
    trial_windows=None,           # list of (start, end). negatives allowed, e.g. (-20, None) = last 20..last
    smooth_window=10, linewidth=1, alpha=0.6,
    figsize=(10, 10), xlim=None, ylim=None, invert_y=True,
    last_odor_colors=None,        # {'A':'red','B':'blue','other':'gray'}
    title=None, save_path=None, show=True
):
    """
    Minimal modes:
      - simple: baseline trace
      - trial_state: in-trial vs outside-trial
      - last_odor: within-trial colored by last odor (A vs B)
      - time_windows: plot only movement within provided clock-time windows (can be multiple)
      - trial_windows: plot only trials in provided windows; supports negatives from the end
    Also auto-creates per-condition facet plots when multiple categories/windows exist.
    """
    assert mode in {"simple", "trial_state", "by_odor", "by_odor_rew", "by_odor_outcome", "time_windows", "trial_windows", "trial_windows_rew"}
    tracking, behavior = _load_tracking_and_behavior(subjid, date)
    def _infer_last_odor_column(trials: pd.DataFrame) -> str | None:
        """
        Try to find a column that represents the last odor identity.
        Returns column name or None.
        """
        cols = set(trials.columns.str.lower())

        # Direct matches
        for cand in ["last_odor", "lastodor", "last_odor_name", "final_odor", "finalodor"]:
            for c in trials.columns:
                if c.lower() == cand:
                    return c

        # If there are odorN columns (odor1, odor2, ...), we will derive per-row later
        has_odorN = any(re.match(r"^odor\d+$", c.lower()) for c in trials.columns)
        if has_odorN:
            return None  # signal to derive from odorN columns

        # If there's an 'odors' list/JSON column
        for c in trials.columns:
            if c.lower() == "odors":
                return c

        return None
    def _last_odor_series(trials: pd.DataFrame) -> pd.Series:
        """
        Build a per-trial Series with the last odor identity.
        Handles:
        - explicit last_odor-like columns,
        - odorN columns (takes last non-null),
        - 'odors' list/JSON.
        Falls back to 'other' if not resolvable.
        """
        if trials.empty:
            return pd.Series([], dtype=object, index=trials.index)

        col = _infer_last_odor_column(trials)
        s = pd.Series(index=trials.index, dtype=object)

        if col is not None and col.lower() not in {"odors"}:
            # Direct column present
            s = trials[col].astype(object).fillna("other")
            return s

        # Try odorN columns
        odorN_cols = sorted(
            [c for c in trials.columns if re.match(r"^odor\d+$", c.lower())],
            key=lambda x: int(re.findall(r"\d+", x)[0]) if re.findall(r"\d+", x) else 0
        )
        if odorN_cols:
            def last_non_null(row):
                vals = [row[c] for c in odorN_cols if pd.notna(row[c])]
                return vals[-1] if vals else "other"
            return trials.apply(last_non_null, axis=1)

        # Try 'odors' as list/JSON
        if col is not None and col.lower() == "odors":
            def from_list(v):
                try:
                    if isinstance(v, str):
                        # maybe JSON-like
                        import json
                        v2 = json.loads(v)
                    else:
                        v2 = v
                    if isinstance(v2, (list, tuple)) and len(v2) > 0:
                        return v2[-1]
                except Exception:
                    pass
                return "other"
            return trials[col].apply(from_list)

        return s.fillna("other")
    # Smoothing
    if smooth_window > 1:
        tracking['X_smooth'] = pd.Series(tracking['X']).rolling(
            window=smooth_window, center=True, min_periods=1
        ).mean()
        tracking['Y_smooth'] = pd.Series(tracking['Y']).rolling(
            window=smooth_window, center=True, min_periods=1
        ).mean()
    else:
        tracking['X_smooth'] = tracking['X']
        tracking['Y_smooth'] = tracking['Y']

    fig, ax = plt.subplots(figsize=figsize)

    def _plot_segments_by_mask(df, mask, color, label=None, axes=None):
        # Plot continuous segments where mask is True
        target_ax = axes if axes is not None else ax
        m = mask.fillna(False).astype(bool)
        if m.sum() == 0:
            return
        seg_id = (m != m.shift(1, fill_value=False)).cumsum()
        first = True
        for _, g in df[m].groupby(seg_id[m]):
            target_ax.plot(g['X_smooth'].values, g['Y_smooth'].values,
                           color=color, linewidth=linewidth, alpha=alpha,
                           label=(label if first else None))
            first = False

    facet_plots = []  # collect per-condition masks to facet later

    if mode == "simple":
        ax.plot(tracking['X_smooth'].values, tracking['Y_smooth'].values,
                color='black', linewidth=linewidth, alpha=alpha, label='Movement trace')

    elif mode == "trial_state":
        trials = behavior.get('initiated_sequences', pd.DataFrame())
        in_trial = pd.Series(False, index=tracking.index)
        if not trials.empty:
            trials = trials.copy()
            trials['sequence_start'] = pd.to_datetime(trials['sequence_start'])
            trials['sequence_end'] = pd.to_datetime(trials['sequence_end'])
            t_time = tracking['time']
            for _, tr in trials.iterrows():
                in_trial |= ((t_time >= tr['sequence_start']) & (t_time <= tr['sequence_end']))

        colors = {'in': 'blue', 'out': 'gray'}
        _plot_segments_by_mask(tracking, in_trial, colors['in'], label='In trial')
        _plot_segments_by_mask(tracking, ~in_trial, colors['out'], label='Between trials')
        facet_plots = [
            ('In trial', in_trial, colors['in']),
            ('Between trials', ~in_trial, colors['out']),
        ]

        
    elif mode in ["by_odor", "by_odor_rew"]:
        if mode == "by_odor":
            comps = behavior.get('completed_sequences', pd.DataFrame())
        elif mode == "by_odor_rew":
            comps = behavior.get('completed_sequence_rewarded', pd.DataFrame())
        if comps.empty:
            raise ValueError("No completed_sequences found; last_odor plot requires completed trials.")
        comps = comps.copy()
        comps['sequence_start'] = pd.to_datetime(comps['sequence_start'])
        comps['sequence_end'] = pd.to_datetime(comps['sequence_end'])

        if 'last_odor' not in comps.columns:
            raise ValueError("The 'last_odor' column is missing in completed_sequences.")
        
        if last_odor_colors is None:
            last_odor_colors = {'OdorA': 'red', 'OdorB': 'blue', 'other': 'lightgray'}

        # Map each tracking frame to its odor category
        t_time = tracking['time']
        trial_odor = pd.Series('', index=tracking.index, dtype=object)
        
        for _, tr in comps.iterrows():
            mask = (t_time >= tr['sequence_start']) & (t_time <= tr['sequence_end'])
            trial_odor.loc[mask] = str(tr['last_odor'])
        
        # Filter to only frames within trials
        in_trial_mask = trial_odor != ''
        tracking_in_trial = tracking[in_trial_mask].copy()
        trial_odor_filtered = trial_odor[in_trial_mask]
        
        unique_odors = sorted(trial_odor_filtered.unique())
        
        # Plot combined view with all odors
        for odor in unique_odors:
            odor_mask = trial_odor_filtered == odor
            full_mask = pd.Series(False, index=tracking.index)
            full_mask.loc[odor_mask.index[odor_mask]] = True
            color = last_odor_colors.get(odor, last_odor_colors.get('other', 'gray'))
            _plot_segments_by_mask(tracking, full_mask, color, label=f"Odors: {odor}", axes=ax)
        
        # Create facet plots for individual odors
        facet_plots = []
        for odor in unique_odors:
            odor_mask = trial_odor_filtered == odor
            full_mask = pd.Series(False, index=tracking.index)
            full_mask.loc[odor_mask.index[odor_mask]] = True
            color = last_odor_colors.get(odor, last_odor_colors.get('other', 'gray'))
            facet_plots.append((f"{odor}", full_mask, color))

    elif mode == "by_odor_outcome":
        comps = behavior.get('completed_sequences', pd.DataFrame())
        if comps.empty:
            raise ValueError("No completed_sequences found; by_odor_outcome plot requires completed trials.")
        comps = comps.copy()
        comps['sequence_start'] = pd.to_datetime(comps['sequence_start'])
        comps['sequence_end'] = pd.to_datetime(comps['sequence_end'])

        if 'last_odor' not in comps.columns:
            raise ValueError("The 'last_odor' column is missing in completed_sequences.")

        # Try to infer the rewarded/outcome column
        rewarded_col = None
        for cand in ['rewarded', 'is_rewarded', 'outcome', 'correct', 'success']:
            if cand in comps.columns:
                rewarded_col = cand
                break
        if rewarded_col is None:
            # Fallback: if completed_sequence_rewarded exists, mark those trials as rewarded
            rewarded_trials = behavior.get('completed_sequence_rewarded', pd.DataFrame())
            if not rewarded_trials.empty and 'sequence_start' in rewarded_trials.columns:
                rewarded_starts = set(pd.to_datetime(rewarded_trials['sequence_start']))
                comps['__rewarded'] = comps['sequence_start'].isin(rewarded_starts)
                rewarded_col = '__rewarded'
            else:
                raise ValueError("No rewarded/outcome column found in completed_sequences and cannot infer from completed_sequence_rewarded.")

        if last_odor_colors is None:
            last_odor_colors = {'OdorA': 'red', 'OdorB': 'blue', 'other': 'lightgray'}

        # Map each tracking frame to its odor and outcome category
        t_time = tracking['time']
        trial_odor = pd.Series('', index=tracking.index, dtype=object)
        trial_outcome = pd.Series('', index=tracking.index, dtype=object)

        for _, tr in comps.iterrows():
            mask = (t_time >= tr['sequence_start']) & (t_time <= tr['sequence_end'])
            trial_odor.loc[mask] = str(tr['last_odor'])
            is_rewarded = bool(tr[rewarded_col])
            trial_outcome.loc[mask] = 'rewarded' if is_rewarded else 'not_rewarded'

        # Filter to only frames within trials
        in_trial_mask = trial_odor != ''
        tracking_in_trial = tracking[in_trial_mask].copy()
        trial_odor_filtered = trial_odor[in_trial_mask]
        trial_outcome_filtered = trial_outcome[in_trial_mask]

        unique_odors = sorted(trial_odor_filtered.unique())

        # Plot combined view: correct in color, incorrect/timeout in grey
        for odor in unique_odors:
            odor_mask = trial_odor_filtered == odor
            # Rewarded trials for this odor
            rewarded_mask = odor_mask & (trial_outcome_filtered == 'rewarded')
            # Not rewarded trials for this odor
            not_rewarded_mask = odor_mask & (trial_outcome_filtered == 'not_rewarded')

            full_rewarded_mask = pd.Series(False, index=tracking.index)
            full_rewarded_mask.loc[rewarded_mask.index[rewarded_mask]] = True
            full_not_rewarded_mask = pd.Series(False, index=tracking.index)
            full_not_rewarded_mask.loc[not_rewarded_mask.index[not_rewarded_mask]] = True

            color = last_odor_colors.get(odor, last_odor_colors.get('other', 'gray'))
            _plot_segments_by_mask(tracking, full_rewarded_mask, color, label=f"{odor} (correct)", axes=ax)
            _plot_segments_by_mask(tracking, full_not_rewarded_mask, 'lightgray', label=f"{odor} (incorrect/timeout)", axes=ax)

        # Create facet plots for each odor/outcome
        facet_plots = []
        for odor in unique_odors:
            odor_mask = trial_odor_filtered == odor
            rewarded_mask = odor_mask & (trial_outcome_filtered == 'rewarded')
            not_rewarded_mask = odor_mask & (trial_outcome_filtered == 'not_rewarded')

            full_rewarded_mask = pd.Series(False, index=tracking.index)
            full_rewarded_mask.loc[rewarded_mask.index[rewarded_mask]] = True
            full_not_rewarded_mask = pd.Series(False, index=tracking.index)
            full_not_rewarded_mask.loc[not_rewarded_mask.index[not_rewarded_mask]] = True

            color = last_odor_colors.get(odor, last_odor_colors.get('other', 'gray'))
            # Use a slightly darker grey for incorrect/timeout
            dark_grey = '#888888'
            facet_plots.append((
                f"{odor} by Outcome",
                [  # list of (mask, color, label)
                    (full_rewarded_mask, color, "Correct"),
                    (full_not_rewarded_mask, dark_grey, "Incorrect/Timeout")
                ]
            ))

    elif mode == "time_windows":
        if not time_windows:
            raise ValueError("time_windows must be provided for mode='time_windows'.")
        # Normalize input
        if isinstance(time_windows, tuple):
            time_windows = [time_windows]
        if isinstance(time_windows, str):
            parts = [p.strip() for p in time_windows.split(",")]
            if len(parts) != 2:
                raise ValueError("time_windows string must be 'HH:MM:SS, HH:MM:SS'")
            time_windows = [tuple(parts)]

        t = tracking['time']
        tz = t.dt.tz if hasattr(t.dt, 'tz') else None
        unique_dates = sorted(pd.to_datetime(t.dt.date).unique())
        cmap = plt.cm.Set2
        facet_plots = []
        for i, (ts, te) in enumerate(time_windows):
            color = cmap(i % 8)
            mask = pd.Series(False, index=t.index)
            for d in unique_dates:
                start_dt = pd.to_datetime(f"{pd.to_datetime(d).date()} {ts}")
                end_dt = pd.to_datetime(f"{pd.to_datetime(d).date()} {te}")
                if tz is not None:
                    if start_dt.tzinfo is None:
                        start_dt = start_dt.tz_localize(tz)
                    if end_dt.tzinfo is None:
                        end_dt = end_dt.tz_localize(tz)
                mask |= ((t >= start_dt) & (t <= end_dt))
            _plot_segments_by_mask(tracking, mask, color, label=f"Window {i+1}: {ts}-{te}")
            facet_plots.append((f"Window {i+1}: {ts}-{te}", mask, color))

    elif mode in ["trial_windows", "trial_windows_rew"]:
        if mode == "trial_windows":
            trials = behavior.get('initiated_sequences', pd.DataFrame())
        elif mode == "trial_windows_rew":
            trials = behavior.get('completed_sequence_rewarded', pd.DataFrame())
        if trials.empty:
            raise ValueError(f"{mode} requires appropriate trial data.")
        trials = trials.copy()
        trials['sequence_start'] = pd.to_datetime(trials['sequence_start'])
        trials['sequence_end'] = pd.to_datetime(trials['sequence_end'])
        # Sort by time
        trials = trials.sort_values('sequence_start').reset_index(drop=True)

        # Normalize input to list
        if isinstance(trial_windows, tuple):
            trial_windows = [trial_windows]

        n = len(trials)

        cmap = plt.cm.Dark2
        facet_plots = []
        t_time = tracking['time']
        for i, (start_idx, end_idx) in enumerate(trial_windows):
            # If end_idx is 0 or None, select to the end
            if end_idx in [0, None]:
                sel = trials.iloc[start_idx:]
            else:
                sel = trials.iloc[start_idx:end_idx]
            
            if sel.empty:
                continue
            
            mask = pd.Series(False, index=tracking.index)
            for _, tr in sel.iterrows():
                mask |= ((t_time >= tr['sequence_start']) & (t_time <= tr['sequence_end']))
            
            color = cmap(i % 8)
            # Display label using actual indices for clarity
            actual_indices = sel.index.tolist()
            label = f"Trials {actual_indices[0]}-{actual_indices[-1]}"
            _plot_segments_by_mask(tracking, mask, color, label=label)
            facet_plots.append((label, mask, color))

    # Axes/styling (overlay)
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    if title is None:
        title = f"Movement - Subject {subjid}, {date} ({mode})"
    ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if invert_y:
        ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    # Faceted per-condition plots when multiple categories/windows exist
    if len(facet_plots) > 1:
        n = len(facet_plots)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        facet_fig, facet_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*1.2, figsize[1]*1.2))
        # Ensure facet_axes is always 2D
        if nrows == 1 and ncols == 1:
            facet_axes = np.array([[facet_axes]])
        elif nrows == 1 or ncols == 1:
            facet_axes = facet_axes.reshape(nrows, ncols)
        
        facet_axes_flat = facet_axes.flatten()
        
        for i, facet in enumerate(facet_plots):
            ax_i = facet_axes_flat[i]
            label = facet[0]
            mask_or_list = facet[1]
            # If mask_or_list is a list, it's the new format (by_odor_outcome)
            if isinstance(mask_or_list, list):
                for mask, color, sublabel in mask_or_list:
                    _plot_segments_by_mask(tracking, mask, color, label=sublabel, axes=ax_i)
            else:
                # Old format: (label, mask, color)
                mask = mask_or_list
                color = facet[2]
                _plot_segments_by_mask(tracking, mask, color, label=label, axes=ax_i)
            ax_i.set_title(label)
            ax_i.set_xlabel('X Position (px)')
            ax_i.set_ylabel('Y Position (px)')
            if xlim:
                ax_i.set_xlim(xlim)
            if ylim:
                ax_i.set_ylim(ylim)
            if invert_y:
                ax_i.invert_yaxis()
            ax_i.set_aspect('equal', adjustable='box')
            ax_i.grid(True, alpha=0.3)
        
        # Hide unused axes if any
        for j in range(len(facet_plots), len(facet_axes_flat)):
            facet_axes_flat[j].axis('off')
        
        facet_fig.suptitle(f"Per-condition views - Subject {subjid}, {date} ({mode})")
        facet_fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()

    return fig, ax


def plot_trial_traces_by_mode(
    subjid,
    dates=None,
    mode="rewarded",
    xlim=None,
    ylim=None,
    show_average=False,
    highlight_hr=False,
    color_by_index=False,
    color_by_speed=False,
    color_by_trial_id=False,
    figsize=(18, 6),
    smooth_window=5,
    linewidth=1.4,
    alpha=0.35,
    fa_types="FA_time_in",
    invert_y=True,
):
    """
    Plot centroid traces (SLEAP) for trials filtered by mode, collapsing multiple dates into one plane.

    Parameters
    ----------
    subjid : int
        Subject ID.
    dates : list | tuple | None
        Single date, list of dates, or inclusive range tuple; all sessions are merged into one plot space.
    mode : str
        One of: rewarded, rewarded_hr, completed, all_trials, fa_by_response, fa_by_odor, hr, hr_only.
        "hr" is accepted as an alias for "hr_only".
    xlim, ylim : tuple | None
        Axis limits.
    show_average : bool
        If True, draw a black mean trace per category with a light-grey SEM tube.
    highlight_hr : bool
        Applies to rewarded/all_trials: recolor HR trials with HR palette; ignored elsewhere unless specified.
    color_by_index : bool
        Debug: ignore A/B colors and instead color each trace by normalized sample index (start→end) using a gradient.
    color_by_speed : bool
        If True, color each line segment by speed bins from speed_analysis.parquet (per-trial, per-bin). Segments
        with no speed data are grey. Overrides color_by_index when enabled.
    color_by_trial_id : bool
        If True (modes: rewarded, rewarded_hr, fa_by_response, fa_by_odor, hr_only), color by normalized
        trial order per reward port (A/B) using a dark→light blue gradient. Overrides color_by_index/speed.
    figsize : tuple
        Figure size.
    smooth_window : int
        Rolling window for centroid smoothing (frames).
    linewidth : float
        Line width for individual traces.
    alpha : float
        Transparency for individual traces.
    fa_types : str
        Comma-separated FA labels to include (e.g., "FA_time_in" or "FA_time_in,FA_time_out" or "all").
    invert_y : bool
        If True, invert Y-axis to match video coordinates.
    """

    allowed_modes = {
        "rewarded",
        "rewarded_hr",
        "completed",
        "all_trials",
        "fa_by_response",
        "fa_by_odor",
        "hr",
        "hr_only",
    }
    if mode not in allowed_modes:
        raise ValueError(f"mode must be one of {sorted(allowed_modes)}")
    if mode == "hr":
        mode = "hr_only"

    # FA filter
    if isinstance(fa_types, str):
        fa_types_list = [t.strip().lower() for t in fa_types.split(",")]
        if fa_types.lower() == "all":
            def fa_filter_fn(lbl):
                return str(lbl).startswith("FA_") if pd.notna(lbl) else False
        else:
            def fa_filter_fn(lbl):
                return str(lbl).lower() in fa_types_list if pd.notna(lbl) else False
    elif isinstance(fa_types, (list, tuple, set)):
        fa_set = {str(t).lower() for t in fa_types}
        def fa_filter_fn(lbl):
            return str(lbl).lower() in fa_set if pd.notna(lbl) else False
    else:
        def fa_filter_fn(lbl):
            return True

    # Colors
    port_colors = {1: "#FF6B6B", 2: "#4ECDC4"}
    port_colors_hr = {1: "#E53935", 2: "#00796B"}
    port_colors_fa = {1: "#FF8E8E", 2: "#7EE9DF"}  # slightly altered
    aborted_color = "#555555"
    timeout_color = "#9E9E9E"
    unrewarded_color = "#000000"
    index_cmap = cm.get_cmap("plasma")
    index_norm = Normalize(vmin=0.0, vmax=1.0)
    trial_cmap = cm.get_cmap("Blues")
    speed_cmap = cm.get_cmap("viridis")
    speed_vals_global = []

    subj_str = f"sub-{str(subjid).zfill(3)}"
    derivatives_dir = get_derivatives_root()
    subj_dirs = list(derivatives_dir.glob(f"{subj_str}_id-*"))
    if not subj_dirs:
        raise FileNotFoundError(f"No subject directory found for {subj_str}")
    subj_dir = subj_dirs[0]

    ses_dirs = _filter_session_dirs(subj_dir, dates)
    if not ses_dirs:
        raise FileNotFoundError(f"No sessions found for subject {subjid} with given dates")

    def _odor_letter(val):
        if pd.isna(val):
            return "Unknown"
        s = str(val)
        return s.replace("Odor", "") if s.startswith("Odor") else s

    def _infer_port(row):
        for col in [
            "response_port",
            "rewarded_port",
            "reward_port",
            "supply_port",
            "choice_port",
            "port",
            "fa_port",
        ]:
            if col in row and pd.notna(row[col]):
                try:
                    return int(row[col])
                except Exception:
                    try:
                        return int(float(row[col]))
                    except Exception:
                        continue
        return None

    def _hr_port_from_identity(val):
        if pd.isna(val):
            return None
        s = str(val).strip().upper()
        if s in {"A", "ODORA", "1"}:
            return 1
        if s in {"B", "ODORB", "2"}:
            return 2
        return None

    def _port_from_first_supply(row):
        return _hr_port_from_identity(row.get("first_supply_odor_identity"))

    def _category_from_row(row):
        # Priority: explicit first_supply_odor_identity -> inferred port -> odor letter fallback
        port = _port_from_first_supply(row)
        if port is None:
            port = _infer_port(row)
        if port in {1, 2}:
            return ("A" if port == 1 else "B"), port
        odor = _odor_letter(row.get("last_odor_name") or row.get("last_odor"))
        category = "A" if odor in {"A", "OdorA"} else "B"
        return category, port

    def _smooth_tracking(df):
        def _as_series(col):
            if isinstance(col, pd.DataFrame):
                # take the first column when duplicate names exist
                return col.iloc[:, 0]
            return pd.Series(col)

        if smooth_window > 1:
            df = df.copy()
            df["X"] = _as_series(df["X"]).rolling(window=smooth_window, center=True, min_periods=1).mean()
            df["Y"] = _as_series(df["Y"]).rolling(window=smooth_window, center=True, min_periods=1).mean()
        return df

    def _extract_segment(tracking_df, start, end):
        if pd.isna(start) or pd.isna(end):
            return None
        m = (tracking_df["time"] >= start) & (tracking_df["time"] <= end)
        if not m.any():
            return None
        seg = tracking_df.loc[m, ["time", "X", "Y"]]
        if seg.empty:
            return None
        return seg["X"].to_numpy(), seg["Y"].to_numpy(), seg["time"].to_numpy()

    def _last_poke_out(row):
        pts = row.get("position_poke_times")
        if isinstance(pts, str):
            try:
                pts = json.loads(pts)
            except Exception:
                pts = None
        if isinstance(pts, dict) and pts:
            vals = list(pts.values())
            if all(isinstance(v, dict) and "position" in v for v in vals):
                vals = sorted(vals, key=lambda v: v.get("position", 0))
            last = vals[-1]
            return pd.to_datetime(last.get("poke_odor_end"), errors="coerce")
        if isinstance(pts, list) and pts:
            last = pts[-1]
            if isinstance(last, dict):
                return pd.to_datetime(last.get("poke_odor_end"), errors="coerce")
        for cand in ["poke_odor_end", "last_poke_out_time", "last_poke_time"]:
            if cand in row:
                return pd.to_datetime(row.get(cand), errors="coerce")
        if "sequence_start" in row:
            return pd.to_datetime(row.get("sequence_start"), errors="coerce")
        return pd.NaT

    def _resample_trace(x, y, n_points=200):
        """Resample a trajectory onto a normalized arc-length grid [0,1]."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size < 2 or y.size < 2:
            return None
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            return None
        dx = np.diff(x)
        dy = np.diff(y)
        seg_len = np.hypot(dx, dy)
        cumlen = np.concatenate(([0.0], np.cumsum(seg_len)))
        total_len = cumlen[-1]
        if total_len <= 0:
            return None
        s = cumlen / total_len  # normalized arc length in [0,1]
        s_new = np.linspace(0.0, 1.0, num=n_points)
        x_new = np.interp(s_new, s, x)
        y_new = np.interp(s_new, s, y)
        return x_new, y_new

    def _add_segment(store, axis_key, category, color, x, y, *, label=None, time=None, t_zero=None, speed_bins=None):
        store[axis_key].append({
            "category": category,
            "color": color,
            "x": x,
            "y": y,
            "time": time,
            "t_zero": t_zero,
            "speed_bins": speed_bins,
            "label": label if label is not None else category,
        })

    # Containers
    segments = defaultdict(list)
    avg_pool = defaultdict(lambda: defaultdict(list))
    hr_odors_seen = set()
    speed_analysis_cache = {}

    def _compute_trial_color_map(df, port_fn):
        """Map (port, global_trial_id) -> RGBA using a dark→light blue gradient per port."""
        per_port_ids = defaultdict(list)
        for _, r in df.iterrows():
            tid = r.get("global_trial_id")
            try:
                tid = int(tid)
            except Exception:
                tid = None
            p = port_fn(r)
            if p in {1, 2} and tid is not None:
                per_port_ids[p].append(tid)

        color_map: dict[tuple[int, int], tuple] = {}
        for p, ids in per_port_ids.items():
            ids_sorted = sorted(ids)
            n = len(ids_sorted)
            for i, tid in enumerate(ids_sorted):
                frac = i / (n - 1) if n > 1 else 0.5
                color_map[(p, tid)] = trial_cmap(frac)
        return color_map

    for ses in ses_dirs:
        date_str = ses.name.split("_date-")[-1]
        results_dir = ses / "saved_analysis_results"
        if not results_dir.exists():
            continue

        # Load tracking/behavior
        tracking, behavior = _load_tracking_and_behavior(subjid, date_str)
        tracking = tracking.copy()
        tracking["time"] = pd.to_datetime(tracking["time"], errors="coerce")
        tracking = tracking.dropna(subset=["time"]).reset_index(drop=True)
        tracking = tracking.rename(columns={"centroid_x": "X", "centroid_y": "Y"})

        # Resolve possible duplicate X/Y columns (e.g., both X and centroid_x) to a single Series
        def _resolve_coord(df, candidates):
            for name in candidates:
                if name in df.columns:
                    col = df.loc[:, df.columns == name]
                    if isinstance(col, pd.DataFrame):
                        if col.shape[1] == 0:
                            continue
                        return col.iloc[:, 0]
                    return df[name]
            return None

        x_series = _resolve_coord(tracking, ["X", "centroid_x", "x"])
        y_series = _resolve_coord(tracking, ["Y", "centroid_y", "y"])
        if x_series is not None:
            tracking["X"] = x_series
        if y_series is not None:
            tracking["Y"] = y_series

        tracking = tracking.dropna(subset=["X", "Y"])
        # Drop duplicate columns to avoid DataFrame returns when selecting by name
        tracking = tracking.loc[:, ~tracking.columns.duplicated()]
        tracking = _smooth_tracking(tracking)

        # Load speed analysis parquet if available (per-bin speeds + threshold times)
        speed_df = _get_from_cache(subjid, date_str, kind="speed_analysis")
        if speed_df is None:
            path_speed = results_dir / "speed_analysis.parquet"
            if path_speed.exists():
                try:
                    speed_df = pd.read_parquet(path_speed)
                    _update_cache(subjid, [date_str], {date_str: speed_df.copy()}, kind="speed_analysis")
                except Exception as e:
                    print(f"Warning: could not read {path_speed.name}: {e}")
        speed_bins_map = {}
        if speed_df is not None and not speed_df.empty:
            if "trial_index" in speed_df.columns:
                speed_df = speed_df.copy()
                for col in ["speed_threshold_time", "bin_mid_time", "bin_start_time", "bin_end_time"]:
                    if col in speed_df.columns:
                        speed_df[col] = pd.to_datetime(speed_df[col], errors="coerce")
                for tidx, group in speed_df.groupby("trial_index"):
                    speed_bins_map[tidx] = group.copy()
                finite_speeds = speed_df["speed"].to_numpy()
                speed_vals_global.extend([v for v in finite_speeds if np.isfinite(v)])
        speed_analysis_cache[date_str] = speed_bins_map

        views = _load_trial_views(results_dir)
        td = views.get("trial_data", pd.DataFrame()).copy()
        if not td.empty:
            for c in ["sequence_start", "sequence_end"]:
                if c in td.columns:
                    td[c] = pd.to_datetime(td[c], errors="coerce")
        else:
            td = pd.DataFrame()

        if td.empty:
            # Fallback to behavior tables if trial_data is unavailable
            comp = behavior.get("completed_sequences", pd.DataFrame()).copy()
            if not comp.empty:
                comp["is_aborted"] = False
            aborted = behavior.get("aborted_sequences", pd.DataFrame()).copy()
            if not aborted.empty:
                aborted["is_aborted"] = True
            td = pd.concat([comp, aborted], ignore_index=True) if not comp.empty or not aborted.empty else pd.DataFrame()
            if not td.empty:
                for c in ["sequence_start", "sequence_end"]:
                    if c in td.columns:
                        td[c] = pd.to_datetime(td[c], errors="coerce")

        if td.empty:
            continue

        hr_flag = "hidden_rule_success" if "hidden_rule_success" in td.columns else ("hit_hidden_rule" if "hit_hidden_rule" in td.columns else None)
        hr_mask = td[hr_flag] == True if hr_flag else pd.Series(False, index=td.index)

        # Helper to iterate trials
        def iter_trials(df):
            for idx_row, row in df.iterrows():
                start = row.get("sequence_start")
                # For false alarms, use fa_time as end; for rewarded, prefer first_supply_time to cap at reward delivery
                fa_label = str(row.get("fa_label", "")).lower()
                fa_time = row.get("fa_time")
                resp_cat = str(row.get("response_time_category", "")).lower()
                first_supply_time = row.get("first_supply_time")
                first_reward_poke_time = row.get("first_reward_poke_time")

                if pd.notna(fa_time) and fa_label.startswith("fa_"):
                    end = fa_time
                elif resp_cat == "rewarded" and pd.notna(first_supply_time):
                    end = first_supply_time
                elif resp_cat == "unrewarded" and pd.notna(first_reward_poke_time):
                    end = first_reward_poke_time
                else:
                    end = row.get("sequence_end")
                seg = _extract_segment(tracking, start, end)
                if seg is None:
                    continue
                t_zero = _last_poke_out(row)
                speed_bins = speed_analysis_cache.get(date_str, {}).get(idx_row)
                yield idx_row, row, seg, t_zero, speed_bins

        # Mode-specific selection
        if mode in {"rewarded", "rewarded_hr"}:
            trials = td[(td.get("response_time_category") == "rewarded") & (td.get("is_aborted") == False)]
            include_hr = (mode == "rewarded_hr") or highlight_hr
            if hr_flag and not include_hr:
                trials = trials[~hr_mask]

            trial_color_map = {}
            if color_by_trial_id:
                def _port_trial(row):
                    p = _port_from_first_supply(row)
                    if p is None:
                        p = _infer_port(row)
                    return p
                trial_color_map = _compute_trial_color_map(trials, _port_trial)

            for idx_row, row, seg, t_zero, speed_bins in iter_trials(trials):
                port = None
                if hr_flag and bool(row.get(hr_flag, False)):
                    port = _port_from_first_supply(row) or _infer_port(row)
                category, port_fallback = _category_from_row(row)
                if port is None:
                    port = port_fallback
                color_map = port_colors_hr if (highlight_hr and hr_flag and bool(row.get(hr_flag, False))) else port_colors
                color = color_map.get(port, port_colors[1 if category == "A" else 2])
                if color_by_trial_id:
                    tid = row.get("global_trial_id")
                    try:
                        tid = int(tid)
                    except Exception:
                        tid = None
                    if port in {1, 2} and tid is not None:
                        color = trial_color_map.get((port, tid), color)
                _add_segment(segments, "combined", category, color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                _add_segment(segments, category, category, color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                resampled = _resample_trace(seg[0], seg[1])
                if resampled is not None:
                    avg_pool["combined"][category].append(resampled)
                    avg_pool[category][category].append(resampled)

        elif mode == "completed":
            trials = td[td.get("is_aborted") == False]
            for idx_row, row, seg, t_zero, speed_bins in iter_trials(trials):
                category, port = _category_from_row(row)
                rtc = str(row.get("response_time_category", "")).lower()
                if rtc == "rewarded":
                    color = port_colors.get(port, port_colors[1 if category == "A" else 2])
                elif rtc == "timeout_delayed":
                    color = timeout_color
                elif rtc == "unrewarded":
                    color = unrewarded_color
                else:
                    color = timeout_color
                _add_segment(segments, "combined", category, color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                _add_segment(segments, category, category, color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                resampled = _resample_trace(seg[0], seg[1])
                if resampled is not None:
                    avg_pool["combined"][category].append(resampled)
                    avg_pool[category][category].append(resampled)

        elif mode == "all_trials":
            trials = td.copy()
            for idx_row, row, seg, t_zero, speed_bins in iter_trials(trials):
                category, port = _category_from_row(row)
                if row.get("is_aborted"):
                    color = aborted_color
                    if highlight_hr and hr_flag and bool(row.get(hr_flag, False)):
                        color = "#000000"
                else:
                    color_map = port_colors_hr if (highlight_hr and hr_flag and bool(row.get(hr_flag, False))) else port_colors
                    color = color_map.get(port, port_colors[1 if category == "A" else 2])
                _add_segment(segments, "combined", category, color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                # Only include completed trials in the averages for all_trials
                if not row.get("is_aborted"):
                    resampled = _resample_trace(seg[0], seg[1])
                    if resampled is not None:
                        avg_pool["combined"][category].append(resampled)

        elif mode == "fa_by_response":
            # Only aborted FA trials, filtered by fa_types; plot sequence_start -> fa_time
            fa_df = td[(td.get("is_aborted") == True) & (td.get("fa_label").notna())].copy()
            if not fa_df.empty:
                fa_df = fa_df[fa_df["fa_label"].apply(fa_filter_fn)]
                # Require fa_time for window end
                if "fa_time" in fa_df.columns:
                    fa_df["fa_time"] = pd.to_datetime(fa_df["fa_time"], errors="coerce")
                    fa_df = fa_df.dropna(subset=["fa_time"])
            if not fa_df.empty:
                label_counts = fa_df["fa_label"].value_counts().to_dict()
                print(f"[fa_by_response] session {date_str}: trials after filter={len(fa_df)}, fa_label counts={label_counts}")
            if fa_df.empty:
                continue

            trial_color_map = {}
            if color_by_trial_id:
                def _port_trial(row):
                    p = row.get("fa_port") if pd.notna(row.get("fa_port")) else _infer_port(row)
                    return p
                trial_color_map = _compute_trial_color_map(fa_df, _port_trial)

            for idx_row, row, seg, t_zero, speed_bins in iter_trials(fa_df):
                # Use FA port first, then supply/response port
                port = row.get("fa_port") if pd.notna(row.get("fa_port")) else _infer_port(row)
                if port not in {1, 2}:
                    continue
                category = "A" if port == 1 else "B"
                color = port_colors_fa.get(port, port_colors_fa[1])
                if color_by_trial_id:
                    tid = row.get("global_trial_id")
                    try:
                        tid = int(tid)
                    except Exception:
                        tid = None
                    if tid is not None:
                        color = trial_color_map.get((port, tid), color)
                _add_segment(segments, "combined", category, color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                _add_segment(segments, category, category, color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                resampled = _resample_trace(seg[0], seg[1])
                if resampled is not None:
                    avg_pool["combined"][category].append(resampled)
                    avg_pool[category][category].append(resampled)

        elif mode == "fa_by_odor":
            # Only aborted FA trials; require fa_time for window end
            fa_df = td[(td.get("is_aborted") == True) & (td.get("fa_label").notna())].copy()
            if not fa_df.empty:
                fa_df = fa_df[fa_df["fa_label"].apply(fa_filter_fn)]
                if "fa_time" in fa_df.columns:
                    fa_df["fa_time"] = pd.to_datetime(fa_df["fa_time"], errors="coerce")
                    fa_df = fa_df.dropna(subset=["fa_time"])
            if fa_df.empty:
                continue

            trial_color_map = {}
            if color_by_trial_id:
                def _port_trial(row):
                    p = row.get("fa_port") if pd.notna(row.get("fa_port")) else _infer_port(row)
                    return p
                trial_color_map = _compute_trial_color_map(fa_df, _port_trial)

            for idx_row, row, seg, t_zero, speed_bins in iter_trials(fa_df):
                odor_name = row.get("last_odor_name") or row.get("last_odor")
                odor = _odor_letter(odor_name)
                if odor in {"A", "B", "OdorA", "OdorB"}:
                    continue
                port = row.get("fa_port") if pd.notna(row.get("fa_port")) else _infer_port(row)
                color = port_colors_fa.get(port, port_colors_fa[1])
                if color_by_trial_id:
                    tid = row.get("global_trial_id")
                    try:
                        tid = int(tid)
                    except Exception:
                        tid = None
                    if port in {1, 2} and tid is not None:
                        color = trial_color_map.get((port, tid), color)
                label = "FA to A" if port == 1 else ("FA to B" if port == 2 else "FA")
                _add_segment(segments, odor, label, color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                resampled = _resample_trace(seg[0], seg[1])
                if resampled is not None:
                    avg_pool[odor][label].append(resampled)

        elif mode == "hr_only":
            if hr_flag is None:
                continue
            # Determine hidden-rule odors for this session (from summary.json)
            hr_odors_raw = []
            summary_path = results_dir / "summary.json"
            if summary_path.exists():
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                    hr_odors_raw = summary.get("params", {}).get("hidden_rule_odors", []) or []
                    if not hr_odors_raw:
                        runs = summary.get("session", {}).get("runs", [])
                        if runs and isinstance(runs[0], dict):
                            stage = runs[0].get("stage", {}) if isinstance(runs[0].get("stage", {}), dict) else {}
                            hr_odors_raw = stage.get("hidden_rule_odors", []) or stage.get("hidden_rule_odors".lower(), []) or []
                except Exception:
                    hr_odors_raw = []
            hr_targets = [_odor_letter(o) for o in hr_odors_raw if o is not None]
            hr_targets = hr_targets[:2]  # only need first two hidden-rule odors

            def _parse_odor_sequence(val):
                if isinstance(val, list):
                    return val
                if pd.isna(val):
                    return []
                if isinstance(val, str):
                    try:
                        obj = json.loads(val)
                        if isinstance(obj, list):
                            return obj
                    except Exception:
                        pass
                    # fallback: split by comma/semicolon
                    return [s.strip().strip("[]'\"") for s in re.split(r"[;,]", val) if s.strip()]
                return []

            hr_trials = td[(hr_mask) & (td.get("is_aborted") == False)]
            if hr_trials.empty:
                continue

            trial_color_map = {}
            if color_by_trial_id:
                def _port_trial(row):
                    p = _hr_port_from_identity(row.get("first_supply_odor_identity"))
                    if p is None:
                        p = _infer_port(row)
                    return p
                trial_color_map = _compute_trial_color_map(hr_trials, _port_trial)
            for idx_row, row, seg, t_zero, speed_bins in iter_trials(hr_trials):
                odor_seq = _parse_odor_sequence(row.get("odor_sequence"))
                odor_match = None
                for o in odor_seq:
                    ol = _odor_letter(o)
                    if hr_targets and ol in hr_targets:
                        odor_match = ol
                        break
                if odor_match is None:
                    # fallback to last_odor if no sequence match
                    odor_match = _odor_letter(row.get("last_odor_name") or row.get("last_odor"))
                hr_odors_seen.add(odor_match)

                port = _hr_port_from_identity(row.get("first_supply_odor_identity"))
                if port is None:
                    port = _infer_port(row)
                rtc = str(row.get("response_time_category", "")).lower()

                axis_key = f"HR {odor_match}"
                label_base = f"{odor_match}"
                if rtc == "rewarded":
                    color = port_colors_hr.get(port, port_colors_hr[1])
                    if color_by_trial_id:
                        tid = row.get("global_trial_id")
                        try:
                            tid = int(tid)
                        except Exception:
                            tid = None
                        if port in {1, 2} and tid is not None:
                            color = trial_color_map.get((port, tid), color)
                    _add_segment(segments, axis_key, f"{label_base} rewarded", color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                    _add_segment(segments, "HR Summary", f"{label_base} rewarded", color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                    resampled = _resample_trace(seg[0], seg[1])
                    if resampled is not None:
                        avg_pool[axis_key][f"{label_base} rewarded"].append(resampled)
                        avg_pool["HR Summary"][f"{label_base} rewarded"].append(resampled)
                elif rtc == "timeout_delayed":
                    color = timeout_color
                    _add_segment(segments, axis_key, f"{label_base} timeout", color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                    _add_segment(segments, "HR Summary", f"{label_base} timeout", color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                    resampled = _resample_trace(seg[0], seg[1])
                    if resampled is not None:
                        avg_pool[axis_key][f"{label_base} timeout"].append(resampled)
                        avg_pool["HR Summary"][f"{label_base} timeout"].append(resampled)
                else:  # unrewarded / other
                    color = unrewarded_color
                    _add_segment(segments, axis_key, f"{label_base} unrewarded", color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                    _add_segment(segments, "HR Summary", f"{label_base} unrewarded", color, seg[0], seg[1], time=seg[2], t_zero=t_zero, speed_bins=speed_bins)
                    resampled = _resample_trace(seg[0], seg[1])
                    if resampled is not None:
                        avg_pool[axis_key][f"{label_base} unrewarded"].append(resampled)

    if not segments:
        print("No matching trials found for the requested mode.")
        return None, None

    speed_norm = None
    if color_by_speed and speed_vals_global and not color_by_trial_id:
        vmin = np.nanmin(speed_vals_global)
        vmax = np.nanmax(speed_vals_global)
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            speed_norm = Normalize(vmin=vmin, vmax=vmax)
    color_by_speed_active = color_by_speed and (speed_norm is not None) and (not color_by_trial_id)
    if color_by_trial_id:
        color_by_index = False

    def _plot_axis(ax, axis_key):
        segs = segments.get(axis_key, [])
        used_labels = set()

        def _plot_segment(seg, label=None):
            x = np.asarray(seg["x"])
            y = np.asarray(seg["y"])
            if color_by_speed_active:
                t_arr = np.asarray(seg.get("time"))
                bins_df = seg.get("speed_bins")
                if t_arr is None or bins_df is None or len(x) < 2:
                    ax.plot(x, y, color="#B0B0B0", alpha=alpha, linewidth=linewidth, label=label)
                    return
                try:
                    seg_mid_times = t_arr[:-1] + (t_arr[1:] - t_arr[:-1]) / 2
                except Exception:
                    ax.plot(x, y, color="#B0B0B0", alpha=alpha, linewidth=linewidth, label=label)
                    return
                seg_arr = np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], axis=1)
                colors = []
                bins_df = bins_df.sort_values("bin_start_s") if not bins_df.empty else bins_df
                for t_mid in seg_mid_times:
                    if bins_df is None or bins_df.empty:
                        colors.append("#B0B0B0")
                        continue
                    hit = bins_df[(bins_df["bin_start_time"] <= t_mid) & (t_mid < bins_df["bin_end_time"])]
                    if hit.empty:
                        colors.append("#B0B0B0")
                        continue
                    spd = float(hit.iloc[0]["speed"])
                    if np.isfinite(spd):
                        colors.append(speed_cmap(speed_norm(spd)))
                    else:
                        colors.append("#B0B0B0")
                lc = LineCollection(seg_arr, colors=colors, linewidth=linewidth, alpha=alpha)
                ax.add_collection(lc)
            elif color_by_index:
                if x.size < 2 or y.size < 2:
                    return
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                if points.shape[0] < 2:
                    return
                seg_arr = np.concatenate([points[:-1], points[1:]], axis=1)
                idx_vals = np.linspace(0, 1, len(seg_arr))
                lc = LineCollection(seg_arr, cmap=index_cmap, norm=index_norm, linewidth=linewidth, alpha=alpha)
                lc.set_array(idx_vals)
                ax.add_collection(lc)
            else:
                ax.plot(x, y, color=seg["color"], alpha=alpha, linewidth=linewidth, label=label)

        for seg in segs:
            label = None
            if (not color_by_index) and (not color_by_speed_active) and (seg["label"] not in used_labels):
                label = seg["label"]
                used_labels.add(seg["label"])
            _plot_segment(seg, label)
        if show_average and axis_key in avg_pool:
            for category, traces in avg_pool[axis_key].items():
                if not traces:
                    continue
                xs = [t[0] for t in traces if t is not None]
                ys = [t[1] for t in traces if t is not None]
                if not xs or not ys:
                    continue
                xs = np.vstack(xs)
                ys = np.vstack(ys)
                mean_x = np.nanmean(xs, axis=0)
                mean_y = np.nanmean(ys, axis=0)
                sem_x = np.nanstd(xs, axis=0) / np.sqrt(xs.shape[0])
                sem_y = np.nanstd(ys, axis=0) / np.sqrt(ys.shape[0])
                sem_r = np.sqrt(np.square(sem_x) + np.square(sem_y))

                dx = np.gradient(mean_x)
                dy = np.gradient(mean_y)
                norm = np.hypot(dx, dy)
                norm[norm == 0] = 1.0
                nx = -dy / norm
                ny = dx / norm

                poly_x = np.concatenate([mean_x + nx * sem_r, (mean_x - nx * sem_r)[::-1]])
                poly_y = np.concatenate([mean_y + ny * sem_r, (mean_y - ny * sem_r)[::-1]])

                ax.fill(poly_x, poly_y, color="#DDDDDD", alpha=0.35, linewidth=0)
                ax.plot(mean_x, mean_y, color="black", linewidth=2.0, label=f"{category} mean")

        if color_by_speed_active:
            sm = cm.ScalarMappable(norm=speed_norm, cmap=speed_cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Speed")
        elif color_by_index:
            sm = cm.ScalarMappable(norm=index_norm, cmap=index_cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Normalized sample index")

        ax.set_xlabel("X Position (px)")
        ax.set_ylabel("Y Position (px)")
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if invert_y:
            ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()

    figs = []
    axes_out = []

    def _make_fig(axis_key, title=None):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        _plot_axis(ax, axis_key)
        if title:
            ax.set_title(title)
        plt.tight_layout()
        figs.append(fig)
        axes_out.append(ax)

    # Layout by mode (separate figure per axis)
    if mode in {"rewarded", "rewarded_hr", "completed", "fa_by_response"}:
        for axis_key, title in zip(["combined", "A", "B"], ["Combined", "Odor A / Port 1", "Odor B / Port 2"]):
            _make_fig(axis_key, title)
    elif mode == "all_trials":
        _make_fig("combined", "All trials")
    elif mode == "fa_by_odor":
        odor_keys = [k for k in segments.keys()]
        if not odor_keys:
            print("No FA trials found for fa_by_odor")
            return None, None
        for key in odor_keys:
            _make_fig(key, f"Odor {key}")
    elif mode == "hr_only":
        axis_keys = [k for k in segments.keys() if k.startswith("HR ")]
        if "HR Summary" in segments:
            axis_keys.append("HR Summary")
        if not axis_keys:
            print("No HR trials found.")
            return None, None
        for key in axis_keys:
            _make_fig(key, key)

    return figs if len(figs) > 1 else (figs[0], axes_out[0])


def run_speed_analysis_batch(
    subjids=None,
    dates=None,
    *,
    bin_ms: int = 100,
    pre_buffer_s: float = 1.0,
    fa_label_filter=None,
    mode: str = "mean",
    threshold: bool = True,
    threshold_alpha: float = 10.0,
    threshold_beta: float = 10.0,
    verbose: bool = True,
):
    """Run compute_speed_analysis over all available subject/date combinations.

    Supports single or multiple subject IDs and date specs (list of dates or
    inclusive (start, end) tuple). Only sessions with existing data are passed
    to compute_speed_analysis. Returns a list of (subjid, date) processed and
    prints a summary when verbose=True.
    """

    derivatives_dir = get_derivatives_root()
    processed: list[tuple[int, Union[int, str]]] = []

    for sid, subj_dir in _iter_subject_dirs(derivatives_dir, subjids):
        ses_dirs = _filter_session_dirs(subj_dir, dates)
        if not ses_dirs:
            if verbose:
                print(f"[run_speed_analysis_batch] No sessions found for sub-{sid:03d} with given dates.")
            continue

        # Extract ordered unique dates from session directories
        date_list = []
        seen_dates = set()
        for ses in ses_dirs:
            date_str = ses.name.split("_date-")[-1]
            try:
                date_val = int(date_str) if str(date_str).isdigit() else date_str
            except Exception:
                date_val = date_str
            if date_val in seen_dates:
                continue
            seen_dates.add(date_val)
            date_list.append(date_val)

        if not date_list:
            if verbose:
                print(f"[run_speed_analysis_batch] No matching dates after filtering for sub-{sid:03d}.")
            continue

        try:
            compute_speed_analysis(
                sid,
                dates=date_list,
                bin_ms=bin_ms,
                pre_buffer_s=pre_buffer_s,
                fa_label_filter=fa_label_filter,
                mode=mode,
                threshold=threshold,
                threshold_alpha=threshold_alpha,
                threshold_beta=threshold_beta,
            )
            processed.extend([(sid, d) for d in date_list])
        except Exception as e:
            if verbose:
                print(f"[run_speed_analysis_batch] Failed for sub-{sid:03d}: {e}")

    if verbose:
        if processed:
            print("[run_speed_analysis_batch] Completed speed analysis for:")
            by_subj: dict[int, list[Union[int, str]]] = defaultdict(list)
            for sid, d in processed:
                by_subj[sid].append(d)
            for sid in sorted(by_subj.keys()):
                dates_sorted = sorted(by_subj[sid], key=lambda x: str(x))
                dates_str = ", ".join(str(d) for d in dates_sorted)
                print(f"  sub-{sid:03d}: {dates_str}")
        else:
            print("[run_speed_analysis_batch] No sessions processed.")

    return processed


def compute_speed_analysis(
    subjid,
    dates=None,
    *,
    bin_ms: int = 100,
    pre_buffer_s: float = 1.0,
    fa_label_filter=None,
    mode: str = "mean",
    threshold: bool = True,
    threshold_alpha: float = 10.0,
    threshold_beta: float = 10.0,
):
    """Compute cue-port speed epochs aligned to last poke-out for rewarded, unrewarded, and FA trials.

        Handles loading data, computing speeds, binning, thresholding, and movement metrics. Writes a single
        speed_analysis.parquet per session containing per-bin records with per-trial metrics repeated.
        Returns the same plotting artifacts as before for backward compatibility.

    Parameters
    ----------
    subjid : int
        Subject ID.
    dates : list | tuple | None
        List of dates or inclusive range tuple; None uses all available.
    bin_ms : int
        Epoch width in milliseconds (default 100).
    pre_buffer_s : float
        Seconds to include before last poke-out (default 0).
    fa_label_filter : str | Iterable | None
        FA labels to include (default {"fa_time_in"}); accepts comma/semicolon-separated
        string or any iterable (e.g., ["fa_time_in", "fa_time_out"]). Case-insensitive.
    mode : {"max", "mean"}
        Aggregation per epoch: max speed (current behavior) or mean speed.
    threshold : bool
        If True, compute baseline (mu, sigma) from [-0.15s, -0.05s] pooled across all trials
        in the session, and overlay baseline plus the single threshold line
        vthresh = max(alpha*mu, mu+beta*sigma).
    threshold_alpha : float
        Multiplier for mu when threshold is enabled (default 6.0).
    threshold_beta : float
        Multiplier for sigma when threshold is enabled (default 6.0).
    figsize : tuple
        Figure size for per-session plots.

    Returns
    -------
    dict with:
                - "per_session": list of dicts per session with keys date, figs (violin, traces)
                    where traces is a dict of condition -> figure, and baseline stats when threshold=True
        - "combined": dict of condition -> fig (only when multiple sessions and data present)
    """

    if mode not in {"max", "mean"}:
        raise ValueError("mode must be 'max' or 'mean'")

    # Normalize FA labels: accept comma-separated string or any iterable of labels
    if fa_label_filter is None:
        fa_labels = {"fa_time_in"}
    elif isinstance(fa_label_filter, str):
        # allow "fa_time_in,fa_time_out" or "fa_time_in; fa_time_out"
        parts = re.split(r"[;,]", fa_label_filter)
        fa_labels = {p.strip().lower() for p in parts if p.strip()}
    else:
        try:
            fa_labels = {str(s).strip().lower() for s in fa_label_filter if str(s).strip()}
        except TypeError:
            fa_labels = {str(fa_label_filter).strip().lower()}

    subj_str = f"sub-{str(subjid).zfill(3)}"
    derivatives_dir = get_derivatives_root()
    subj_dirs = list(derivatives_dir.glob(f"{subj_str}_id-*"))
    if not subj_dirs:
        raise FileNotFoundError(f"No subject directory found for {subj_str}")
    subj_dir = subj_dirs[0]

    ses_dirs = _filter_session_dirs(subj_dir, dates)
    if not ses_dirs:
        raise FileNotFoundError(f"No sessions found for subject {subjid} with given dates")

    bin_s = bin_ms / 1000.0
    baseline_window = (-0.15, -0.05)  # seconds relative to last poke-out
    start_target_s = -bin_ms / 2000.0  # target mid-bin time (e.g., -0.05s for 100 ms bins)

    def _safe_dt(val):
        try:
            return pd.to_datetime(val)
        except Exception:
            return pd.NaT

    def _last_poke_out(row):
        pts = row.get("position_poke_times")
        if isinstance(pts, str):
            try:
                pts = json.loads(pts)
            except Exception:
                pts = None

        entries = []
        if isinstance(pts, dict) and pts:
            vals = list(pts.values())
            if all(isinstance(v, dict) and "position" in v for v in vals):
                vals = sorted(vals, key=lambda v: v.get("position", 0))
            entries = vals
        elif isinstance(pts, list) and pts:
            entries = [p for p in pts if isinstance(p, dict)]

        for poke in reversed(entries):
            dt_val = _safe_dt(poke.get("poke_odor_end"))
            if pd.notna(dt_val):
                return dt_val

        return pd.NaT

    def _end_time(row, cond):
        if cond == "rewarded":
            return _safe_dt(row.get("first_supply_time")) or _safe_dt(row.get("sequence_end"))
        if cond == "unrewarded":
            return _safe_dt(row.get("first_reward_poke_time"))
        if cond == "fa":
            return _safe_dt(row.get("fa_time")) or _safe_dt(row.get("sequence_end"))
        return _safe_dt(row.get("sequence_end"))

    def _parse_position_entries(val):
        """Normalize position_* collections to a list of dicts sorted by position when present."""
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except Exception:
                val = None

        entries = []
        if isinstance(val, dict) and val:
            vals = list(val.values())
            if all(isinstance(v, dict) and "position" in v for v in vals):
                vals = sorted(vals, key=lambda v: v.get("position", 0))
            entries = vals
        elif isinstance(val, list) and val:
            entries = [v for v in val if isinstance(v, dict)]
        return entries

    def _last_valve_start(row):
        """Return the last valve_start that has a matching poke_odor_start for the same position.

        Iterates positions from highest to lowest; requires both valve_start and poke_odor_start
        for that position. If none meet both criteria, returns NaT.
        """
        pvt_entries = _parse_position_entries(row.get("position_valve_times"))
        ppt_entries = _parse_position_entries(row.get("position_poke_times"))

        poke_by_pos = {}
        for entry in ppt_entries:
            try:
                pos_key = int(entry.get("position"))
            except Exception:
                continue
            poke_start = _safe_dt(entry.get("poke_odor_start"))
            if pd.notna(poke_start):
                poke_by_pos[pos_key] = poke_start

        for entry in reversed(pvt_entries):
            try:
                pos = int(entry.get("position"))
            except Exception:
                pos = None
            valve_ts = _safe_dt(entry.get("valve_start"))
            if pd.isna(valve_ts):
                continue
            if pos is not None and pos in poke_by_pos:
                return valve_ts

        return pd.NaT

    def _condition_label(row):
        rtc = str(row.get("response_time_category", "")).lower()
        if rtc == "rewarded" and not row.get("is_aborted", False):
            return "rewarded"
        if rtc == "unrewarded" and not row.get("is_aborted", False):
            return "unrewarded"
        fa_label = str(row.get("fa_label", "")).lower()
        if fa_label.startswith("fa_") and fa_label in fa_labels:
            return "fa"
        return None

    def _speed_by_bins(tracking_df, zero_dt, end_dt, edges):
        # legacy helper for plotting on shared edges; threshold now uses per-trial binning
        if pd.isna(zero_dt) or pd.isna(end_dt) or end_dt <= zero_dt:
            return None
        start_dt = zero_dt - pd.Timedelta(seconds=pre_buffer_s)
        seg = tracking_df[(tracking_df["time"] >= start_dt) & (tracking_df["time"] <= end_dt)].copy()
        if seg.empty or {"X", "Y", "time"} - set(seg.columns):
            return None
        if len(seg) < 2:
            return None
        t_rel = (seg["time"] - zero_dt).dt.total_seconds().to_numpy()
        if not np.isfinite(t_rel).all() or np.ptp(t_rel) == 0:
            return None
        x = seg["X"].to_numpy()
        y = seg["Y"].to_numpy()
        vx = np.gradient(x, t_rel)
        vy = np.gradient(y, t_rel)
        speed = np.hypot(vx, vy)
        seg["speed"] = speed
        seg["t_rel"] = t_rel
        seg["bin"] = pd.cut(seg["t_rel"], bins=edges, right=False, include_lowest=True)
        grouped = seg.groupby("bin", observed=False)["speed"]
        agg_series = grouped.max() if mode == "max" else grouped.mean()
        mids = edges[:-1] + (edges[1] - edges[0]) / 2
        arr = np.full_like(mids, np.nan, dtype=float)
        bin_to_idx = {b: i for i, b in enumerate(agg_series.index.categories)}
        for b, v in agg_series.items():
            idx = bin_to_idx.get(b)
            if idx is not None:
                arr[idx] = v
        return mids, arr

    def _speed_series(tracking_df, zero_dt, end_dt, pre_buffer_s_local):
        """Return per-sample relative time (s) and speed for a trial segment."""
        if pd.isna(zero_dt) or pd.isna(end_dt) or end_dt <= zero_dt:
            return None, None
        start_dt = zero_dt - pd.Timedelta(seconds=pre_buffer_s_local)
        seg = tracking_df[(tracking_df["time"] >= start_dt) & (tracking_df["time"] <= end_dt)].copy()
        if len(seg) < 2 or {"X", "Y", "time"} - set(seg.columns):
            return None, None
        t_rel = (seg["time"] - zero_dt).dt.total_seconds().to_numpy()
        if not np.isfinite(t_rel).all() or np.ptp(t_rel) == 0:
            return None, None
        x = seg["X"].to_numpy()
        y = seg["Y"].to_numpy()
        vx = np.gradient(x, t_rel)
        vy = np.gradient(y, t_rel)
        speed = np.hypot(vx, vy)
        return t_rel, speed


    def _compute_tortuosity(tracking_df, t_zero, mids_trial, t_end):
        """Compute tortuosity using per-trial bins (non-fixed coords)."""
        if mids_trial is None or len(mids_trial) == 0 or pd.isna(t_zero) or pd.isna(t_end):
            return np.nan
        # choose start bin near target (-0.05s) else first
        mids_sorted = np.asarray(mids_trial)
        start_idx = np.where(np.abs(mids_sorted - start_target_s) <= (bin_ms / 1000.0) * 0.01)[0]
        if start_idx.size == 0:
            start_idx = np.array([0])
        mid_start = float(mids_sorted[start_idx[0]])
        half = bin_s / 2.0
        # Use bin end times (start/end of tortuosity window) clamped to the actual trial span
        start_time = t_zero + pd.Timedelta(seconds=mid_start + half)
        start_time = max(start_time, t_zero)
        if pd.notna(t_end):
            start_time = min(start_time, t_end)

        mid_last = float(mids_sorted[-1])
        end_time = t_zero + pd.Timedelta(seconds=mid_last + half)
        if pd.notna(t_end):
            end_time = min(end_time, t_end)

        if pd.isna(start_time) or pd.isna(end_time) or end_time <= start_time:
            return np.nan

        seg_mask = (tracking_df["time"] >= start_time) & (tracking_df["time"] <= end_time)
        seg = tracking_df.loc[seg_mask, ["X", "Y", "time"]].copy()
        if len(seg) < 2:
            return np.nan
        seg = seg.sort_values("time")

        start_frame_idx = int(np.argmin(np.abs((seg["time"] - start_time).dt.total_seconds())))
        end_frame_idx = int(np.argmin(np.abs((seg["time"] - end_time).dt.total_seconds())))
        start_xy = seg.iloc[start_frame_idx][["X", "Y"]].to_numpy(dtype=float)
        end_xy = seg.iloc[end_frame_idx][["X", "Y"]].to_numpy(dtype=float)

        x_arr = seg["X"].to_numpy(dtype=float)
        y_arr = seg["Y"].to_numpy(dtype=float)
        path_len = float(np.sum(np.hypot(np.diff(x_arr), np.diff(y_arr))))
        straight_len = float(np.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]))
        return path_len / straight_len if straight_len > 0 else np.nan

    def _path_length(x_arr, y_arr):
        x = np.asarray(x_arr, float)
        y = np.asarray(y_arr, float)
        if x.size < 2 or y.size < 2:
            return np.nan
        dx = np.diff(x)
        dy = np.diff(y)
        return float(np.sum(np.hypot(dx, dy)))

    per_session = []
    combined_data = {"rewarded": [], "unrewarded": [], "fa": []}

    for ses in ses_dirs:
        date_str = ses.name.split("_date-")[-1]
        results_dir = ses / "saved_analysis_results"
        if not results_dir.exists():
            continue

        try:
            tracking, _ = _load_tracking_and_behavior(subjid, date_str)
        except Exception as e:
            print(f"Skipping {date_str}: tracking load failed ({e})")
            continue

        tracking = tracking.copy()
        tracking["time"] = pd.to_datetime(tracking["time"], errors="coerce")
        tracking = tracking.dropna(subset=["time"]).reset_index(drop=True)

        # Resolve X/Y columns
        for cand in [("centroid_x", "centroid_y"), ("X", "Y")]:
            if cand[0] in tracking.columns and cand[1] in tracking.columns:
                tracking["X"] = tracking[cand[0]]
                tracking["Y"] = tracking[cand[1]]
                break
        tracking = tracking.dropna(subset=["X", "Y"])
        tracking = tracking.loc[:, ~tracking.columns.duplicated()]
        if tracking.empty:
            continue

        # Smoothed copy for path-length computation (centered rolling, ~5-frame default)
        smooth_window_frames = 5
        tracking_smooth = tracking.copy()
        if smooth_window_frames > 1:
            tracking_smooth["X"] = (
                pd.Series(tracking_smooth["X"]).rolling(window=smooth_window_frames, center=True, min_periods=1).mean()
            )
            tracking_smooth["Y"] = (
                pd.Series(tracking_smooth["Y"]).rolling(window=smooth_window_frames, center=True, min_periods=1).mean()
            )
        tracking_smooth = tracking_smooth.dropna(subset=["X", "Y"])

        views = _load_trial_views(results_dir)
        trial_data = views.get("trial_data", pd.DataFrame()).copy()
        if trial_data.empty:
            print(f"No trial_data for {date_str}; skipping")
            continue
        for c in ["sequence_start", "sequence_end", "first_supply_time", "first_reward_poke_time", "fa_time"]:
            if c in trial_data.columns:
                trial_data[c] = pd.to_datetime(trial_data[c], errors="coerce")

        trials_info = []
        skipped_no_poke_end = []
        for idx_row, row in trial_data.iterrows():
            cond = _condition_label(row)
            if cond is None:
                continue
            t_zero = _last_poke_out(row)
            if pd.isna(t_zero):
                trial_id = row.get("trial_id", idx_row) if hasattr(row, "get") else idx_row
                skipped_no_poke_end.append(trial_id)
                continue
            t_end = _end_time(row, cond)
            if pd.isna(t_end) or t_end <= t_zero:
                continue
            dur_post = (t_end - t_zero).total_seconds()
            if dur_post <= 0:
                continue
            trials_info.append((idx_row, cond, t_zero, t_end, dur_post))

        if skipped_no_poke_end:
            print(f"Warning [{date_str}]: skipped trials with no poke_odor_end in position_poke_times: {skipped_no_poke_end}")

        if not trials_info:
            print(f"No usable trials for {date_str}")
            continue

        max_post = max(dur for _, _, _, _, dur in trials_info)
        edges = np.arange(-pre_buffer_s, max_post + bin_s, bin_s)
        if len(edges) < 2:
            continue

        epoch_series = {"rewarded": [], "unrewarded": [], "fa": []}
        speeds_flat = {"rewarded": [], "unrewarded": [], "fa": []}
        epoch_records = []  # flattened per-trial, per-bin speeds for downstream use
        baseline_vals = []
        # store per-trial threshold times
        trial_data["speed_threshold_time"] = pd.NaT
        trial_data["movement_onset_from_valve_s"] = np.nan
        # cache per-trial bins for threshold computation without writing arrays into the DataFrame
        trial_bins = {}
        mids_common = None
        movement_records = []

        for idx_row, cond, t_zero, t_end, _ in trials_info:
            # per-trial binning for threshold/baseline
            mids_trial, arr_trial = _binned_speed(tracking, t_zero, t_end, pre_buffer_s, bin_s, mode)
            if mids_trial is None:
                continue
            if threshold:
                mask_base = (mids_trial >= baseline_window[0]) & (mids_trial <= baseline_window[1])
                if mask_base.any():
                    baseline_vals.extend([v for v in arr_trial[mask_base] if not np.isnan(v)])

            # legacy/global binning for plotting alignment
            res_plot = _speed_by_bins(tracking, t_zero, t_end, edges)
            if res_plot is None:
                continue
            mids_plot, arr_plot = res_plot
            if mids_common is None:
                mids_common = mids_plot
            epoch_series[cond].append(arr_plot)
            speeds_flat[cond].extend([v for v in arr_plot if not np.isnan(v)])

            # Write per-trial per-bin records using the per-trial bins (mids_trial/arr_trial) to avoid extending past t_end.
            if mids_trial is not None and arr_trial is not None:
                for mid, val in zip(mids_trial, arr_trial):
                    mid_td = pd.Timedelta(seconds=float(mid))
                    half_bin = pd.Timedelta(seconds=float(bin_s / 2))
                    bin_mid_time = t_zero + mid_td
                    bin_start_time = bin_mid_time - half_bin
                    bin_end_time = bin_mid_time + half_bin
                    # Clamp end to t_end to avoid overshooting when global edges are longer than this trial
                    if bin_end_time > t_end:
                        bin_end_time = t_end
                    epoch_records.append({
                        "trial_index": idx_row,
                        "condition": cond,
                        "bin_mid_s": float(mid),
                        "bin_start_s": float(mid - bin_s / 2),
                        "bin_end_s": float(mid + bin_s / 2),
                        "bin_mid_time": bin_mid_time,
                        "bin_start_time": bin_start_time,
                        "bin_end_time": bin_end_time,
                        "speed": float(val) if not np.isnan(val) else np.nan,
                        "date": date_str,
                        "subjid": subjid,
                        "speed_threshold_time": pd.NaT,
                        "latency_s": np.nan,
                    })

            trial_bins[idx_row] = {
                "mids": mids_trial,
                "arr": arr_trial,
                "t_zero": t_zero,
            }

            # Path length (smoothed) and travel time between t_zero and t_end
            seg_path = tracking_smooth[(tracking_smooth["time"] >= t_zero) & (tracking_smooth["time"] <= t_end)]
            path_len = _path_length(seg_path["X"], seg_path["Y"]) if len(seg_path) >= 2 else np.nan
            travel_time_s = (t_end - t_zero).total_seconds() if pd.notna(t_end) and pd.notna(t_zero) else np.nan
            tortuosity_val = _compute_tortuosity(tracking, t_zero, mids_trial, t_end)
            movement_records.append({
                "trial_index": idx_row,
                "condition": cond,
                "path_length_px": path_len,
                "travel_time_s": travel_time_s,
                "tortuosity": tortuosity_val,
                "start_time": t_zero,
                "end_time": t_end,
                "date": date_str,
                "subjid": subjid,
            })

        conds_with_data = [c for c in ["rewarded", "unrewarded", "fa"] if epoch_series[c]]
        if not conds_with_data:
            print(f"No epoch data for {date_str}")
            continue

        baseline_mean = np.nanmean(baseline_vals) if baseline_vals else None
        baseline_sd = np.nanstd(baseline_vals) if baseline_vals else None
        thr_alpha_mu = baseline_mean * threshold_alpha if baseline_mean is not None else None
        thr_mu_plus_beta_sigma = (
            baseline_mean + threshold_beta * baseline_sd
            if baseline_mean is not None and baseline_sd is not None
            else None
        )
        thr_max = None
        if threshold and baseline_mean is not None:
            candidates = [v for v in [thr_alpha_mu, thr_mu_plus_beta_sigma] if v is not None]
            if candidates:
                thr_max = max(candidates)

        # compute and store per-trial crossing times using per-trial bins
        if "latency_s" not in trial_data.columns:
            trial_data["latency_s"] = np.nan

        if threshold and thr_max is not None:
            for idx_row, cond, t_zero, t_end, _ in trials_info:
                # Bin-gated crossing: find first bin (mean) above threshold, then refine within that bin using per-sample speed
                crossing_time = pd.NaT
                latency_val = np.nan
                movement_from_valve = np.nan
                valve_start = _last_valve_start(trial_data.loc[idx_row]) if "position_valve_times" in trial_data.columns else pd.NaT

                bins = trial_bins.get(idx_row, {})
                mids_trial = bins.get("mids")
                arr_trial = bins.get("arr")

                # Identify first bin whose mean/max (per mode) exceeds threshold, after t=0
                bin_idx = None
                if mids_trial is not None and arr_trial is not None:
                    crossing_bins = np.where((mids_trial >= 0) & (arr_trial > thr_max))[0]
                    if crossing_bins.size > 0:
                        bin_idx = crossing_bins[0]

                if bin_idx is not None:
                    bin_mid = float(mids_trial[bin_idx])
                    half = bin_s / 2.0
                    win_start = bin_mid - half
                    win_end = bin_mid + half

                    # Per-sample refinement within the bin window
                    t_rel_series, speed_series = _speed_series(tracking, t_zero, t_end, pre_buffer_s)
                    if t_rel_series is not None and speed_series is not None:
                        mask = (t_rel_series >= win_start) & (t_rel_series <= win_end) & np.isfinite(speed_series)
                        if mask.any():
                            idx_cross = np.where(speed_series[mask] > thr_max)[0]
                            if idx_cross.size > 0:
                                idx_masked = idx_cross[0]
                                idx_global = np.where(mask)[0][idx_masked]

                                if idx_masked > 0:
                                    i1 = np.where(mask)[0][idx_masked - 1]
                                    i2 = idx_global
                                    t1, t2 = t_rel_series[i1], t_rel_series[i2]
                                    s1, s2 = speed_series[i1], speed_series[i2]
                                    if s2 != s1 and np.isfinite([t1, t2, s1, s2]).all():
                                        frac = (thr_max - s1) / (s2 - s1)
                                        frac = np.clip(frac, 0.0, 1.0)
                                        t_cross = t1 + frac * (t2 - t1)
                                    else:
                                        t_cross = t_rel_series[i2]
                                else:
                                    t_cross = t_rel_series[idx_global]

                                crossing_time = t_zero + pd.Timedelta(seconds=float(t_cross))
                                latency_val = float(t_cross)
                                if pd.notna(valve_start):
                                    movement_from_valve = (crossing_time - valve_start).total_seconds()

                    # If bin mean crossed but no per-sample crossing found within window, fallback to bin midpoint
                    if pd.isna(latency_val):
                        crossing_time = t_zero + pd.Timedelta(seconds=float(bin_mid))
                        latency_val = float(bin_mid)
                        if pd.notna(valve_start):
                            movement_from_valve = (crossing_time - valve_start).total_seconds()

                if pd.notna(crossing_time):
                    trial_data.at[idx_row, "speed_threshold_time"] = crossing_time
                trial_data.at[idx_row, "latency_s"] = latency_val
                trial_data.at[idx_row, "movement_onset_from_valve_s"] = movement_from_valve

        # Fill per-bin records with per-trial metrics (repeat per bin so one file carries both)
        if "speed_threshold_time" in trial_data.columns:
            thr_map = trial_data["speed_threshold_time"].to_dict()
            lat_map = trial_data["latency_s"].to_dict() if "latency_s" in trial_data.columns else {}
            mov_map = trial_data["movement_onset_from_valve_s"].to_dict() if "movement_onset_from_valve_s" in trial_data.columns else {}
            # Movement metrics per trial
            path_map = {}
            travel_map = {}
            tort_map = {}
            for rec_mov in movement_records:
                tid = rec_mov.get("trial_index")
                path_map[tid] = rec_mov.get("path_length_px")
                travel_map[tid] = rec_mov.get("travel_time_s")
                tort_map[tid] = rec_mov.get("tortuosity")

            for rec in epoch_records:
                tid = rec["trial_index"]
                rec["speed_threshold_time"] = thr_map.get(tid, pd.NaT)
                rec["latency_s"] = lat_map.get(tid, np.nan)
                rec["movement_onset_from_valve_s"] = mov_map.get(tid, np.nan)
                rec["path_length_px"] = path_map.get(tid, np.nan)
                rec["travel_time_s"] = travel_map.get(tid, np.nan)
                rec["tortuosity"] = tort_map.get(tid, np.nan)

        analysis_path = results_dir / "speed_analysis.parquet"
        try:
            analysis_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(epoch_records).to_parquet(analysis_path, index=False)
        except Exception as e:
            print(f"Warning: failed to write {analysis_path.name}: {e}")
        else:
            _update_cache(subjid, [date_str], {date_str: pd.DataFrame(epoch_records)}, kind="speed_analysis")


        per_session.append({
            "date": date_str,
            "baseline": {
                "mu": baseline_mean,
                "sigma": baseline_sd,
                "alpha": threshold_alpha,
                "beta": threshold_beta,
                "alpha_mu": thr_alpha_mu,
                "mu_plus_beta_sigma": thr_mu_plus_beta_sigma,
                "max_alpha_mu_mu_plus_beta_sigma": thr_max,
            } if threshold else None,
            "trial_data_with_threshold": trial_data.copy(),
        })

        for cond in conds_with_data:
            stack = np.vstack(epoch_series[cond])
            session_mean = np.nanmean(stack, axis=0)
            mids_combined = mids_common if mids_common is not None else (edges[:-1] + (edges[1] - edges[0]) / 2)
            combined_data[cond].append((date_str, mids_combined, session_mean))


    return {"per_session": per_session}


def plot_epoch_speeds_by_condition(
    subjid,
    dates=None,
    *,
    bin_ms: int = 100,
    fa_label_filter=None,
    mode: str = "mean",
    threshold: bool = True,
    threshold_alpha: float = 10.0,
    threshold_beta: float = 10.0,
    figsize=(8, 5),
):
    """Plot cue-port speed epochs from precomputed speed_analysis.parquet.

    Uses outputs from compute_speed_analysis (same parameters) to build per-session, per-condition
    per-trial traces with session mean overlay and optional threshold lines. Violin plots are omitted.
    """

    if mode not in {"max", "mean"}:
        raise ValueError("mode must be 'max' or 'mean'")

    # Normalize FA labels: accept comma-separated string or any iterable of labels (used at compute time)
    if fa_label_filter is None:
        fa_labels = {"fa_time_in"}
    elif isinstance(fa_label_filter, str):
        parts = re.split(r"[;,]", fa_label_filter)
        fa_labels = {p.strip().lower() for p in parts if p.strip()}
    else:
        try:
            fa_labels = {str(s).strip().lower() for s in fa_label_filter if str(s).strip()}
        except TypeError:
            fa_labels = {str(fa_label_filter).strip().lower()}

    subj_str = f"sub-{str(subjid).zfill(3)}"
    derivatives_dir = get_derivatives_root()
    subj_dirs = list(derivatives_dir.glob(f"{subj_str}_id-*") )
    if not subj_dirs:
        raise FileNotFoundError(f"No subject directory found for {subj_str}")
    subj_dir = subj_dirs[0]

    ses_dirs = _filter_session_dirs(subj_dir, dates)
    if not ses_dirs:
        raise FileNotFoundError(f"No sessions found for subject {subjid} with given dates")

    bin_s = bin_ms / 1000.0
    baseline_window = (-0.15, -0.05)

    per_session = []
    combined_data = {"rewarded": [], "unrewarded": [], "fa": []}

    for ses in ses_dirs:
        date_str = ses.name.split("_date-")[-1]
        results_dir = ses / "saved_analysis_results"
        if not results_dir.exists():
            continue

        df_speed = _get_from_cache(subjid, date_str, kind="speed_analysis")
        if df_speed is None:
            path_speed = results_dir / "speed_analysis.parquet"
            if not path_speed.exists():
                raise FileNotFoundError(f"Missing speed_analysis.parquet for {date_str}; run compute_speed_analysis first")
            df_speed = pd.read_parquet(path_speed)
            _update_cache(subjid, [date_str], {date_str: df_speed.copy()}, kind="speed_analysis")

        df_speed = df_speed.copy()
        conds_with_data = [c for c in ["rewarded", "unrewarded", "fa"] if not df_speed[df_speed["condition"] == c].empty]
        if not conds_with_data:
            continue

        # Baseline stats from stored speeds
        baseline_mask = (df_speed["bin_mid_s"] >= baseline_window[0]) & (df_speed["bin_mid_s"] <= baseline_window[1])
        baseline_vals = df_speed.loc[baseline_mask, "speed"].dropna().to_numpy()
        baseline_mean = np.nanmean(baseline_vals) if baseline_vals.size else None
        baseline_sd = np.nanstd(baseline_vals) if baseline_vals.size else None
        thr_alpha_mu = baseline_mean * threshold_alpha if baseline_mean is not None else None
        thr_mu_plus_beta_sigma = (
            baseline_mean + threshold_beta * baseline_sd
            if baseline_mean is not None and baseline_sd is not None
            else None
        )
        thr_max = None
        if threshold and baseline_mean is not None:
            candidates = [v for v in [thr_alpha_mu, thr_mu_plus_beta_sigma] if v is not None]
            if candidates:
                thr_max = max(candidates)

        figs_by_cond = {}
        for cond in conds_with_data:
            sub = df_speed[df_speed["condition"] == cond].copy()
            if sub.empty:
                continue
            # Trial-wise traces
            trials = []
            trial_arrays = []
            mids_all = np.sort(sub["bin_mid_s"].unique())
            fig_t, ax_t = plt.subplots(figsize=figsize)

            for tid, g in sub.groupby("trial_index"):
                g = g.sort_values("bin_mid_s")
                mids = g["bin_mid_s"].to_numpy(float)
                speeds = g["speed"].to_numpy(float)
                if mids.size and speeds.size:
                    ax_t.plot(mids, speeds, color="gray", alpha=0.2)
                trials.append((tid, mids, speeds))

                arr_full = np.full_like(mids_all, np.nan, dtype=float)
                mid_to_idx = {m: i for i, m in enumerate(mids_all)}
                for m, s in zip(mids, speeds):
                    idx = mid_to_idx.get(m)
                    if idx is not None:
                        arr_full[idx] = s
                trial_arrays.append(arr_full)

            if trial_arrays:
                stack = np.vstack(trial_arrays)
                mean_speeds = np.nanmean(stack, axis=0)
                ax_t.plot(mids_all, mean_speeds, color="blue", linewidth=2, label="session mean")

            if threshold and baseline_mean is not None:
                ax_t.axhline(baseline_mean, color="red", linestyle="-", linewidth=1.5, label="baseline μ")
                if thr_max is not None:
                    ax_t.axhline(thr_max, color="#2F4F4F", linestyle="--", linewidth=1.4, label=f"max(αμ, μ+βσ), α={threshold_alpha:g}, β={threshold_beta:g}")

            ax_t.set_title(f"{cond} — sub {subjid}, {date_str} ({mode})")
            ax_t.set_xlabel("Time from last poke-out (s)")
            ax_t.set_ylabel("Speed (units/s)")
            ax_t.legend()
            fig_t.tight_layout()
            figs_by_cond[cond] = fig_t

            if trial_arrays:
                combined_data[cond].append((date_str, mids_all, np.nanmean(np.vstack(trial_arrays), axis=0)))

        per_session.append({
            "date": date_str,
            "fig_traces": figs_by_cond,
            "baseline": {
                "mu": baseline_mean,
                "sigma": baseline_sd,
                "alpha": threshold_alpha,
                "beta": threshold_beta,
                "alpha_mu": thr_alpha_mu,
                "mu_plus_beta_sigma": thr_mu_plus_beta_sigma,
                "max_alpha_mu_mu_plus_beta_sigma": thr_max,
            } if threshold else None,
        })

    combined_figs = {}
    if len(per_session) > 1:
        colors = plt.cm.tab10.colors
        for idx, cond in enumerate(["rewarded", "unrewarded", "fa"]):
            if not combined_data[cond]:
                continue
            fig, ax = plt.subplots(figsize=figsize)
            for j, (date_str, mids, session_mean) in enumerate(combined_data[cond]):
                ax.plot(mids, session_mean, color=colors[j % len(colors)], label=date_str)
            ax.set_title(f"Session means — {cond} ({mode})")
            ax.set_xlabel("Time from last poke-out (s)")
            ax.set_ylabel("Speed (units/s)")
            ax.legend()
            fig.tight_layout()
            combined_figs[cond] = fig

    return {"per_session": per_session, "combined": combined_figs}


def plot_traces_with_speed_threshold(
    subjid,
    dates=None,
    *,
    fa_types="FA_time_in",
    bin_ms: int = 100,
    pre_buffer_s: float = 0.2,
    threshold_alpha: float = 6.0,
    threshold_beta: float = 6.0,
    mode: str = "mean",
    smooth_window: int = 5,
    figsize=(10, 8),
    invert_y: bool = True,
):
    """Plot spatial traces for rewarded, unrewarded, and FA trials with a speed threshold marker.

    For the selected sessions, builds three figures (rewarded, unrewarded, fa). Traces are overlaid
    across sessions. Each trial trace gets a black dot at the first time after last poke-out when
    speed exceeds vthresh = max(alpha*mu, mu+beta*sigma), where mu/sigma come from the pooled
    baseline window [-0.15s, -0.05s] relative to last poke-out across all trials in the session.
    If a parquet with `speed_threshold_time` exists for the session (written by
    compute_speed_analysis), it is loaded (and cached) and used directly; otherwise the
    threshold is recomputed and the result is saved + cached.

    Parameters
    ----------
    subjid : int
        Subject ID.
    dates : list | tuple | None
        Dates list or inclusive range; None uses all available for the subject.
    fa_types : str | Iterable
        FA labels to include (default "FA_time_in"). Case-insensitive; accepts comma/semicolon list.
    bin_ms : int
        Epoch/bin width in milliseconds for speed aggregation (default 100).
    pre_buffer_s : float
        Seconds to include before last poke-out when computing speed (default 0.2). Needs >=0.15s
        to populate the baseline window.
    threshold_alpha : float
        Multiplier for mu in the threshold definition (default 6.0).
    threshold_beta : float
        Multiplier for sigma in the threshold definition (default 6.0).
    mode : {"max", "mean"}
        Aggregation per bin when computing speeds.
    smooth_window : int
        Rolling window (frames) for smoothing X/Y before speed computation and plotting.
    figsize : tuple
        Figure size for each condition plot.
    invert_y : bool
        If True, invert Y-axis to match video coordinates.

    Returns
    -------
    dict with keys "rewarded", "unrewarded", "fa" mapping to matplotlib figures.
    """

    # Ensure helper is available even if an old module version was loaded
    try:
        binned_speed_fn = _binned_speed
    except NameError:
        import hypnose_analysis.utils.movement_analysis_utils as _mau
        binned_speed_fn = getattr(_mau, "_binned_speed", None)
    if binned_speed_fn is None:
        raise RuntimeError("_binned_speed helper not available; reload hypnose_analysis.utils.movement_analysis_utils")

    # Color palette consistent with plot_trial_traces_by_mode
    port_colors = {1: "#FF6B6B", 2: "#4ECDC4"}
    port_colors_fa = {1: "#FF8E8E", 2: "#7EE9DF"}
    aborted_color = "#555555"

    # Normalize FA labels
    if isinstance(fa_types, str):
        if fa_types.lower() == "all":
            def fa_filter_fn(lbl):
                return str(lbl).lower().startswith("fa_") if pd.notna(lbl) else False
        else:
            fa_set = {s.strip().lower() for s in re.split(r"[;,]", fa_types) if s.strip()}
            def fa_filter_fn(lbl):
                return str(lbl).lower() in fa_set if pd.notna(lbl) else False
    else:
        fa_set = {str(s).strip().lower() for s in fa_types}
        def fa_filter_fn(lbl):
            return str(lbl).lower() in fa_set if pd.notna(lbl) else False

    if mode not in {"max", "mean"}:
        raise ValueError("mode must be 'max' or 'mean'")
    if pre_buffer_s < 0.15:
        print("pre_buffer_s < 0.15s: baseline window [-0.15, -0.05] may be empty")

    subj_str = f"sub-{str(subjid).zfill(3)}"
    derivatives_dir = get_derivatives_root()
    subj_dirs = list(derivatives_dir.glob(f"{subj_str}_id-*"))
    if not subj_dirs:
        raise FileNotFoundError(f"No subject directory found for {subj_str}")
    subj_dir = subj_dirs[0]

    ses_dirs = _filter_session_dirs(subj_dir, dates)
    if not ses_dirs:
        raise FileNotFoundError(f"No sessions found for subject {subjid} with given dates")

    baseline_window = (-0.15, -0.05)
    bin_s = bin_ms / 1000.0

    def _safe_dt(val):
        try:
            return pd.to_datetime(val)
        except Exception:
            return pd.NaT

    def _last_poke_out(row):
        pts = row.get("position_poke_times")
        if isinstance(pts, str):
            try:
                pts = json.loads(pts)
            except Exception:
                pts = None

        entries = []
        if isinstance(pts, dict) and pts:
            vals = list(pts.values())
            if all(isinstance(v, dict) and "position" in v for v in vals):
                vals = sorted(vals, key=lambda v: v.get("position", 0))
            entries = vals
        elif isinstance(pts, list) and pts:
            entries = [p for p in pts if isinstance(p, dict)]

        for poke in reversed(entries):
            dt_val = _safe_dt(poke.get("poke_odor_end"))
            if pd.notna(dt_val):
                return dt_val

        return pd.NaT

    def _end_time(row, cond):
        if cond == "rewarded":
            return _safe_dt(row.get("first_supply_time")) or _safe_dt(row.get("sequence_end"))
        if cond == "unrewarded":
            return _safe_dt(row.get("first_reward_poke_time"))
        if cond == "fa":
            return _safe_dt(row.get("fa_time")) or _safe_dt(row.get("sequence_end"))
        return _safe_dt(row.get("sequence_end"))

    def _infer_port(row):
        # Try explicit port fields first
        for col in [
            "response_port", "rewarded_port", "reward_port", "supply_port",
            "choice_port", "port", "fa_port", "last_reward_port", "odor_port",
        ]:
            if col in row and pd.notna(row[col]):
                try:
                    return int(row[col])
                except Exception:
                    try:
                        return int(float(row[col]))
                    except Exception:
                        continue
        # Try odor-number style fields
        for col in ["last_odor_num", "odor_num", "odor_index", "odor_position"]:
            if col in row and pd.notna(row[col]):
                try:
                    val = int(row[col])
                    if val == 2:
                        return 2
                    if val == 1:
                        return 1
                except Exception:
                    continue
        # Try odor labels
        odor = str(row.get("last_odor_name") or row.get("last_odor") or row.get("odor_name") or row.get("odor") or "").strip().lower()
        if odor in {"b", "odorb", "odor_b", "2", "portb", "port_b"}:
            return 2
        if odor in {"a", "odora", "odor_a", "1", "porta", "port_a"}:
            return 1
        return None

    def _category_from_row(row):
        odor = str(row.get("last_odor_name") or row.get("last_odor") or "A")
        if odor in {"A", "OdorA", "1"}:
            return "A"
        if odor in {"B", "OdorB", "2"}:
            return "B"
        return "A"

    def _smooth_tracking(df):
        if smooth_window > 1:
            df = df.copy()
            df["X"] = pd.Series(df["X"]).rolling(window=smooth_window, center=True, min_periods=1).mean()
            df["Y"] = pd.Series(df["Y"]).rolling(window=smooth_window, center=True, min_periods=1).mean()
        return df

    traces = {"rewarded": [], "unrewarded": [], "fa": []}
    markers = {"rewarded": [], "unrewarded": [], "fa": []}
    for ses in ses_dirs:
        date_str = ses.name.split("_date-")[-1]
        results_dir = ses / "saved_analysis_results"
        if not results_dir.exists():
            continue
        skipped_no_poke_end = []
        analysis_path = results_dir / "speed_analysis.parquet"

        trial_data = None
        use_saved_thresholds = False

        cached_df = _get_from_cache(subjid, date_str, kind="speed_analysis")
        if cached_df is None and analysis_path.exists():
            try:
                cached_df = pd.read_parquet(analysis_path)
                _update_cache(subjid, [date_str], {date_str: cached_df.copy()}, kind="speed_analysis")
            except Exception as e:
                print(f"Failed to read {analysis_path.name}: {e}")

        if cached_df is not None:
            # Extract per-trial threshold times from per-bin records
            thr_series = (cached_df.dropna(subset=["speed_threshold_time"])
                                       .drop_duplicates(subset=["trial_index"])
                                       .set_index("trial_index")["speed_threshold_time"])
        else:
            thr_series = None

        views = _load_trial_views(results_dir)
        trial_data = views.get("trial_data", pd.DataFrame()).copy()
        if trial_data.empty:
            print(f"No trial_data for {date_str}; skipping")
            continue
        for c in ["sequence_start", "sequence_end", "first_supply_time", "first_reward_poke_time", "fa_time", "speed_threshold_time"]:
            if c in trial_data.columns:
                trial_data[c] = pd.to_datetime(trial_data[c], errors="coerce")

        if thr_series is not None:
            trial_data["speed_threshold_time"] = trial_data.index.map(thr_series)
            use_saved_thresholds = True

        # Load tracking per session
        try:
            tracking, _ = _load_tracking_and_behavior(subjid, date_str)
        except Exception as e:
            print(f"Skipping {date_str}: tracking load failed ({e})")
            continue

        tracking = tracking.copy()
        tracking["time"] = pd.to_datetime(tracking["time"], errors="coerce")
        tracking = tracking.dropna(subset=["time"]).reset_index(drop=True)
        for cand in [("centroid_x", "centroid_y"), ("X", "Y")]:
            if cand[0] in tracking.columns and cand[1] in tracking.columns:
                tracking["X"] = tracking[cand[0]]
                tracking["Y"] = tracking[cand[1]]
                break
        tracking = tracking.dropna(subset=["X", "Y"])
        tracking = tracking.loc[:, ~tracking.columns.duplicated()]
        if tracking.empty:
            continue
        tracking = _smooth_tracking(tracking)

        if use_saved_thresholds:
            # Strictly use stored threshold times; no recomputation
            for idx, row in trial_data.iterrows():
                rtc = str(row.get("response_time_category", "")).lower()
                is_aborted = bool(row.get("is_aborted", False))
                fa_label = str(row.get("fa_label", "")).lower()

                if rtc == "rewarded" and not is_aborted:
                    cond = "rewarded"
                elif rtc == "unrewarded" and not is_aborted:
                    cond = "unrewarded"
                elif fa_label.startswith("fa_") and fa_filter_fn(fa_label):
                    cond = "fa"
                else:
                    continue

                t_zero = _last_poke_out(row)
                if pd.isna(t_zero):
                    trial_id = row.get("trial_id", idx) if hasattr(row, "get") else idx
                    skipped_no_poke_end.append(trial_id)
                    continue
                t_end = _end_time(row, cond)
                if pd.isna(t_end) or t_end <= t_zero:
                    continue

                start_dt = t_zero - pd.Timedelta(seconds=pre_buffer_s)
                seg = tracking[(tracking["time"] >= start_dt) & (tracking["time"] <= t_end)].copy()
                if len(seg) < 2 or {"X", "Y", "time"} - set(seg.columns):
                    continue
                t_rel = (seg["time"] - t_zero).dt.total_seconds().to_numpy()
                if not np.isfinite(t_rel).all() or np.ptp(t_rel) == 0:
                    continue
                x = seg["X"].to_numpy()
                y = seg["Y"].to_numpy()

                marker = None
                thr_time = row.get("speed_threshold_time") if "speed_threshold_time" in trial_data.columns else pd.NaT
                if pd.notna(thr_time):
                    nearest_idx = int(np.argmin(np.abs((seg["time"] - thr_time).dt.total_seconds())))
                    marker = (x[nearest_idx], y[nearest_idx])

                port = _infer_port(row)
                if cond == "fa":
                    color = port_colors_fa.get(port, port_colors_fa[1])
                else:
                    color = port_colors.get(port, port_colors[1 if _category_from_row(row) == "A" else 2])

                traces[cond].append({"x": x, "y": y, "color": color, "session": date_str})
                if marker is not None:
                    markers[cond].append({"xy": marker, "color": "black", "session": date_str})
            if skipped_no_poke_end:
                print(f"Warning [{date_str}]: skipped trials with no poke_odor_end in position_poke_times: {skipped_no_poke_end}")
            # done with this session
            continue

        baseline_vals = []
        baseline_mask = None
        trial_cache = {}

        # First pass: per-trial binned speeds to build baseline and cache for later reuse
        for idx, row in trial_data.iterrows():
            rtc = str(row.get("response_time_category", "")).lower()
            is_aborted = bool(row.get("is_aborted", False))
            fa_label = str(row.get("fa_label", "")).lower()

            if rtc == "rewarded" and not is_aborted:
                cond = "rewarded"
            elif rtc == "unrewarded" and not is_aborted:
                cond = "unrewarded"
            elif fa_label.startswith("fa_") and fa_filter_fn(fa_label):
                cond = "fa"
            else:
                continue

            t_zero = _last_poke_out(row)
            if pd.isna(t_zero):
                trial_id = row.get("trial_id", idx) if hasattr(row, "get") else idx
                skipped_no_poke_end.append(trial_id)
                continue
            t_end = _end_time(row, cond)
            if pd.isna(t_end) or t_end <= t_zero:
                continue

            mids_trial, arr_trial = binned_speed_fn(tracking, t_zero, t_end, pre_buffer_s, bin_s, mode)
            if mids_trial is None or arr_trial is None:
                continue

            if baseline_mask is None:
                baseline_mask = (mids_trial >= baseline_window[0]) & (mids_trial <= baseline_window[1])
            if baseline_mask is not None and baseline_mask.any():
                baseline_vals.extend([v for v in arr_trial[baseline_mask] if not np.isnan(v)])

            trial_cache[idx] = {
                "cond": cond,
                "t_zero": t_zero,
                "t_end": t_end,
                "mids": mids_trial,
                "arr": arr_trial,
            }

        if not baseline_vals:
            print(f"No baseline window data for {date_str}; skipping session")
            if skipped_no_poke_end:
                print(f"Warning [{date_str}]: skipped trials with no poke_odor_end in position_poke_times: {skipped_no_poke_end}")
            continue

        baseline_vals_arr = np.asarray([v for v in baseline_vals if np.isfinite(v)])
        if baseline_vals_arr.size == 0:
            print(f"Baseline values not finite for {date_str}; skipping session")
            if skipped_no_poke_end:
                print(f"Warning [{date_str}]: skipped trials with no poke_odor_end in position_poke_times: {skipped_no_poke_end}")
            continue
        mu = float(np.nanmean(baseline_vals_arr))
        sigma = float(np.nanstd(baseline_vals_arr))
        thr_alpha_mu = mu * threshold_alpha
        thr_mu_plus_beta_sigma = mu + threshold_beta * sigma
        vthresh = max(thr_alpha_mu, thr_mu_plus_beta_sigma)

        if "speed_threshold_time" not in trial_data.columns:
            trial_data["speed_threshold_time"] = pd.NaT

        # Second pass: build traces and threshold markers using computed threshold
        for idx, meta in trial_cache.items():
            cond = meta["cond"]
            t_zero = meta["t_zero"]
            t_end = meta["t_end"]
            row = trial_data.loc[idx]

            thr_time = pd.NaT
            saved_thr = trial_data.at[idx, "speed_threshold_time"] if "speed_threshold_time" in trial_data.columns else pd.NaT
            if pd.notna(saved_thr):
                thr_time = saved_thr
            else:
                mids_trial = meta.get("mids")
                arr_trial = meta.get("arr")
                if mids_trial is not None and arr_trial is not None:
                    crossing_idx = np.where((mids_trial >= 0) & (arr_trial > vthresh))[0]
                    if crossing_idx.size > 0:
                        k = crossing_idx[0]
                        thr_time = t_zero + pd.Timedelta(seconds=float(mids_trial[k]))

            start_dt = t_zero - pd.Timedelta(seconds=pre_buffer_s)
            seg = tracking[(tracking["time"] >= start_dt) & (tracking["time"] <= t_end)].copy()
            if len(seg) < 2 or {"X", "Y", "time"} - set(seg.columns):
                continue
            t_rel = (seg["time"] - t_zero).dt.total_seconds().to_numpy()
            if not np.isfinite(t_rel).all() or np.ptp(t_rel) == 0:
                continue
            x = seg["X"].to_numpy()
            y = seg["Y"].to_numpy()

            marker = None
            trial_data.at[idx, "speed_threshold_time"] = thr_time
            if pd.notna(thr_time):
                nearest_idx = int(np.argmin(np.abs((seg["time"] - thr_time).dt.total_seconds())))
                marker = (x[nearest_idx], y[nearest_idx])

            port = _infer_port(row)
            if cond == "fa":
                color = port_colors_fa.get(port, port_colors_fa[1])
            else:
                color = port_colors.get(port, port_colors[1 if _category_from_row(row) == "A" else 2])

            traces[cond].append({"x": x, "y": y, "color": color, "session": date_str})
            if marker is not None:
                markers[cond].append({"xy": marker, "color": "black", "session": date_str})

        if skipped_no_poke_end:
            print(f"Warning [{date_str}]: skipped trials with no poke_odor_end in position_poke_times: {skipped_no_poke_end}")

    figs = {}
    for cond, label in [("rewarded", "Rewarded"), ("unrewarded", "Unrewarded"), ("fa", "False Alarms")]:
        if not traces[cond]:
            continue
        fig, ax = plt.subplots(figsize=figsize)
        for tr in traces[cond]:
            ax.plot(tr["x"], tr["y"], color=tr["color"], alpha=0.35, linewidth=1.2)
        for mk in markers[cond]:
            ax.scatter(mk["xy"][0], mk["xy"][1], color="black", s=18, zorder=5)
        ax.set_title(f"{label} traces with speed-threshold crossing")
        ax.set_xlabel("X Position (px)")
        ax.set_ylabel("Y Position (px)")
        if invert_y:
            ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        figs[cond] = fig

    return figs


def plot_tortuosity_lines_overlay(
    subjid,
    dates=None,
    *,
    fa_types="FA_time_in",
    bin_ms: int = 100,
    fixed_start_xy=(575, 90),
    fixed_goal_a_xy=(208, 930),
    fixed_goal_b_xy=(973, 930),
    figsize=(8, 8),
):
    """Plot traces by condition with both data-derived tortuosity lines and fixed lines overlaid.

    Uses speed_analysis.parquet to align start/end times per trial. For each trial, draws the trajectory,
    a line from start→goal derived from tracking, and a fixed start→goal line (A/B) using provided coordinates.
    Returns a dict of figures keyed by (date, condition).
    """

    # FA filter
    if isinstance(fa_types, str):
        if fa_types.lower() == "all":
            def fa_filter_fn(lbl):
                return str(lbl).lower().startswith("fa_") if pd.notna(lbl) else False
        else:
            fa_set = {s.strip().lower() for s in re.split(r"[;,]", fa_types) if s.strip()}
            def fa_filter_fn(lbl):
                return str(lbl).lower() in fa_set if pd.notna(lbl) else False
    else:
        fa_set = {str(s).strip().lower() for s in fa_types}
        def fa_filter_fn(lbl):
            return str(lbl).lower() in fa_set if pd.notna(lbl) else False

    subj_str = f"sub-{str(subjid).zfill(3)}"
    subj_dirs = list(get_derivatives_root().glob(f"{subj_str}_id-*"))
    if not subj_dirs:
        raise FileNotFoundError(f"No subject directory found for {subj_str}")
    subj_dir = subj_dirs[0]

    ses_dirs = _filter_session_dirs(subj_dir, dates)
    if not ses_dirs:
        raise FileNotFoundError(f"No sessions found for subject {subjid} with given dates")

    start_target_s = -bin_ms / 2000.0
    cond_colors = {"rewarded": "#4CAF50", "unrewarded": "#F44336", "fa": "#2196F3"}
    data_line_color = "#424242"
    fixed_line_color = "#9C27B0"

    def _infer_port(row):
        for col in [
            "response_port", "rewarded_port", "reward_port", "supply_port",
            "choice_port", "port", "fa_port",
        ]:
            if col in row and pd.notna(row[col]):
                try:
                    return int(row[col])
                except Exception:
                    try:
                        return int(float(row[col]))
                    except Exception:
                        continue
        return None

    def _condition_label(row):
        rtc = str(row.get("response_time_category", "")).lower()
        is_aborted = bool(row.get("is_aborted", False))
        fa_label = str(row.get("fa_label", "")).lower()
        if rtc == "rewarded" and not is_aborted:
            return "rewarded"
        if rtc == "unrewarded" and not is_aborted:
            return "unrewarded"
        if fa_label.startswith("fa_") and fa_filter_fn(fa_label):
            return "fa"
        return None

    figs = {}

    for ses in ses_dirs:
        date_str = ses.name.split("_date-")[-1]
        results_dir = ses / "saved_analysis_results"
        if not results_dir.exists():
            continue

        views = _load_trial_views(results_dir)
        trial_data = views.get("trial_data", pd.DataFrame()).copy()
        if trial_data.empty:
            continue
        for c in ["sequence_start", "sequence_end", "fa_time", "first_supply_time", "first_reward_poke_time"]:
            if c in trial_data.columns:
                trial_data[c] = pd.to_datetime(trial_data[c], errors="coerce")

        try:
            tracking, _ = _load_tracking_and_behavior(subjid, date_str)
        except Exception as e:
            print(f"Skipping {date_str}: tracking load failed ({e})")
            continue
        tracking = tracking.copy()
        tracking["time"] = pd.to_datetime(tracking["time"], errors="coerce")
        tracking = tracking.dropna(subset=["time"]).reset_index(drop=True)
        for cand in [("centroid_x", "centroid_y"), ("X", "Y")]:
            if cand[0] in tracking.columns and cand[1] in tracking.columns:
                tracking["X"] = tracking[cand[0]]
                tracking["Y"] = tracking[cand[1]]
                break
        tracking = tracking.dropna(subset=["X", "Y"])
        tracking = tracking.loc[:, ~tracking.columns.duplicated()]
        if tracking.empty:
            continue

        speed_df = _get_from_cache(subjid, date_str, kind="speed_analysis")
        if speed_df is None:
            path_speed = results_dir / "speed_analysis.parquet"
            if not path_speed.exists():
                print(f"No speed_analysis.parquet for {date_str}; run plot_epoch_speeds_by_condition first")
                continue
            try:
                speed_df = pd.read_parquet(path_speed)
                _update_cache(subjid, [date_str], {date_str: speed_df.copy()}, kind="speed_analysis")
            except Exception as e:
                print(f"Failed to read speed_analysis for {date_str}: {e}")
                continue
        speed_df = speed_df.copy()
        for col in ["bin_mid_time", "bin_start_time", "bin_end_time"]:
            if col in speed_df.columns:
                speed_df[col] = pd.to_datetime(speed_df[col], errors="coerce")

        traces = {"rewarded": [], "unrewarded": [], "fa": []}
        data_lines = {"rewarded": [], "unrewarded": [], "fa": []}
        fixed_lines = {"rewarded": [], "unrewarded": [], "fa": []}

        for idx_row, row in trial_data.iterrows():
            cond = _condition_label(row)
            if cond is None:
                continue
            bins_df = speed_df[speed_df["trial_index"] == idx_row].sort_values("bin_mid_s")
            if bins_df.empty:
                continue
            start_bin = bins_df.loc[(bins_df["bin_mid_s"].sub(start_target_s).abs() <= (bin_ms / 1000.0) * 0.01)]
            if start_bin.empty:
                start_bin = bins_df.head(1)
            if start_bin.empty or "bin_end_time" not in start_bin.columns:
                continue
            start_time = pd.to_datetime(start_bin.iloc[0]["bin_end_time"], errors="coerce")
            end_time = pd.to_datetime(bins_df.sort_values("bin_end_time").iloc[-1]["bin_end_time"], errors="coerce") if "bin_end_time" in bins_df.columns else pd.NaT
            if pd.isna(start_time) or pd.isna(end_time) or end_time <= start_time:
                continue

            seg = tracking[(tracking["time"] >= start_time) & (tracking["time"] <= end_time)][["X", "Y", "time"]].copy()
            if len(seg) < 2:
                continue
            seg = seg.sort_values("time")
            x_arr = seg["X"].to_numpy(dtype=float)
            y_arr = seg["Y"].to_numpy(dtype=float)

            start_idx = int(np.argmin(np.abs((seg["time"] - start_time).dt.total_seconds())))
            end_idx = int(np.argmin(np.abs((seg["time"] - end_time).dt.total_seconds())))
            start_xy = seg.iloc[start_idx][["X", "Y"]].to_numpy(dtype=float)
            end_xy = seg.iloc[end_idx][["X", "Y"]].to_numpy(dtype=float)

            port = _infer_port(row)
            fixed_start = np.asarray(fixed_start_xy, dtype=float)
            fixed_goal = np.asarray(fixed_goal_b_xy if port == 2 else fixed_goal_a_xy, dtype=float)

            traces[cond].append((x_arr, y_arr))
            data_lines[cond].append((start_xy, end_xy))
            fixed_lines[cond].append((fixed_start, fixed_goal))

        for cond in ["rewarded", "unrewarded", "fa"]:
            if not traces[cond]:
                continue
            fig, ax = plt.subplots(figsize=figsize)
            for (x_arr, y_arr), (sxy, gxy), (fsxy, fgxy) in zip(traces[cond], data_lines[cond], fixed_lines[cond]):
                ax.plot(x_arr, y_arr, color=cond_colors[cond], alpha=0.35, linewidth=1.2)
                ax.plot([sxy[0], gxy[0]], [sxy[1], gxy[1]], color=data_line_color, linestyle="--", linewidth=1.4, alpha=0.9)
                ax.plot([fsxy[0], fgxy[0]], [fsxy[1], fgxy[1]], color=fixed_line_color, linestyle="-", linewidth=1.4, alpha=0.9)
            # Always show a reference fixed B line for visual comparison
            ax.plot(
                [fixed_start_xy[0], fixed_goal_b_xy[0]],
                [fixed_start_xy[1], fixed_goal_b_xy[1]],
                color=fixed_line_color,
                linestyle="-",
                linewidth=1.6,
                alpha=0.6,
            )
            ax.set_title(f"{cond.capitalize()} traces with data vs fixed lines — {date_str}")
            ax.set_xlabel("X (px)")
            ax.set_ylabel("Y (px)")
            ax.set_aspect("equal", adjustable="box")
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            figs[(date_str, cond)] = fig

    return figs


def plot_movement_analysis_statistics(
    subjid,
    dates=None,
    *,
    fa_types="FA_time_in",
    figsize=(10, 6),
    clean_graph: bool = False,
):
    """Scatter movement-related metrics per condition with mean±SEM.

    Produces five figures per session when data are present:
    - Movement onset latency relative to poke_out (latency_s from speed_analysis.parquet)
    - Animal's Consideration Time (Valve Onset - Movement Onset) (movement_onset_from_valve_s from speed_analysis.parquet)
    - Path length traveled per trial (path_length_px from speed_analysis.parquet)
    - Movement duration per trial (travel_time_s from speed_analysis.parquet)
    - Tortuosity per trial (tortuosity from speed_analysis.parquet)

    Returns dict with per-session figs and combined figs when multiple dates are provided.
    """

    # FA filter
    if isinstance(fa_types, str):
        if fa_types.lower() == "all":
            def fa_filter_fn(lbl):
                return str(lbl).lower().startswith("fa_") if pd.notna(lbl) else False
        else:
            fa_set = {s.strip().lower() for s in re.split(r"[;,]", fa_types) if s.strip()}
            def fa_filter_fn(lbl):
                return str(lbl).lower() in fa_set if pd.notna(lbl) else False
    else:
        fa_set = {str(s).strip().lower() for s in fa_types}
        def fa_filter_fn(lbl):
            return str(lbl).lower() in fa_set if pd.notna(lbl) else False

    subj_str = f"sub-{str(subjid).zfill(3)}"
    subj_dirs = list(get_derivatives_root().glob(f"{subj_str}_id-*"))
    if not subj_dirs:
        raise FileNotFoundError(f"No subject directory found for {subj_str}")
    subj_dir = subj_dirs[0]

    ses_dirs = _filter_session_dirs(subj_dir, dates)
    if not ses_dirs:
        raise FileNotFoundError(f"No sessions found for subject {subjid} with given dates")

    def _condition_label(row):
        rtc = str(row.get("response_time_category", "")).lower()
        is_aborted = bool(row.get("is_aborted", False))
        fa_label = str(row.get("fa_label", "")).lower()
        if rtc == "rewarded" and not is_aborted:
            return "rewarded"
        if rtc == "unrewarded" and not is_aborted:
            return "unrewarded"
        if fa_label.startswith("fa_") and fa_filter_fn(fa_label):
            return "fa"
        return None

    def _kw_mwu_by_group(df, value_col, group_col="condition", min_pair_n=5):
        """Run Kruskal-Wallis across groups, then pairwise Mann-Whitney U with Holm correction if KW is significant.

        Returns dict with keys:
          - kruskal: {"stat": H, "p": p, "n_per_group": {...}, "groups": [...]} or None
          - pairwise: list of {g1, g2, n1, n2, u_stat, p_raw, p_corr} (only if KW significant and n>=min_pair_n in both).
        """
        if df is None or df.empty or value_col not in df.columns or group_col not in df.columns:
            return {"kruskal": None, "pairwise": []}

        clean_df = df[[group_col, value_col]].dropna()
        clean_df = clean_df[np.isfinite(clean_df[value_col].astype(float))]
        if clean_df.empty:
            return {"kruskal": None, "pairwise": []}

        groups = {}
        for g, sub in clean_df.groupby(group_col):
            vals = sub[value_col].astype(float).to_numpy()
            if vals.size > 0:
                groups[g] = vals

        if len(groups) < 2:
            return {"kruskal": None, "pairwise": []}

        # Kruskal-Wallis
        try:
            kw_stat, kw_p = kruskal(*groups.values())
        except Exception:
            return {"kruskal": None, "pairwise": []}

        kruskal_res = {
            "stat": float(kw_stat),
            "p": float(kw_p),
            "n_per_group": {k: int(len(v)) for k, v in groups.items()},
            "groups": list(groups.keys()),
        }

        # Pairwise only if significant
        pairwise = []
        if kw_p < 0.05:
            pairs = [("rewarded", "unrewarded"), ("rewarded", "fa"), ("unrewarded", "fa")]
            raw_ps = []
            stats_tmp = []
            for g1, g2 in pairs:
                v1 = groups.get(g1)
                v2 = groups.get(g2)
                n1 = len(v1) if v1 is not None else 0
                n2 = len(v2) if v2 is not None else 0
                if v1 is None or v2 is None or n1 < min_pair_n or n2 < min_pair_n:
                    continue
                try:
                    u_stat, p_raw = mannwhitneyu(v1, v2, alternative="two-sided")
                except Exception:
                    continue
                raw_ps.append(p_raw)
                stats_tmp.append({"g1": g1, "g2": g2, "n1": n1, "n2": n2, "u_stat": float(u_stat), "p_raw": float(p_raw)})

            # Holm-Bonferroni on the collected raw p-values
            m = len(raw_ps)
            if m > 0:
                order = np.argsort(raw_ps)
                adjusted = np.empty(m)
                max_adj = 0.0
                for rank, idx in enumerate(order):
                    adj = raw_ps[idx] * (m - rank)
                    adj = min(adj, 1.0)
                    max_adj = max(max_adj, adj) # this should enfore monotonicity, as each p_corr should be >= the previous one
                    adjusted[idx] = max_adj
                # map back
                for i, entry in enumerate(stats_tmp):
                    entry["p_corr"] = float(adjusted[i])
                    pairwise.append(entry)

        return {"kruskal": kruskal_res, "pairwise": pairwise}

    per_session = []
    combined_rows = []
    combined_valve_rows = []
    combined_path_rows = []
    combined_travel_rows = []
    combined_tortuosity_rows = []

    cond_positions = {"rewarded": 0.0, "unrewarded": 0.4, "fa": 0.8}
    cond_colors = {"rewarded": "#4CAF50", "unrewarded": "#F44336", "fa": "#2196F3"}
    jitter_span = 0.06  # tighter jitter to match closer grouping

    def _style_axis(ax, *, ylabel: str, xticklabels=None):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(2.5)
        ax.spines["bottom"].set_linewidth(2.5)
        ax.tick_params(axis="y", width=2.3, labelsize=13)
        ax.tick_params(axis="x", width=2.0, labelsize=13)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=16)
        if clean_graph:
            _clean_graph(ax, ylabel=ylabel)

    def _plot_by_trial_sequence(df, value_col, ylabel):
        df_seq = df.copy()
        df_seq["seq_in_condition"] = df_seq.groupby("condition").cumcount() + 1

        fig_seq, ax_seq = plt.subplots(figsize=figsize)
        for cond, color in cond_colors.items():
            sub = df_seq[df_seq["condition"] == cond]
            if sub.empty:
                continue
            x_vals = sub["seq_in_condition"].astype(float).to_numpy()
            y_vals = sub[value_col].astype(float).to_numpy()
            ax_seq.scatter(x_vals, y_vals, color=color, alpha=0.7)
            if len(x_vals) >= 2:
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                x_line = np.array([x_vals.min(), x_vals.max()])
                y_line = slope * x_line + intercept
                ax_seq.plot(x_line, y_line, color=color, linewidth=2.0, alpha=0.9,
                             label=f"{cond}: y={slope:.3f}x+{intercept:.3f}")
            else:
                ax_seq.plot([], [], color=color, linewidth=0, label=f"{cond}: n={len(x_vals)}")

        ax_seq.set_xlabel("Trial # (within condition)", fontsize=14)
        _style_axis(ax_seq, ylabel=ylabel)
        ax_seq.legend()
        fig_seq.tight_layout()
        return fig_seq

    for ses in ses_dirs:
        date_str = ses.name.split("_date-")[-1]
        results_dir = ses / "saved_analysis_results"
        if not results_dir.exists():
            continue

        # load trial_data
        views = _load_trial_views(results_dir)
        trial_data = views.get("trial_data", pd.DataFrame()).copy()
        if trial_data.empty:
            continue
        for c in ["response_time_category", "fa_label", "is_aborted"]:
            if c in trial_data.columns:
                continue

        # load speed_analysis
        speed_df = _get_from_cache(subjid, date_str, kind="speed_analysis")
        if speed_df is None:
            path_speed = results_dir / "speed_analysis.parquet"
            if not path_speed.exists():
                print(f"No speed_analysis.parquet for {date_str}")
                continue
            speed_df = pd.read_parquet(path_speed)
            _update_cache(subjid, [date_str], {date_str: speed_df.copy()}, kind="speed_analysis")
        speed_df = speed_df.copy()

        latencies = []
        valve_latencies = []
        path_lengths = []
        travel_times = []
        tortuosities = []
        for idx_row, row in trial_data.iterrows():
            cond = _condition_label(row)
            if cond is None:
                continue
            bins = speed_df[speed_df["trial_index"] == idx_row]
            if bins.empty:
                continue
            lat = bins["latency_s"].dropna()
            if not lat.empty:
                lat_val = float(lat.iloc[0])
                latencies.append({"date": date_str, "condition": cond, "latency_s": lat_val})

            if "movement_onset_from_valve_s" in bins.columns:
                mov = bins["movement_onset_from_valve_s"].dropna()
                if not mov.empty:
                    valve_latencies.append({"date": date_str, "condition": cond, "movement_from_valve_s": float(mov.iloc[0])})

            if "path_length_px" in bins.columns:
                pl = bins["path_length_px"].dropna()
                if not pl.empty:
                    path_lengths.append({
                        "date": date_str,
                        "condition": cond,
                        "path_length_px": float(pl.iloc[0]),
                    })
            if "travel_time_s" in bins.columns:
                tt = bins["travel_time_s"].dropna()
                if not tt.empty:
                    travel_times.append({
                        "date": date_str,
                        "condition": cond,
                        "travel_time_s": float(tt.iloc[0]),
                    })
            if "tortuosity" in bins.columns:
                tor = bins["tortuosity"].dropna()
                if not tor.empty:
                    tortuosities.append({
                        "date": date_str,
                        "condition": cond,
                        "tortuosity": float(tor.iloc[0]),
                    })

        if not any([latencies, valve_latencies, path_lengths, travel_times, tortuosities]):
            continue

        entry = {"date": date_str}

        if latencies:
            df_ses = pd.DataFrame(latencies)
            entry["data"] = df_ses
            combined_rows.append(df_ses)

            fig, ax = plt.subplots(figsize=figsize)
            for cond, color in [("rewarded", "#4CAF50"), ("unrewarded", "#F44336"), ("fa", "#2196F3")]:
                sub = df_ses[df_ses["condition"] == cond]
                if sub.empty:
                    continue
                y = sub["latency_s"].astype(float)
                x0 = cond_positions[cond]
                x_jit = x0 + (np.random.rand(len(y)) - 0.5) * jitter_span
                ax.scatter(x_jit, y, color=color, alpha=0.6, label=f"{cond} trials")
                mean = y.mean()
                sem = y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else np.nan
                ax.errorbar(x0, mean, yerr=sem, fmt="o", color="black", capsize=4)
            ax.set_xticks(list(cond_positions.values()))
            ax.set_xticklabels(["Rewarded", "Unrewarded", "FA"])
            ax.set_xlim(-0.2, 1.0)
            _style_axis(ax, ylabel="Latency (s)", xticklabels=["Rewarded", "Unrewarded", "FA"])
            fig.tight_layout()
            entry["fig"] = fig

            entry["fig_latency_by_trial"] = _plot_by_trial_sequence(df_ses, "latency_s", "Latency (s)")

        if valve_latencies:
            df_valve = pd.DataFrame(valve_latencies)
            entry["valve_data"] = df_valve
            combined_valve_rows.append(df_valve)

            fig_v, ax_v = plt.subplots(figsize=figsize)
            for cond, color in [("rewarded", "#4CAF50"), ("unrewarded", "#F44336"), ("fa", "#2196F3")]:
                sub = df_valve[df_valve["condition"] == cond]
                if sub.empty:
                    continue
                y = sub["movement_from_valve_s"].astype(float)
                x0 = cond_positions[cond]
                x_jit = x0 + (np.random.rand(len(y)) - 0.5) * jitter_span
                ax_v.scatter(x_jit, y, color=color, alpha=0.6, label=f"{cond} trials")
                mean = y.mean()
                sem = y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else np.nan
                ax_v.errorbar(x0, mean, yerr=sem, fmt="o", color="black", capsize=4)
            ax_v.set_xticks(list(cond_positions.values()))
            ax_v.set_xticklabels(["Rewarded", "Unrewarded", "FA"])
            ax_v.set_xlim(-0.2, 1.0)
            _style_axis(ax_v, ylabel="Consideration Time (s)", xticklabels=["Rewarded", "Unrewarded", "FA"])
            fig_v.tight_layout()
            entry["fig_valve"] = fig_v

            entry["fig_valve_by_trial"] = _plot_by_trial_sequence(df_valve, "movement_from_valve_s", "Consideration Time (s)")

        if path_lengths:
            df_path = pd.DataFrame(path_lengths)
            entry["path_data"] = df_path
            combined_path_rows.append(df_path)

            fig_p, ax_p = plt.subplots(figsize=figsize)
            for cond, color in [("rewarded", "#4CAF50"), ("unrewarded", "#F44336"), ("fa", "#2196F3")]:
                sub = df_path[df_path["condition"] == cond]
                if sub.empty:
                    continue
                y = sub["path_length_px"].astype(float)
                x0 = cond_positions[cond]
                x_jit = x0 + (np.random.rand(len(y)) - 0.5) * jitter_span
                ax_p.scatter(x_jit, y, color=color, alpha=0.6, label=f"{cond} trials")
                mean = y.mean()
                sem = y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else np.nan
                ax_p.errorbar(x0, mean, yerr=sem, fmt="o", color="black", capsize=4)
            ax_p.set_xticks(list(cond_positions.values()))
            ax_p.set_xticklabels(["Rewarded", "Unrewarded", "FA"])
            ax_p.set_xlim(-0.2, 1.0)
            _style_axis(ax_p, ylabel="Path length (px)", xticklabels=["Rewarded", "Unrewarded", "FA"])
            fig_p.tight_layout()
            entry["fig_path"] = fig_p

            entry["fig_path_by_trial"] = _plot_by_trial_sequence(df_path, "path_length_px", "Path length (px)")

        if travel_times:
            df_travel = pd.DataFrame(travel_times)
            entry["travel_data"] = df_travel
            combined_travel_rows.append(df_travel)

            fig_t, ax_t = plt.subplots(figsize=figsize)
            for cond, color in [("rewarded", "#4CAF50"), ("unrewarded", "#F44336"), ("fa", "#2196F3")]:
                sub = df_travel[df_travel["condition"] == cond]
                if sub.empty:
                    continue
                y = sub["travel_time_s"].astype(float)
                x0 = cond_positions[cond]
                x_jit = x0 + (np.random.rand(len(y)) - 0.5) * jitter_span
                ax_t.scatter(x_jit, y, color=color, alpha=0.6, label=f"{cond} trials")
                mean = y.mean()
                sem = y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else np.nan
                ax_t.errorbar(x0, mean, yerr=sem, fmt="o", color="black", capsize=4)
            ax_t.set_xticks(list(cond_positions.values()))
            ax_t.set_xticklabels(["Rewarded", "Unrewarded", "FA"])
            ax_t.set_xlim(-0.2, 1.0)
            _style_axis(ax_t, ylabel="Duration (s)", xticklabels=["Rewarded", "Unrewarded", "FA"])
            fig_t.tight_layout()
            entry["fig_travel"] = fig_t

            entry["fig_travel_by_trial"] = _plot_by_trial_sequence(df_travel, "travel_time_s", "Duration (s)")

        if tortuosities:
            df_tort = pd.DataFrame(tortuosities)
            entry["tortuosity_data"] = df_tort
            combined_tortuosity_rows.append(df_tort)

            fig_to, ax_to = plt.subplots(figsize=figsize)
            for cond, color in [("rewarded", "#4CAF50"), ("unrewarded", "#F44336"), ("fa", "#2196F3")]:
                sub = df_tort[df_tort["condition"] == cond]
                if sub.empty:
                    continue
                y = sub["tortuosity"].astype(float)
                x0 = cond_positions[cond]
                x_jit = x0 + (np.random.rand(len(y)) - 0.5) * jitter_span
                ax_to.scatter(x_jit, y, color=color, alpha=0.6, label=f"{cond} trials")
                mean = y.mean()
                sem = y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else np.nan
                ax_to.errorbar(x0, mean, yerr=sem, fmt="o", color="black", capsize=4)
            ax_to.set_xticks(list(cond_positions.values()))
            ax_to.set_xticklabels(["Rewarded", "Unrewarded", "FA"])
            ax_to.set_xlim(-0.2, 1.0)
            _style_axis(ax_to, ylabel="Tortuosity", xticklabels=["Rewarded", "Unrewarded", "FA"])
            fig_to.tight_layout()
            entry["fig_tortuosity"] = fig_to

            entry["fig_tortuosity_by_trial"] = _plot_by_trial_sequence(df_tort, "tortuosity", "Tortuosity")

        per_session.append(entry)

    combined_fig = None
    combined_valve_fig = None
    combined_path_fig = None
    combined_travel_fig = None
    combined_tortuosity_fig = None
    if len(per_session) > 1 and combined_rows:
        all_df = pd.concat(combined_rows, ignore_index=True)
        fig, ax = plt.subplots(figsize=figsize)
        for cond, color in [("rewarded", "#4CAF50"), ("unrewarded", "#F44336"), ("fa", "#2196F3")]:
            sub = all_df[all_df["condition"] == cond]
            if sub.empty:
                continue
            per_ses = sub.groupby("date")["latency_s"]
            means = per_ses.mean()
            sems = per_ses.sem()
            x_vals = np.arange(len(means))
            ax.errorbar(x_vals, means.values, yerr=sems.values, fmt="o", color=color, label=f"{cond} session means")
        ax.set_xticks(np.arange(len(all_df["date"].unique())))
        ax.set_xticklabels(sorted(all_df["date"].unique()), rotation=45, ha="right", fontsize=12)
        _style_axis(ax, ylabel="Latency (s)")
        ax.legend()
        fig.tight_layout()
        combined_fig = fig

    if len(per_session) > 1 and combined_valve_rows:
        all_valve = pd.concat(combined_valve_rows, ignore_index=True)
        fig_v, ax_v = plt.subplots(figsize=figsize)
        for cond, color in [("rewarded", "#4CAF50"), ("unrewarded", "#F44336"), ("fa", "#2196F3")]:
            sub = all_valve[all_valve["condition"] == cond]
            if sub.empty:
                continue
            per_ses = sub.groupby("date")["movement_from_valve_s"]
            means = per_ses.mean()
            sems = per_ses.sem()
            x_vals = np.arange(len(means))
            ax_v.errorbar(x_vals, means.values, yerr=sems.values, fmt="o", color=color, label=f"{cond} session means")
        ax_v.set_xticks(np.arange(len(all_valve["date"].unique())))
        ax_v.set_xticklabels(sorted(all_valve["date"].unique()), rotation=45, ha="right", fontsize=12)
        _style_axis(ax_v, ylabel="Consideration Time (s)")
        ax_v.legend()
        fig_v.tight_layout()
        combined_valve_fig = fig_v

    if len(per_session) > 1 and combined_path_rows:
        all_path = pd.concat(combined_path_rows, ignore_index=True)
        fig_p, ax_p = plt.subplots(figsize=figsize)
        for cond, color in [("rewarded", "#4CAF50"), ("unrewarded", "#F44336"), ("fa", "#2196F3")]:
            sub = all_path[all_path["condition"] == cond]
            if sub.empty:
                continue
            per_ses = sub.groupby("date")["path_length_px"]
            means = per_ses.mean()
            sems = per_ses.sem()
            x_vals = np.arange(len(means))
            ax_p.errorbar(x_vals, means.values, yerr=sems.values, fmt="o", color=color, label=f"{cond} session means")
        ax_p.set_xticks(np.arange(len(all_path["date"].unique())))
        ax_p.set_xticklabels(sorted(all_path["date"].unique()), rotation=45, ha="right", fontsize=12)
        _style_axis(ax_p, ylabel="Path length (px)")
        ax_p.legend()
        fig_p.tight_layout()
        combined_path_fig = fig_p

    if len(per_session) > 1 and combined_travel_rows:
        all_travel = pd.concat(combined_travel_rows, ignore_index=True)
        fig_t, ax_t = plt.subplots(figsize=figsize)
        for cond, color in [("rewarded", "#4CAF50"), ("unrewarded", "#F44336"), ("fa", "#2196F3")]:
            sub = all_travel[all_travel["condition"] == cond]
            if sub.empty:
                continue
            per_ses = sub.groupby("date")["travel_time_s"]
            means = per_ses.mean()
            sems = per_ses.sem()
            x_vals = np.arange(len(means))
            ax_t.errorbar(x_vals, means.values, yerr=sems.values, fmt="o", color=color, label=f"{cond} session means")
        ax_t.set_xticks(np.arange(len(all_travel["date"].unique())))
        ax_t.set_xticklabels(sorted(all_travel["date"].unique()), rotation=45, ha="right", fontsize=12)
        _style_axis(ax_t, ylabel="Duration (s)")
        ax_t.legend()
        fig_t.tight_layout()
        combined_travel_fig = fig_t

    if len(per_session) > 1 and combined_tortuosity_rows:
        all_tort = pd.concat(combined_tortuosity_rows, ignore_index=True)
        fig_to, ax_to = plt.subplots(figsize=figsize)
        for cond, color in [("rewarded", "#4CAF50"), ("unrewarded", "#F44336"), ("fa", "#2196F3")]:
            sub = all_tort[all_tort["condition"] == cond]
            if sub.empty:
                continue
            per_ses = sub.groupby("date")["tortuosity"]
            means = per_ses.mean()
            sems = per_ses.sem()
            x_vals = np.arange(len(means))
            ax_to.errorbar(x_vals, means.values, yerr=sems.values, fmt="o", color=color, label=f"{cond} session means")
        ax_to.set_xticks(np.arange(len(all_tort["date"].unique())))
        ax_to.set_xticklabels(sorted(all_tort["date"].unique()), rotation=45, ha="right", fontsize=12)
        _style_axis(ax_to, ylabel="Tortuosity")
        ax_to.legend()
        fig_to.tight_layout()
        combined_tortuosity_fig = fig_to

    # Statistical summaries across all pooled sessions/trials (by condition)
    stats_summary = {}
    stats_summary["latency_s"] = _kw_mwu_by_group(pd.concat(combined_rows, ignore_index=True) if combined_rows else pd.DataFrame(), "latency_s")
    stats_summary["movement_from_valve_s"] = _kw_mwu_by_group(pd.concat(combined_valve_rows, ignore_index=True) if combined_valve_rows else pd.DataFrame(), "movement_from_valve_s")
    stats_summary["path_length_px"] = _kw_mwu_by_group(pd.concat(combined_path_rows, ignore_index=True) if combined_path_rows else pd.DataFrame(), "path_length_px")
    stats_summary["travel_time_s"] = _kw_mwu_by_group(pd.concat(combined_travel_rows, ignore_index=True) if combined_travel_rows else pd.DataFrame(), "travel_time_s")
    stats_summary["tortuosity"] = _kw_mwu_by_group(pd.concat(combined_tortuosity_rows, ignore_index=True) if combined_tortuosity_rows else pd.DataFrame(), "tortuosity")

    # Print statistical summary
    print("\n" + "="*60)
    print("STATISTICAL SUMMARY (Kruskal-Wallis + Pairwise Mann-Whitney U with Holm-Bonferroni correction)")
    print("="*60)

    for variable, results in stats_summary.items():
        if results["kruskal"] is None:
            print(f"{variable}: No data")
            continue
        
        kw_p = results["kruskal"]["p"]
        print(f"\n{variable}: Kruskal-Wallis: p = {kw_p:.4f}")
        
        # Only print pairwise comparisons if KW is significant
        if kw_p < 0.05 and results["pairwise"]:
            for comparison in results["pairwise"]:
                g1 = comparison["g1"]
                g2 = comparison["g2"]
                p_corr = comparison["p_corr"]
                print(f"      {g1.capitalize()} vs {g2.capitalize()}: p = {p_corr:.4f} (corrected)")
        elif kw_p >= 0.05:
            print("      (not significant)")

    print("\n" + "="*60 + "\n")

    return {
        "per_session": per_session,
        "combined": combined_fig,
        "combined_valve": combined_valve_fig,
        "combined_path": combined_path_fig,
        "combined_travel": combined_travel_fig,
        "combined_tortuosity": combined_tortuosity_fig,
        "stats": stats_summary,
    }

