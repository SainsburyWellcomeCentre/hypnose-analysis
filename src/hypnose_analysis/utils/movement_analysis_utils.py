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
            for idx_row, row, seg, t_zero, speed_bins in iter_trials(trials):
                port = None
                if hr_flag and bool(row.get(hr_flag, False)):
                    port = _port_from_first_supply(row) or _infer_port(row)
                category, port_fallback = _category_from_row(row)
                if port is None:
                    port = port_fallback
                color_map = port_colors_hr if (highlight_hr and hr_flag and bool(row.get(hr_flag, False))) else port_colors
                color = color_map.get(port, port_colors[1 if category == "A" else 2])
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
            for idx_row, row, seg, t_zero, speed_bins in iter_trials(fa_df):
                # Use FA port first, then supply/response port
                port = row.get("fa_port") if pd.notna(row.get("fa_port")) else _infer_port(row)
                if port not in {1, 2}:
                    continue
                category = "A" if port == 1 else "B"
                color = port_colors_fa.get(port, port_colors_fa[1])
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
            for idx_row, row, seg, t_zero, speed_bins in iter_trials(fa_df):
                odor_name = row.get("last_odor_name") or row.get("last_odor")
                odor = _odor_letter(odor_name)
                if odor in {"A", "B", "OdorA", "OdorB"}:
                    continue
                port = row.get("fa_port") if pd.notna(row.get("fa_port")) else _infer_port(row)
                color = port_colors_fa.get(port, port_colors_fa[1])
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
    if color_by_speed and speed_vals_global:
        vmin = np.nanmin(speed_vals_global)
        vmax = np.nanmax(speed_vals_global)
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            speed_norm = Normalize(vmin=vmin, vmax=vmax)
    color_by_speed_active = color_by_speed and (speed_norm is not None)

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


def plot_epoch_speeds_by_condition(
    subjid,
    dates=None,
    *,
    bin_ms: int = 100,
    pre_buffer_s: float = 0.0,
    fa_label_filter=None,
    mode: str = "max",
    threshold: bool = False,
    threshold_alpha: float = 6.0,
    threshold_beta: float = 6.0,
    figsize=(8, 5),
):
    """Plot cue-port speed epochs aligned to last poke-out for rewarded, unrewarded, and FA trials.

    For each session, builds two figures:
      1) Violin of epoch speeds pooled by condition.
      2) Per-trial epoch traces per condition with a session mean overlay.
    If multiple dates are provided, also builds combined figures (one per condition)
    showing the session-level mean trace for each session with data.

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
        if isinstance(pts, dict) and pts:
            vals = list(pts.values())
            if all(isinstance(v, dict) and "position" in v for v in vals):
                vals = sorted(vals, key=lambda v: v.get("position", 0))
            last = vals[-1]
            return _safe_dt(last.get("poke_odor_end"))
        if isinstance(pts, list) and pts:
            last = pts[-1]
            if isinstance(last, dict):
                return _safe_dt(last.get("poke_odor_end"))
        for cand in ["poke_odor_end", "last_poke_out_time", "last_poke_time"]:
            if cand in row:
                return _safe_dt(row.get(cand))
        if "sequence_start" in row:
            return _safe_dt(row.get("sequence_start"))
        return pd.NaT

    def _end_time(row, cond):
        if cond == "rewarded":
            return _safe_dt(row.get("first_supply_time")) or _safe_dt(row.get("sequence_end"))
        if cond == "unrewarded":
            return _safe_dt(row.get("first_reward_poke_time"))
        if cond == "fa":
            return _safe_dt(row.get("fa_time")) or _safe_dt(row.get("sequence_end"))
        return _safe_dt(row.get("sequence_end"))

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

        views = _load_trial_views(results_dir)
        trial_data = views.get("trial_data", pd.DataFrame()).copy()
        if trial_data.empty:
            print(f"No trial_data for {date_str}; skipping")
            continue
        for c in ["sequence_start", "sequence_end", "first_supply_time", "first_reward_poke_time", "fa_time"]:
            if c in trial_data.columns:
                trial_data[c] = pd.to_datetime(trial_data[c], errors="coerce")

        trials_info = []
        for idx_row, row in trial_data.iterrows():
            cond = _condition_label(row)
            if cond is None:
                continue
            t_zero = _last_poke_out(row)
            t_end = _end_time(row, cond)
            if pd.isna(t_zero) or pd.isna(t_end) or t_end <= t_zero:
                continue
            dur_post = (t_end - t_zero).total_seconds()
            if dur_post <= 0:
                continue
            trials_info.append((idx_row, cond, t_zero, t_end, dur_post))

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
        # cache per-trial bins for threshold computation without writing arrays into the DataFrame
        trial_bins = {}
        mids_common = None

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
                    })

            trial_bins[idx_row] = {
                "mids": mids_trial,
                "arr": arr_trial,
                "t_zero": t_zero,
            }

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
        if threshold and thr_max is not None:
            for idx_row, cond, t_zero, t_end, _ in trials_info:
                bins = trial_bins.get(idx_row)
                if not bins:
                    continue
                mids_trial = bins.get("mids")
                arr_trial = bins.get("arr")
                if mids_trial is None or arr_trial is None:
                    continue
                crossing = np.where((mids_trial >= 0) & (arr_trial > thr_max))[0]
                if crossing.size > 0:
                    k = crossing[0]
                    crossing_time = t_zero + pd.Timedelta(seconds=float(mids_trial[k]))
                    trial_data.at[idx_row, "speed_threshold_time"] = crossing_time

        # Fill per-bin records with per-trial threshold (repeat per bin so one file carries both)
        if "speed_threshold_time" in trial_data.columns:
            thr_map = trial_data["speed_threshold_time"].to_dict()
            for rec in epoch_records:
                rec["speed_threshold_time"] = thr_map.get(rec["trial_index"], pd.NaT)

        analysis_path = results_dir / "speed_analysis.parquet"
        try:
            analysis_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(epoch_records).to_parquet(analysis_path, index=False)
        except Exception as e:
            print(f"Warning: failed to write {analysis_path.name}: {e}")
        else:
            _update_cache(subjid, [date_str], {date_str: pd.DataFrame(epoch_records)}, kind="speed_analysis")

        # Per-session violin plot
        fig_violin, ax_violin = plt.subplots(figsize=figsize)
        data = [speeds_flat[c] for c in conds_with_data]
        ax_violin.violinplot(data, showmeans=True, showextrema=True, widths=0.8)
        ax_violin.set_xticks(range(1, len(conds_with_data) + 1))
        ax_violin.set_xticklabels(conds_with_data)
        ax_violin.set_ylabel("Speed (units/s)")
        ax_violin.set_title(f"Epoch speeds — sub {subjid}, {date_str} ({mode})")
        fig_violin.tight_layout()

        # Per-session per-trial traces with mean (one figure per condition for readability)
        fig_traces_by_cond = {}
        for cond in conds_with_data:
            trials = epoch_series[cond]
            fig_t, ax_t = plt.subplots(figsize=figsize)
            if not trials:
                ax_t.text(0.5, 0.5, "No trials", ha="center")
            else:
                mids_use = mids_common
                if mids_use is None:
                    mids_use = edges[:-1] + (edges[1] - edges[0]) / 2
                for arr in trials:
                    ax_t.plot(mids_use, arr, color="gray", alpha=0.2)
                stack = np.vstack(trials)
                mean_speeds = np.nanmean(stack, axis=0)
                ax_t.plot(mids_use, mean_speeds, color="blue", linewidth=2, label="session mean")

            if threshold and baseline_mean is not None:
                ax_t.axhline(baseline_mean, color="red", linestyle="-", linewidth=1.5, label="baseline μ")
                if thr_max is not None:
                    ax_t.axhline(thr_max, color="#2F4F4F", linestyle="--", linewidth=1.4, label=f"max(αμ, μ+βσ), α={threshold_alpha:g}, β={threshold_beta:g}")

            ax_t.set_title(f"{cond} — sub {subjid}, {date_str} ({mode})")
            ax_t.set_xlabel("Time from last poke-out (s)")
            ax_t.set_ylabel("Speed (units/s)")
            ax_t.legend()
            fig_t.tight_layout()
            fig_traces_by_cond[cond] = fig_t

        per_session.append({
            "date": date_str,
            "fig_violin": fig_violin,
            "fig_traces": fig_traces_by_cond,
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
    plot_epoch_speeds_by_condition), it is loaded (and cached) and used directly; otherwise the
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
        if isinstance(pts, dict) and pts:
            vals = list(pts.values())
            if all(isinstance(v, dict) and "position" in v for v in vals):
                vals = sorted(vals, key=lambda v: v.get("position", 0))
            last = vals[-1]
            return _safe_dt(last.get("poke_odor_end"))
        if isinstance(pts, list) and pts:
            last = pts[-1]
            if isinstance(last, dict):
                return _safe_dt(last.get("poke_odor_end"))
        for cand in ["poke_odor_end", "last_poke_out_time", "last_poke_time"]:
            if cand in row:
                return _safe_dt(row.get(cand))
        if "sequence_start" in row:
            return _safe_dt(row.get("sequence_start"))
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
                t_end = _end_time(row, cond)
                if pd.isna(t_zero) or pd.isna(t_end) or t_end <= t_zero:
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
            t_end = _end_time(row, cond)
            if pd.isna(t_zero) or pd.isna(t_end) or t_end <= t_zero:
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
            continue

        baseline_vals_arr = np.asarray([v for v in baseline_vals if np.isfinite(v)])
        if baseline_vals_arr.size == 0:
            print(f"Baseline values not finite for {date_str}; skipping session")
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


def plot_tortuosity_by_condition(
    subjid,
    dates=None,
    *,
    fa_types="FA_time_in",
    bin_ms: int = 100,
    use_fixed_coords: bool = False,
    fixed_start_xy=(575, 90),
    fixed_goal_a_xy=(208, 930),
    fixed_goal_b_xy=(973, 930),
    figsize=(10, 6),
):
    """Compute and plot tortuosity per trial (path length / straight line) for rewarded, unrewarded, and FA.

    Requires speed_analysis.parquet (generated by plot_epoch_speeds_by_condition) to align bin times.
    Adds a 'tortuosity' column to trial_data and writes updated parquet/CSV if present in the session.
    
    Returns a dict with per-session figures and an optional combined figure when multiple dates are provided.
    """

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
    derivatives_dir = get_derivatives_root()
    subj_dirs = list(derivatives_dir.glob(f"{subj_str}_id-*"))
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

    per_session = []
    combined_rows = []
    start_target_s = -bin_ms / 2000.0  # target mid-bin time (e.g., -0.05s for 100 ms bins)

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

    for ses in ses_dirs:
        date_str = ses.name.split("_date-")[-1]
        results_dir = ses / "saved_analysis_results"
        if not results_dir.exists():
            continue

        # Load tracking
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

        # Load trial_data
        views = _load_trial_views(results_dir)
        trial_data = views.get("trial_data", pd.DataFrame()).copy()
        if trial_data.empty:
            print(f"No trial_data for {date_str}; skipping")
            continue
        for c in ["sequence_start", "sequence_end", "fa_time", "first_supply_time", "first_reward_poke_time"]:
            if c in trial_data.columns:
                trial_data[c] = pd.to_datetime(trial_data[c], errors="coerce")

        # Load speed_analysis
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

        tortuosity_list = []

        for idx_row, row in trial_data.iterrows():
            cond = _condition_label(row)
            if cond is None:
                continue
            bins_df = speed_df[speed_df["trial_index"] == idx_row]
            if bins_df.empty:
                continue

            # start: bin with mid at -0.05s
            bins_df = bins_df.sort_values("bin_mid_s").reset_index(drop=True)
            start_bin = bins_df.loc[(bins_df["bin_mid_s"].sub(start_target_s).abs() <= (bin_ms / 1000.0) * 0.01)]
            if start_bin.empty:
                start_bin = bins_df.head(1)
            if start_bin.empty or "bin_end_time" not in start_bin.columns:
                continue
            start_time = pd.to_datetime(start_bin.iloc[0]["bin_end_time"], errors="coerce")
            end_time = pd.to_datetime(bins_df.sort_values("bin_end_time").iloc[-1]["bin_end_time"], errors="coerce") if "bin_end_time" in bins_df.columns else pd.NaT
            if pd.isna(start_time) or pd.isna(end_time) or end_time <= start_time:
                continue

            seg_mask = (tracking["time"] >= start_time) & (tracking["time"] <= end_time)
            seg = tracking.loc[seg_mask, ["X", "Y", "time"]].copy()
            if len(seg) < 2:
                continue
            seg = seg.sort_values("time")

            # Coordinates for start/goal as nearest frames to those times
            if use_fixed_coords:
                port = _infer_port(row)
                start_xy = np.asarray(fixed_start_xy, dtype=float)
                end_xy = np.asarray(fixed_goal_b_xy if port == 2 else fixed_goal_a_xy, dtype=float)
            else:
                start_idx = int(np.argmin(np.abs((seg["time"] - start_time).dt.total_seconds())))
                end_idx = int(np.argmin(np.abs((seg["time"] - end_time).dt.total_seconds())))
                start_xy = seg.iloc[start_idx][["X", "Y"]].to_numpy(dtype=float)
                end_xy = seg.iloc[end_idx][["X", "Y"]].to_numpy(dtype=float)

            x_arr = seg["X"].to_numpy(dtype=float)
            y_arr = seg["Y"].to_numpy(dtype=float)
            path_len = float(np.sum(np.hypot(np.diff(x_arr), np.diff(y_arr))))
            straight_len = float(np.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]))
            tortuosity = path_len / straight_len if straight_len > 0 else np.nan

            trial_data.at[idx_row, "tortuosity"] = tortuosity
            tortuosity_list.append({"date": date_str, "condition": cond, "tortuosity": tortuosity})

        if "tortuosity" in trial_data.columns:
            # Persist updated trial_data
            parquet_path = results_dir / "trial_data.parquet"
            csv_path = results_dir / "trial_data.csv"
            try:
                trial_data.to_parquet(parquet_path, index=False)
            except Exception:
                pass
            try:
                trial_data.to_csv(csv_path, index=False)
            except Exception:
                pass

        if not tortuosity_list:
            print(f"No tortuosity computed for {date_str}")
            continue

        df_ses = pd.DataFrame(tortuosity_list)
        per_session.append({"date": date_str, "data": df_ses})
        combined_rows.append(df_ses)

        # Plot per-session scatter with mean±SEM
        fig, ax = plt.subplots(figsize=figsize)
        for cond, color in [("rewarded", "#4CAF50"), ("unrewarded", "#F44336"), ("fa", "#2196F3")]:
            sub = df_ses[df_ses["condition"] == cond]
            if sub.empty:
                continue
            y = sub["tortuosity"].astype(float)
            x_jitter = np.full_like(y, {"rewarded": 0, "unrewarded": 1, "fa": 2}[cond], dtype=float)
            x_jitter = x_jitter + (np.random.rand(len(y)) - 0.5) * 0.1
            ax.scatter(x_jitter, y, color=color, alpha=0.6, label=f"{cond} trials")
            mean = y.mean()
            sem = y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else np.nan
            ax.errorbar({"rewarded": 0, "unrewarded": 1, "fa": 2}[cond], mean, yerr=sem, fmt="o", color="black", capsize=4)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Rewarded", "Unrewarded", "FA"])
        ax.set_ylabel("Tortuosity")
        ax.set_title(f"Tortuosity by condition — {date_str}")
        fig.tight_layout()
        per_session[-1]["fig"] = fig

    combined_fig = None
    if len(per_session) > 1 and combined_rows:
        all_df = pd.concat(combined_rows, ignore_index=True)
        fig, ax = plt.subplots(figsize=figsize)
        for cond, color in [("rewarded", "#4CAF50"), ("unrewarded", "#F44336"), ("fa", "#2196F3")]:
            sub = all_df[all_df["condition"] == cond]
            if sub.empty:
                continue
            # mean per session
            per_ses = sub.groupby("date")
            means = per_ses["tortuosity"].mean()
            sems = per_ses["tortuosity"].sem()
            x_vals = np.arange(len(means))
            ax.errorbar(x_vals, means.values, yerr=sems.values, fmt="o", color=color, label=f"{cond} session means")
        ax.set_xticks(np.arange(len(all_df["date"].unique())))
        ax.set_xticklabels(sorted(all_df["date"].unique()), rotation=45, ha="right")
        ax.set_ylabel("Tortuosity")
        ax.set_title("Tortuosity by condition across sessions")
        ax.legend()
        fig.tight_layout()
        combined_fig = fig

    return {"per_session": per_session, "combined": combined_fig}


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