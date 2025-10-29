import sys
import os
project_root = os.path.abspath("")
if project_root not in sys.path:
    sys.path.append(project_root)
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, Optional, Union
from utils.metrics_utils import load_session_results
from datetime import timedelta, datetime
from utils.classification_utils import load_all_streams, load_experiment
from pathlib import Path
import re
import numpy as np

def plot_cumulative_rewards(subjids, dates, split_days=False, figsize=(12, 6), title=None, save_path=None):
    """
    Plot accumulated rewards over time for one or more sessions.
    
    Parameters:
    -----------
    subjids : int or list of int
        Subject ID(s) to plot
    dates : int/str or list of int/str
        Date(s) to plot for each subject
    split_days : bool, optional
        If True, reset cumulative count to 0 for each day
        If False, accumulate rewards continuously across days
    figsize : tuple, optional
        Figure size (width, height)
    title : str, optional
        Custom plot title. If None, uses default title
    save_path : str or Path, optional
        If provided, saves the plot to this path
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Ensure subjids and dates are lists
    if not isinstance(subjids, (list, tuple)):
        subjids = [subjids]
    if not isinstance(dates, (list, tuple)):
        dates = [dates]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors for different subjects
    colors = plt.cm.tab10(range(len(subjids)))
    
    # Plot each subject
    for subj_idx, subjid in enumerate(subjids):
        all_rewarded = []
        session_info = []
        
        # Load data for each date
        for date in dates:
            try:
                results = load_session_results(subjid, date)
                rewarded_trials = results.get('completed_sequence_rewarded', pd.DataFrame())
                manifest = results.get('manifest', {})
                
                if not rewarded_trials.empty:
                    rewarded_trials = rewarded_trials.copy()
                    rewarded_trials['sequence_start'] = pd.to_datetime(rewarded_trials['sequence_start'])
                    rewarded_trials['date'] = date
                    all_rewarded.append(rewarded_trials)
                    
                # Store session timing info from manifest
                runs = manifest.get('session', {}).get('runs', [])
                if runs:
                    session_info.append({
                        'date': date,
                        'runs': runs,
                        'manifest': manifest
                    })
                    
            except FileNotFoundError:
                print(f"Warning: No data found for subject {subjid}, date {date}")
                continue
        
        if not all_rewarded:
            print(f"No rewarded trials found for subject {subjid}")
            continue
        
        # Combine all dates
        combined = pd.concat(all_rewarded, ignore_index=True)
        combined = combined.sort_values('sequence_start')
        
        # Subject-specific session boundaries and gaps
        subj_session_starts = []
        subj_gaps = []
        
        # Calculate continuous time axis with collapsed inter-session gaps for this subject
        if session_info:
            # Build time offset mapping for each session
            time_offset = 0  # Cumulative offset to add to timestamps
            session_offsets = {}  # Maps session date to its time offset
            
            for sess_idx, sess in enumerate(session_info):
                runs = sess['runs']
                session_date = sess['date']
                
                if sess_idx == 0:
                    # First session: no offset, use actual start time
                    first_run_start = pd.to_datetime(runs[0]['start_time']).tz_localize(None)
                    global_start_time = first_run_start
                    session_offsets[session_date] = 0
                else:
                    # Get end of previous session
                    prev_sess = session_info[sess_idx - 1]
                    prev_last_run = prev_sess['runs'][-1]
                    prev_end = pd.to_datetime(prev_last_run['end_time']).tz_localize(None)
                    
                    # Get start of current session
                    curr_start = pd.to_datetime(runs[0]['start_time']).tz_localize(None)
                    
                    # Calculate the gap between sessions (in seconds)
                    inter_session_gap = (curr_start - prev_end).total_seconds()
                    
                    # Add this gap to our cumulative offset (we want to subtract it from timestamps)
                    time_offset += inter_session_gap - 1  # Keep 1 second buffer
                    session_offsets[session_date] = time_offset
                    
                    # Mark session boundary (time where new session "starts" in plot)
                    boundary_seconds = (prev_end - global_start_time).total_seconds() - session_offsets[prev_sess['date']] + 1
                    subj_session_starts.append(boundary_seconds)
                
                # Calculate gaps within this session
                for run in runs:
                    if 'gap_to_next_run' in run and run['gap_to_next_run']:
                        gap_str = run['gap_to_next_run']
                        try:
                            # Parse format like "0:27:53.342496"
                            parts = gap_str.split(':')
                            if len(parts) == 3:
                                hours = int(parts[0])
                                minutes = int(parts[1])
                                seconds = float(parts[2])
                                gap_duration = hours * 3600 + minutes * 60 + seconds
                            else:
                                gap_duration = float(gap_str)
                            
                            # Gap starts at end_time of this run
                            run_end = pd.to_datetime(run['end_time']).tz_localize(None)
                            gap_start_seconds = (run_end - global_start_time).total_seconds() - session_offsets[session_date]
                            gap_end_seconds = gap_start_seconds + gap_duration
                            
                            subj_gaps.append((gap_start_seconds, gap_end_seconds))
                        except Exception as e:
                            print(f"Warning: Could not parse gap '{gap_str}': {e}")
            
            # Apply time offsets to trial data
            combined['time_seconds'] = combined.apply(
                lambda row: (row['sequence_start'] - global_start_time).total_seconds() - session_offsets[row['date']],
                axis=1
            )
        else:
            # Fallback if no manifest info
            global_start_time = combined['sequence_start'].iloc[0]
            combined['time_seconds'] = (combined['sequence_start'] - global_start_time).dt.total_seconds()
        
        # Add grey shading for gaps between runs (subject-specific)
        for gap_start, gap_end in subj_gaps:
            ax.axvspan(gap_start, gap_end, alpha=0.2, color='gray', zorder=0)
        
        # Add vertical dashed lines at session boundaries (subject-specific, colored)
        for boundary in subj_session_starts:
            ax.axvline(x=boundary, color=colors[subj_idx], linestyle='--', linewidth=1, alpha=0.7)
        
        if split_days:
            # Reset count for each day
            combined['day_group'] = combined['date'].astype(str)
            combined['cumulative_rewards'] = combined.groupby('day_group').cumcount() + 1
            
            # Plot each day separately (no connecting lines)
            for day in combined['day_group'].unique():
                day_data = combined[combined['day_group'] == day]
                ax.plot(day_data['time_seconds'], 
                       day_data['cumulative_rewards'],
                       color=colors[subj_idx],
                       marker='o',
                       markersize=3,
                       label=f'Subject {subjid}' if day == combined['day_group'].iloc[0] else None)
        else:
            # Continuous accumulation across days
            combined['cumulative_rewards'] = range(1, len(combined) + 1)
            ax.plot(combined['time_seconds'], 
                   combined['cumulative_rewards'],
                   color=colors[subj_idx],
                   marker='o',
                   markersize=3,
                   label=f'Subject {subjid}')
    
    ax.set_xlabel('Time (seconds from session start)')
    ax.set_ylabel('Cumulative Rewards')
    ax.set_title(title if title else 'Accumulated Rewards Over Time')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return fig, ax



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
    base_path = Path(project_root) / "data" / "rawdata"
    derivatives_dir = base_path.resolve().parent / "derivatives"
    
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


def load_tracking_with_behavior(subjid, date):
    """
    Load combined tracking data and behavior results for a session.
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    date : int or str
        Session date (e.g., 20251017)
    
    Returns:
    --------
    dict containing:
        - 'tracking': pd.DataFrame with tracking data (X, Y, time)
        - 'behavior': dict from load_session_results()
        - 'tracking_labeled': pd.DataFrame with added 'in_trial' column
    """
    # Load tracking data
    base_path = Path(project_root) / "data" / "rawdata"
    derivatives_dir = base_path.resolve().parent / "derivatives"
    
    sub_str = f"sub-{str(subjid).zfill(3)}"
    subject_dirs = list(derivatives_dir.glob(f"{sub_str}_id-*"))
    if not subject_dirs:
        raise FileNotFoundError(f"No subject directory found for {sub_str}")
    subject_dir = subject_dirs[0]
    
    date_str = str(date)
    session_dirs = list(subject_dir.glob(f"ses-*_date-{date_str}"))
    if not session_dirs:
        raise FileNotFoundError(f"No session found for date {date_str}")
    session_dir = session_dirs[0]
    
    results_dir = session_dir / "saved_analysis_results"
    tracking_files = [f for f in results_dir.glob(f"*_combined_tracking_with_timestamps.csv") 
                      if not f.name.startswith('._')]
    if not tracking_files:
        raise FileNotFoundError(
            f"No combined tracking file found. Run add_timestamps_to_tracking({subjid}, {date}) first."
        )
    
    # Load tracking CSV
    try:
        tracking = pd.read_csv(tracking_files[0], encoding='utf-8')
    except UnicodeDecodeError:
        tracking = pd.read_csv(tracking_files[0], encoding='latin1')
    
    # Convert time column to datetime
    tracking['time'] = pd.to_datetime(tracking['time'])
    
    # Load behavior data
    behavior = load_session_results(subjid, date)
    
    # Label tracking frames as in_trial or not
    tracking_labeled = tracking.copy()
    tracking_labeled['in_trial'] = False
    tracking_labeled['trial_type'] = None
    
    # Get all completed trials (can customize this to use different trial types)
    trials = behavior.get('completed_sequences', pd.DataFrame())
    
    if not trials.empty:
        trials = trials.copy()
        trials['sequence_start'] = pd.to_datetime(trials['sequence_start'])
        trials['sequence_end'] = pd.to_datetime(trials['sequence_end'])
        
        # Mark frames that fall within any trial
        for idx, trial in trials.iterrows():
            mask = (tracking_labeled['time'] >= trial['sequence_start']) & \
                   (tracking_labeled['time'] <= trial['sequence_end'])
            tracking_labeled.loc[mask, 'in_trial'] = True
            tracking_labeled.loc[mask, 'trial_type'] = trial.get('trial_type', 'trial')
    
    return {
        'tracking': tracking,
        'behavior': behavior,
        'tracking_labeled': tracking_labeled
    }


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


def _load_tracking_and_behavior(subjid, date):
    """
    Load combined tracking CSV and behavior results for a session.
    Returns tracking_df (with parsed time) and behavior dict.
    """
    base_path = Path(project_root) / "data" / "rawdata"
    derivatives_dir = base_path.resolve().parent / "derivatives"

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

    tracking_files = [f for f in results_dir.glob("*_combined_tracking_with_timestamps.csv")
                      if not f.name.startswith("._")]
    if not tracking_files:
        raise FileNotFoundError(
            f"No combined tracking file found in {results_dir}. "
            f"Run add_timestamps_to_tracking({subjid}, {date}) first."
        )
    csv_path = tracking_files[0]

    try:
        tracking = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        tracking = pd.read_csv(csv_path, encoding='latin1')

    tracking['time'] = pd.to_datetime(tracking['time'], errors='coerce')

    behavior = load_session_results(subjid, date)
    return tracking, behavior





def plot_movement_with_behavior(
    subjid, date,
    mode="simple",                # "simple" | "trial_state" | "last_odor" | "time_windows" | "trial_windows"
    time_windows=None,            # list of ("HH:MM:SS","HH:MM:SS")
    trial_windows=None,           # list of (start, end). negatives allowed, e.g. (-20, -0) = last 20..last
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
    assert mode in {"simple", "trial_state", "last_odor", "time_windows", "trial_windows"}
    import numpy as np
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

    def _plot_segments_by_category(df, category_series, colors, axes=None, legend_name_func=lambda c: str(c)):
        target_ax = axes if axes is not None else ax
        cat = category_series.fillna('other')
        cat_reset = cat.reset_index(drop=True)
        seg_boundaries = (cat_reset != cat_reset.shift()).cumsum()
        used = set()
        for seg_id, seg_idx in seg_boundaries.groupby(seg_boundaries).groups.items():
            seg_idx = sorted(seg_idx)
            c = cat_reset.iloc[seg_idx[0]]
            color = colors.get(c, colors.get('other', 'gray'))
            segment_df = df.iloc[seg_idx]
            label = None if c in used else legend_name_func(c)
            target_ax.plot(segment_df['X_smooth'].values, segment_df['Y_smooth'].values,
                           color=color, linewidth=linewidth, alpha=alpha, label=label)
            used.add(c)

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

        
    elif mode == "last_odor":
        comps = behavior.get('completed_sequences', pd.DataFrame())
        if comps.empty:
            raise ValueError("No completed_sequences found; last_odor plot requires completed trials.")
        comps = comps.copy()
        comps['sequence_start'] = pd.to_datetime(comps['sequence_start'])
        comps['sequence_end'] = pd.to_datetime(comps['sequence_end'])

        # Use the `last_odor` column directly
        if 'last_odor' not in comps.columns:
            raise ValueError("The 'last_odor' column is missing in completed_sequences.")
        
        if last_odor_colors is None:
            last_odor_colors = {'A': 'red', 'B': 'blue', 'other': 'lightgray'}

        # Build a series mapping each tracking frame to its odor category
        t_time = tracking['time']
        trial_odor = pd.Series('', index=tracking.index, dtype=object)
        
        for _, tr in comps.iterrows():
            mask = (t_time >= tr['sequence_start']) & (t_time <= tr['sequence_end'])
            trial_odor.loc[mask] = str(tr['last_odor'])
        
        # Filter to only frames within trials
        in_trial_mask = trial_odor != ''
        tracking_in_trial = tracking[in_trial_mask].copy()
        trial_odor_filtered = trial_odor[in_trial_mask]
        
        trial_odor_filtered = trial_odor_filtered.str.strip() 
        unique_odors = sorted(trial_odor_filtered.unique())       
        # Plot combined view
        _plot_segments_by_category(
            tracking_in_trial, 
            trial_odor_filtered, 
            last_odor_colors, 
            axes=ax,
            legend_name_func=lambda c: f"Last odor {c}"
        )
        
        # Create individual subplot for each odor - use full tracking with proper masks
        facet_plots = []
        for odor in unique_odors:
            # Create mask directly from the filtered series
            odor_mask = trial_odor_filtered == odor
            # Map back to full tracking index
            full_mask = pd.Series(False, index=tracking.index)
            full_mask.loc[odor_mask.index[odor_mask]] = True
            color = last_odor_colors.get(odor, last_odor_colors.get('other', 'gray'))
            facet_plots.append((f"Last odor {odor}", full_mask, color))


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

    elif mode == "trial_windows":
        # Use initiated_sequences as the complete list of trials
        trials = behavior.get('initiated_sequences', pd.DataFrame())
        if trials.empty:
            raise ValueError("initiated_sequences required for trial_windows.")
        trials = trials.copy()
        trials['sequence_start'] = pd.to_datetime(trials['sequence_start'])
        trials['sequence_end'] = pd.to_datetime(trials['sequence_end'])
        # Create 1-based ordinal index within session by time
        trials = trials.sort_values('sequence_start').reset_index(drop=True)
        trials['trial_idx'] = trials.index + 1

        # Normalize input to list
        if isinstance(trial_windows, tuple):
            trial_windows = [trial_windows]

        n = len(trials)

        def resolve_range(lo, hi):
            # negatives are from the end; -0 means last trial
            def to_idx(v):
                if v < 0:
                    # -1 -> last, -20 -> last-19
                    return n + v + 1
                return v
            lo_i = to_idx(int(lo))
            hi_i = to_idx(int(hi)) if hi is not None else n
            lo_i = max(1, lo_i)
            hi_i = min(n, hi_i if hi_i != 0 else n)
            if lo_i > hi_i:
                lo_i, hi_i = hi_i, lo_i
            return lo_i, hi_i

        cmap = plt.cm.Dark2
        facet_plots = []
        t_time = tracking['time']
        for i, (lo, hi) in enumerate(trial_windows):
            lo_i, hi_i = resolve_range(lo, hi)
            sel = trials[(trials['trial_idx'] >= lo_i) & (trials['trial_idx'] <= hi_i)]
            if sel.empty:
                continue
            mask = pd.Series(False, index=tracking.index)
            for _, tr in sel.iterrows():
                mask |= ((t_time >= tr['sequence_start']) & (t_time <= tr['sequence_end']))
            color = cmap(i % 8)
            _plot_segments_by_mask(tracking, mask, color, label=f"Trials {lo}-{hi}")
            facet_plots.append((f"Trials {lo}-{hi}", mask, color))

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
        facet_axes = np.array(facet_axes).reshape(-1)  # flatten
        for i, (label, mask, color) in enumerate(facet_plots):
            ax_i = facet_axes[i]
            _plot_segments_by_mask(tracking, mask, color, label=label, axes=ax_i)
            ax_i.set_title(label)
            ax_i.set_xlabel('X Position (px)')
            ax_i.set_ylabel('Y Position (px)')
            if invert_y:
                ax_i.invert_yaxis()
            ax_i.set_aspect('equal', adjustable='box')
            ax_i.grid(True, alpha=0.3)
        # Hide unused axes if any
        for j in range(len(facet_plots), len(facet_axes)):
            facet_axes[j].axis('off')
        facet_fig.suptitle(f"Per-condition views - Subject {subjid}, {date} ({mode})")
        facet_fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()

    return fig, ax

def get_video_frame_times(root, verbose=True):
    """
    Get synchronized timestamps for all video frames across all video files.
    
    Returns:
    --------
    pd.DataFrame with columns:
        - frame: global frame index across all videos
        - local_frame: frame index within the video file
        - time: synchronized timestamp (tz-aware Europe/London)
        - video_path: path to the .avi file
        - video_file: basename of the video file
    """
    
    # Load video metadata (already time-synchronized by load_all_streams)
    data = load_all_streams(root, verbose=verbose)
    video_data = data.get('video_data', pd.DataFrame())
    
    if video_data.empty:
        if verbose:
            print("No video data found")
        return pd.DataFrame()
    
    # video_data.index is already the synchronized time
    # _frame column is the frame index within each video file
    # _path column is the path to the video file
    
    result = pd.DataFrame({
        'time': video_data.index,
        'local_frame': video_data['_frame'],
        'video_path': video_data['_path'],
    })
    
    # Add global frame index (continuous across all videos)
    result['frame'] = range(len(result))
    
    # Add video file basename for convenience
    result['video_file'] = result['video_path'].apply(lambda x: Path(x).name if pd.notna(x) else None)
    
    # Reorder columns
    result = result[['frame', 'local_frame', 'time', 'video_path', 'video_file']]
    
    if verbose:
        print(f"Found {len(result)} video frames across {result['video_file'].nunique()} video files")
        print(f"Time range: {result['time'].min()} to {result['time'].max()}")
    
    return result


def get_frame_for_time(video_frame_times, target_time):
    """
    Find the video frame closest to a given timestamp.
    
    Parameters:
    -----------
    video_frame_times : pd.DataFrame
        Output from get_video_frame_times()
    target_time : pd.Timestamp or datetime
        The time to find the frame for
    
    Returns:
    --------
    dict with frame info: frame, local_frame, time, video_path, video_file
    """
    if video_frame_times.empty:
        return None
    
    # Ensure target_time is tz-aware
    if isinstance(target_time, pd.Timestamp) and target_time.tz is None:
        target_time = target_time.tz_localize('Europe/London')
    
    # Find closest frame by time
    idx = (video_frame_times['time'] - target_time).abs().idxmin()
    
    return video_frame_times.loc[idx].to_dict()


def get_frames_for_trial(video_frame_times, trial_row, padding_sec=0.0):
    """
    Get all video frames for a trial (with optional padding).
    
    Parameters:
    -----------
    video_frame_times : pd.DataFrame
        Output from get_video_frame_times()
    trial_row : dict or pd.Series
        Trial data containing 'sequence_start' and 'sequence_end'
    padding_sec : float
        Seconds to add before/after trial times
    
    Returns:
    --------
    pd.DataFrame of frames within the trial window
    """
    start_time = trial_row.get('sequence_start')
    end_time = trial_row.get('sequence_end')
    
    if start_time is None or end_time is None:
        return pd.DataFrame()
    
    # Add padding
    if padding_sec > 0:
        start_time = start_time - pd.Timedelta(seconds=padding_sec)
        end_time = end_time + pd.Timedelta(seconds=padding_sec)
    
    # Filter frames in time window
    mask = (video_frame_times['time'] >= start_time) & (video_frame_times['time'] <= end_time)
    return video_frame_times[mask].copy()


def add_timestamps_to_tracking_simple(csv_path, root, save_output=True, output_suffix='_with_timestamps'):
    """
    Add synchronized timestamps to ezTrack location tracking CSV.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to the ezTrack output CSV file
    root : Path or experiment object
        Root directory or experiment object for load_all_streams
    save_output : bool, optional
        If True, saves the timestamped CSV to a new file (default: True)
    output_suffix : str, optional
        Suffix to add to output filename (default: '_with_timestamps')
    
    Returns:
    --------
    pd.DataFrame : Tracking data with added columns:
        - 'time': synchronized timestamp (tz-aware Europe/London)
        - 'global_frame': frame index across all videos in session
        - 'video_file': video filename
    """
    from pathlib import Path
    
    # Load tracking CSV
    tracking_df = pd.read_csv(csv_path)
    
    # Get video filename from CSV
    if 'File' in tracking_df.columns:
        video_file = tracking_df['File'].iloc[0]
    else:
        # Infer from CSV filename
        csv_name = Path(csv_path).stem
        if '_LocationOutput' in csv_name:
            video_file = csv_name.replace('_LocationOutput', '.avi')
        else:
            raise ValueError("Cannot determine video file from CSV")
    
    # Get all video frame times for this session
    frame_times = get_video_frame_times(root, verbose=False)
    
    if frame_times.empty:
        raise ValueError("No video metadata found for this session")
    
    # Filter for this specific video file
    video_frames = frame_times[frame_times['video_file'] == video_file].copy()
    
    if video_frames.empty:
        raise ValueError(f"Video file '{video_file}' not found in video metadata")
    
    # Validate frame count
    if len(tracking_df) != len(video_frames):
        print(f"Warning: Frame count mismatch!")
        print(f"  Tracking CSV: {len(tracking_df)}")
        print(f"  Video data:   {len(video_frames)}")
        print(f"  Proceeding with merge, but some frames may not have timestamps")
    
    # Merge tracking data with timestamps
    # Match on local frame index
    result = tracking_df.merge(
        video_frames[['local_frame', 'time', 'frame', 'video_file']],
        left_on='Frame',
        right_on='local_frame',
        how='left'
    )
    
    # Rename 'frame' to 'global_frame' for clarity
    if 'frame' in result.columns:
        result = result.rename(columns={'frame': 'global_frame'})
    
    # Drop redundant 'local_frame' column (same as 'Frame')
    if 'local_frame' in result.columns:
        result = result.drop(columns=['local_frame'])
    
    # Reorder columns to put time info near the front
    cols = result.columns.tolist()
    time_cols = ['Frame', 'time', 'global_frame', 'video_file']
    other_cols = [c for c in cols if c not in time_cols]
    result = result[time_cols + other_cols]
    
    # Save output if requested
    if save_output:
        output_path = Path(csv_path).parent / f"{Path(csv_path).stem}{output_suffix}.csv"
        result.to_csv(output_path, index=False)
        print(f"Saved timestamped tracking data to: {output_path}")
    
    print(f"Added timestamps to {len(result)} tracking frames")
    print(f"Time range: {result['time'].min()} to {result['time'].max()}")
    
    return result


def add_timestamps_to_tracking(subjid, date, save_output=True, output_suffix='_with_timestamps'):
    """
    Add synchronized timestamps to all ezTrack location tracking CSVs for a session.
    Automatically finds all tracking files and combines them in temporal order.
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    date : int or str
        Session date (e.g., 20251017)
    save_output : bool, optional
        If True, saves the combined timestamped CSV (default: True)
    output_suffix : str, optional
        Suffix to add to output filename (default: '_with_timestamps')
    
    Returns:
    --------
    pd.DataFrame : Combined tracking data from all videos with added columns:
        - 'time': synchronized timestamp (tz-aware Europe/London)
        - 'global_frame': frame index across all videos in session
        - 'video_file': video filename
    """
    
    # Build path to derivatives directory
    base_path = Path(project_root) / "data" / "rawdata"
    derivatives_dir = base_path.resolve().parent / "derivatives"
    
    # Find subject and session directories
    sub_str = f"sub-{str(subjid).zfill(3)}"
    subject_dirs = list(derivatives_dir.glob(f"{sub_str}_id-*"))
    if not subject_dirs:
        raise FileNotFoundError(f"No subject directory found for {sub_str}")
    subject_dir = subject_dirs[0]
    
    date_str = str(date)
    session_dirs = list(subject_dir.glob(f"ses-*_date-{date_str}"))
    if not session_dirs:
        raise FileNotFoundError(f"No session found for date {date_str}")
    session_dir = session_dirs[0]
    
    # Find all LocationOutput CSV files (exclude macOS metadata files)
    results_dir = session_dir / "saved_analysis_results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    tracking_csvs = [f for f in sorted(results_dir.glob("*_LocationOutput.csv")) 
                     if not f.name.startswith('._')]
    if not tracking_csvs:
        raise FileNotFoundError(f"No LocationOutput CSV files found in {results_dir}")
    
    print(f"Found {len(tracking_csvs)} tracking file(s)")
    
    # Step 1: Load ALL experiment folders and get video frame times
    all_frame_times = []
    exp_index = 0
    while True:
        try:
            root = load_experiment(subjid, date, index=exp_index)
            frame_times = get_video_frame_times(root, verbose=False)
            if not frame_times.empty:
                all_frame_times.append(frame_times)
            exp_index += 1
        except (FileNotFoundError, IndexError):
            break
    
    if not all_frame_times:
        raise ValueError("No video metadata found for this session")
    
    # Step 2: Combine and sort all video frame times
    combined_frame_times = pd.concat(all_frame_times, ignore_index=True)
    combined_frame_times = combined_frame_times.sort_values('time').reset_index(drop=True)
    combined_frame_times['frame'] = range(len(combined_frame_times))
    
    print(f"Loaded {len(combined_frame_times):,} frames from {combined_frame_times['video_file'].nunique()} video file(s)")
    
    # Step 3: Add timestamps to each tracking CSV
    all_tracking = []
    for csv_path in tracking_csvs:
        # Load tracking CSV
        try:
            tracking_df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            tracking_df = pd.read_csv(csv_path, encoding='latin1')
        
        # Get video filename
        video_file = tracking_df['File'].iloc[0] if 'File' in tracking_df.columns else \
                     csv_path.stem.replace('_LocationOutput', '.avi')
        
        # Match to video metadata
        video_frames = combined_frame_times[combined_frame_times['video_file'] == video_file].copy()
        if video_frames.empty:
            print(f"Warning: {csv_path.name} - video '{video_file}' not found, skipping")
            continue
        
        # Merge tracking with timestamps
        result = tracking_df.merge(
            video_frames[['local_frame', 'time', 'frame', 'video_file']],
            left_on='Frame',
            right_on='local_frame',
            how='left'
        ).drop(columns=['local_frame']).rename(columns={'frame': 'global_frame'})
        
        all_tracking.append(result)
    
    if not all_tracking:
        raise ValueError("No tracking data could be matched to video metadata")
    
    # Step 4: Combine and sort all tracking data by time
    combined = pd.concat(all_tracking, ignore_index=True)
    combined = combined.sort_values('time').reset_index(drop=True)
    
    # Reorder columns
    time_cols = ['Frame', 'time', 'global_frame', 'video_file']
    other_cols = [c for c in combined.columns if c not in time_cols]
    combined = combined[time_cols + other_cols]
    
    # Save output
    if save_output:
        output_path = results_dir / f"sub-{str(subjid).zfill(3)}_ses-{date_str}_combined_tracking{output_suffix}.csv"
        combined.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
    
    duration = (combined['time'].max() - combined['time'].min()).total_seconds()
    print(f"Combined: {len(combined):,} frames, {duration/60:.1f} minutes")
    
    return combined