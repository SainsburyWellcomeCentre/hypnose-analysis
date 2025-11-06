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


def _load_tracking_and_behavior(subjid, date, tracking_source='auto'):
    """
    Load combined tracking CSV and behavior results for a session.
    Supports both ezTrack (*_combined_tracking_with_timestamps.csv) 
    and SLEAP (*_combined_sleap_tracking_timestamps.csv) formats.
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    date : int or str
        Session date
    tracking_source : str, optional
        'auto' (default): prefer SLEAP if available, else ezTrack
        'sleap': only load SLEAP
        'eztrack': only load ezTrack
    
    Returns:
    --------
    tracking_df, behavior_dict
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

    # Find tracking files
    sleap_files = [f for f in results_dir.glob("*_combined_sleap_tracking_timestamps.csv")
                   if not f.name.startswith("._")]
    eztrack_files = [f for f in results_dir.glob("*_combined_tracking_with_timestamps.csv")
                     if not f.name.startswith("._")]
    
    csv_path = None
    source_used = None
    
    if tracking_source == 'auto':
        # Prefer SLEAP if available
        if sleap_files:
            csv_path = sleap_files[0]
            source_used = 'sleap'
        elif eztrack_files:
            csv_path = eztrack_files[0]
            source_used = 'eztrack'
    elif tracking_source == 'sleap':
        if sleap_files:
            csv_path = sleap_files[0]
            source_used = 'sleap'
    elif tracking_source == 'eztrack':
        if eztrack_files:
            csv_path = eztrack_files[0]
            source_used = 'eztrack'
    
    if csv_path is None:
        raise FileNotFoundError(
            f"No tracking file found for {tracking_source}. "
            f"Available: {len(sleap_files)} SLEAP, {len(eztrack_files)} ezTrack"
        )

    try:
        tracking = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        tracking = pd.read_csv(csv_path, encoding='latin1')

    tracking['time'] = pd.to_datetime(tracking['time'], errors='coerce')

    # For SLEAP data: use 'centroid_x' and 'centroid_y' if available, else 'X'/'Y'
    if source_used == 'sleap':
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
    
    print(f"Loaded {source_used.upper()} tracking: {len(tracking)} frames from {csv_path.name}")
    
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
        
        # Group by consecutive same categories
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
        _plot_segments_by_category(
            tracking_in_trial, 
            trial_odor_filtered, 
            last_odor_colors, 
            axes=ax,
            legend_name_func=lambda c: f"Last odor {c}"
        )
        
        # Create facet plots for individual odors
        facet_plots = []
        for odor in unique_odors:
            odor_mask = trial_odor_filtered == odor
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
            # Use standard Python slicing - negative indices work naturally with iloc
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


def add_timestamps_to_sleap_tracking(subjid, date, save_output=True):
    """
    Add synchronized timestamps to all SLEAP tracking CSVs for a session.
    Matches sleap_tracking_videox.csv files to video files by index (1-indexed filename to 0-indexed video order).
    Handles gaps in frame coverage and partial tracking.
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    date : int or str
        Session date (e.g., 20251027)
    save_output : bool, optional
        If True, saves the combined timestamped CSV (default: True)
    
    Returns:
    --------
    pd.DataFrame : Combined SLEAP tracking data with added columns:
        - 'time': synchronized timestamp (tz-aware Europe/London)
        - 'global_frame': frame index across all videos in session
        - 'video_file': video filename
    """
    import re
    
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
    
    # Find all sleap_tracking_videox.csv files
    results_dir = session_dir / "saved_analysis_results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    tracking_csvs = sorted([f for f in results_dir.glob("sleap_tracking_video*.csv") 
                           if not f.name.startswith('._')])
    if not tracking_csvs:
        raise FileNotFoundError(f"No sleap_tracking_videox.csv files found in {results_dir}")
    
    print(f"Found {len(tracking_csvs)} SLEAP tracking file(s)")
    
    # Step 1: Find behavior directory and experiment folders
    behav_dirs = list(base_path.glob(f"{sub_str}_id-*/{Path(session_dirs[0]).name}/behav"))
    if not behav_dirs:
        raise FileNotFoundError(f"Behavior directory not found")
    behav_dir = behav_dirs[0]
    
    # Find all experiment folders (numbered time-stamped directories)
    exp_folders = sorted([d for d in behav_dir.iterdir() 
                         if d.is_dir() and re.match(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', d.name)])
    
    if not exp_folders:
        raise FileNotFoundError(f"No experiment folders found in {behav_dir}")
    
    print(f"Found {len(exp_folders)} experiment folder(s)")
    
    # Step 2: Load video frame times from each experiment
    all_frame_times = []
    for exp_idx, exp_folder in enumerate(exp_folders):
        try:
            frame_times = get_video_frame_times(exp_folder, verbose=False)
            if not frame_times.empty:
                all_frame_times.append(frame_times)
                print(f"  Loaded {len(frame_times)} frames from experiment {exp_idx}")
        except Exception as e:
            print(f"  Warning: Could not load experiment {exp_idx}: {e}")
            continue
    
    if not all_frame_times:
        raise ValueError("No video metadata found for this session")
    
    # Step 3: Combine and sort all video frame times
    combined_frame_times = pd.concat(all_frame_times, ignore_index=True)
    combined_frame_times = combined_frame_times.sort_values('time').reset_index(drop=True)
    combined_frame_times['global_frame'] = range(len(combined_frame_times))
    
    print(f"Total: {len(combined_frame_times):,} frames from {combined_frame_times['video_file'].nunique()} video file(s)")
    
    # Get unique video files in order
    video_files_ordered = combined_frame_times['video_file'].unique()
    print(f"\nVideo files in order:")
    for i, vf in enumerate(video_files_ordered):
        print(f"  {i+1}. {vf}")
    
    # Step 4: Match SLEAP files to videos by index
    # Extract video number from filename (sleap_tracking_video2.csv -> 2)
    sleap_video_mapping = {}  # maps video_file to SLEAP CSV path
    
    for csv_path in tracking_csvs:
        match = re.search(r'sleap_tracking_video(\d+)', csv_path.name)
        if not match:
            print(f"Warning: Could not extract video number from {csv_path.name}, skipping")
            continue
        
        video_num = int(match.group(1))  # 1-indexed from filename
        video_idx = video_num - 1  # Convert to 0-indexed
        
        if video_idx >= len(video_files_ordered):
            print(f"Warning: {csv_path.name} references video {video_num}, but only {len(video_files_ordered)} video(s) exist")
            continue
        
        video_file = video_files_ordered[video_idx]
        sleap_video_mapping[video_file] = csv_path
        print(f"Matched {csv_path.name} (video {video_num}) to {video_file}")
    
    if not sleap_video_mapping:
        raise ValueError("No SLEAP files could be matched to videos")
    
    # Step 5: Add timestamps to each SLEAP tracking CSV
    all_tracking = []
    
    for video_file, csv_path in sleap_video_mapping.items():
        try:
            tracking_df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            tracking_df = pd.read_csv(csv_path, encoding='latin1')
        
        # Get frame times for this specific video
        video_frames = combined_frame_times[combined_frame_times['video_file'] == video_file].copy()
        if video_frames.empty:
            print(f"Warning: Video '{video_file}' not found in frame times, skipping {csv_path.name}")
            continue
        
        print(f"\nProcessing {csv_path.name}:")
        print(f"  SLEAP frames: {tracking_df['frame'].min():.0f} - {tracking_df['frame'].max():.0f} ({len(tracking_df)} rows)")
        print(f"  Video local frames: {video_frames['local_frame'].min():.0f} - {video_frames['local_frame'].max():.0f}")
        
        # Merge to add timestamps - match on 'frame' column (local frame index within video)
        # Use left join to keep all SLEAP frames
        result = tracking_df.merge(
            video_frames[['local_frame', 'time', 'global_frame']],
            left_on='frame',
            right_on='local_frame',
            how='left'
        ).drop(columns=['local_frame'], errors='ignore')
        
        result['video_file'] = video_file
        
        # Check how many frames got timestamps
        matched_frames = result['time'].notna().sum()
        print(f"  Matched {matched_frames}/{len(result)} frames to timestamps")
        
        all_tracking.append(result)
    
    if not all_tracking:
        raise ValueError("No SLEAP tracking data could be matched to video metadata")
    
    # Step 6: Combine and sort all tracking data by time
    combined = pd.concat(all_tracking, ignore_index=True)
    combined = combined.sort_values('time', na_position='last').reset_index(drop=True)
    
    # Reorder columns - put frame/time/video info first
    priority_cols = ['frame', 'time', 'global_frame', 'video_file', 'instance']
    other_cols = [c for c in combined.columns if c not in priority_cols]
    priority_cols = [c for c in priority_cols if c in combined.columns]
    combined = combined[priority_cols + other_cols]
    
    # Save output
    if save_output:
        output_filename = f"sub-{str(subjid).zfill(3)}_ses-{date_str}_combined_sleap_tracking_timestamps.csv"
        output_path = results_dir / output_filename
        combined.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")
    
    if 'time' in combined.columns and combined['time'].notna().any():
        duration = (combined['time'].max() - combined['time'].min()).total_seconds()
        print(f"Combined: {len(combined):,} frames, {duration/60:.1f} minutes")
        print(f"  Frames with timestamps: {combined['time'].notna().sum()}")
        print(f"  Frames without timestamps (gaps): {combined['time'].isna().sum()}")
    else:
        print(f"Combined: {len(combined):,} frames (no timestamps matched)")
    
    return combined




def annotate_videos_with_sleap_and_trials(subjid, date, base_dir=None, output_suffix="sleap_visualization", 
                                          centroid_radius=8, centroid_color='red', show_fps_counter=True,
                                          show_timestamp=True, show_trial_state=True):
    """
    Annotate behavior videos with SLEAP centroid tracking and trial state overlays.
    
    Creates MP4 videos with:
    - Red dot at animal centroid position (from SLEAP tracking)
    - Frame counter (bottom right)
    - Synchronized timestamp from video metadata (bottom left)
    - Trial state indicator: "WITHIN TRIAL" (white) or "OUTSIDE TRIAL" (blue)
    
    Parameters:
    -----------
    subjid : int
        Subject ID
    date : int or str
        Session date (e.g., 20251029)
    base_dir : str or Path, optional
        Base data directory (default: /Volumes/harris/hypnose)
    output_suffix : str, optional
        Suffix for output files (default: "sleap_visualization")
    centroid_radius : int, optional
        Radius of centroid marker in pixels (default: 8)
    centroid_color : str, optional
        Color of centroid marker (default: 'red')
    show_fps_counter : bool, optional
        Show frame counter (default: True)
    show_timestamp : bool, optional
        Show synchronized timestamp (default: True)
    show_trial_state : bool, optional
        Show trial state overlay (default: True)
    
    Returns:
    --------
    list : Paths to generated output MP4 files
    """
    from pathlib import Path
    import pandas as pd
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    from tqdm import tqdm
    
    # Default base directory
    if base_dir is None:
        base_dir = Path("/Volumes/harris/hypnose")
    else:
        base_dir = Path(base_dir)
    
    # Load behavior data for trial state
    behavior = load_session_results(subjid, date)
    
    # Find directories
    deriv_dir = base_dir / "derivatives"
    subj_pattern = f"sub-{subjid:03d}_*"
    subj_dirs = list(deriv_dir.glob(subj_pattern))
    
    if not subj_dirs:
        raise FileNotFoundError(f"No subject directory found for pattern {subj_pattern}")
    
    subj_dir = subj_dirs[0]
    session_pattern = f"ses-*_date-{date}"
    session_dirs = list(subj_dir.glob(session_pattern))
    
    if not session_dirs:
        raise FileNotFoundError(f"No session found for date {date}")
    
    session_dir = session_dirs[0]
    results_dir = session_dir / "saved_analysis_results"
    
    # Find video files
    rawdata_dir = base_dir / "rawdata"
    rawdata_subj_dirs = list(rawdata_dir.glob(f"sub-{subjid:03d}_*"))
    
    if not rawdata_subj_dirs:
        raise FileNotFoundError(f"No rawdata subject directory found")
    
    rawdata_subj_dir = rawdata_subj_dirs[0]
    rawdata_session_dir = next(rawdata_subj_dir.glob(f"ses-*_date-{date}"), None)
    
    if not rawdata_session_dir:
        raise FileNotFoundError(f"No rawdata session directory found for date {date}")
    
    video_files = sorted(rawdata_session_dir.glob("behav/*/VideoData/*.avi"))
    
    if not video_files:
        raise FileNotFoundError(f"No video files found in {rawdata_session_dir}")
    
    # Load combined timestamps file
    combined_ts_file = list(results_dir.glob("*_combined_sleap_tracking_timestamps.csv"))
    if not combined_ts_file:
        raise FileNotFoundError(f"No combined timestamps file found in {results_dir}")
    
    combined_df = pd.read_csv(combined_ts_file[0])
    print(f"Loaded combined timestamps: {len(combined_df)} frames")
    
    # Get trial times
    trials = behavior.get('initiated_sequences', pd.DataFrame())
    if not trials.empty:
        trials = trials.copy()
        trials['sequence_start'] = pd.to_datetime(trials['sequence_start'])
        trials['sequence_end'] = pd.to_datetime(trials['sequence_end'])
        print(f"Loaded {len(trials)} trials")
    else:
        print("⚠️ No trials found - will show OUTSIDE TRIAL for all frames")
    
    # Load fonts
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    output_paths = []
    
    # Process each video
    for video_idx, video_path in enumerate(video_files, 1):
        print(f"\nProcessing video {video_idx}/{len(video_files)}: {video_path.name}")
        
        # Filter combined_df for this video
        video_name = video_path.stem  # e.g., VideoData_1904-01-02T03-00-00
        df_video = combined_df[combined_df['video_file'].str.contains(video_name, na=False)].copy()
        
        if df_video.empty:
            print(f"  ⚠️ No timestamps found for video {video_name}, skipping")
            continue
        
        # Convert time column to datetime
        df_video['time'] = pd.to_datetime(df_video['time'])
        df_video = df_video.reset_index(drop=True)
        
        print(f"  Found {len(df_video)} frames with timestamps")
        
        # Load video with OpenCV for frame-by-frame processing
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"  Video: {width}x{height} @ {fps} fps, {total_frames} frames")
        
        # Output video writer
        output_path = results_dir / f"{output_suffix}_video{video_idx}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        output_paths.append(output_path)
        
        print(f"  Saving to: {output_path.name}")
        
        # Use tqdm for progress bar
        with tqdm(total=total_frames, desc="  Encoding", unit="frames") as pbar:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB for PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                draw = ImageDraw.Draw(frame_pil)
                
                # Plot centroid if available
                if show_fps_counter and frame_idx < len(df_video):
                    row = df_video.iloc[frame_idx]
                    if not pd.isna(row['centroid_x']) and not pd.isna(row['centroid_y']):
                        cx = int(row['centroid_x'])
                        cy = int(row['centroid_y'])
                        # Draw simple red dot
                        draw.ellipse([cx - centroid_radius, cy - centroid_radius, 
                                     cx + centroid_radius, cy + centroid_radius], 
                                    fill=centroid_color, outline=centroid_color)
                
                # Add frame counter in bottom right
                if show_fps_counter:
                    counter_text = f"{frame_idx}/{total_frames}"
                    counter_bbox = draw.textbbox((0, 0), counter_text, font=font_small)
                    counter_width = counter_bbox[2] - counter_bbox[0]
                    counter_x = width - counter_width - 20
                    counter_y = height - 50
                    draw.rectangle([counter_x - 10, counter_y - 10, counter_x + counter_width + 10, counter_y + 40],
                                  fill='black', outline='white', width=2)
                    draw.text((counter_x, counter_y), counter_text, fill='white', font=font_small)
                
                # Add timestamp in bottom left from CSV
                if show_timestamp and frame_idx < len(df_video):
                    row = df_video.iloc[frame_idx]
                    frame_time = row['time']
                    timestamp_text = frame_time.strftime("%H:%M:%S")
                    
                    timestamp_bbox = draw.textbbox((0, 0), timestamp_text, font=font_small)
                    timestamp_width = timestamp_bbox[2] - timestamp_bbox[0]
                    timestamp_x = 20
                    timestamp_y = height - 50
                    draw.rectangle([timestamp_x - 10, timestamp_y - 10, timestamp_x + timestamp_width + 10, timestamp_y + 40],
                                  fill='black', outline='white', width=2)
                    draw.text((timestamp_x, timestamp_y), timestamp_text, fill='white', font=font_small)
                
                # Determine trial state from timestamps
                if show_trial_state:
                    trial_state = "OUTSIDE TRIAL"
                    trial_color = (100, 100, 255)  # Blue RGB
                    
                    if not trials.empty and frame_idx < len(df_video):
                        row = df_video.iloc[frame_idx]
                        frame_time = row['time']
                        
                        for _, trial in trials.iterrows():
                            if trial['sequence_start'] <= frame_time <= trial['sequence_end']:
                                trial_state = "WITHIN TRIAL"
                                trial_color = (255, 255, 255)  # White RGB
                                break
                    
                    # Add trial state in top center
                    state_bbox = draw.textbbox((0, 0), trial_state, font=font_large)
                    state_width = state_bbox[2] - state_bbox[0]
                    state_x = (width - state_width) // 2
                    state_y = 20
                    draw.rectangle([state_x - 15, state_y - 10, state_x + state_width + 15, state_y + 50],
                                  fill='black', outline=trial_color, width=3)
                    draw.text((state_x, state_y), trial_state, fill=trial_color, font=font_large)
                
                # Convert back to BGR for OpenCV
                frame_annotated = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                out.write(frame_annotated)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        out.release()
        
        print(f"  ✓ Completed!")
    
    print(f"\n✅ All videos processed and saved!")
    return output_paths