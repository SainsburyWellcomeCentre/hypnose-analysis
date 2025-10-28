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



def plot_movement_trace(csv_path, smooth_window=10, linewidth=1, alpha=0.5, figsize=(10, 10), 
                       xlim=None, ylim=None, invert_y=True, title=None, save_path=None):
    """
    Plot animal movement trace from ezTrack location tracking CSV.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to the ezTrack output CSV file
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
    # Load the tracking data
    df = pd.read_csv(csv_path)
    
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
    ax.set_title(title if title else 'Animal Movement Trace')
    
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
    from pathlib import Path
    
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