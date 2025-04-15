import argparse
import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path
import harp
import src.utils as utils
from src.analysis import detect_stage  # Changed from relative to absolute import
import numpy as np

def process_subject_sessions(subject_folder):
    """
    Adapted from plot-rewards-single-subject.py
    Processes all sessions for a subject and returns a DataFrame with rewards data.
    """
    base_dir = Path(subject_folder)
    session_dirs = [d for d in base_dir.glob('ses-*_date-*/behav/*') if d.is_dir()]
    sessions_data = {}
    session_groups = {}
    for session_dir in session_dirs:
        session_match = re.search(r'ses-(\d+)_date-(\d+)', str(session_dir))
        if session_match:
            session_id = session_match.group(1)
        else:
            session_id = session_dir.parent.name
        if session_id not in session_groups:
            session_groups[session_id] = []
        session_groups[session_id].append(session_dir)

    behavior_reader = harp.reader.create_reader('device_schemas/behavior.yml', epoch=harp.io.REFERENCE_EPOCH)
    for session_id, dirs in sorted(session_groups.items(), key=lambda x: int(x[0]) if x[0].isdigit() else float('inf')):
        try:
            r1_poke_count_total, r2_poke_count_total = 0, 0
            r1_reward_count_total, r2_reward_count_total = 0, 0
            total_session_duration_sec = 0
            earliest_timestamp, latest_timestamp = None, None
            first_dir = dirs[0]
            date_match = re.search(r'date-(\d{8})', str(first_dir))
            if date_match:
                date_str = date_match.group(1)
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            else:
                formatted_date = str(first_dir.stem)
            try:
                stage = detect_stage(first_dir)
            except:
                stage = "Unknown"
            for session_dir in dirs:
                try:
                    digital_input_data = utils.load(behavior_reader.DigitalInputState, session_dir/"Behavior")
                    if digital_input_data is not None:
                        r1_poke_count_total += digital_input_data.get('DIPort1', pd.Series()).sum()
                        r2_poke_count_total += digital_input_data.get('DIPort2', pd.Series()).sum()
                except:
                    pass
                try:
                    heartbeat = utils.load(behavior_reader.TimestampSeconds, session_dir/"Behavior")
                    if not heartbeat.empty:
                        start_time = heartbeat['TimestampSeconds'].iloc[0]
                        end_time = heartbeat['TimestampSeconds'].iloc[-1]
                        dir_duration = end_time - start_time
                        total_session_duration_sec += dir_duration
                        if earliest_timestamp is None or start_time < earliest_timestamp:
                            earliest_timestamp = start_time
                        if latest_timestamp is None or end_time > latest_timestamp:
                            latest_timestamp = end_time
                except:
                    pass
                try:
                    pulse_supply_1 = utils.load(behavior_reader.PulseSupplyPort1, session_dir/"Behavior")
                    r1_reward_count_total += len(pulse_supply_1)
                except:
                    pass
                try:
                    pulse_supply_2 = utils.load(behavior_reader.PulseSupplyPort2, session_dir/"Behavior")
                    r2_reward_count_total += len(pulse_supply_2)
                except:
                    pass
            session_duration_sec = total_session_duration_sec if total_session_duration_sec else None
            duration_str = "Unknown"
            if session_duration_sec:
                hours = int(session_duration_sec // 3600)
                minutes = int((session_duration_sec % 3600) // 60)
                seconds = int(session_duration_sec % 60)
                duration_str = f"{hours}h {minutes}m {seconds}s"
            sessions_data[session_id] = {
                'date': formatted_date,
                'duration': duration_str,
                'r1_rewards': r1_reward_count_total,
                'r2_rewards': r2_reward_count_total,
                'stage': stage
            }
        except:
            pass
    df = pd.DataFrame.from_dict(sessions_data, orient='index')
    df.index.name = 'session_id'
    return df

def plot_subject_on_ax(sessions_df, ax, subject_name):
    """
    Plots total rewards for one subject on a given Axes object.
    """
    if sessions_df.empty:
        return
    required_columns = ['r1_rewards', 'r2_rewards', 'date']
    if not all(col in sessions_df.columns for col in required_columns):
        return
    sessions_df['total_rewards'] = sessions_df['r1_rewards'] + sessions_df['r2_rewards']
    sessions_df['date'] = pd.to_datetime(sessions_df['date'])
    sessions_df['calendar_date'] = sessions_df['date'].dt.date
    full_date_range = pd.date_range(start=sessions_df['calendar_date'].min(),
                                    end=sessions_df['calendar_date'].max(), freq='D')
    full_sessions_df = pd.DataFrame({'calendar_date': full_date_range.date})
    full_sessions_df = full_sessions_df.merge(sessions_df, on='calendar_date', how='left')
    
    stage_colors = {
        '1': 'lightblue',
        '2': 'lightgreen',
        '3': 'salmon',
        '4': 'purple',
        '5': 'orange',
        '6': 'brown',
        '7': 'pink',
        '8': 'gray',
        'Unknown': 'black'
    }
    has_data = ~full_sessions_df['total_rewards'].isna()
    for stage, color in stage_colors.items():
        mask = (full_sessions_df['stage'] == stage) & has_data
        if mask.any():
            scatter_obj = ax.scatter(
                full_sessions_df.loc[mask, 'calendar_date'],
                full_sessions_df.loc[mask, 'total_rewards'],
                color=color, alpha=0.7, edgecolor='black', s=40
            )
            scatter_obj.set_label(f"Stage {stage}")
    ax.set_ylabel('Total Rewards')
    ax.set_title(subject_name if subject_name else '')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    dates = full_sessions_df['calendar_date']
    date_labels = []
    for date in dates:
        dt = pd.to_datetime(date)
        month_day = dt.strftime('%m/%d')
        weekday = dt.strftime('%a')
        date_labels.append(f"{month_day} - {weekday}")
    ax.set_xticks([pd.Timestamp(d) for d in dates])
    ax.set_xticklabels(date_labels, rotation=45, ha='right')

def main():
    parser = argparse.ArgumentParser(description="Plot rewards for multiple subjects in subplots.")
    parser.add_argument("subjects", nargs="+", help="Paths to each subject's data folder")
    parser.add_argument("--output", "-o", help="Optional path to save the figure")
    parser.add_argument("--grid-rows", type=int, default=3, help="Number of rows in subplot grid")
    parser.add_argument("--grid-cols", type=int, default=3, help="Number of columns in subplot grid")
    args = parser.parse_args()

    fig, axes = plt.subplots(args.grid_rows, args.grid_cols, figsize=(15, 15), sharex=True)
    axes = axes.flatten()

    for ax, subject_folder in zip(axes, args.subjects):
        subject_path = Path(subject_folder)
        subject_name_match = re.match(r'(sub-\d+).*', subject_path.name)
        if subject_name_match:
            subject_name = subject_name_match.group(1)
        else:
            subject_name = subject_path.name
        df = process_subject_sessions(subject_folder)
        plot_subject_on_ax(df, ax, subject_name)

    # Combine handles and labels from all subplots
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Remove duplicate labels
    unique_entries = {}
    for h, l in zip(handles, labels):
        if l not in unique_entries:
            unique_entries[l] = h

    # Sort labels so numeric stages are ascending, with Unknown last
    def sort_key(label):
        parts = label.split()
        if len(parts) == 2 and parts[0] == "Stage" and parts[1].isdigit():
            return int(parts[1])
        return float('inf')  # for 'Stage Unknown'
    sorted_labels = sorted(unique_entries.keys(), key=sort_key)
    sorted_handles = [unique_entries[l] for l in sorted_labels]

    fig.legend(sorted_handles, sorted_labels, loc='upper right')
    #fig.subplots_adjust(right=0.8)
    fig.tight_layout(rect=[0, 0, 0.96, 1])

    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()