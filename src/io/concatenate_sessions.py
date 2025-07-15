import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import harp
import re
from collections import defaultdict
import matplotlib.pyplot as plt

from src import utils
from src.io import loaders

# NOTE: Under development
# Filter out specific warnings
warnings.filterwarnings(
    "ignore", 
    message="Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated",
    category=FutureWarning
)

def remove_uncompleted_trials(session_folder):
    """
    Remove uncompleted trials, where odour sampling started but no decision was made at the end of the session
    
    Parameters:
    -----------
    session_folder : str or Path
        Path to the session folder (e.g., ses-YYY_date-YYYYMMDD/)
    """

    session_folder = Path(session_folder)
    behav_folder = session_folder / "behav"
    
    if not behav_folder.exists():
        print(f"No behavior folder found at {behav_folder}")
        return None
    
    # Collect all session directories
    session_dirs = [d for d in behav_folder.iterdir() if d.is_dir()]
    print(f"Found {len(session_dirs)} behavioral sessions in {behav_folder}")
    
    # Process each session
    for session_dir in sorted(session_dirs):
        print(f"\nProcessing session: {session_dir.name}")
        
        # Only process if session settings exist
        if not (session_dir / "SessionSettings").exists():
            print(f"No SessionSettings found in {session_dir.name}, skipping")
            continue

        # Load session 
        session_data = loaders.load_session_data(session_dir)
        
        experiment_events = session_data['processed_events']

        all_experiment_events_id = np.zeros((len(experiment_events['end_initiation']) + len(experiment_events['initiation_sequence']) + len(experiment_events['await_reward']) + len(experiment_events['reset'])))
        all_experiment_events_time = np.zeros((len(experiment_events['end_initiation']) + len(experiment_events['initiation_sequence']) + len(experiment_events['await_reward']) + len(experiment_events['reset'])))

        # Combine the lists into a dictionary for cleaner looping
        event_dict = {
            'EndInitiation': experiment_events['end_initiation'],
            'InitiationSequence': experiment_events['initiation_sequence'],
            'AwaitReward': experiment_events['await_reward'],
            'Reset': experiment_events['reset']
        }

        # for i, (event_name, df) in enumerate(event_dict.items()):
        #     event_counter = 0
        #     for j in range(len(df['Time'])):
        #         # print(len(d))
        #         all_experiment_events_id[idx] = i
        #         all_experiment_events_time[idx] = df['Time'][j]
                 
        # print(all_experiment_events_id)
        # print(all_experiment_events_time) 
        ##### Plot experiment events #####
        # Combine the lists into a dictionary for cleaner looping
    #     event_dict = {
    #         'EndInitiation': experiment_events['end_initiation'],
    #         'InitiationSequence': experiment_events['initiation_sequence'],
    #         'AwaitReward': experiment_events['await_reward'],
    #         'Reset': experiment_events['reset']
    #     }

    #     # Assign a color to each event type
    #     colors = {
    #         'EndInitiation': 'red',
    #         'InitiationSequence': 'blue',
    #         'AwaitReward': 'green',
    #         'Reset': 'purple'
    #     }

    #     # Create subplots
    #     num_events = len(event_dict)
    #     fig, ax = plt.subplots(figsize=(12, 6), sharex=True)
    #     # fig, axes = plt.subplots(num_events, 1, figsize=(10, 3 * num_events), sharex=True)

    #     # if num_events == 1:
    #     #     axes = [axes]  # ensure it's iterable if only one subplot

    #     # Loop through each event type and plot its DataFrames
    #     # for ax, (event_name, df) in zip(axes, event_dict.items()):
    #     for i, (event_name, df) in enumerate(event_dict.items()):
    #         if not df.empty and 'Time' in df.columns:
    #             ax.plot(df['Time'], [1]*len(df), 'o', label=event_name, color=colors[event_name], alpha=0.5)
    #         ax.set_ylabel(event_name)
    #         ax.legend(loc='upper right')
    #         ax.grid(True)

    #     # Set common x-axis label
    #     ax.set_xlabel("Time (s)")
    #     fig.suptitle("Event Timings by Type", fontsize=16)
    #     plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()

        
        
# def concatenate_sessions(session_folder):

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("/Volumes/harris/hypnose/rawdata/sub-026_id-077/ses-71_date-20250702")

    parser = argparse.ArgumentParser(description="Analyze all behavioral sessions in a folder")
    parser.add_argument("session_folder", help="Path to the session folder (e.g., sub-XXX/ses-YYY_date-YYYYMMDD)")
    
    args = parser.parse_args()
    
    result = remove_uncompleted_trials(
        args.session_folder
    )