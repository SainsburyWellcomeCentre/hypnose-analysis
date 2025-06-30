import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import matplotlib.dates as mdates
from src.analysis import detect_stage, get_false_alarm_bias
import src.utils as utils
from collections import defaultdict

def plot_false_alarm_time_bias(results_df, num_intervals=5, stage=None, plot_file=None, subject_id=None):
    """
    Create a scatterplot of false alarm time bias (# odours since rewarded odour) across sessions.
    
    Args:
        results_df (DataFrame): Contains session_id, session_date, and false alarm bias data
        plot_file (str): Path to save the plot, if provided
        subject_id (str): Subject ID to use in the plot title
    """
    # Convert date strings to datetime objects for better x-axis formatting
    results_df['date'] = pd.to_datetime(results_df['session_date'], format='%Y%m%d')
    results_df = results_df.sort_values('date')
    
    # Horizontal layout: one column per interval
    fig, ax = plt.subplots(1, num_intervals, sharey=True, figsize=(16, 8))  
    ax = ax.ravel()

    colors = ['blue', 'deepskyblue', 'green', 'orange', 'red']
    nonR_odours = ['C', 'D', 'E', 'F', 'G']

    # Same interval setup
    sample_row = results_df['total_odour_interval_false_alarm'].dropna().iloc[0]
    all_intervals = sorted({intv for odour_data in sample_row.values() for intv in odour_data})[1:num_intervals+1]
    interval_to_axis = {interval: idx for idx, interval in enumerate(all_intervals)}

    # Loop over rows
    for idx, row in results_df.iterrows():
        if not isinstance(row['total_odour_interval_false_alarm'], dict):
            continue

        for o, odour in enumerate(nonR_odours):
            for interval in all_intervals:
                val = row['total_odour_interval_false_alarm'][odour].get(interval, None)
                if val is not None:
                    i = interval_to_axis[interval]
                    ax[i].scatter(val, row['date'],
                                label=f'{odour}' if idx == 0 else "",
                                color=colors[o], marker='o', s=60, alpha=0.8)
                    ax[i].plot([row['total_odour_interval_false_alarm'][odour].get(interval, None) for _, row in results_df.iterrows()], results_df['date'],  
                            color=colors[o], linestyle='--', alpha=0.3)
                    ax[i].plot([row['total_interval_false_alarm'].get(interval, None) for _, row in results_df.iterrows()], results_df['date'],
                            color='black', linestyle='-')
                    
    # Format plots
    for i, axis in enumerate(ax):
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.set_xlim(0, 105)
        axis.set_xlabel(f'FA (%)\nInterval {all_intervals[i]}')
        axis.axvline(x=40, color='gray', linestyle='--', alpha=0.7)
        # axis.invert_yaxis()
        if i == 0:
            axis.set_ylabel('Session Date')
        else:
            axis.set_yticklabels([])  # Only show y-labels on first plot

    # Format y-axis (session date)
    axis.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axis.tick_params(axis='y', rotation=0) 

    # Format legend
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    # Add session annotations (optional)
    for _, row in results_df.iterrows():
        if isinstance(row['total_interval_false_alarm'], dict):
            for interval, x in row['total_interval_false_alarm'].items():
                if interval in interval_to_axis and pd.notna(x) and x > 0:
                    i = interval_to_axis[interval]
                    ax[i].annotate(
                        f"Ses-{row['session_id']}",
                        (x, row['date']),
                        textcoords="offset points",
                        xytext=(5, 0),
                        ha='left',
                        va='center')
    plt.suptitle(f'Decision False Alarm Time Bias (# odours since rewarded odour) Across Sessions - sub-{subject_id}' if subject_id else 'False Alarm Time Bias (# odours since rewarded odour) Across Sessions')
    
    # Save if requested
    if plot_file:
        filename = f"sub-{subject_id}_FalseAlarmTimeBias" + (f"_stage{stage}" if stage is not None else "") + ".png"
        plot_file = os.path.join(plot_file, filename)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_file}")
    
    plt.tight_layout()
    plt.show()


def plot_false_alarm_olfactometer_bias(results_df, stage=None, plot_file=None, subject_id=None):
    """
    Create a scatterplot of false alarm olfactometer bias (preceding odour) across sessions.
    
    Args:
        results_df (DataFrame): Contains session_id, session_date, and false alarm bias data
        plot_file (str): Path to save the plot, if provided
        subject_id (str): Subject ID to use in the plot title
    """
    # Convert date strings to datetime objects for better x-axis formatting
    results_df['date'] = pd.to_datetime(results_df['session_date'], format='%Y%m%d')
    results_df = results_df.sort_values('date')
    
    # Create the plot
    fig2, ax = plt.subplots(6, 1, sharex=True, figsize=(12,8))
    ax = ax.ravel()

    colors = ['blue', 'deepskyblue', 'green', 'orange', 'red']
    nonR_odours = ['C', 'D', 'E', 'F', 'G']  

    # Loop over rows
    for idx, row in results_df.iterrows():
        for o, odour in enumerate(nonR_odours):
            ax[o].scatter(row['date'], row['all_odour_same_olf_false_alarm'][odour],
                        color=colors[o], marker='o', s=60, alpha=1)
            ax[o].scatter(row['date'], row['all_odour_diff_olf_false_alarm'][odour],
                        color=colors[o], marker='o', s=60, alpha=0.4)
            ax[o].plot(results_df['date'], [row['all_odour_same_olf_false_alarm'][odour] for _, row in results_df.iterrows()], 
                    label='same-olf' if idx == 0 else "", color=colors[o], linestyle='-', alpha=1)
            ax[o].plot(results_df['date'], [row['all_odour_diff_olf_false_alarm'][odour] for _, row in results_df.iterrows()],
                    label='diff-olf' if idx == 0 else "", color=colors[o], linestyle='--', alpha=0.4)
            ax[o].set_ylabel(f'FA (%)\nodour {odour}')
            ax[o].annotate(f"Ses-{row['session_id']}",
                        (row['date'], row['all_odour_same_olf_false_alarm'][odour]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center')

        # Overall false alarm bias 
        ax[-1].scatter(row['date'], row['all_same_olf_false_alarm'],
                    color='black', marker='o', s=60, alpha=1)
        ax[-1].scatter(row['date'], row['all_diff_olf_false_alarm'],
                    color='black', marker='o', s=60, alpha=0.4)
        ax[-1].plot(results_df['date'], [row['all_same_olf_false_alarm'] for _, row in results_df.iterrows()], 
                label='same-olf' if idx == 0 else "", color='black', linestyle='-', alpha=1)
        ax[-1].plot(results_df['date'], [row['all_diff_olf_false_alarm'] for _, row in results_df.iterrows()],
                label='diff-olf' if idx == 0 else "", color='black', linestyle='--', alpha=0.4)
        ax[-1].set_ylabel(f'FA (%)\n average')
        ax[-1].annotate(f"Ses-{row['session_id']}",
                        (row['date'], row['all_same_olf_false_alarm']),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center')
        
    # Format the plot
    for i, axis in enumerate(ax):
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.set_ylim(0, 105)   # Assuming false alarm is a percentage between 0 and 100
        axis.grid(False)
        axis.axhline(y=40, color='gray', linestyle='--', alpha=0.7)
        axis.legend(loc='upper right')
    
    # Set title with subject ID if provided
    if subject_id:
        plt.suptitle(f'Decision False Alarm Olfactometer Bias Across Sessions - sub-{subject_id}')
    else:
        plt.suptitle('Decision False Alarm Olfactometer Bias Across Sessions')
    
    axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axis.tick_params(axis='x', rotation=45)  
    axis.set_xlabel('Session Date')

    # # Add session annotations
    for _, row in results_df.iterrows():
        if isinstance(row['all_odour_same_olf_false_alarm'], dict):
            for i, (odour, y) in enumerate(row['all_odour_same_olf_false_alarm'].items()): 
                if pd.notna(y) and y > 0:
                    ax[i].annotate(
                        f"Ses-{row['session_id']}",
                        (row['date'], y),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center')

    # Save if requested
    if plot_file:
        filename = f"sub-{subject_id}_FalseAlarmOlfactometerBias" + (f"_stage{stage}" if stage is not None else "") + ".png"
        plot_file = os.path.join(plot_file, filename)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_file}")
    
    plt.tight_layout()
    plt.show()

    return

def main(subject_folder, sessions=None, num_intervals=None, stage=None, output_file=None, plot_file=None):
    """
    Process a subject folder and calculate false alarm time bias for all sessions.
    Combines results from multiple directories within the same session.
    Saves results to CSV file if output_file is provided.
    """
    subject_path = Path(subject_folder)
    print(f"Processing subject folder: {subject_path}")
    
    # Extract subject ID from the folder path
    subject_id = subject_path.name
    if 'sub-' in subject_id:
        subject_id = subject_id.split('sub-')[1]
    
    # Use utils.find_session_roots instead of the local function
    session_roots = utils.find_session_roots(subject_path)
    
    if not session_roots:
        print(f"No valid session directories found in {subject_path}")
        return
     
    # Group sessions by session_id and session_date
    grouped_sessions = {}
    for session_id, session_date, session_path in session_roots:
        if sessions is not None:
            if int(session_id) in sessions:
                key = (session_id, session_date)
                if key not in grouped_sessions:
                    grouped_sessions[key] = []
                grouped_sessions[key].append(session_path)
    
    # Create a list to store combined results
    results = []
    
    # Process each grouped session
    for (session_id, session_date), session_paths in sorted(grouped_sessions.items()):
        print(f"\nProcessing Session ID: {session_id}, Date: {session_date}")
        print(f"Found {len(session_paths)} directories within this session")
        
        # Initialize combined metrics
        total_odour_interval_pokes = defaultdict(lambda: defaultdict(int))
        total_odour_interval_trials = defaultdict(lambda: defaultdict(int))
        total_odour_interval_false_alarm = defaultdict(lambda: defaultdict(lambda: None))
        total_interval_pokes = {}
        total_interval_trials = {}
        total_interval_false_alarm = {}

        all_odour_same_olf_pokes = defaultdict(int)
        all_odour_same_olf_trials = defaultdict(int)
        all_odour_same_olf_false_alarm = {}
        all_odour_diff_olf_pokes = defaultdict(int)
        all_odour_diff_olf_trials = defaultdict(int)
        all_odour_diff_olf_false_alarm = {}
        all_same_olf_pokes = 0
        all_same_olf_trials = 0
        all_same_olf_false_alarm = {}
        all_diff_olf_pokes = 0
        all_diff_olf_trials = 0
        all_diff_olf_false_alarm = {}
        
        # Directory-specific results for detailed information
        dir_results = []
        
        # Process each directory within the session
        for session_path in session_paths:
            detected_stage = int(detect_stage(session_path))

            if stage is not None and detected_stage != stage:
                print('Continue to next session...')
                continue
        
            print(f"  Processing directory: {session_path.name}")
            try:
                # Get false alarm data for this directory
                false_alarm_bias = get_false_alarm_bias(session_path)

                if false_alarm_bias and false_alarm_bias != {'odour_interval_pokes': 0, 
                                                            'odour_interval_trials': 0,
                                                            'odour_interval_false_alarm': 0, 
                                                            'interval_pokes': 0,
                                                            'interval_trials': 0, 
                                                            'interval_false_alarm': 0,
                                                            'odour_same_olf_pokes': 0, 
                                                            'odour_same_olf_trials': 0, 
                                                            'odour_same_olf_false_alarm': 0, 
                                                            'odour_diff_olf_pokes': 0, 
                                                            'odour_diff_olf_trials': 0, 
                                                            'odour_diff_olf_false_alarm': 0, 
                                                            'same_olf_pokes': 0, 
                                                            'same_olf_trials': 0, 
                                                            'same_olf_false_alarm': 0, 
                                                            'diff_olf_pokes': 0, 
                                                            'diff_olf_trials': 0, 
                                                            'diff_olf_false_alarm': 0
                    }:
                    # Add to totals 
                    nonR_odours = false_alarm_bias['odour_interval_pokes'].keys()
                    first_odour = next(iter(false_alarm_bias['odour_interval_pokes']))
                    intervals = false_alarm_bias['odour_interval_pokes'][first_odour].keys()
            
                    for odour in nonR_odours:
                        # time bias
                        for interval in intervals:
                            total_odour_interval_pokes[odour][interval] += false_alarm_bias['odour_interval_pokes'][odour][interval]
                            total_odour_interval_trials[odour][interval] += false_alarm_bias['odour_interval_trials'][odour][interval]
                            
                        # olfactometer bias 
                        all_odour_same_olf_pokes[odour] += false_alarm_bias['odour_same_olf_pokes'][odour]
                        all_odour_same_olf_trials[odour] += false_alarm_bias['odour_same_olf_trials'][odour]
                        all_odour_diff_olf_pokes[odour] += false_alarm_bias['odour_diff_olf_pokes'][odour]
                        all_odour_diff_olf_trials[odour] += false_alarm_bias['odour_diff_olf_trials'][odour]

                    all_same_olf_pokes += false_alarm_bias['same_olf_pokes']
                    all_same_olf_trials += false_alarm_bias['same_olf_trials']
                    all_diff_olf_pokes += false_alarm_bias['diff_olf_pokes']
                    all_diff_olf_trials += false_alarm_bias['diff_olf_trials']

                    dir_info = {
                        'directory': session_path.name,
                        'odour_interval_pokes': false_alarm_bias['odour_interval_pokes'],
                        'odour_interval_trials': false_alarm_bias['odour_interval_trials'],
                        'odour_interval_false_alarm': false_alarm_bias['odour_interval_false_alarm'],
                        'interval_pokes': false_alarm_bias['interval_pokes'],
                        'interval_trials': false_alarm_bias['interval_trials'],
                        'interval_false_alarm': false_alarm_bias['interval_false_alarm'],
                        'odour_same_olf_pokes': false_alarm_bias['odour_same_olf_pokes'], 
                        'odour_same_olf_trials': false_alarm_bias['odour_same_olf_trials'], 
                        'odour_same_olf_false_alarm': false_alarm_bias['odour_same_olf_false_alarm'], 
                        'odour_diff_olf_pokes': false_alarm_bias['odour_diff_olf_pokes'], 
                        'odour_diff_olf_trials': false_alarm_bias['odour_diff_olf_trials'], 
                        'odour_diff_olf_false_alarm': false_alarm_bias['odour_diff_olf_false_alarm'], 
                        'same_olf_pokes': false_alarm_bias['same_olf_pokes'], 
                        'same_olf_trials': false_alarm_bias['same_olf_trials'], 
                        'same_olf_false_alarm': false_alarm_bias['same_olf_false_alarm'], 
                        'diff_olf_pokes': false_alarm_bias['diff_olf_pokes'], 
                        'diff_olf_trials': false_alarm_bias['diff_olf_trials'], 
                        'diff_olf_false_alarm': false_alarm_bias['diff_olf_false_alarm']
                    }
                    
                    for odour in nonR_odours:
                        print(f"\nFalse alarm bias rates for odour {odour}:")
                        for interval, rate in false_alarm_bias['odour_interval_false_alarm'][odour].items():
                            print(f"  Interval {interval}: {rate:.1f}%")

                    for interval, rate in false_alarm_bias['interval_false_alarm'].items():
                        print(f"  Overall false alarm bias rate for Interval {interval}: {rate:.1f}%")
                
                    dir_results.append(dir_info)
                else:
                    print(f"    No valid false alarm time bias data found")
                    
            except Exception as e:
                print(f"    Error processing directory {session_path.name}: {str(e)}")
        
        # Calculate combined false alarm time and olfactometer bias values
        for odour in nonR_odours:
            for interval in intervals:
                total_odour_interval_false_alarm[odour][interval] = (total_odour_interval_pokes[odour][interval] / total_odour_interval_trials[odour][interval] * 100) if total_odour_interval_trials[odour][interval] else 0
        
            all_odour_same_olf_false_alarm[odour] = (all_odour_same_olf_pokes[odour] / all_odour_same_olf_trials[odour] * 100) if all_odour_same_olf_trials[odour] else 0
            all_odour_diff_olf_false_alarm[odour] = (all_odour_diff_olf_pokes[odour] / all_odour_diff_olf_trials[odour] * 100) if all_odour_diff_olf_trials[odour] else 0

        all_same_olf_false_alarm = (all_same_olf_pokes / all_same_olf_trials * 100) if all_same_olf_trials else 0
        all_diff_olf_false_alarm = (all_diff_olf_pokes / all_diff_olf_trials * 100) if all_diff_olf_trials else 0

        for interval in intervals:
            total_interval_pokes[interval] = np.sum([total_odour_interval_pokes[odour][interval] for odour in nonR_odours])
            total_interval_trials[interval] = np.sum([total_odour_interval_trials[odour][interval] for odour in nonR_odours])
            total_interval_false_alarm[interval] = (total_interval_pokes[interval] / total_interval_trials[interval] * 100) if total_interval_trials[interval] > 0 else 0
    
        if any(interval_trials > 0 for interval_trials in total_interval_trials.values()):
            # Store combined session results
            session_result = {
                'session_id': session_id,
                'session_date': session_date,
                'total_odour_interval_pokes': total_odour_interval_pokes,
                'total_odour_interval_trials': total_odour_interval_trials,
                'total_odour_interval_false_alarm': total_odour_interval_false_alarm,
                'total_interval_pokes': total_interval_pokes,
                'total_interval_trials': total_interval_trials,
                'total_interval_false_alarm': total_interval_false_alarm,
                'all_odour_same_olf_pokes': all_odour_same_olf_pokes, 
                'all_odour_same_olf_trials': all_odour_same_olf_trials, 
                'all_odour_same_olf_false_alarm': all_odour_same_olf_false_alarm, 
                'all_odour_diff_olf_pokes': all_odour_diff_olf_pokes, 
                'all_odour_diff_olf_trials': all_odour_diff_olf_trials, 
                'all_odour_diff_olf_false_alarm': all_odour_diff_olf_false_alarm, 
                'all_same_olf_pokes': all_same_olf_pokes, 
                'all_same_olf_trials': all_same_olf_trials, 
                'all_same_olf_false_alarm': all_same_olf_false_alarm, 
                'all_diff_olf_pokes': all_diff_olf_pokes, 
                'all_diff_olf_trials': all_diff_olf_trials, 
                'all_diff_olf_false_alarm': all_diff_olf_false_alarm,
                'directory_count': len(session_paths),
                'directories': dir_results
            }
            
            print(f"  Combined results for Session {session_id}:")
            for odour in nonR_odours:
                print(f"\nFalse alarm bias rates for odour {odour}:")
                for interval, rate in total_odour_interval_false_alarm[odour].items():
                    print(f"  Interval {interval}: {rate:.1f}%")

            for interval, rate in total_interval_false_alarm.items():
                print(f"  Overall false alarm bias rate for Interval {interval}: {rate:.1f}%")
        
            print(f"False alarm same-olfactometer bias: {all_same_olf_false_alarm:.1f}%")
            print(f"False alarm diff-olfactometer bias: {all_diff_olf_false_alarm:.1f}%")

            results.append(session_result)
        else:
            print(f"  No valid non-rewarded trials found for Session {session_id}")
            results.append({
                'session_id': session_id,
                'session_date': session_date,
                'total_odour_interval_pokes': total_odour_interval_pokes,
                'total_odour_interval_trials': total_odour_interval_trials,
                'total_odour_interval_false_alarm': total_odour_interval_false_alarm,
                'total_interval_pokes': total_interval_pokes,
                'total_interval_trials': total_interval_trials,
                'total_interval_false_alarm': total_interval_false_alarm,
                'all_odour_same_olf_pokes': all_odour_same_olf_pokes, 
                'all_odour_same_olf_trials': all_odour_same_olf_trials, 
                'all_odour_same_olf_false_alarm': all_odour_same_olf_false_alarm, 
                'all_odour_diff_olf_pokes': all_odour_diff_olf_pokes, 
                'all_odour_diff_olf_trials': all_odour_diff_olf_trials, 
                'all_odour_diff_olf_false_alarm': all_odour_diff_olf_false_alarm, 
                'all_same_olf_pokes': all_same_olf_pokes, 
                'all_same_olf_trials': all_same_olf_trials, 
                'all_same_olf_false_alarm': all_same_olf_false_alarm, 
                'all_diff_olf_pokes': all_diff_olf_pokes, 
                'all_diff_olf_trials': all_diff_olf_trials, 
                'all_diff_olf_false_alarm': all_diff_olf_false_alarm,
                'directory_count': len(session_paths),
                'directories': dir_results
            })
    
    # Create DataFrame from combined results
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'directories'} for r in results])
    
    # Print summary
    print("\nSummary of Combined Session False Alarms:")
    print("=======================================")
    for _, row in results_df.iterrows():
        if pd.notna(row['total_interval_false_alarm']):
            print(f"Session {row['session_id']} ({row['session_date']}): ") 
            
            for odour in nonR_odours:
                print(f"\nFalse alarm bias rates for odour {odour}:")
                for interval, rate in row['total_odour_interval_false_alarm'][odour].items():
                    print(f"  Interval {interval}: {rate:.1f}%")

            for interval, rate in row['total_interval_false_alarm'].items():
                print(f"  Overall false alarm bias rate for Interval {interval}: {rate:.1f}%")
            
            print(f"False alarm same-olfactometer bias: {row['all_same_olf_false_alarm']:.1f}%")
            print(f"False alarm diff-olfactometer bias: {row['all_diff_olf_false_alarm']:.1f}%")

        else:
            print(f"Session {row['session_id']} ({row['session_date']}): No valid trials found")
    
    # Save results to CSV if requested
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        # Also save detailed directory results
        detailed_output = Path(output_file).with_name(f"{Path(output_file).stem}_detailed.json")
        with open(detailed_output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to {detailed_output}")
    
    # Generate plot for time bias if we have data
    if not results_df.empty and not results_df['total_interval_false_alarm'].isna().all():
        plot_false_alarm_time_bias(results_df.dropna(subset=['total_interval_false_alarm']), num_intervals, stage, plot_file, subject_id)
    else:
        print("No valid false alarm time bias data to plot")

    # Generate plot for olfactometer bias if we have data
    if not results_df.empty and not results_df['all_same_olf_false_alarm'].isna().all() and not results_df['all_diff_olf_false_alarm'].isna().all():
        plot_false_alarm_olfactometer_bias(results_df.dropna(subset=['all_same_olf_false_alarm', 'all_diff_olf_false_alarm']), stage, plot_file, subject_id)
    else:
        print("No valid false alarm olfactometer bias data to plot")

    return results_df

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("/Volumes/harris/hypnose/rawdata/sub-020_id-072")

    parser = argparse.ArgumentParser(description="Calculate and plot decision false alarm time bias across sessions")
    parser.add_argument("subject_folder", help="Path to the subject's folder containing session data")
    parser.add_argument("--sessions", default=np.arange(55, 68), help="List of session IDs (optional)") 
    parser.add_argument("--intervals", "--i", default=8, help="How many intervals to plot (optional)")
    parser.add_argument("--stage", "--s", default=8, help="Stage to be analysed (optional)")
    parser.add_argument("--output", "-o", help="Path to save CSV output (optional)")
    parser.add_argument("--plot", "-p", help="Path to save plot image (optional)")
    args = parser.parse_args()
    
    main(args.subject_folder, args.sessions, args.intervals, args.stage, args.output, args.plot)