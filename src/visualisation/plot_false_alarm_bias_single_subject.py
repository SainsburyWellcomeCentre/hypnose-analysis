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
from src.analyse.combined_analyses import calculate_combined_session_false_alarm_bias
import src.utils as utils
from collections import defaultdict

def plot_false_alarm_time_bias(results_df, num_intervals=5, stage=None, plot_path=None, subject_id=None):
    """
    Create a scatterplot of false alarm time bias (# odours since rewarded odour) across sessions.
    
    Args:
        results_df (DataFrame): Contains session_id, session_date, and false alarm bias data
        stage (float): Behavioural stage, if provided
        plot_path (str): Path to save the plot, if provided
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
    total_intervals = sorted({intv for odour_data in sample_row.values() for intv in odour_data})[1:num_intervals+1]
    interval_to_axis = {interval: idx for idx, interval in enumerate(total_intervals)}

    # Loop over rows
    for idx, row in results_df.iterrows():
        if not isinstance(row['total_odour_interval_false_alarm'], dict):
            continue

        for o, odour in enumerate(nonR_odours):
            for interval in total_intervals:
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
        axis.set_xlabel(f'FA (%)\nInterval {total_intervals[i]}')
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
                if i != 0 and i != len(results_df) - 1:
                    continue

                if interval in interval_to_axis and pd.notna(x) and x > 0:
                    i = interval_to_axis[interval]
                    ax[i].annotate(
                        f"Ses-{row['session_id']}",
                        (x, row['date']),
                        textcoords="offset points",
                        xytext=(5, 0),
                        ha='left',
                        va='center')
    plt.suptitle(f'False Alarm Time Bias (# odours since rewarded odour) Across Sessions - sub-{subject_id}' if subject_id else 'False Alarm Time Bias (# odours since rewarded odour) Across Sessions')
    
    # Save if requested
    if plot_path:
        first_session_id = results_df['session_id'].iloc[0]
        last_session_id = results_df['session_id'].iloc[-1]

        plot_file = os.path.join(plot_path, f"time_side_bias_sub-{subject_id}" + (f"_stage{stage}" if stage is not None else "") + f"_ses{first_session_id}-ses{last_session_id}" + ".png")
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)

        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_file}")
    
    plt.tight_layout()
    plt.show()


def plot_false_alarm_olfactometer_bias(results_df, stage=None, plot_path=None, subject_id=None):
    """
    Create a scatterplot of false alarm olfactometer bias (preceding odour) across sessions.
    
    Args:
        results_df (DataFrame): Contains session_id, session_date, and false alarm bias data
        stage (float): Behavioural stage, if provided
        plot_path (str): Path to save the plot, if provided
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
            ax[o].scatter(row['date'], row['total_odour_same_olf_false_alarm'][odour],
                        color=colors[o], marker='o', s=60, alpha=1)
            ax[o].scatter(row['date'], row['total_odour_diff_olf_false_alarm'][odour],
                        color=colors[o], marker='o', s=60, alpha=0.4)
            ax[o].plot(results_df['date'], [row['total_odour_same_olf_false_alarm'][odour] for _, row in results_df.iterrows()], 
                    label='same-olf' if idx == 0 else "", color=colors[o], linestyle='-', alpha=1)
            ax[o].plot(results_df['date'], [row['total_odour_diff_olf_false_alarm'][odour] for _, row in results_df.iterrows()],
                    label='diff-olf' if idx == 0 else "", color=colors[o], linestyle='--', alpha=0.4)
            ax[o].set_ylabel(f'FA (%)\nodour {odour}')
            ax[o].annotate(f"Ses-{row['session_id']}",
                        (row['date'], row['total_odour_same_olf_false_alarm'][odour]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center')

        # Overall false alarm bias 
        ax[-1].scatter(row['date'], row['total_same_olf_false_alarm'],
                    color='black', marker='o', s=60, alpha=1)
        ax[-1].scatter(row['date'], row['total_diff_olf_false_alarm'],
                    color='black', marker='o', s=60, alpha=0.4)
        ax[-1].plot(results_df['date'], [row['total_same_olf_false_alarm'] for _, row in results_df.iterrows()], 
                label='same-olf' if idx == 0 else "", color='black', linestyle='-', alpha=1)
        ax[-1].plot(results_df['date'], [row['total_diff_olf_false_alarm'] for _, row in results_df.iterrows()],
                label='diff-olf' if idx == 0 else "", color='black', linestyle='--', alpha=0.4)
        ax[-1].set_ylabel(f'FA (%)\n average')
        ax[-1].annotate(f"Ses-{row['session_id']}",
                        (row['date'], row['total_same_olf_false_alarm']),
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
        plt.suptitle(f'False Alarm Olfactometer Bias Across Sessions - sub-{subject_id}')
    else:
        plt.suptitle('False Alarm Olfactometer Bias Across Sessions')
    
    axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axis.tick_params(axis='x', rotation=45)  
    axis.set_xlabel('Session Date')

    # Add session annotations
    for idx, (_, row) in enumerate(results_df.iterrows()):
        if isinstance(row['total_odour_same_olf_false_alarm'], dict):
            for i, (odour, y) in enumerate(row['total_odour_same_olf_false_alarm'].items()): 
                if i != 0 and i != len(results_df) - 1:
                    continue

                if pd.notna(y) and y > 0:
                    ax[i].annotate(
                        f"Ses-{row['session_id']}",
                        (row['date'], y),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center')

    # Save if requested
    if plot_path:
        first_session_id = results_df['session_id'].iloc[0]
        last_session_id = results_df['session_id'].iloc[-1]

        plot_file = os.path.join(plot_path, f"olfactometer_bias_sub-{subject_id}" + (f"_stage{stage}" if stage is not None else "") + f"_ses{first_session_id}-ses{last_session_id}" + ".png")
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)

        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_file}")
    
    plt.tight_layout()
    plt.show()

    return


def plot_false_alarm_olfactometer_reward_side_bias(results_df, stage=None, plot_path=None, subject_id=None):
    """
    Create a scatterplot of false alarm reward side bias (port chosen based on olfactometer) 
    
    Args:
        results_df (DataFrame): Contains session_id, session_date, and false alarm bias data
        stage (float): Behavioural stage, if provided
        plot_path (str): Path to save the plot, if provided
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
            ax[o].scatter(row['date'], row['total_odour_same_olf_rew_false_alarm'][odour],
                        color=colors[o], marker='o', s=60, alpha=1)
            ax[o].scatter(row['date'], row['total_odour_diff_olf_rew_false_alarm'][odour],
                        color=colors[o], marker='o', s=60, alpha=0.4)
            ax[o].plot(results_df['date'], [row['total_odour_same_olf_rew_false_alarm'][odour] for _, row in results_df.iterrows()], 
                    label='same-olf-rew' if idx == 0 else "", color=colors[o], linestyle='-', alpha=1)
            ax[o].plot(results_df['date'], [row['total_odour_diff_olf_rew_false_alarm'][odour] for _, row in results_df.iterrows()],
                    label='diff-olf-rew' if idx == 0 else "", color=colors[o], linestyle='--', alpha=0.4)
            ax[o].set_ylabel(f'FA (%)\nodour {odour}')
            ax[o].annotate(f"Ses-{row['session_id']}",
                        (row['date'], row['total_odour_same_olf_rew_false_alarm'][odour]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center')

        # Overall false alarm bias 
        ax[-1].scatter(row['date'], row['total_same_olf_rew_false_alarm'],
                    color='black', marker='o', s=60, alpha=1)
        ax[-1].scatter(row['date'], row['total_diff_olf_rew_false_alarm'],
                    color='black', marker='o', s=60, alpha=0.4)
        ax[-1].plot(results_df['date'], [row['total_same_olf_rew_false_alarm'] for _, row in results_df.iterrows()], 
                label='same-olf-rew' if idx == 0 else "", color='black', linestyle='-', alpha=1)
        ax[-1].plot(results_df['date'], [row['total_diff_olf_rew_false_alarm'] for _, row in results_df.iterrows()],
                label='diff-olf-rew' if idx == 0 else "", color='black', linestyle='--', alpha=0.4)
        ax[-1].set_ylabel(f'FA (%)\n average')
        ax[-1].annotate(f"Ses-{row['session_id']}",
                        (row['date'], row['total_same_olf_rew_false_alarm']),
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
        plt.suptitle(f'False Alarm Olfactometer-Reward side Bias Across Sessions - sub-{subject_id}')
    else:
        plt.suptitle('False Alarm Olfactometer-Reward side Bias Across Sessions')
    
    axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axis.tick_params(axis='x', rotation=45)  
    axis.set_xlabel('Session Date')

    # Add session annotations
    for idx, (_, row) in enumerate(results_df.iterrows()):
        if isinstance(row['total_odour_same_olf_rew_false_alarm'], dict):
            for i, (odour, y) in enumerate(row['total_odour_same_olf_rew_false_alarm'].items()): 
                if i != 0 and i != len(results_df) - 1:
                    continue

                if pd.notna(y) and y > 0:
                    ax[i].annotate(
                        f"Ses-{row['session_id']}",
                        (row['date'], y),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center')

    # Save if requested
    if plot_path:
        first_session_id = results_df['session_id'].iloc[0]
        last_session_id = results_df['session_id'].iloc[-1]

        plot_file = os.path.join(plot_path, f"olf_reward_side_bias_sub-{subject_id}" + (f"_stage{stage}" if stage is not None else "") + f"_ses{first_session_id}-ses{last_session_id}" + ".png")
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)

        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_file}")
    
    plt.tight_layout()
    plt.show()

    return


def main(subject_folder, sessions=None, num_intervals=None, stage=None, output_path=None, plot_path=None):
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
    
    # Calculate false alarm bias rates for all sessions
    results_df = calculate_combined_session_false_alarm_bias(subject_path, stage, sessions)

    # Save results to CSV if requested
    if output_path:
        first_session_id = results_df['session_id'].iloc[0]
        last_session_id = results_df['session_id'].iloc[-1]

        output_file = os.path.join(output_path, f"false_alarm_bias_sub-{subject_id}" + (f"_stage{stage}" if stage is not None else "") + f"_ses{first_session_id}-ses{last_session_id}" + ".csv")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

    # Plot results if requested
    # Time bias
    if not results_df.empty and not results_df['total_interval_false_alarm'].isna().all():
        plot_false_alarm_time_bias(results_df.dropna(subset=['total_interval_false_alarm']), num_intervals, stage, plot_path, subject_id)
    else:
        print("No valid false alarm time bias data to plot")

    # Olfactometer bias 
    if not results_df.empty and not results_df['total_same_olf_false_alarm'].isna().all() and not results_df['total_diff_olf_false_alarm'].isna().all():
        plot_false_alarm_olfactometer_bias(results_df.dropna(subset=['total_same_olf_false_alarm', 'total_diff_olf_false_alarm']), stage, plot_path, subject_id)
    else:
        print("No valid false alarm olfactometer bias data to plot")

    # Olfactometer-reward pair bias 
    if not results_df.empty and not results_df['total_same_olf_rew_false_alarm'].isna().all() and not results_df['total_diff_olf_rew_false_alarm'].isna().all():
        plot_false_alarm_olfactometer_reward_side_bias(results_df.dropna(subset=['total_same_olf_rew_false_alarm', 'total_diff_olf_rew_false_alarm']), stage, plot_path, subject_id)
    else:
        print("No valid false alarm olfactometer-reward side bias data to plot")

    return results_df

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("/Volumes/harris/hypnose/rawdata/sub-020_id-072")
        sys.argv.append("--output")
        sys.argv.append("/Volumes/harris/Athina/hypnose/analysis/sub-020_id-072/false_alarm")
        sys.argv.append("--plot")
        sys.argv.append("/Volumes/harris/Athina/hypnose/analysis/sub-020_id-072/false_alarm")

    parser = argparse.ArgumentParser(description="Calculate and plot false alarm biases across sessions")
    parser.add_argument("subject_folder", help="Path to the subject's folder containing session data")
    parser.add_argument("--sessions", default=np.arange(31, 94), help="List of session IDs (optional)") 
    parser.add_argument("--intervals", "--i", default=5, help="How many intervals to plot (optional)")
    parser.add_argument("--stage", "--s", default=8, help="Stage to be analysed (optional)")
    parser.add_argument("--output", "-o", help="Path to CSV output folder (optional)")
    parser.add_argument("--plot", "-p", help="Path to plot output folder (optional)")
    args = parser.parse_args()
    
    main(args.subject_folder, args.sessions, args.intervals, args.stage, args.output, args.plot)