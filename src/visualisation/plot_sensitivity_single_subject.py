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
from src.analysis import get_decision_sensitivity, detect_stage
from src.analyse.combined_analyses import calculate_combined_session_sensitivity
import src.utils as utils

def plot_sensitivity(results_df, stage=None, plot_path=None, subject_id=None):
    """
    Create a scatterplot of decision sensitivity across sessions.
    
    Args:
        results_df (DataFrame): Contains session_id, session_date, and sensitivity data
        stage (float): Behavioural stage, if provided
        plot_path (str): Path to save the plot, if provided
        subject_id (str): Subject ID to use in the plot title
    """
    # Convert date strings to datetime objects for better x-axis formatting
    results_df['date'] = pd.to_datetime(results_df['session_date'], format='%Y%m%d')
    results_df = results_df.sort_values('date')
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot each sensitivity type
    plt.scatter(results_df['date'], results_df['overall_sensitivity'], 
               label='Overall sensitivity', marker='o', s=60, color='black')
    plt.plot(results_df['date'], results_df['overall_sensitivity'], 
            color='black', linestyle='-', alpha=0.7)
    
    plt.scatter(results_df['date'], results_df['r1_sensitivity'], 
               label='R1 sensitivity', marker='s', s=60, color='blue')
    plt.plot(results_df['date'], results_df['r1_sensitivity'], 
            color='blue', linestyle='--', alpha=0.7)
    
    plt.scatter(results_df['date'], results_df['r2_sensitivity'], 
               label='R2 sensitivity', marker='^', s=60, color='red')
    plt.plot(results_df['date'], results_df['r2_sensitivity'], 
            color='red', linestyle='-.', alpha=0.7)
    
    # Format the plot
    plt.xlabel('Session Date')
    plt.ylabel('Decision Sensitivity')
    
    # Set title with subject ID if provided
    if subject_id:
        plt.title(f'Decision Sensitivity Across Sessions - sub-{subject_id}')
    else:
        plt.title('Decision Sensitivity Across Sessions')
    
    plt.ylim(0, 1.05)  # Assuming sensitivity is a percentage between 0 and 105
    
    # Remove grid and add single line at 0.8
    plt.grid(False)
    plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.7)
    
    plt.legend()
    
    # Format the x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Add session IDs as annotations
    for i, row in results_df.iterrows():
        y = row['overall_sensitivity']
        if pd.notna(y) and y > 0:
            plt.annotate(f"Ses-{row['session_id']}", 
                        (row['date'], y),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
     
    # Save if requested
    if plot_path:
        first_session_id = results_df['session_id'].iloc[0]
        last_session_id = results_df['session_id'].iloc[-1]

        plot_file = os.path.join(plot_path, f"sensitivity_sub-{subject_id}" + (f"_stage{stage}" if stage is not None else "") + f"_ses{first_session_id}-ses{last_session_id}" + ".png")
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)

        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_file}")
    
    plt.tight_layout()
    plt.show()

def main(subject_folder, sessions=None, stage=None, output_path=None, plot_path=None):
    """
    Process a subject folder and calculate decision sensitivity for all sessions.
    Combines results from multiple directories within the same session.
    Saves results to CSV file if output_path is provided.
    """
    subject_path = Path(subject_folder)
    print(f"Processing subject folder: {subject_path}")
    
    # Extract subject ID from the folder path
    subject_id = subject_path.name
    if 'sub-' in subject_id:
        subject_id = subject_id.split('sub-')[1]
    
    # Calculate session sensitivity for all sessions
    results_df = calculate_combined_session_sensitivity(subject_path, stage, sessions)

    # Plot results if requested
    if not results_df.empty and not results_df['overall_sensitivity'].isna().all():
        plot_sensitivity(results_df.dropna(subset=['overall_sensitivity']), stage, plot_path, subject_id)
    else:
        print("No valid sensitivity data to plot")
    
    # Save results to CSV if requested 
    if output_path:
        first_session_id = results_df['session_id'].iloc[0]
        last_session_id = results_df['session_id'].iloc[-1]

        output_file = os.path.join(output_path, f"sensitivity_sub-{subject_id}" + (f"_stage{stage}" if stage is not None else "") + f"_ses{first_session_id}-ses{last_session_id}" + ".csv")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

        return results_df


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("/Volumes/harris/hypnose/rawdata/sub-020_id-072")
        sys.argv.append("--output")
        sys.argv.append("/Volumes/harris/Athina/hypnose/analysis/sub-020_id-072/sensitivity")
        sys.argv.append("--plot")
        sys.argv.append("/Volumes/harris/Athina/hypnose/analysis/sub-020_id-072/sensitivity")

    parser = argparse.ArgumentParser(description="Calculate and plot decision sensitivity across sessions")
    parser.add_argument("subject_folder", help="Path to the subject's folder containing session data")
    parser.add_argument("--sessions", default=np.arange(31, 94), help="List of session IDs (optional)") 
    parser.add_argument("--stage", "--s", default=8, help="Stage to be analysed (optional)")
    parser.add_argument("--output", "-o", help="Path to CSV output folder (optional)")
    parser.add_argument("--plot", "-p", help="Path to plot output folder (optional)")
    args = parser.parse_args()
    
    main(args.subject_folder, args.sessions, args.stage, args.output, args.plot)