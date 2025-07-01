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
import src.utils as utils

def plot_sensitivity(results_df, stage=None, plot_file=None, subject_id=None):
    """
    Create a scatterplot of decision sensitivity across sessions.
    
    Args:
        results_df (DataFrame): Contains session_id, session_date, and sensitivity data
        stage (float): Behavioural stage, if provided
        plot_file (str): Path to save the plot, if provided
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
    if plot_file:
        filename = f"sub-{subject_id}_sensitivity" + (f"_stage{stage}" if stage is not None else "") + ".png"
        plot_file = os.path.join(plot_file, filename)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_file}")
    
    plt.tight_layout()
    plt.show()

def main(subject_folder, sessions=None, stage=None, output_file=None, plot_file=None):
    """
    Process a subject folder and calculate decision sensitivity for all sessions.
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
        total_r1_respond = 0
        total_r1_trials = 0
        total_r2_respond = 0
        total_r2_trials = 0
        
        # Directory-specific results for detailed information
        dir_results = []
        
        # Process each directory within the session
        for session_path in session_paths:
            detected_stage = float(detect_stage(session_path))

            if stage is not None and detected_stage not in stage:
                print('Continue to next session...')
                continue
        
            print(f"  Processing directory: {session_path.name}")
            try:
                # Get sensitivity data for this directory
                sensitivity = get_decision_sensitivity(session_path)

                if sensitivity and sensitivity != {
                    'r1_total': 0, 'r1_respond': 0, 'r1_sensitivity': 0,
                    'r2_total': 0, 'r2_respond': 0, 'r2_sensitivity': 0,
                    'overall_sensitivity': 0
                }:
                    # Add to totals
                    total_r1_respond += sensitivity['r1_respond']
                    total_r1_trials += sensitivity['r1_total']
                    total_r2_respond += sensitivity['r2_respond']
                    total_r2_trials += sensitivity['r2_total']
                    
                    dir_info = {
                        'directory': session_path.name,
                        'r1_respond': sensitivity['r1_respond'],
                        'r1_trials': sensitivity['r1_total'],
                        'r1_sensitivity': sensitivity['r1_sensitivity'],
                        'r2_respond': sensitivity['r2_respond'],
                        'r2_trials': sensitivity['r2_total'],
                        'r2_sensitivity': sensitivity['r2_sensitivity'],
                        'overall_sensitivity': sensitivity['overall_sensitivity']
                    }
                    
                    print(f"    R1: {sensitivity['r1_respond']}/{sensitivity['r1_total']} ({sensitivity['r1_sensitivity']:.1f}%), "
                          f"R2: {sensitivity['r2_respond']}/{sensitivity['r2_total']} ({sensitivity['r2_sensitivity']:.1f}%)")
                    
                    dir_results.append(dir_info)
                else:
                    print(f"    No valid sensitivity data found")
                    
            except Exception as e:
                print(f"    Error processing directory {session_path.name}: {str(e)}")
        
        # Calculate combined sensitivity values
        if total_r1_trials + total_r2_trials > 0:
            # Calculate overall sensitivity across all directories in this session
            r1_sensitivity = (total_r1_respond / total_r1_trials) if total_r1_trials > 0 else 0
            r2_sensitivity = (total_r2_respond / total_r2_trials) if total_r2_trials > 0 else 0
            overall_sensitivity = ((total_r1_respond + total_r2_respond) / 
                               (total_r1_trials + total_r2_trials)) if (total_r1_trials + total_r2_trials) > 0 else 0
            
            # Store combined session results
            session_result = {
                'session_id': session_id,
                'session_date': session_date,
                'overall_sensitivity': overall_sensitivity,
                'r1_sensitivity': r1_sensitivity,
                'r2_sensitivity': r2_sensitivity,
                'r1_trials': total_r1_trials,
                'r2_trials': total_r2_trials,
                'total_trials': total_r1_trials + total_r2_trials,
                'r1_respond': total_r1_respond,
                'r2_respond': total_r2_respond,
                'directory_count': len(session_paths),
                'directories': dir_results
            }
            
            print(f"  Combined results for Session {session_id}:")
            print(f"    R1: {total_r1_respond}/{total_r1_trials} ({r1_sensitivity:.1%})")
            print(f"    R2: {total_r2_respond}/{total_r2_trials} ({r2_sensitivity:.1%})")
            print(f"    Overall: {total_r1_respond + total_r2_respond}/{total_r1_trials + total_r2_trials} ({overall_sensitivity:.1%})")
            
            results.append(session_result)
        else:
            print(f"  No valid trials found for Session {session_id}")
            results.append({
                'session_id': session_id,
                'session_date': session_date,
                'overall_sensitivity': np.nan,
                'r1_sensitivity': np.nan,
                'r2_sensitivity': np.nan,
                'r1_trials': 0,
                'r2_trials': 0,
                'total_trials': 0,
                'r1_respond': 0,
                'r2_respond': 0,
                'directory_count': len(session_paths),
                'directories': dir_results
            })
    
    # Create DataFrame from combined results
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'directories'} for r in results])
    
    # Print summary
    print("\nSummary of Combined Session Accuracies:")
    print("=======================================")
    for _, row in results_df.iterrows():
        if pd.notna(row['overall_sensitivity']):
            print(f"Session {row['session_id']} ({row['session_date']}): " 
                  f"Overall {row['overall_sensitivity']:.1%}, "
                  f"R1 {row['r1_sensitivity']:.1%} ({row['r1_trials']} trials), "
                  f"R2 {row['r2_sensitivity']:.1%} ({row['r2_trials']} trials)")
        else:
            print(f"Session {row['session_id']} ({row['session_date']}): No valid trials found")
    
    # Save results to CSV if requested TODO
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        # Also save detailed directory results
        detailed_output = Path(output_file).with_name(f"{Path(output_file).stem}_detailed.json")
        with open(detailed_output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to {detailed_output}")
    
    # Generate plot if we have data
    if not results_df.empty and not results_df['overall_sensitivity'].isna().all():
        plot_sensitivity(results_df.dropna(subset=['overall_sensitivity']), stage, plot_file, subject_id)
    else:
        print("No valid sensitivity data to plot")
    
    return results_df

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("/Volumes/harris/hypnose/rawdata/sub-020_id-072")

    parser = argparse.ArgumentParser(description="Calculate and plot decision sensitivity across sessions")
    parser.add_argument("subject_folder", help="Path to the subject's folder containing session data")
    parser.add_argument("--sessions", default=np.arange(66, 70), help="List of session IDs (optional)") 
    parser.add_argument("--stage", "--s", default=[8.2, 8.3], help="Stage to be analysed (optional)")
    parser.add_argument("--output", "-o", help="Path to save CSV output (optional)")
    parser.add_argument("--plot", "-p", help="Path to save plot image (optional)")
    args = parser.parse_args()
    
    main(args.subject_folder, args.sessions, args.stage, args.output, args.plot)