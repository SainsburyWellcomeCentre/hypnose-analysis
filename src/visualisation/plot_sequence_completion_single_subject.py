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
from src.analysis import get_sequence_completion, detect_stage
import src.utils as utils


def plot_sequence_completion(results_df, stage=None, plot_file=None, subject_id=None):
    """
    Create a scatterplot of sequence completion ratio across sessions.
    
    Args:
        results_df (DataFrame): Contains session_id, session_date, and sequence completion data
        plot_file (str): Path to save the plot, if provided
        subject_id (str): Subject ID to use in the plot title
    """
    # Convert date strings to datetime objects for better x-axis formatting
    results_df['date'] = pd.to_datetime(results_df['session_date'], format='%Y%m%d')
    results_df = results_df.sort_values('date')
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot sequence completion
    plt.scatter(results_df['date'], results_df['overall_completion_ratio'], 
               label='Sequence Completion ratio', marker='o', s=60, color='black')
    plt.plot(results_df['date'], results_df['overall_completion_ratio'], 
            color='black', linestyle='-', alpha=0.7)
    
    # Format the plot
    plt.xlabel('Session Date')
    plt.ylabel('Sequence Completion')
    
    # Set title with subject ID if provided
    if subject_id:
        plt.title(f'Sequence Completion Across Sessions - sub-{subject_id}')
    else:
        plt.title('Sequence Completion Across Sessions')
    
    plt.ylim(0, 105)  # Assuming sequence completion ratio is a percentage between 0 and 100
    
    # Remove grid and add single line at 70
    plt.grid(False)
    plt.axhline(y=70, color='gray', linestyle='--', alpha=0.7)
    
    plt.legend()
    
    # Format the x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Add session IDs as annotations
    for i, row in results_df.iterrows():
        y = row['overall_completion_ratio']
        if pd.notna(y) and y > 0:
            plt.annotate(f"Ses-{row['session_id']}", 
                        (row['date'], y),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
    
    # Save if requested
    if plot_file:
        filename = f"sub-{subject_id}_Completion" + (f"_stage{stage}" if stage is not None else "") + ".png"
        plot_file = os.path.join(plot_file, filename)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_file}")
    
    plt.tight_layout()
    plt.show()

def main(subject_folder, sessions=None, stage=None, output_file=None, plot_file=None):
    """
    Process a subject folder and calculate sequence completion ratio for all sessions.
    Combines results from multiple directories within the same session.
    Saves results to CSV file if output_file is provided.
    """
    if stage < 9:
        raise ValueError("Sequence completion is only valid for stage 9 (Doubles)") # TODO
    
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
        total_rew_trials = 0
        total_non_rew_trials = 0
        
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
                # Get sequence completion data for this directory
                sequence_completion = get_sequence_completion(session_path)

                if sequence_completion and sequence_completion != {
                    'rew_trials': 0, 'non_rew_trials': 0, 'completion_ratio': 0
                }:
                    # Add to totals
                    total_rew_trials += sequence_completion['rew_trials']
                    total_non_rew_trials += sequence_completion['non_rew_trials']
                    
                    dir_info = {
                        'directory': session_path.name,
                        'rew_trials': sequence_completion['rew_trials'],
                        'non_rew_trials': sequence_completion['non_rew_trials'],
                        'completion_ratio': sequence_completion['completion_ratio'],
                        }
                    
                    print(f"  Sequence completion: {sequence_completion['completion_ratio']:.1f}%")
                    
                    dir_results.append(dir_info)
                else:
                    print(f"    No valid sequence completion data found")
                    
            except Exception as e:
                print(f"    Error processing directory {session_path.name}: {str(e)}")
        
        # Calculate combined sequence completion values
        if total_rew_trials + total_non_rew_trials > 0:
            # Calculate overall sequence completion across all directories in this session
            overall_completion_ratio = total_rew_trials / (total_rew_trials + total_non_rew_trials) * 100 
  
            # Store combined session results
            session_result = {
                'session_id': session_id,
                'session_date': session_date,
                'overall_completion_ratio': overall_completion_ratio,
                'directory_count': len(session_paths),
                'directories': dir_results
            }
            
            print(f"  Combined results for Session {session_id}:")
            print(f"    Overall sequence completion: {overall_completion_ratio:.1f}%")
            
            results.append(session_result)
        else:
            print(f"  No valid trials found for Session {session_id}")
            results.append({
                'session_id': session_id,
                'session_date': session_date,
                'overall_completion_ratio': np.nan,
                'directory_count': len(session_paths),
                'directories': dir_results
            })
    
    # Create DataFrame from combined results
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'directories'} for r in results])
    
    # Print summary
    print("\nSummary of Combined Session Sequence Completion:")
    print("=======================================")
    for _, row in results_df.iterrows():
        if pd.notna(row['overall_completion_ratio']):
            print(f"Session {row['session_id']} ({row['session_date']}): " 
                  f"Overall {row['overall_completion_ratio']:.1f}%, ")
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
    
    # Generate plot if we have data
    if not results_df.empty and not results_df['overall_completion_ratio'].isna().all():
        plot_sequence_completion(results_df.dropna(subset=['overall_completion_ratio']), stage, plot_file, subject_id)
    else:
        print("No valid sequence completion data to plot")
    
    return results_df

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("/Volumes/harris/hypnose/rawdata/sub-026_id-077")

    parser = argparse.ArgumentParser(description="Calculate and plot sequence completion data across sessions")
    parser.add_argument("subject_folder", help="Path to the subject's folder containing session data")
    parser.add_argument("--sessions", default=np.arange(55, 66), help="List of session IDs (optional)") 
    parser.add_argument("--stage", "--s", default=9, help="Stage to be analysed (optional)")
    parser.add_argument("--output", "-o", help="Path to save CSV output (optional)")
    parser.add_argument("--plot", "-p", help="Path to save plot image (optional)")
    args = parser.parse_args()
    
    main(args.subject_folder, args.sessions, args.stage, args.output, args.plot)