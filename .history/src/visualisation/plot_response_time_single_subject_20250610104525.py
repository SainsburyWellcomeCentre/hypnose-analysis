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
from src.analysis import get_decision_accuracy, get_response_time
import src.utils as utils
from src.analyse.response_time_analyser_single_session import analyze_session_folder

def plot_session_response_time(results_df, plot_file=None, subject_id=None):
    """
    Create a scatterplot of response time for each session.
    
    Args:
        results_df (DataFrame): Contains session_id, session_date, accuracy and response time data
        output_file (str): Path to save the plot, if provided
        subject_id (str): Subject ID to use in the plot title
    """    
    # Use trial IDs to find if trial is R1/R2 and correct/incorrect
    trial_id = np.array(results_df['all_trial_id'][0])

    # Create the plot
    _, ax = plt.subplots(1, 1, sharex=True, figsize=(12,6))
    
    # Plot each response time type
    r1_correct_trials = np.where((trial_id[:, 0] == 1) & (trial_id[:, 1] == 1))[0]
    r2_correct_trials = np.where((trial_id[:, 0] == 2) & (trial_id[:, 1] == 2))[0]
    r1_incorrect_trials = np.where((trial_id[:, 0] == 1) & (trial_id[:, 1] == 2))[0]
    r2_incorrect_trials = np.where((trial_id[:, 0] == 2) & (trial_id[:, 1] == 1))[0]

    ax.scatter(r1_correct_trials, results_df['all_r1_correct_rt'], label='R1 correct RT', color='blue')
    ax.scatter(r2_correct_trials, results_df['all_r2_correct_rt'], label='R2 correct RT', color='red')
    ax.scatter(r1_incorrect_trials, results_df['all_r1_incorrect_rt'], label='R1 incorrect RT', color='cyan')
    ax.scatter(r2_incorrect_trials, results_df['all_r2_incorrect_rt'], label='R2 incorrect RT', color='orange')
    
    # Add average response time
    ax.plot(results_df['avg_response_time'], label='avg response time', color='black', linestyle='--', alpha=0.7)
    
    # Format the plot
    ax.set_xlabel('Trial')
    ax.set_ylabel('Response time (s)')
    
    # Set title with subject ID if provided
    session_id = os.path.basename(results_df['session_folder'])
    if subject_id:
        plt.title(f'Response time - sub-{subject_id} - session {session_id}')
    else:
        plt.title(f'Decision Accuracy - session {session_id}')
        
    plt.grid(False)    
    plt.legend()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Save if requested
    if plot_file:
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_file}")
    
    plt.tight_layout()
    plt.show()


def plot_response_time(results_df, output_file=None, subject_id=None):
    return 

def main(session_folder, across_sessions=True, output_file=None, plot_file=None, subject_id=None):
    """
    Process a subject folder and calculate response time for a given session or across sessions.
    Combines results from multiple directories within the same session.
    Saves results to CSV file if output_file is provided.
    """

    print(across_sessions)
    

    if across_sessions:
        print('yes')
        # TODO: fix this and also the main definition 
        subject_path = os.path.split(os.path.split(session_folder)[0])[1]
        subject_path = Path(subject_path)
        print(f"Processing subject folder: {subject_path}")
        
    #     # Extract subject ID from the folder path
    #     subject_id = subject_path.name
    #     if 'sub-' in subject_id:
    #         subject_id = subject_id.split('sub-')[1]
        
    #     # Use utils.find_session_roots instead of the local function
    #     session_roots = utils.find_session_roots(subject_path)
        
    #     if not session_roots:
    #         print(f"No valid session directories found in {subject_path}")
    #         return
        
    #     # Group sessions by session_id and session_date
    #     grouped_sessions = {}
    #     for session_id, session_date, session_path in session_roots:
    #         key = (session_id, session_date)
    #         if key not in grouped_sessions:
    #             grouped_sessions[key] = []
    #         grouped_sessions[key].append(session_path)
        
    #     # Create a list to store combined results
    #     results = []
        
    #     # Process each grouped session
    #     for (session_id, session_date), session_paths in sorted(grouped_sessions.items()):
    #         print(f"\nProcessing Session ID: {session_id}, Date: {session_date}")
    #         print(f"Found {len(session_paths)} directories within this session")
            
    #         # Initialize combined metrics
    #         total_r1_correct = 0
    #         total_r1_trials = 0
    #         total_r2_correct = 0
    #         total_r2_trials = 0
            
    #         # Directory-specific results for detailed information
    #         dir_results = []
            
    #         # Process each directory within the session
    #         for session_path in session_paths:
    #             print(f"  Processing directory: {session_path.name}")
    #             try:
    #                 # Get accuracy data for this directory
    #                 accuracy = get_decision_accuracy(session_path)

    #                 if accuracy and accuracy != {
    #                     'r1_total': 0, 'r1_correct': 0, 'r1_accuracy': 0,
    #                     'r2_total': 0, 'r2_correct': 0, 'r2_accuracy': 0,
    #                     'overall_accuracy': 0
    #                 }:
    #                     # Add to totals
    #                     total_r1_correct += accuracy['r1_correct']
    #                     total_r1_trials += accuracy['r1_total']
    #                     total_r2_correct += accuracy['r2_correct']
    #                     total_r2_trials += accuracy['r2_total']
                        
    #                     dir_info = {
    #                         'directory': session_path.name,
    #                         'r1_correct': accuracy['r1_correct'],
    #                         'r1_trials': accuracy['r1_total'],
    #                         'r1_accuracy': accuracy['r1_accuracy'],
    #                         'r2_correct': accuracy['r2_correct'],
    #                         'r2_trials': accuracy['r2_total'],
    #                         'r2_accuracy': accuracy['r2_accuracy'],
    #                         'overall_accuracy': accuracy['overall_accuracy']
    #                     }
                        
    #                     print(f"    R1: {accuracy['r1_correct']}/{accuracy['r1_total']} ({accuracy['r1_accuracy']:.1f}%), "
    #                         f"R2: {accuracy['r2_correct']}/{accuracy['r2_total']} ({accuracy['r2_accuracy']:.1f}%)")
                        
    #                     dir_results.append(dir_info)
    #                 else:
    #                     print(f"    No valid accuracy data found")
                        
    #             except Exception as e:
    #                 print(f"    Error processing directory {session_path.name}: {str(e)}")
            
    #         # Calculate combined accuracy values
    #         if total_r1_trials + total_r2_trials > 0:
    #             # Calculate overall accuracy across all directories in this session
    #             r1_accuracy = (total_r1_correct / total_r1_trials) if total_r1_trials > 0 else 0
    #             r2_accuracy = (total_r2_correct / total_r2_trials) if total_r2_trials > 0 else 0
    #             overall_accuracy = ((total_r1_correct + total_r2_correct) / 
    #                             (total_r1_trials + total_r2_trials)) if (total_r1_trials + total_r2_trials) > 0 else 0
                
    #             # Store combined session results
    #             session_result = {
    #                 'session_id': session_id,
    #                 'session_date': session_date,
    #                 'overall_accuracy': overall_accuracy,
    #                 'r1_accuracy': r1_accuracy,
    #                 'r2_accuracy': r2_accuracy,
    #                 'r1_trials': total_r1_trials,
    #                 'r2_trials': total_r2_trials,
    #                 'total_trials': total_r1_trials + total_r2_trials,
    #                 'r1_correct': total_r1_correct,
    #                 'r2_correct': total_r2_correct,
    #                 'directory_count': len(session_paths),
    #                 'directories': dir_results
    #             }
                
    #             print(f"  Combined results for Session {session_id}:")
    #             print(f"    R1: {total_r1_correct}/{total_r1_trials} ({r1_accuracy:.1%})")
    #             print(f"    R2: {total_r2_correct}/{total_r2_trials} ({r2_accuracy:.1%})")
    #             print(f"    Overall: {total_r1_correct + total_r2_correct}/{total_r1_trials + total_r2_trials} ({overall_accuracy:.1%})")
                
    #             results.append(session_result)
    #         else:
    #             print(f"  No valid trials found for Session {session_id}")
    #             results.append({
    #                 'session_id': session_id,
    #                 'session_date': session_date,
    #                 'overall_accuracy': np.nan,
    #                 'r1_accuracy': np.nan,
    #                 'r2_accuracy': np.nan,
    #                 'r1_trials': 0,
    #                 'r2_trials': 0,
    #                 'total_trials': 0,
    #                 'r1_correct': 0,
    #                 'r2_correct': 0,
    #                 'directory_count': len(session_paths),
    #                 'directories': dir_results
    #             })
        
    #     # Create DataFrame from combined results
    #     results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'directories'} for r in results])

    #     # Plot response time across sessions
    #     plot_response_time(results_df, plot_file, subject_id)
    # else:
    #     # Get the accuracy and response time data
    #     results = analyze_session_folder(session_folder)

    #     # Plot response time for the session
    #     plot_session_response_time(results, plot_file, subject_id)

    # return results
    

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("/Volumes/harris/hypnose/rawdata/sub-026_id-077/ses-50_date-20250603")

    parser = argparse.ArgumentParser(description="Calculate and plot response time for a session or across sessions")
    parser.add_argument("session_folder", help="Path to the session folder containing data")
    parser.add_argument("--across_sessions", default=True, help="Whether to plot results for a single session or across sessions")
    parser.add_argument("--output", "-o", help="Path to save CSV output (optional)")
    parser.add_argument("--plot", "-p", help="Path to save plot image (optional)")
    args = parser.parse_args()
    
    main(args.session_folder, args.across_sessions, args.output, args.plot)











































































































































































































