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
from src.analysis import get_decision_accuracy, detect_stage, get_decision_specificity
import src.utils as utils

def calculate_session_specificity(session_path):
    """
    Calculate specificity for the session.
    
    Args:
        session_path (Path): Path to the session directory
        
    Returns:
        dict: Contains specificity values for each non-rewarded odour
    """
    # Don't look for JSON files - send the session path directly to get_decision_specificity
    try:
        # Get decision specificity for this session
        specificity = get_decision_specificity(session_path)
        
        if specificity is None:
            print(f"No specificity data available for {session_path}")
            return {
                'C_pokes': 0, 'C_trials': 0,
                'D_pokes': 0, 'D_trials': 0, 'E_pokes': 0, 'E_trials': 0,
                'F_pokes': 0, 'F_trials': 0, 'G_pokes': 0, 'G_trials': 0,
                'C_false_alarm': 0,
                'D_false_alarm': 0, 'E_false_alarm': 0,
                'F_false_alarm': 0, 'G_false_alarm': 0, 
                'overall_false_alarm': 0
            }
        
        # Convert from percentages to proportions if needed
        r1_accuracy = accuracy['r1_accuracy'] / 100 if accuracy['r1_accuracy'] > 1 else accuracy['r1_accuracy']
        r2_accuracy = accuracy['r2_accuracy'] / 100 if accuracy['r2_accuracy'] > 1 else accuracy['r2_accuracy']
        overall_accuracy = accuracy['overall_accuracy'] / 100 if accuracy['overall_accuracy'] > 1 else accuracy['overall_accuracy']
        
        return {
            'overall': overall_accuracy,
            'r1': r1_accuracy,
            'r2': r2_accuracy,
            'r1_trials': accuracy['r1_total'],
            'r2_trials': accuracy['r2_total'],
            'total_trials': accuracy['r1_total'] + accuracy['r2_total']
        }
        
    except Exception as e:
        print(f"Error processing session {session_path}: {str(e)}")
        return {
            'overall': np.nan,
            'r1': np.nan,
            'r2': np.nan,
            'r1_trials': 0,
            'r2_trials': 0,
            'total_trials': 0
        }

def calculate_session_accuracy(session_path):
    """
    Calculate decision accuracy for the session.
    
    Args:
        session_path (Path): Path to the session directory
        
    Returns:
        dict: Contains 'overall', 'r1', and 'r2' accuracy values
    """
    # Don't look for JSON files - send the session path directly to get_decision_accuracy
    try:
        # Get decision accuracy for this session
        accuracy = get_decision_accuracy(session_path)
        
        if accuracy is None:
            print(f"No accuracy data available for {session_path}")
            return {
                'overall': np.nan,
                'r1': np.nan,
                'r2': np.nan,
                'r1_trials': 0,
                'r2_trials': 0,
                'total_trials': 0
            }
        
        # Convert from percentages to proportions if needed
        r1_accuracy = accuracy['r1_accuracy'] / 100 if accuracy['r1_accuracy'] > 1 else accuracy['r1_accuracy']
        r2_accuracy = accuracy['r2_accuracy'] / 100 if accuracy['r2_accuracy'] > 1 else accuracy['r2_accuracy']
        overall_accuracy = accuracy['overall_accuracy'] / 100 if accuracy['overall_accuracy'] > 1 else accuracy['overall_accuracy']
        
        return {
            'overall': overall_accuracy,
            'r1': r1_accuracy,
            'r2': r2_accuracy,
            'r1_trials': accuracy['r1_total'],
            'r2_trials': accuracy['r2_total'],
            'total_trials': accuracy['r1_total'] + accuracy['r2_total']
        }
        
    except Exception as e:
        print(f"Error processing session {session_path}: {str(e)}")
        return {
            'overall': np.nan,
            'r1': np.nan,
            'r2': np.nan,
            'r1_trials': 0,
            'r2_trials': 0,
            'total_trials': 0
        }

def plot_accuracy(results_df, output_file=None, subject_id=None):
    """
    Create a scatterplot of decision accuracy across sessions.
    
    Args:
        results_df (DataFrame): Contains session_id, session_date, and accuracy data
        output_file (str): Path to save the plot, if provided
        subject_id (str): Subject ID to use in the plot title
    """
    # Convert date strings to datetime objects for better x-axis formatting
    results_df['date'] = pd.to_datetime(results_df['session_date'], format='%Y%m%d')
    results_df = results_df.sort_values('date')
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot each accuracy type
    plt.scatter(results_df['date'], results_df['overall_accuracy'], 
               label='Overall Accuracy', marker='o', s=60, color='black')
    plt.plot(results_df['date'], results_df['overall_accuracy'], 
            color='black', linestyle='-', alpha=0.7)
    
    plt.scatter(results_df['date'], results_df['r1_accuracy'], 
               label='R1 Accuracy', marker='s', s=60, color='blue')
    plt.plot(results_df['date'], results_df['r1_accuracy'], 
            color='blue', linestyle='--', alpha=0.7)
    
    plt.scatter(results_df['date'], results_df['r2_accuracy'], 
               label='R2 Accuracy', marker='^', s=60, color='red')
    plt.plot(results_df['date'], results_df['r2_accuracy'], 
            color='red', linestyle='-.', alpha=0.7)
    
    # Format the plot
    plt.xlabel('Session Date')
    plt.ylabel('Decision Accuracy')
    
    # Set title with subject ID if provided
    if subject_id:
        plt.title(f'Decision Accuracy Across Sessions - sub-{subject_id}')
    else:
        plt.title('Decision Accuracy Across Sessions')
    
    plt.ylim(0, 1.05)  # Assuming accuracy is a proportion between 0 and 1
    
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
        plt.annotate(f"Ses-{row['session_id']}", 
                    (row['date'], row['overall_accuracy']),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    plt.tight_layout()
    plt.show()

def main(subject_folder, sessions=None, stage=None, output_file=None, plot_file=None):
    """
    Process a subject folder and calculate decision accuracy for all sessions.
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
        total_C_pokes = 0
        total_D_pokes = 0
        total_E_pokes = 0
        total_F_pokes = 0
        total_G_pokes = 0
        total_C_trials = 0
        total_D_trials = 0
        total_E_trials = 0
        total_F_trials = 0
        total_G_trials = 0
        # total_D_false_alarm = 0
        # total_E_false_alarm = 0
        # total_F_false_alarm = 0
        # total_G_false_alarm = 0
        # total_overall_false_alarm = 0
        
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
                # Get specificity data for this directory
                specificity = get_decision_specificity(session_path)

                if specificity and specificity != {'C_pokes': 0, 'C_trials': 0,
                        'D_pokes': 0, 'D_trials': 0, 'E_pokes': 0, 'E_trials': 0,
                        'F_pokes': 0, 'F_trials': 0, 'G_pokes': 0, 'G_trials': 0,
                        'C_false_alarm': 0,
                        'D_false_alarm': 0, 'E_false_alarm': 0,
                        'F_false_alarm': 0, 'G_false_alarm': 0, 
                        'overall_false_alarm': 0
                    }:
                    # Add to totals # TODO: Calculate using single trials 
                    total_C_pokes += specificity['C_pokes']
                    total_D_pokes += specificity['D_pokes']
                    total_E_pokes += specificity['E_pokes']
                    total_F_pokes += specificity['F_pokes']
                    total_G_pokes += specificity['G_pokes']
                    total_C_trials += specificity['C_trials']
                    total_D_trials += specificity['D_trials']
                    total_E_trials += specificity['E_trials']
                    total_F_trials += specificity['F_trials']
                    total_G_trials += specificity['G_trials']

                    dir_info = {
                        'directory': session_path.name,
                        'C_pokes': specificity['C_pokes'],
                        'C_trials': specificity['C_trials'],
                        'D_pokes': specificity['D_pokes'],
                        'D_trials': specificity['D_trials'],
                        'E_pokes': specificity['E_pokes'],
                        'E_trials': specificity['E_trials'],
                        'F_pokes': specificity['F_pokes'],
                        'F_trials': specificity['F_trials'],
                        'G_pokes': specificity['G_pokes'],
                        'G_trials': specificity['G_trials'],
                        'C_false_alarm': specificity['C_false_alarm'],
                        'D_false_alarm': specificity['D_false_alarm'],
                        'E_false_alarm': specificity['E_false_alarm'],
                        'F_false_alarm': specificity['F_false_alarm'],
                        'G_false_alarm': specificity['G_false_alarm'],
                        'overall_false_alarm': specificity['overall_false_alarm']
                    }
                    
                    print(f"    C: {specificity['C_false_alarm']:.1f}%, D: {specificity['D_false_alarm']:.1f}%, "
                          f"    E: {specificity['E_false_alarm']:.1f}%, F: {specificity['F_false_alarm']:.1f}%, "
                          f"    G: {specificity['G_false_alarm']:.1f}%, overall: {specificity['overall_false_alarm']:.1f}%, ")
                    
                    dir_results.append(dir_info)
                else:
                    print(f"    No valid specificity data found")
                    
            except Exception as e:
                print(f"    Error processing directory {session_path.name}: {str(e)}")
        
        # Calculate combined accuracy values
        if total_r1_trials + total_r2_trials > 0:
            # Calculate overall accuracy across all directories in this session
            r1_accuracy = (total_r1_correct / total_r1_trials) if total_r1_trials > 0 else 0
            r2_accuracy = (total_r2_correct / total_r2_trials) if total_r2_trials > 0 else 0
            overall_accuracy = ((total_r1_correct + total_r2_correct) / 
                               (total_r1_trials + total_r2_trials)) if (total_r1_trials + total_r2_trials) > 0 else 0
            
            # Store combined session results
            session_result = {
                'session_id': session_id,
                'session_date': session_date,
                'overall_accuracy': overall_accuracy,
                'r1_accuracy': r1_accuracy,
                'r2_accuracy': r2_accuracy,
                'r1_trials': total_r1_trials,
                'r2_trials': total_r2_trials,
                'total_trials': total_r1_trials + total_r2_trials,
                'r1_correct': total_r1_correct,
                'r2_correct': total_r2_correct,
                'directory_count': len(session_paths),
                'directories': dir_results
            }
            
            print(f"  Combined results for Session {session_id}:")
            print(f"    R1: {total_r1_correct}/{total_r1_trials} ({r1_accuracy:.1%})")
            print(f"    R2: {total_r2_correct}/{total_r2_trials} ({r2_accuracy:.1%})")
            print(f"    Overall: {total_r1_correct + total_r2_correct}/{total_r1_trials + total_r2_trials} ({overall_accuracy:.1%})")
            
            results.append(session_result)
        else:
            print(f"  No valid trials found for Session {session_id}")
            results.append({
                'session_id': session_id,
                'session_date': session_date,
                'overall_accuracy': np.nan,
                'r1_accuracy': np.nan,
                'r2_accuracy': np.nan,
                'r1_trials': 0,
                'r2_trials': 0,
                'total_trials': 0,
                'r1_correct': 0,
                'r2_correct': 0,
                'directory_count': len(session_paths),
                'directories': dir_results
            })
    
    # Create DataFrame from combined results
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'directories'} for r in results])
    
    # Print summary
    print("\nSummary of Combined Session Accuracies:")
    print("=======================================")
    for _, row in results_df.iterrows():
        if pd.notna(row['overall_accuracy']):
            print(f"Session {row['session_id']} ({row['session_date']}): " 
                  f"Overall {row['overall_accuracy']:.2%}, "
                  f"R1 {row['r1_accuracy']:.2%} ({row['r1_trials']} trials), "
                  f"R2 {row['r2_accuracy']:.2%} ({row['r2_trials']} trials)")
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
    if not results_df.empty and not results_df['overall_accuracy'].isna().all():
        plot_accuracy(results_df.dropna(subset=['overall_accuracy']), plot_file, subject_id)
    else:
        print("No valid accuracy data to plot")
    
    return results_df

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("/Volumes/harris/hypnose/rawdata/sub-026_id-077")

    parser = argparse.ArgumentParser(description="Calculate and plot decision accuracy across sessions")
    parser.add_argument("subject_folder", help="Path to the subject's folder containing session data")
    parser.add_argument("--sessions", default=np.arange(31, 54), help="List of session IDs (optional)") 
    parser.add_argument("--stage", "--s", default=7, help="Stage to be analysed (optional)")
    parser.add_argument("--output", "-o", help="Path to save CSV output (optional)")
    parser.add_argument("--plot", "-p", help="Path to save plot image (optional)")
    args = parser.parse_args()
    
    main(args.subject_folder, args.sessions, args.stage, args.output, args.plot)