import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import matplotlib.dates as mdates
from analysis import get_decision_accuracy
import utils

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

def plot_accuracy(results_df, output_file=None):
    """
    Create a scatterplot of decision accuracy across sessions.
    
    Args:
        results_df (DataFrame): Contains session_id, session_date, and accuracy data
        output_file (str): Path to save the plot, if provided
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
    plt.title('Decision Accuracy Across Sessions')
    plt.ylim(0, 1.05)  # Assuming accuracy is a proportion between 0 and 1
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Format the x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
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

def main(subject_folder, output_file=None, plot_file=None):
    """
    Process a subject folder and calculate decision accuracy for all sessions.
    Saves results to CSV file if output_file is provided.
    """
    subject_path = Path(subject_folder)
    print(f"Processing subject folder: {subject_path}")
    
    # Use utils.find_session_roots instead of the local function
    session_roots = utils.find_session_roots(subject_path)
    
    if not session_roots:
        print(f"No valid session directories found in {subject_path}")
        return
    
    # Create a list to store results
    results = []
    
    # Process each session (already sorted by session ID)
    for session_id, session_date, session_path in session_roots:
        try:
            accuracy = calculate_session_accuracy(session_path)
            print(f"Session ID: {session_id}, Date: {session_date}, Path: {session_path.name}")
            print(f"  Overall Accuracy: {accuracy['overall']:.2%}")
            print(f"  R1 Accuracy: {accuracy['r1']:.2%} ({accuracy['r1_trials']} trials)")
            print(f"  R2 Accuracy: {accuracy['r2']:.2%} ({accuracy['r2_trials']} trials)")
            
            results.append({
                'session_id': session_id,
                'session_date': session_date,
                'session_path': str(session_path),
                'overall_accuracy': accuracy['overall'],
                'r1_accuracy': accuracy['r1'],
                'r2_accuracy': accuracy['r2'],
                'r1_trials': accuracy['r1_trials'],
                'r2_trials': accuracy['r2_trials'],
                'total_trials': accuracy['total_trials']
            })
        except Exception as e:
            print(f"Error processing session {session_path}: {str(e)}")
            results.append({
                'session_id': session_id,
                'session_date': session_date,
                'session_path': str(session_path),
                'overall_accuracy': np.nan,
                'r1_accuracy': np.nan,
                'r2_accuracy': np.nan,
                'r1_trials': 0,
                'r2_trials': 0,
                'total_trials': 0,
                'error': str(e)
            })
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Print summary
    print("\nSummary of Session Accuracies:")
    print("==============================")
    for _, row in results_df.iterrows():
        if pd.notna(row['overall_accuracy']):
            print(f"Session {row['session_id']} ({row['session_date']}): " 
                  f"Overall {row['overall_accuracy']:.2%}, "
                  f"R1 {row['r1_accuracy']:.2%}, "
                  f"R2 {row['r2_accuracy']:.2%}")
        else:
            print(f"Session {row['session_id']} ({row['session_date']}): Error - {row.get('error', 'Unknown error')}")
    
    # Save results to CSV if requested
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    
    # Generate plot
    plot_accuracy(results_df.dropna(subset=['overall_accuracy']), plot_file)
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and plot decision accuracy across sessions")
    parser.add_argument("subject_folder", help="Path to the subject's folder containing session data")
    parser.add_argument("--output", "-o", help="Path to save CSV output (optional)")
    parser.add_argument("--plot", "-p", help="Path to save plot image (optional)")
    args = parser.parse_args()
    
    main(args.subject_folder, args.output, args.plot)