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
from src.analyse import analyze_session_folder


# def calculate_session_response_time(session_path):
#     """
#     Calculate decision response time for the session.
    
#     Args:
#         session_path (Path): Path to the session directory
        
#     Returns:
#         dict: Contains 'r1_correct_rt', 'r1_incorrect_rt', 'r1_avg_correct_rt', 'r1_avg_incorrect_rt', 'r1_avg_rt',
#         'r2_correct_rt', 'r2_incorrect_rt', 'r2_avg_correct_rt', 'r2_avg_incorrect_rt', 'r2_avg_rt', 
#         'hit_rt', 'false_alarm_rt' response time values
#     """
#     # Don't look for JSON files - send the session path directly to get_response_time
#     try:
#         # Get response times for this session
#         response_time = get_response_time(session_path)
        
#         if response_time is None:
#             print(f"No response time data available for {session_path}")
#             return {
#                 'r1_correct_rt': np.nan,
#                 'r1_incorrect_rt': np.nan,
#                 'r1_avg_correct_rt': np.nan,
#                 'r1_avg_incorrect_rt': np.nan,
#                 'r1_avg_rt': np.nan,
#                 'r2_correct_rt': np.nan,
#                 'r2_incorrect_rt': np.nan,
#                 'r2_avg_correct_rt': np.nan,
#                 'r2_avg_incorrect_rt': np.nan,
#                 'r2_avg_rt': np.nan,
#                 'hit_rt': np.nan,
#                 'false_alarm_rt': np.nan 
#             }
        
#         return {
#             'r1_correct_rt': response_time['r1_correct_rt'],
#             'r1_incorrect_rt': response_time['r1_incorrect_rt'],
#             'r1_avg_correct_rt': response_time['r1_avg_correct_rt'],
#             'r1_avg_incorrect_rt': response_time['r1_avg_incorrect_rt'],
#             'r1_avg_rt': response_time['r1_avg_rt'],
#             'r2_correct_rt': response_time['r2_correct_rt'],
#             'r2_incorrect_rt': response_time['r2_incorrect_rt'],
#             'r2_avg_correct_rt': response_time['r2_avg_correct_rt'],
#             'r2_avg_incorrect_rt': response_time['r2_avg_incorrect_rt'],
#             'r2_avg_rt': response_time['r2_avg_rt'],
#             'hit_rt': response_time['hit_rt'],
#             'false_alarm_rt': response_time['false_alarm_rt']
#         }
        
#     except Exception as e:
#         print(f"Error processing session {session_path}: {str(e)}")
#         return {
#             'r1_correct_rt': np.nan,
#             'r1_incorrect_rt': np.nan,
#             'r1_avg_correct_rt': np.nan,
#             'r1_avg_incorrect_rt': np.nan,
#             'r1_avg_rt': np.nan,
#             'r2_correct_rt': np.nan,
#             'r2_incorrect_rt': np.nan,
#             'r2_avg_correct_rt': np.nan,
#             'r2_avg_incorrect_rt': np.nan,
#             'r2_avg_rt': np.nan,
#             'hit_rt': np.nan,
#             'false_alarm_rt': np.nan 
#         }

def plot_session_response_time(results_df, output_file=None, subject_id=None):
    """
    Create a scatterplot of response time for each session.
    
    Args:
        results_df (DataFrame): Contains session_id, session_date, accuracy and response time data
        output_file (str): Path to save the plot, if provided
        subject_id (str): Subject ID to use in the plot title
    """
    # Convert date strings to datetime objects for better x-axis formatting
    results_df['date'] = pd.to_datetime(results_df['session_date'], format='%Y%m%d')
    results_df = results_df.sort_values('date')
    
    # Create the plot
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(12,6))
    ax = ax.ravel()
    
    # Plot each response time type
    ax[0].scatter(results_df['r1_correct_rt'], label='R1 correct RT', color='blue')
    ax[1].scatter(results_df['r2_correct_rt'], label='R2 correct RT', color='red')

    ax[2].scatter(results_df['r1_incorrect_rt'], label='R1 incorrect RT', color='blue')
    ax[3].scatter(results_df['r2_incorrect_rt'], label='R2 incorrect RT', color='red')

    
    
    # plt.scatter(results_df['date'], results_df['overall_accuracy'], 
    #            label='Overall Accuracy', marker='o', s=60, color='black')
    # plt.plot(results_df['date'], results_df['overall_accuracy'], 
    #         color='black', linestyle='-', alpha=0.7)
    
    # plt.scatter(results_df['date'], results_df['r1_accuracy'], 
    #            label='R1 Accuracy', marker='s', s=60, color='blue')
    # plt.plot(results_df['date'], results_df['r1_accuracy'], 
    #         color='blue', linestyle='--', alpha=0.7)
    
    # plt.scatter(results_df['date'], results_df['r2_accuracy'], 
    #            label='R2 Accuracy', marker='^', s=60, color='red')
    # plt.plot(results_df['date'], results_df['r2_accuracy'], 
    #         color='red', linestyle='-.', alpha=0.7)
    
    # Format the plot
    # plt.xlabel('Session Date')
    # plt.ylabel('Decision Accuracy')
    
    # Set title with subject ID if provided
    # if subject_id:
    #     plt.title(f'Decision Accuracy Across Sessions - sub-{subject_id}')
    # else:
    #     plt.title('Decision Accuracy Across Sessions')
    
    plt.ylim(0, 1.05)  # Assuming accuracy is a proportion between 0 and 1
    
    # Remove grid and add single line at 0.8
    plt.grid(False)
    plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.7)
    
    plt.legend()
    
    # Format the x-axis to show dates nicely
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.gcf().autofmt_xdate()
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Add session IDs as annotations
    # for i, row in results_df.iterrows():
    #     plt.annotate(f"Ses-{row['session_id']}", 
    #                 (row['date'], row['overall_accuracy']),
    #                 textcoords="offset points", 
    #                 xytext=(0,10), 
    #                 ha='center')
    
    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    plt.tight_layout()
    plt.show()

def main(session_folder, output_file=None, plot_file=None):
    """
    Process a subject folder and calculate response time for a given session.
    Combines results from multiple directories within the same session.
    Saves results to CSV file if output_file is provided.
    """

    results = analyze_session_folder(session_folder)
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'directories'} for r in results])
    print(results_df)
    if not results_df.empty and not results_df['response_time'].isna().all():
        plot_session_response_time(results_df.dropna(subset=['response_time']))
    else:
        print("No valid response time data to plot")

    return results_df
    
# def main(session_folder, output_file=None, plot_file=None):
#     """
#     Process a subject folder and calculate response time for all sessions.
#     Combines results from multiple directories within the same session.
#     Saves results to CSV file if output_file is provided.
#     """
#     session_path = Path(session_folder)
#     print(f"Processing session folder: {session_path}")
    
#     # Use utils.find_session_roots instead of the local function
#     session_roots = utils.find_session_roots(session_path)
    
#     if not session_roots:
#         print(f"No valid session directories found in {session_path}")
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

#             dir_info = {'directory': session_path.name}  # Start with directory name

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
                    
#                     dir_info.update({
#                         'r1_correct': accuracy['r1_correct'],
#                         'r1_trials': accuracy['r1_total'],
#                         'r1_accuracy': accuracy['r1_accuracy'],
#                         'r2_correct': accuracy['r2_correct'],
#                         'r2_trials': accuracy['r2_total'],
#                         'r2_accuracy': accuracy['r2_accuracy'],
#                         'overall_accuracy': accuracy['overall_accuracy']
#                     })
                    
#                     print(f"    R1: {accuracy['r1_correct']}/{accuracy['r1_total']} ({accuracy['r1_accuracy']:.1f}%), "
#                           f"R2: {accuracy['r2_correct']}/{accuracy['r2_total']} ({accuracy['r2_accuracy']:.1f}%)")

#                 else:
#                     print(f"    No valid accuracy data found")

#                 # Get response time data for this directory
#                 response_time = get_response_time(session_path)

#                 if response_time and response_time != {
#                     'r1_correct_rt': np.nan,
#                     'r1_incorrect_rt': np.nan,
#                     'r1_avg_correct_rt': np.nan,
#                     'r1_avg_incorrect_rt': np.nan,
#                     'r1_avg_rt': np.nan,
#                     'r2_correct_rt': np.nan,
#                     'r2_incorrect_rt': np.nan,
#                     'r2_avg_correct_rt': np.nan,
#                     'r2_avg_incorrect_rt': np.nan,
#                     'r2_avg_rt': np.nan,
#                     'hit_rt': np.nan,
#                     'false_alarm_rt': np.nan 
#                 }:
                     
#                     dir_info.update({
#                         'r1_correct_rt': response_time['r1_correct_rt'],
#                         'r1_incorrect_rt': response_time['r1_incorrect_rt'],
#                         'r1_avg_correct_rt': response_time['r1_avg_correct_rt'],
#                         'r1_avg_incorrect_rt': response_time['r1_avg_incorrect_rt'],
#                         'r1_avg_rt': response_time['r1_avg_rt'],
#                         'r2_correct_rt': response_time['r2_correct_rt'],
#                         'r2_incorrect_rt': response_time['r2_incorrect_rt'],
#                         'r2_avg_correct_rt': response_time['r2_avg_correct_rt'],
#                         'r2_avg_incorrect_rt': response_time['r2_avg_incorrect_rt'],
#                         'r2_avg_rt': response_time['r2_avg_rt'],
#                         'hit_rt': response_time['hit_rt'],
#                         'false_alarm_rt': response_time['false_alarm_rt']
#                     })

#                 else:
#                     print(f"    No valid response time data found")

#                 if len(dir_info) > 1:  # more than just 'directory'
#                     dir_results.append(dir_info)           
            
#             except Exception as e:
#                 print(f"    Error processing directory {session_path.name}: {str(e)}")
        
    #     # Calculate combined accuracy values
    #     if total_r1_trials + total_r2_trials > 0:
    #         # Calculate overall accuracy across all directories in this session
    #         r1_accuracy = (total_r1_correct / total_r1_trials) if total_r1_trials > 0 else 0
    #         r2_accuracy = (total_r2_correct / total_r2_trials) if total_r2_trials > 0 else 0
    #         overall_accuracy = ((total_r1_correct + total_r2_correct) / 
    #                            (total_r1_trials + total_r2_trials)) if (total_r1_trials + total_r2_trials) > 0 else 0
            
    #         # Store combined session results
    #         session_result = {
    #             'session_id': session_id,
    #             'session_date': session_date,
    #             'overall_accuracy': overall_accuracy,
    #             'r1_accuracy': r1_accuracy,
    #             'r2_accuracy': r2_accuracy,
    #             'r1_trials': total_r1_trials,
    #             'r2_trials': total_r2_trials,
    #             'total_trials': total_r1_trials + total_r2_trials,
    #             'r1_correct': total_r1_correct,
    #             'r2_correct': total_r2_correct,
    #             'directory_count': len(session_paths),
    #             'directories': dir_results
    #         }
            
    #         print(f"  Combined results for Session {session_id}:")
    #         print(f"    R1: {total_r1_correct}/{total_r1_trials} ({r1_accuracy:.1%})")
    #         print(f"    R2: {total_r2_correct}/{total_r2_trials} ({r2_accuracy:.1%})")
    #         print(f"    Overall: {total_r1_correct + total_r2_correct}/{total_r1_trials + total_r2_trials} ({overall_accuracy:.1%})")
            
    #         results.append(session_result)
    #     else:
    #         print(f"  No valid trials found for Session {session_id}")
    #         results.append({
    #             'session_id': session_id,
    #             'session_date': session_date,
    #             'overall_accuracy': np.nan,
    #             'r1_accuracy': np.nan,
    #             'r2_accuracy': np.nan,
    #             'r1_trials': 0,
    #             'r2_trials': 0,
    #             'total_trials': 0,
    #             'r1_correct': 0,
    #             'r2_correct': 0,
    #             'directory_count': len(session_paths),
    #             'directories': dir_results
    #         })
    
    # # Create DataFrame from combined results
    # results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'directories'} for r in results])
    
    # # Print summary
    # print("\nSummary of Combined Session Accuracies:")
    # print("=======================================")
    # for _, row in results_df.iterrows():
    #     if pd.notna(row['overall_accuracy']):
    #         print(f"Session {row['session_id']} ({row['session_date']}): " 
    #               f"Overall {row['overall_accuracy']:.2%}, "
    #               f"R1 {row['r1_accuracy']:.2%} ({row['r1_trials']} trials), "
    #               f"R2 {row['r2_accuracy']:.2%} ({row['r2_trials']} trials)")
    #     else:
    #         print(f"Session {row['session_id']} ({row['session_date']}): No valid trials found")
    
    # # Save results to CSV if requested
    # if output_file:
    #     results_df.to_csv(output_file, index=False)
    #     print(f"\nResults saved to {output_file}")
        
    #     # Also save detailed directory results
    #     detailed_output = Path(output_file).with_name(f"{Path(output_file).stem}_detailed.json")
    #     with open(detailed_output, 'w') as f:
    #         json.dump(results, f, indent=2)
    #     print(f"Detailed results saved to {detailed_output}")
    
    # # Generate plot if we have data
    # if not results_df.empty and not results_df['overall_accuracy'].isna().all():
    #     plot_accuracy(results_df.dropna(subset=['overall_accuracy']), plot_file, subject_id)
    # else:
    #     print("No valid accuracy data to plot")
    
    # return results_df
    # print(dir_results)
    # return dir_results

# TODO: for all sessions
# def main(subject_folder, output_file=None, plot_file=None):
#     """
#     Process a subject folder and calculate respomse time for all sessions.
#     Combines results from multiple directories within the same session.
#     Saves results to CSV file if output_file is provided.
#     """
#     session_path = Path(subject_folder)
#     print(f"Processing subject folder: {session_path}")
    
#     # Extract subject ID from the folder path
#     subject_id = session_path.name
#     if 'sub-' in subject_id:
#         subject_id = subject_id.split('sub-')[1]
    
#     # Use utils.find_session_roots instead of the local function
#     session_roots = utils.find_session_roots(session_path)
    
#     if not session_roots:
#         print(f"No valid session directories found in {session_path}")
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

#             dir_info = {'directory': session_path.name}  # Start with directory name

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
                    
#                     dir_info.update({
#                         'r1_correct': accuracy['r1_correct'],
#                         'r1_trials': accuracy['r1_total'],
#                         'r1_accuracy': accuracy['r1_accuracy'],
#                         'r2_correct': accuracy['r2_correct'],
#                         'r2_trials': accuracy['r2_total'],
#                         'r2_accuracy': accuracy['r2_accuracy'],
#                         'overall_accuracy': accuracy['overall_accuracy']
#                     })
                    
#                     print(f"    R1: {accuracy['r1_correct']}/{accuracy['r1_total']} ({accuracy['r1_accuracy']:.1f}%), "
#                           f"R2: {accuracy['r2_correct']}/{accuracy['r2_total']} ({accuracy['r2_accuracy']:.1f}%)")

#                 else:
#                     print(f"    No valid accuracy data found")

#                 # Get response time data for this directory
#                 response_time = get_response_time(session_path)

#                 if response_time and response_time != {
#                     'r1_correct_rt': np.nan,
#                     'r1_incorrect_rt': np.nan,
#                     'r1_avg_correct_rt': np.nan,
#                     'r1_avg_incorrect_rt': np.nan,
#                     'r1_avg_rt': np.nan,
#                     'r2_correct_rt': np.nan,
#                     'r2_incorrect_rt': np.nan,
#                     'r2_avg_correct_rt': np.nan,
#                     'r2_avg_incorrect_rt': np.nan,
#                     'r2_avg_rt': np.nan,
#                     'hit_rt': np.nan,
#                     'false_alarm_rt': np.nan 
#                 }:
                     
#                     dir_info.update({
#                         'r1_correct_rt': response_time['r1_correct_rt'],
#                         'r1_incorrect_rt': response_time['r1_incorrect_rt'],
#                         'r1_avg_correct_rt': response_time['r1_avg_correct_rt'],
#                         'r1_avg_incorrect_rt': response_time['r1_avg_incorrect_rt'],
#                         'r1_avg_rt': response_time['r1_avg_rt'],
#                         'r2_correct_rt': response_time['r2_correct_rt'],
#                         'r2_incorrect_rt': response_time['r2_incorrect_rt'],
#                         'r2_avg_correct_rt': response_time['r2_avg_correct_rt'],
#                         'r2_avg_incorrect_rt': response_time['r2_avg_incorrect_rt'],
#                         'r2_avg_rt': response_time['r2_avg_rt'],
#                         'hit_rt': response_time['hit_rt'],
#                         'false_alarm_rt': response_time['false_alarm_rt']
#                     })



#                 else:
#                     print(f"    No valid response time data found")

#                 if len(dir_info) > 1:  # more than just 'directory'
#                     dir_results.append(dir_info)           
            

#             except Exception as e:
#                 print(f"    Error processing directory {session_path.name}: {str(e)}")
        
#         # Calculate combined accuracy values
#         if total_r1_trials + total_r2_trials > 0:
#             # Calculate overall accuracy across all directories in this session
#             r1_accuracy = (total_r1_correct / total_r1_trials) if total_r1_trials > 0 else 0
#             r2_accuracy = (total_r2_correct / total_r2_trials) if total_r2_trials > 0 else 0
#             overall_accuracy = ((total_r1_correct + total_r2_correct) / 
#                                (total_r1_trials + total_r2_trials)) if (total_r1_trials + total_r2_trials) > 0 else 0
            
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
    
#     # Print summary
#     print("\nSummary of Combined Session Accuracies:")
#     print("=======================================")
#     for _, row in results_df.iterrows():
#         if pd.notna(row['overall_accuracy']):
#             print(f"Session {row['session_id']} ({row['session_date']}): " 
#                   f"Overall {row['overall_accuracy']:.2%}, "
#                   f"R1 {row['r1_accuracy']:.2%} ({row['r1_trials']} trials), "
#                   f"R2 {row['r2_accuracy']:.2%} ({row['r2_trials']} trials)")
#         else:
#             print(f"Session {row['session_id']} ({row['session_date']}): No valid trials found")
    
#     # Save results to CSV if requested
#     if output_file:
#         results_df.to_csv(output_file, index=False)
#         print(f"\nResults saved to {output_file}")
        
#         # Also save detailed directory results
#         detailed_output = Path(output_file).with_name(f"{Path(output_file).stem}_detailed.json")
#         with open(detailed_output, 'w') as f:
#             json.dump(results, f, indent=2)
#         print(f"Detailed results saved to {detailed_output}")
    
#     # Generate plot if we have data
#     if not results_df.empty and not results_df['overall_accuracy'].isna().all():
#         plot_accuracy(results_df.dropna(subset=['overall_accuracy']), plot_file, subject_id)
#     else:
#         print("No valid accuracy data to plot")
    
#     return results_df

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("/Volumes/harris/hypnose/rawdata/sub-026_id-077/ses-50_date-20250603")

    parser = argparse.ArgumentParser(description="Calculate and plot decision accuracy across sessions")
    parser.add_argument("subject_folder", help="Path to the subject's folder containing session data")
    parser.add_argument("--output", "-o", help="Path to save CSV output (optional)")
    parser.add_argument("--plot", "-p", help="Path to save plot image (optional)")
    args = parser.parse_args()
    
    main(args.subject_folder, args.output, args.plot)











































































































































































































