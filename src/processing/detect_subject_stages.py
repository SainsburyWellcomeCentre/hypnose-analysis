import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from src.processing.detect_stage import detect_stage
import src.utils as utils  # Changed from relative to absolute import

def main(subject_folder, output_file=None):
    """
    Process a subject folder and detect stages for all sessions.
    Saves results to CSV file if output_file is provided.
    """
    subject_path = Path(subject_folder)
    print(f"Processing subject folder: {subject_path}")
    
    session_roots = utils.find_session_roots(subject_path)
    
    if not session_roots:
        print(f"No valid session directories found in {subject_path}")
        return
    
    # Load existing results if output_file exists
    existing_results = pd.DataFrame()
    processed_session_ids = set()
    
    if output_file and Path(output_file).exists():
        print(f"Loading existing results from {output_file}")
        existing_results = pd.read_csv(output_file)
        if 'session_id' in existing_results.columns:
            processed_session_ids = set(existing_results['session_id'].astype(str))
    
    # Create a list to store results
    results = []
    
    # Process each session (already sorted by session ID)
    for session_id, session_date, session_path in session_roots:
        if str(session_id) in processed_session_ids:
            print(f"Skipping already processed session {session_id}")
            continue
        try:
            stage = detect_stage(session_path)
            print(f"Session ID: {session_id}, Date: {session_date}, Path: {session_path.name}, Stage: {stage}")
            
            results.append({
                'session_id': session_id,
                'session_date': session_date,
                'session_path': str(session_path),
                'stage': stage
            })
        except Exception as e:
            print(f"Error processing session {session_path}: {str(e)}")
            results.append({
                'session_id': session_id,
                'session_date': session_date,
                'session_path': str(session_path),
                'stage': 'Error',
                'error': str(e)
            })
    
    # Print summary in ascending session number order (sorted above)
    print("\nSummary of Session Stages:")
    print("==========================")
    for result in results:
        print(f"Session {result['session_id']}: Stage {result['stage']} ({result['session_date']})")
    
    # Append to output file
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        new_results_df = pd.DataFrame(results)

        if not existing_results.empty:
            combined_df = pd.concat([existing_results, new_results_df], ignore_index=True)
        else:
            combined_df = new_results_df

        combined_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("/Volumes/harris/hypnose/rawdata/sub-020_id-072/")
        sys.argv.append("--output")
        sys.argv.append("/Volumes/harris/Athina/hypnose/analysis/sub-020_id-072/stages.csv")

    parser = argparse.ArgumentParser(description="Detect training stage of behavioral sessions")
    parser.add_argument("subject_folder", help="Path to the subject's folder containing session data")
    parser.add_argument("--output", "-o", help="Path to save CSV output (optional)")
    args = parser.parse_args()
    
    main(args.subject_folder, args.output)