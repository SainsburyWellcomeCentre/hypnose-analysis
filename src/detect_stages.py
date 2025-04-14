import os
import json
import argparse
import pandas as pd
from pathlib import Path
from analysis import detect_stage

def find_session_roots(subject_folder):
    """
    Find all session root directories for a subject following the structure:
    subject_folder/ses-*_date-*/behav/*
    Returns a list of tuples: (session_id, session_date, session_path)
    """
    subject_path = Path(subject_folder)
    session_roots = []
    
    # Find all session directories following pattern 'ses-*_date-*'
    session_dirs = list(subject_path.glob('ses-*_date-*/behav/*'))
    
    for session_dir in session_dirs:
        if not session_dir.is_dir() or not (session_dir / "SessionSettings").exists():
            continue
            
        # Extract session ID and date from parent directory names
        parent_dir = session_dir.parent.parent.name
        try:
            # Parse the ses-X_date-YYYYMMDD format
            parts = parent_dir.split('_')
            session_id = parts[0].replace('ses-', '')
            session_date = parts[1].replace('date-', '')
            session_roots.append((session_id, session_date, session_dir))
        except (IndexError, AttributeError):
            print(f"Warning: Could not parse session information from {parent_dir}")
            session_roots.append(("unknown", "unknown", session_dir))
    
    # Sort session roots by session_id (numerically if possible)
    session_roots.sort(key=lambda x: int(x[0]) if x[0].isdigit() else float('inf'))
    
    return session_roots

def main(subject_folder, output_file=None):
    """
    Process a subject folder and detect stages for all sessions.
    Saves results to CSV file if output_file is provided.
    """
    subject_path = Path(subject_folder)
    print(f"Processing subject folder: {subject_path}")
    
    session_roots = find_session_roots(subject_path)
    
    if not session_roots:
        print(f"No valid session directories found in {subject_path}")
        return
    
    # Create a list to store results
    results = []
    
    # Process each session (already sorted by session ID)
    for session_id, session_date, session_path in session_roots:
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
    
    # Save results to CSV if requested
    if output_file:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect training stage of behavioral sessions")
    parser.add_argument("subject_folder", help="Path to the subject's folder containing session data")
    parser.add_argument("--output", "-o", help="Path to save CSV output (optional)")
    args = parser.parse_args()
    
    main(args.subject_folder, args.output)