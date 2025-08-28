import re
import sys
import argparse
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import zoneinfo
import src.utils as utils  # Changed from relative to absolute import
import harp
import yaml
from functools import reduce

def detect_stage(root):
    """
    Extracts the stage from metadata if available.
    Handles structure of settings and schema files.
    Handles nested structure of sequences in metadata.
    """

    path_root = Path(root)
    metadata_reader = utils.SessionData()
    session_settings = utils.load_json(metadata_reader, path_root/"SessionSettings")
    stage_found = None
    metadata = session_settings.iloc[0]['metadata']

    # Handle schema and session settings format
    if not hasattr(metadata, 'sequences') or (hasattr(metadata, 'sequences') and not metadata.sequences):  # separate files
        try: 
            sequence_schema = utils.load_json(metadata_reader, path_root/"Schema") 
            sequence_metadata = sequence_schema['metadata'].iloc[0]
            sequences = sequence_metadata['sequences']
        except:
            try:
                schema_filename = metadata.metadata.initialSequence.split("/")[-1]
                with open(path_root/"Schema"/schema_filename, 'r') as file:
                    sequence_schema = yaml.load(file, Loader=yaml.SafeLoader)
                    sequences = sequence_schema['sequences']
            except Exception as e:
                raise RuntimeError(f"Error loading session schema in {path_root}: {e}")
    else:
        sequences = metadata.sequences
        
    # Handle the nested list structure
    if isinstance(sequences, list):
        # Iterate through outer list
        for seq_group in sequences:
            if isinstance(seq_group, list):
                # Iterate through inner list
                for seq in seq_group:
                    if isinstance(seq, dict) and 'name' in seq:
                        print(f"Found sequence name: {seq['name']}")
                        match = re.search(r'_Stage(\d+)', seq['name'], re.IGNORECASE)
                        if match:
                            stage_number = int(match.group(1))
                            if 'FreeRun' in seq['name']:
                                stage_found = 8 + 0.1 * stage_number
                            elif 'Doubles' in seq['name']:
                                stage_found = 9 + 0.1 * stage_number
                            elif 'Triples' in seq['name']:
                                stage_found = 10 + 0.1 * stage_number
                            elif 'Quadruple' in seq['name']:
                                stage_found = 11 + 0.1 * stage_number
                            elif 'Quintuple' in seq['name']:
                                stage_found = 12 + 0.1 * stage_number
                            else:
                                stage_found = stage_number
                            return stage_found
                        else:
                            if 'Doubles' in seq['name']:
                                stage_found = 9
            elif isinstance(seq_group, dict) and 'name' in seq_group:
                # Handle case where outer list contains dicts directly
                print(f"Found sequence name: {seq_group['name']}")
                match = re.search(r'_Stage(\d+)', seq_group['name'], re.IGNORECASE)
                if match:
                    stage_number = int(match.group(1))
                    if 'FreeRun' in seq_group['name']:
                        stage_found = 8 + 0.1 * stage_number
                    elif 'Doubles' in seq_group['name']:
                        stage_found = 9 + 0.1 * stage_number
                    elif 'Triples' in seq_group['name']:
                        stage_found = 10 + 0.1 * stage_number
                    elif 'Quadruple' in seq_group['name']:
                        stage_found = 11 + 0.1 * stage_number
                    elif 'Quintuple' in seq_group['name']:
                        stage_found = 12 + 0.1 * stage_number
                    else:
                        stage_found = stage_number
                    return stage_found
                else:
                    if 'Doubles' in seq_group['name']:
                        stage_found = 9
                        return stage_found
            
    return stage_found if stage_found else "Unknown"


if __name__ == "__main__":
    # Deal with inputs
    if len(sys.argv) == 1:
        sys.argv.append("/Volumes/harris/hypnose/rawdata/sub-020_id-072/ses-90_date-20250728")

    parser = argparse.ArgumentParser(description="Get stage of a behavioral session")
    parser.add_argument("session_folder", help="Path to the session folder (e.g., sub-XXX/ses-YYY_date-YYYYMMDD)")
    
    args = parser.parse_args()
    
    # Check session folder structure
    session_folder = Path(args.session_folder)
    behav_folder = session_folder / "behav"
    
    if not behav_folder.exists():
        print(f"No behavior folder found at {behav_folder}")
    
    # Collect all session directories
    session_dirs = [d for d in behav_folder.iterdir() if d.is_dir()]
    print(f"Found {len(session_dirs)} behavioral sessions in {behav_folder}")

    # Get stage for each session directory
    for session_dir in sorted(session_dirs):
        print(f"\nProcessing session: {session_dir.name}")
        
         # Only process if session settings exist
        if not (session_dir / "SessionSettings").exists():
            print(f"No SessionSettings found in {session_dir.name}, skipping")
            continue

        try:
            stage = detect_stage(session_dir) 
        except Exception as e:
            print(f"Something went wrong with {session_dir}: {e}. Continuing to next session.")
            continue
