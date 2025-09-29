import re
import sys
import argparse
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import zoneinfo
import src.session_utils as utils
import harp
import yaml
from functools import reduce

def detect_stage(root):
    """
    Extracts stage information from sequence schema.
    Returns stage_name and hidden_rule_index.
    """
    
    path_root = Path(root)
    metadata_reader = utils.SessionData()
    
    try:
        # Try loading from SessionSettings first
        session_settings = utils.load_json(metadata_reader, path_root/"SessionSettings")
        metadata = session_settings.iloc[0]['metadata']
        
        # Check if sequences exist in session settings
        if hasattr(metadata, 'sequences') and metadata.sequences:
            sequences = metadata.sequences
        else:
            # Load from Schema files
            try:
                sequence_schema = utils.load_json(metadata_reader, path_root/"Schema") 
                sequence_metadata = sequence_schema['metadata'].iloc[0]
                sequences = sequence_metadata['sequences']
            except:
                # Fallback to YAML schema file
                schema_filename = metadata.metadata.initialSequence.split("/")[-1]
                with open(path_root/"Schema"/schema_filename, 'r') as file:
                    sequence_schema = yaml.load(file, Loader=yaml.SafeLoader)
                    sequences = sequence_schema['sequences']
        
        # Extract the first sequence name
        stage_name = None
        hidden_rule_index = None
        
        if isinstance(sequences, list) and len(sequences) > 0:
            # Navigate through the nested structure to find the first 'name'
            for seq_group in sequences:
                if isinstance(seq_group, list):
                    for seq in seq_group:
                        if isinstance(seq, dict) and 'name' in seq:
                            stage_name = seq['name']
                            break
                elif isinstance(seq_group, dict) and 'name' in seq_group:
                    stage_name = seq_group['name']
                    break
                
                if stage_name:
                    break
        
        # Extract hidden rule index if present (format: name_StageX where X is the index)
        if stage_name:
            match = re.search(r'_Location(\d+)', stage_name, re.IGNORECASE)
            if match:
                hidden_rule_index = int(match.group(1))
            
            print(f"Detected stage: {stage_name}")
            if hidden_rule_index is not None:
                print(f"Hidden rule index: {hidden_rule_index}")
            else:
                print("No hidden rule index found")
            
            return {
                'stage_name': stage_name,
                'hidden_rule_index': hidden_rule_index
            }
        else:
            print("No stage name found in sequences")
            return {
                'stage_name': "Unknown",
                'hidden_rule_index': None
            }
            
    except Exception as e:
        print(f"Error detecting stage: {e}")
        return {
            'stage_name': "Unknown", 
            'hidden_rule_index': None
        }


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
        sys.exit(1)
    
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
            stage_info = detect_stage(session_dir)
            print(f"Stage: {stage_info['stage_name']}")
            if stage_info['hidden_rule_index'] is not None:
                print(f"Hidden rule index: {stage_info['hidden_rule_index']}")
        except Exception as e:
            print(f"Something went wrong with {session_dir}: {e}. Continuing to next session.")
            continue