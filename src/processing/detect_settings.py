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

def detect_settings(root):
    """
    Handles structure of settings and schema files.
    Extracts parameters from schema settings.

    # TODO currently the script assumes the schema settings are the same for all odours and sequences
    """

    path_root = Path(root)
    metadata_reader = utils.SessionData()
    session_settings = utils.load_json(metadata_reader, path_root/"SessionSettings")
    minimumSamplingTime = None
    metadata = session_settings.iloc[0]['metadata']

    # Handle schema and session settings format
    schema_settings = {}
    if not hasattr(metadata, 'sequences') or (hasattr(metadata, 'sequences') and not metadata.sequences):  # separate files
        try: 
            sequence_schema = utils.load_json(metadata_reader, path_root/"Schema") # TODO
            sequence_metadata = sequence_schema['metadata'].iloc[0]
            
            minimumSamplingTime = sequence_metadata['sequences'][0][0]['rewardConditions'][0]['definition'][0][0]['minimumSamplingTime']
            completionRequiresEngagement = sequence_metadata['sequences'][0][0]['completionRequiresEngagement']
        except:
            try:
                schema_filename = metadata.metadata.initialSequence.split("/")[-1]
                with open(path_root/"Schema"/schema_filename, 'r') as file:
                    sequence_schema = yaml.load(file, Loader=yaml.SafeLoader)
                    
                    minimumSamplingTime = sequence_schema['sequences'][0][0]['rewardConditions'][0]['definition'][0][0]['minimumSamplingTime']
                    completionRequiresEngagement = sequence_schema['sequences'][0][0]['completionRequiresEngagement']
            except Exception as e:
                print(f"Error loading session schema: {e}")
    else:  
        try:
            minimumSamplingTime = metadata.sequences[0][0]['rewardConditions'][0]['definition'][0][0]['minimumSamplingTime']
            completionRequiresEngagement = metadata.sequences[0][0]['completionRequiresEngagement']
        except Exception as e:
            print(f"minimumSamplingTime was not a parameter in this session: {e}")
            try:
                minimumSamplingTime = metadata.sequences[0][0]['minimumEngagementTime']
                completionRequiresEngagement = metadata.sequences[0][0]['completionRequiresEngagement']
            except Exception as e:
                print(f"minimumEngagementTime was not a parameter in this session either: {e}")
            
                minimumSamplingTime = 0 
                completionRequiresEngagement = None
    
    
    schema_settings['minimumSamplingTime'] = minimumSamplingTime
    schema_settings['sampleOffsetTime'] = metadata.metadata.sampleOffsetTime
    schema_settings['completionRequiresEngagement'] = completionRequiresEngagement
    
    return session_settings, schema_settings

if __name__ == "__main__":
    # Deal with inputs
    if len(sys.argv) == 1:
        # sys.argv.append("/Volumes/harris/hypnose/rawdata/sub-025_id-076/ses-81_date-20250715")
        sys.argv.append("/Volumes/harris/hypnose/rawdata/sub-020_id-072/ses-36_date-20250513")

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

        session_settings = detect_settings(session_dir) 