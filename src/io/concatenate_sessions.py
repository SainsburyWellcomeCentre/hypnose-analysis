import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import harp
import re
from collections import defaultdict

from src import utils

# Filter out specific warnings
warnings.filterwarnings(
    "ignore", 
    message="Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated",
    category=FutureWarning
)

def remove_uncompleted_trials(session_folder):
    """
    Remove uncompleted trials, where odour sampling started but no decision was made at the end of the session
    
    Parameters:
    -----------
    session_folder : str or Path
        Path to the session folder (e.g., ses-YYY_date-YYYYMMDD/)
    """

    session_folder = Path(session_folder)
    behav_folder = session_folder / "behav"
    
    if not behav_folder.exists():
        print(f"No behavior folder found at {behav_folder}")
        return None
    
    # Collect all session directories
    session_dirs = [d for d in behav_folder.iterdir() if d.is_dir()]
    print(f"Found {len(session_dirs)} behavioral sessions in {behav_folder}")
    
    # Process each session
    for session_dir in sorted(session_dirs):
        print(f"\nProcessing session: {session_dir.name}")
        
        # Only process if session settings exist
        if not (session_dir / "SessionSettings").exists():
            print(f"No SessionSettings found in {session_dir.name}, skipping")
            continue

        # Load session settings
        metadata_reader = utils.SessionData()
        try:
            session_settings = utils.load_json(metadata_reader, session_dir / "SessionSettings")
            
            # Detect stage if not already set TODO
            if stage is None:
                stage = detect_stage(session_dir)
        except Exception as e:
            print(f"Error loading session settings: {e}")
            continue
        
        # Create analyzer instance: TODO
        try:
            analyzer = RewardAnalyser(session_settings)
        except Exception as e:
            print(f"Error creating analyzer: {e}")
            continue
    
        # Create readers
        # behavior_reader = harp.reader.create_reader('device_schemas/behavior.yml', epoch=harp.io.REFERENCE_EPOCH)
        # olfactometer_reader = harp.reader.create_reader('device_schemas/olfactometer.yml', epoch=harp.io.REFERENCE_EPOCH)
        
# def concatenate_sessions(session_folder):
