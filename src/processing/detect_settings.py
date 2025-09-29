import re
import sys
import argparse
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import zoneinfo
import src.session_utils as utils  # Changed from relative to absolute import
import harp
import yaml
from functools import reduce
from collections import defaultdict

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
    sequence_obj = None
    if not hasattr(metadata, 'sequences') or (hasattr(metadata, 'sequences') and not metadata.sequences):  # separate files
        try: 
            sequence_schema = utils.load_json(metadata_reader, path_root/"Schema") # TODO
            sequence_metadata = sequence_schema['metadata'].iloc[0]
            
            minimumSamplingTime = sequence_metadata['sequences'][0][0]['rewardConditions'][0]['definition'][0][0]['minimumSamplingTime']
            completionRequiresEngagement = sequence_metadata['sequences'][0][0]['completionRequiresEngagement']
            responseTime = sequence_metadata['sequences'][0][0].get('responseTime') 
            sequences_obj = sequence_metadata.get('sequences', None)

        except:
            try:
                schema_filename = metadata.metadata.initialSequence.split("/")[-1]
                with open(path_root/"Schema"/schema_filename, 'r') as file:
                    sequence_schema = yaml.load(file, Loader=yaml.SafeLoader)
                    
                    minimumSamplingTime = sequence_schema['sequences'][0][0]['rewardConditions'][0]['definition'][0][0]['minimumSamplingTime']
                    completionRequiresEngagement = sequence_schema['sequences'][0][0]['completionRequiresEngagement']
                    responseTime = sequence_schema['sequences'][0][0].get('responseTime') 
                    sequences_obj = sequence_schema.get('sequences', None)
            except Exception as e:
                print(f"Error loading session schema: {e}")
    else:  
        try:
            minimumSamplingTime = metadata.sequences[0][0]['rewardConditions'][0]['definition'][0][0]['minimumSamplingTime']
            completionRequiresEngagement = metadata.sequences[0][0]['completionRequiresEngagement']
            responseTime = metadata.sequences[0][0].get('responseTime')
            sequences_obj = metadata.sequences
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
    schema_settings['responseTime'] = responseTime
    

    # --- Infer Hidden Rule index and odors from schema sequences ---
    def _ci_get(d, key):
        # Case-insensitive dict get
        if not isinstance(d, dict):
            return None
        lk = str(key).lower()
        for k, v in d.items():
            if str(k).lower() == lk:
                return v
        return None

    def _flatten_list(x):
        # Flatten arbitrarily nested lists to a flat list
        out = []
        stack = [x]
        while stack:
            cur = stack.pop()
            if isinstance(cur, list):
                stack.extend(cur)
            else:
                out.append(cur)
        return out

    def _iter_definitions(sequences):
        # Yield each rewardConditions[0].definition list from the sequences tree
        if sequences is None:
            return
        # sequences is typically a list of blocks; each block is a list of segments (dicts)
        for block in sequences:
            segs = block if isinstance(block, list) else [block]
            for seg in segs:
                if not isinstance(seg, dict):
                    continue
                rc = _ci_get(seg, "rewardConditions")
                if isinstance(rc, list) and rc and isinstance(rc[0], dict):
                    definition = _ci_get(rc[0], "definition")
                    if isinstance(definition, list):
                        yield definition

    def _command_name(obj):
        if not isinstance(obj, dict):
            return None
        for k in ("command", "name", "odor", "odour"):
            v = _ci_get(obj, k)
            if isinstance(v, str):
                return v
        return None

    def _is_rewarded(obj):
        if not isinstance(obj, dict):
            return False
        v = _ci_get(obj, "rewarded")
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1")
        if isinstance(v, (int, float)):
            return v != 0
        return False

    hidden_rule_index_inferred = None
    hidden_rule_odors_inferred = []
    try:
        # Accumulate rewarded odors per position across all blocks
        rewarded_by_pos = defaultdict(set)
        for definition in _iter_definitions(sequences_obj):
            for pos_idx, choices in enumerate(definition):
                items = _flatten_list(choices) if isinstance(choices, list) else [choices]
                for it in items:
                    if _is_rewarded(it):
                        name = _command_name(it)
                        if name:
                            rewarded_by_pos[pos_idx].add(name)
        # Choose position(s) with >= 2 rewarded odors as Hidden Rule candidates
        candidates = [(idx, sorted(list(odors))) for idx, odors in rewarded_by_pos.items() if len(odors) >= 2]
        if candidates:
            # Prefer the first candidate deterministically
            candidates.sort(key=lambda x: x[0])
            hidden_rule_index_inferred, hidden_rule_odors_inferred = candidates[0]
    except Exception as e:
        # Silent fallback; leave as None / []
        pass

    schema_settings['hiddenRuleIndexInferred'] = hidden_rule_index_inferred
    schema_settings['hiddenRuleOdorsInferred'] = hidden_rule_odors_inferred
     
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