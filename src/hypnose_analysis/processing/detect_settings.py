import re
import sys
import argparse
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import zoneinfo
import hypnose_analysis.session_utils as utils
import harp
import yaml
from functools import reduce
from collections import defaultdict

def detect_settings(root):
    """
    Handles structure of settings and schema files.
    Extracts parameters from schema settings.
    """

    path_root = Path(root)
    metadata_reader = utils.SessionData()
    session_settings = utils.load_json(metadata_reader, path_root/"SessionSettings")
    
    metadata = session_settings.iloc[0]['metadata']

    # Handle schema and session settings format
    schema_settings = {}
    
    # Extract minimum sampling times per odor and other parameters
    minimumSamplingTime_by_odor = {}
    completionRequiresEngagement = None
    responseTime = None
    sequences_obj = None
    
    if not hasattr(metadata, 'sequences') or (hasattr(metadata, 'sequences') and not metadata.sequences):
        # Separate Schema files case
        sequence_schema = utils.load_json(metadata_reader, path_root/"Schema")
        sequence_metadata = sequence_schema['metadata'].iloc[0]
        sequences_obj = sequence_metadata.get('sequences', None)
    else:
        # Inline sequences case
        sequences_obj = metadata.sequences
    
    # Extract parameters from first segment (same for all)
    try:
        first_segment = sequences_obj[0][0] if sequences_obj else None
        if first_segment:
            completionRequiresEngagement = first_segment.get('completionRequiresEngagement')
            responseTime = first_segment.get('responseTime')
    except:
        pass
    
    # Extract minimum sampling time per odor by iterating through all definitions
    target_odors = {'OdorA', 'OdorB', 'OdorC', 'OdorD', 'OdorE', 'OdorF', 'OdorG'}
    found_odors = set()
    sequence_length_index = None
    sequence_length_from_def = None
    max_index_seen = None
    
    try:
        if sequences_obj:
            for block in sequences_obj:
                segs = block if isinstance(block, list) else [block]
                for seg in segs:
                    if not isinstance(seg, dict):
                        continue
                    
                    rc = seg.get('rewardConditions')
                    if isinstance(rc, list) and rc and isinstance(rc[0], dict):
                        definition = rc[0].get('definition')
                        if isinstance(definition, list):
                            # definition is a list of position lists
                            # Track length from definition as fallback for sequence length
                            if sequence_length_from_def is None:
                                sequence_length_from_def = len(definition)
                            for position_choices in definition:
                                items = position_choices if isinstance(position_choices, list) else [position_choices]
                                for item in items:
                                    if isinstance(item, dict):
                                        odor = item.get('command')
                                        if odor in target_odors and odor not in found_odors:
                                            mst = item.get('minimumSamplingTime')
                                            if mst is not None:
                                                minimumSamplingTime_by_odor[odor] = mst
                                                found_odors.add(odor)
                                            try:
                                                idx_val = item.get('index')
                                                if isinstance(idx_val, (int, float)):
                                                    max_index_seen = int(idx_val) if max_index_seen is None else max(max_index_seen, int(idx_val))
                                            except Exception:
                                                pass
                                
                                # Early exit if all odors found
                                if len(found_odors) == len(target_odors):
                                    break
                            
                            if len(found_odors) == len(target_odors):
                                break
                    
                    # Capture explicit sequenceLengthIndex if present
                    try:
                        if sequence_length_index is None and isinstance(seg.get('sequenceLengthIndex'), (int, float)):
                            sequence_length_index = int(seg.get('sequenceLengthIndex'))
                    except Exception:
                        pass

                    if len(found_odors) == len(target_odors):
                        break
                
                if len(found_odors) == len(target_odors):
                    break
    except Exception as e:
        print(f"Error extracting minimum sampling times: {e}")

    # Resolve sequence length with priority: explicit index -> max item index -> definition length
    sequence_length = None
    sequence_length_source = None

    if sequence_length_index is not None:
        sequence_length = sequence_length_index + 1
        sequence_length_source = 'sequenceLengthIndex'
    elif max_index_seen is not None:
        sequence_length = max_index_seen + 1
        sequence_length_source = 'maxItemIndex'
    elif sequence_length_from_def is not None:
        sequence_length = sequence_length_from_def
        sequence_length_source = 'definitionLength'

    schema_settings['sequenceLengthIndex'] = sequence_length_index
    schema_settings['sequenceLengthMaxIndex'] = max_index_seen
    schema_settings['sequenceLengthFromDefinition'] = sequence_length_from_def
    schema_settings['sequenceLengthSource'] = sequence_length_source
    schema_settings['sequenceLength'] = sequence_length
    
    schema_settings['minimumSamplingTime_by_odor'] = minimumSamplingTime_by_odor
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

    hidden_rule_indices_inferred = []
    hidden_rule_odors_inferred = []
    try:
        # Find the maximum position index across all definitions
        max_position = -1
        for definition in _iter_definitions(sequences_obj):
            max_position = max(max_position, len(definition) - 1)
        
        # Accumulate rewarded odors per position, excluding the last position
        rewarded_by_pos = defaultdict(set)
        for definition in _iter_definitions(sequences_obj):
            for pos_idx, choices in enumerate(definition):
                # Skip the last position
                if pos_idx >= max_position:
                    continue
                    
                items = _flatten_list(choices) if isinstance(choices, list) else [choices]
                for it in items:
                    if _is_rewarded(it):
                        name = _command_name(it)
                        if name:
                            rewarded_by_pos[pos_idx].add(name)
        
        # Find all positions with exactly 2 rewarded odors (hidden rule candidates)
        candidates = [(idx, sorted(list(odors))) for idx, odors in rewarded_by_pos.items() if len(odors) == 2]
        
        if candidates:
            candidates.sort(key=lambda x: x[0])
            hidden_rule_indices_inferred = [idx for idx, _ in candidates]
            odors_union = []
            seen = set()
            for _, odors in candidates:
                for odor in odors:
                    if odor not in seen:
                        seen.add(odor)
                        odors_union.append(odor)
            hidden_rule_odors_inferred = odors_union
    except Exception as e:
        # Silent fallback; leave as empty list
        pass

    schema_settings['hiddenRuleIndicesInferred'] = hidden_rule_indices_inferred
    schema_settings['hiddenRuleIndexInferred'] = hidden_rule_indices_inferred[0] if hidden_rule_indices_inferred else None
    schema_settings['hiddenRuleOdorsInferred'] = hidden_rule_odors_inferred
     
    return session_settings, schema_settings

if __name__ == "__main__":
    # Deal with inputs
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