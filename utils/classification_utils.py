import sys
import os
project_root = os.path.abspath("")
if project_root not in sys.path:
    sys.path.append(project_root)
import os
import json
from dotmap import DotMap
import pandas as pd
import numpy as np
import math
from pathlib import Path
from glob import glob
from aeon.io.reader import Reader, Csv
import aeon.io.api as api
import re
import yaml
import harp
import datetime
from datetime import timezone
import zoneinfo
import src.processing.detect_settings as detect_settings
from datetime import datetime, timezone, date
from collections import defaultdict
from bisect import bisect_left, bisect_right
from typing import Iterable, Optional
import io
import contextlib
from collections.abc import Mapping
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import cv2 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from IPython import get_ipython




# ============== General Utility Functions and Class Definitions =======================================
class SessionData(Reader):
    """Extracts metadata information from a settings .jsonl file."""

    def __init__(self, pattern="Metadata"):
        super().__init__(pattern, columns=["metadata"], extension="jsonl")

    def read(self, file):
        """Returns metadata for the specified epoch."""
        with open(file) as fp:
            metadata = [json.loads(line) for line in fp] 

        data = {
            "metadata": [DotMap(entry['value']) for entry in metadata]
        }
        timestamps = [api.aeon(entry['seconds']) for entry in metadata]

        return pd.DataFrame(data, index=timestamps, columns=self.columns)

class Video(Csv):
    """Extracts video frame metadata."""

    def __init__(self, pattern="VideoData"):
        super().__init__(pattern, columns=["hw_counter", "hw_timestamp", "_frame", "_path", "_epoch"])
        self._rawcolumns = ["Time"] + self.columns[0:2]

    def read(self, file):
        """Reads video metadata from the specified file."""
        data = pd.read_csv(file, header=0, names=self._rawcolumns)
        data["_frame"] = data.index
        data["_path"] = os.path.splitext(file)[0] + ".avi"
        data["_epoch"] = file.parts[-3]
        data["Time"] = data["Time"].transform(lambda x: api.aeon(x))
        data.set_index("Time", inplace=True)
        return data
    
class TimestampedCsvReader(Csv):
    def __init__(self, pattern, columns):
        super().__init__(pattern, columns, extension="csv")
        self._rawcolumns = ["Time"] + columns

    def read(self, file):
        data = pd.read_csv(file, header=0, names=self._rawcolumns)
        data["Seconds"] = data["Time"]
        data["Time"] = data["Time"].transform(lambda x: api.aeon(x))
        data.set_index("Time", inplace=True)
        return data
    

def vprint(verbose: bool, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def load_json(reader: SessionData, root: Path) -> pd.DataFrame:
    root = Path(root)
    pattern = f"{root.joinpath(root.name)}_*.{reader.extension}"
    files = sorted(glob(pattern))
    chunks = []
    for file in files:
        try:
            df = reader.read(Path(file))
            if df is None or (hasattr(df, "empty") and df.empty):
                continue
            chunks.append(df)
        except Exception:
            # skip bad file
            continue
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, axis=0)
    try:
        out = out.sort_index()
    except Exception:
        pass
    return out


def load(reader: Reader, root: Path) -> pd.DataFrame:
    root = Path(root)
    pattern = f"{root.joinpath(root.name)}_{reader.register.address}_*.bin"
    files = sorted(glob(pattern))
    chunks = []
    for file in files:
        try:
            df = reader.read(file)
            if df is None or (hasattr(df, "empty") and df.empty):
                continue
            chunks.append(df)
        except Exception:
            # skip bad file
            continue
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, axis=0)
    try:
        out = out.sort_index()
    except Exception:
        pass
    return out


def load_video(reader: Video, root: Path) -> pd.DataFrame:
    root = Path(root)
    pattern = f"{root.joinpath(root.name)}_*.csv"
    files = sorted(glob(pattern))
    chunks = []
    for file in files:
        try:
            df = reader.read(Path(file))
            if df is None or (hasattr(df, "empty") and df.empty):
                continue
            chunks.append(df)
        except Exception:
            # skip bad file
            continue
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, axis=0)
    try:
        out = out.sort_index()
    except Exception:
        pass
    return out


def concat_digi_events(series_low: pd.DataFrame, series_high: pd.DataFrame) -> pd.DataFrame:
    """Concatenate seperate high and low dataframes to produce on/off vector"""
    data_off = ~series_low[series_low==True]
    data_on = series_high[series_high==True]
    return pd.concat([data_off, data_on]).sort_index()


def load_csv(reader: Csv, root: Path) -> pd.DataFrame:
    root = Path(root)
    pattern = f"{root.joinpath(reader.pattern).joinpath(reader.pattern)}_*.{reader.extension}"
    files = sorted(glob(pattern))
    chunks = []
    for file in files:
        try:
            df = reader.read(Path(file))
            if df is None or (hasattr(df, "empty") and df.empty):
                continue
            chunks.append(df)
        except Exception:
            # skip bad file
            continue
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, axis=0)
    try:
        out = out.sort_index()
    except Exception:
        pass
    return out

def load_experiment(subjid, date, index=None):
    """
    Load experiment data with automatic session detection
    
    Parameters:
    -----------
    subjid : str or int
        Subject ID (e.g., '025' or 25)
    date : str or int  
        Date in format YYYYMMDD (e.g., '20250730' or 20250730)
    index : int, optional
        If multiple experiments exist, specify which one (0, 1, 2, etc.)
    
    Returns:
    --------
    Path object to experiment root, or None if selection needed
    """
    
    base_path = Path(project_root) / "data" / "rawdata"
    
    # Format inputs
    subjid_str = f"sub-{str(subjid).zfill(3)}"  
    date_str = str(date)
    
    subject_dirs = list(base_path.glob(f"{subjid_str}_id-*"))
    
    if not subject_dirs:
        raise FileNotFoundError(f"No subject directory found for {subjid_str}")
    
    if len(subject_dirs) > 1:
        print(f"Warning: Multiple subject directories found for {subjid_str}, using first one")
    
    subject_dir = subject_dirs[0]
    print(f"Using subject directory: {subject_dir}")

    session_dirs = list(subject_dir.glob(f"ses-*_date-{date_str}"))
    
    if not session_dirs:
        # Better error reporting - show what sessions actually exist
        all_sessions = list(subject_dir.glob("ses-*"))
        session_names = [d.name for d in all_sessions]
        raise FileNotFoundError(f"No session found for date {date_str} in {subject_dir}.\n"
                              f"Available sessions: {session_names}")
    
    if len(session_dirs) > 1:
        print(f"Warning: Multiple sessions found for date {date_str}, using first one")
    
    session_dir = session_dirs[0]
    
    behav_dir = session_dir / "behav"
    if not behav_dir.exists():
        raise FileNotFoundError(f"No behav directory found in {session_dir}")
    
    # Find experiment folders (timestamp folders)
    experiment_dirs = [d for d in behav_dir.iterdir() 
                      if d.is_dir() and re.match(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', d.name)]
    
    if not experiment_dirs:
        # Better error reporting - show what directories actually exist
        all_dirs = [d.name for d in behav_dir.iterdir() if d.is_dir()]
        raise FileNotFoundError(f"No experiment directories found in {behav_dir}.\n"
                              f"Available directories: {all_dirs}")
    
    # Sort by timestamp (chronological order)
    experiment_dirs.sort(key=lambda x: x.name)
    
    # Handle multiple experiments
    if len(experiment_dirs) == 1:
        # Single experiment - return it directly
        root = experiment_dirs[0]
        print(f"Loaded experiment: {root}")
        return root
    
    elif index is None:
        # Multiple experiments, no index specified
        print(f"Multiple experiments detected for subject {subjid_str} on {date_str}:")
        for i, exp_dir in enumerate(experiment_dirs):
            print(f"  Index {i}: {exp_dir.name}")
        print(f"\nPlease run again with index parameter:")
        print(f"root = load_experiment({subjid}, {date}, index=0)  # for first experiment")
        print(f"root = load_experiment({subjid}, {date}, index=1)  # for second experiment")
        return None
    
    else:
        # Index specified
        if index >= len(experiment_dirs) or index < 0:
            raise IndexError(f"Index {index} out of range. Available indices: 0-{len(experiment_dirs)-1}")
        
        root = experiment_dirs[index]
        print(f"Loaded experiment {index}: {root}")
        return root

#Helper function with shorter name 
def exp_data(subjid, date, index=None):
    """Alias for load_experiment with shorter name"""
    return load_experiment(subjid, date, index)


def load_all_streams(root, apply_corrections = True, *args, verbose: bool = True, **kwargs):
    """
    Load all behavioral data streams with proper timestamp synchronization
    """
    vprint(verbose, f"Loading data streams from: {root}")
    
    # Create readers
    behavior_reader = harp.create_reader('device_schemas/behavior.yml', epoch=harp.REFERENCE_EPOCH)
    olfactometer_reader = harp.create_reader('device_schemas/olfactometer.yml', epoch=harp.REFERENCE_EPOCH)
    
    data = {}
    
    # === TIMESTAMP SYNCHRONIZATION ===
    # Load heartbeat for timestamp conversion
    try:
        heartbeat = load(behavior_reader.TimestampSeconds, root/"Behavior")
        if not heartbeat.empty:
            heartbeat.reset_index(inplace=True)
        vprint(verbose, "Loaded heartbeat data")
    except Exception as e:
        print(f"Failed to load heartbeat: {e}")
        heartbeat = pd.DataFrame(columns=['Time', 'TimestampSeconds'])
    
    # Calculate real-time offset
    real_time_offset = pd.Timedelta(0)
    if not heartbeat.empty and 'Time' in heartbeat.columns and len(heartbeat) > 0:
        try:
            # Extract timestamp from root folder name
            real_time_str = root.name
            match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', real_time_str)
            if not match:
                real_time_str = root.parent.name
                match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', real_time_str)
            
            if match:
                real_time_str = match.group(0)
                real_time_ref_utc = datetime.strptime(real_time_str, '%Y-%m-%dT%H-%M-%S')
                real_time_ref_utc = real_time_ref_utc.replace(tzinfo=timezone.utc)
                uk_tz = zoneinfo.ZoneInfo("Europe/London")
                real_time_ref = real_time_ref_utc.astimezone(uk_tz)
                
                start_time_hardware = heartbeat['Time'].iloc[0]
                start_time_dt = start_time_hardware.to_pydatetime()
                if start_time_dt.tzinfo is None:
                    start_time_dt = start_time_dt.replace(tzinfo=uk_tz)
                real_time_offset = real_time_ref - start_time_dt
                vprint(verbose, f"Calculated real-time offset: {real_time_offset}")
        except Exception as e:
            print(f"Error calculating real-time offset: {e}")
    
    # Create timestamp interpolation mapping
    timestamp_to_time = pd.Series()
    if not heartbeat.empty and 'Time' in heartbeat.columns and 'TimestampSeconds' in heartbeat.columns:
        heartbeat['Time'] = pd.to_datetime(heartbeat['Time'], errors='coerce')
        timestamp_to_time = pd.Series(data=heartbeat['Time'].values, index=heartbeat['TimestampSeconds'])
        vprint(verbose, "Created timestamp interpolation mapping")
    
    def interpolate_time(seconds):
        """Interpolate timestamps from seconds, with safety checks"""
        if timestamp_to_time.empty:
            return pd.NaT
        int_seconds = int(seconds)
        fractional_seconds = seconds % 1
        if int_seconds in timestamp_to_time.index:
            base_time = timestamp_to_time.loc[int_seconds]
            return base_time + pd.to_timedelta(fractional_seconds, unit='s')
        return pd.NaT
    
    # Store timing data
    data['heartbeat'] = heartbeat
    data['real_time_offset'] = real_time_offset
    data['timestamp_to_time'] = timestamp_to_time
    data['interpolate_time'] = interpolate_time
    
    # === LOAD ALL OTHER DATA STREAMS ===

    # Core behavioral data
    try:
        data['digital_input_data'] = load(behavior_reader.DigitalInputState, root/"Behavior")
        vprint(verbose, "Loaded digital_input_data")
    except Exception as e:
        print(f"Failed to load digital_input_data: {e}")
        data['digital_input_data'] = pd.DataFrame()
    
    try:
        data['output_set'] = load(behavior_reader.OutputSet, root/"Behavior")
        vprint(verbose, "Loaded output_set")
    except Exception as e:
        print(f"Failed to load output_set: {e}")
        data['output_set'] = pd.DataFrame()
    
    try:
        data['output_clear'] = load(behavior_reader.OutputClear, root/"Behavior")
        vprint(verbose, "Loaded output_clear")
    except Exception as e:
        print(f"Failed to load output_clear: {e}")
        data['output_clear'] = pd.DataFrame()
    
    # Olfactometer valve data
    try:
        data['olfactometer_valves_0'] = load(olfactometer_reader.OdorValveState, root/"Olfactometer0")
        vprint(verbose, "Loaded olfactometer_valves_0")
    except Exception as e:
        print(f"Failed to load olfactometer_valves_0: {e}")
        data['olfactometer_valves_0'] = pd.DataFrame()
    
    try:
        data['olfactometer_valves_1'] = load(olfactometer_reader.OdorValveState, root/"Olfactometer1")
        vprint(verbose, "Loaded olfactometer_valves_1")
    except Exception as e:
        print(f"Failed to load olfactometer_valves_1: {e}")
        data['olfactometer_valves_1'] = pd.DataFrame()
    
    # End valve states (commented in original but included for completeness)
    try:
        data['olfactometer_end_0'] = load(olfactometer_reader.EndValveState, root/"Olfactometer0")
        vprint(verbose, "Loaded olfactometer_end_0")
    except Exception as e:
        print(f"Failed to load olfactometer_end_0: {e}")
        data['olfactometer_end_0'] = pd.DataFrame()
    
    # Analog data
    try:
        data['analog_data'] = load(behavior_reader.AnalogData, root/"Behavior")
        vprint(verbose, "Loaded analog_data")
    except Exception as e:
        print(f"Failed to load analog_data: {e}")
        data['analog_data'] = pd.DataFrame()
    
    # Flow meter data
    try:
        data['flow_meter'] = load(olfactometer_reader.Flowmeter, root/"Olfactometer0")
        vprint(verbose, "Loaded flow_meter")
    except Exception as e:
        print(f"Failed to load flow_meter: {e}")
        data['flow_meter'] = pd.DataFrame()
    
    # Video data
    try:
        video_reader = Video()
        data['video_reader'] = video_reader
        data['video_data'] = load_video(video_reader, root/"VideoData")
        vprint(verbose, "Loaded video_data")
    except Exception as e:
        print(f"Failed to load video_data: {e}")
        data['video_reader'] = None
        data['video_data'] = pd.DataFrame()
    
    # Pulse supply (reward delivery)
    try:
        data['pulse_supply_1'] = load(behavior_reader.PulseSupplyPort1, root/"Behavior")
        vprint(verbose, "Loaded pulse_supply_1")
    except Exception as e:
        print(f"Failed to load pulse_supply_1: {e}")
        data['pulse_supply_1'] = pd.DataFrame()
    
    try:
        data['pulse_supply_2'] = load(behavior_reader.PulseSupplyPort2, root/"Behavior")
        vprint(verbose, "Loaded pulse_supply_2")
    except Exception as e:
        print(f"Failed to load pulse_supply_2: {e}")
        data['pulse_supply_2'] = pd.DataFrame()
    
    # Create combined odour LED signal
    try:
        if not data['output_clear'].empty and not data['output_set'].empty:
            data['odour_led'] = concat_digi_events(data['output_clear']['DOPort0'], data['output_set']['DOPort0'])
            vprint(verbose, "Created odour_led")
        else:
            data['odour_led'] = pd.Series()
            print("Could not create odour_led (missing output data)")
    except Exception as e:
        print(f"Failed to create odour_led: {e}")
        data['odour_led'] = pd.Series()
    
    # Store readers for later use
    data['behavior_reader'] = behavior_reader
    data['olfactometer_reader'] = olfactometer_reader
    
    if apply_corrections and real_time_offset != pd.Timedelta(0):
        vprint(verbose, "\nApplying time corrections to all data streams...")
        
        time_indexed_streams = [
            'digital_input_data', 'output_set', 'output_clear',
            'olfactometer_valves_0', 'olfactometer_valves_1', 
            'olfactometer_end_0', 'analog_data', 'flow_meter',
            'video_data', 'pulse_supply_1', 'pulse_supply_2', 
            'odour_led'
        ]
        
        
        for stream_name in time_indexed_streams:
            if stream_name in data and not data[stream_name].empty:
                try:
                    if isinstance(data[stream_name], pd.DataFrame):
                        # Check if index is datetime-like
                        if hasattr(data[stream_name].index, 'dtype') and pd.api.types.is_datetime64_any_dtype(data[stream_name].index):
                            data[stream_name].index = data[stream_name].index + real_time_offset
                            vprint(verbose, f"Applied correction to {stream_name}")
                        else:
                            print(f"Skipped {stream_name} (not datetime index)")
                            
                    elif isinstance(data[stream_name], pd.Series):
                        # Check if index is datetime-like
                        if hasattr(data[stream_name].index, 'dtype') and pd.api.types.is_datetime64_any_dtype(data[stream_name].index):
                            data[stream_name].index = data[stream_name].index + real_time_offset
                            vprint(verbose, f"Applied correction to {stream_name}")
                        else:
                            print(f"Skipped {stream_name} (not datetime index)")
                except Exception as e:
                    print(f"Failed to apply correction to {stream_name}: {e}")
    


    vprint(verbose, f"\nData loading complete! Loaded {len([k for k, v in data.items() if not (isinstance(v, pd.DataFrame) and v.empty) and not (isinstance(v, pd.Series) and v.empty)])} streams successfully.")
    
    return data

def load_experiment_events(root, *args, verbose: bool = True, **kwargs):
    """
    Load and process experiment events with automatic time synchronization
    matching load_all_streams() timing corrections
    """
    
    vprint(verbose, "Loading experiment events...")
    
    # === LOAD TIMING DATA ===
    try:
        behavior_reader = harp.create_reader('device_schemas/behavior.yml', epoch=harp.REFERENCE_EPOCH)
        heartbeat = load(behavior_reader.TimestampSeconds, root/"Behavior")
        if not heartbeat.empty:
            heartbeat.reset_index(inplace=True)
        vprint(verbose, "Loaded heartbeat data for timing synchronization")
    except Exception as e:
        print(f"Failed to load heartbeat: {e}")
        heartbeat = pd.DataFrame(columns=['Time', 'TimestampSeconds'])
    
    # Calculate real-time offset (same logic as load_all_streams)
    real_time_offset = pd.Timedelta(0)
    if not heartbeat.empty and 'Time' in heartbeat.columns and len(heartbeat) > 0:
        try:
            # Extract timestamp from root folder name
            real_time_str = root.name
            match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', real_time_str)
            if not match:
                real_time_str = root.parent.name
                match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', real_time_str)
            
            if match:
                real_time_str = match.group(0)
                real_time_ref_utc = datetime.strptime(real_time_str, '%Y-%m-%dT%H-%M-%S')
                real_time_ref_utc = real_time_ref_utc.replace(tzinfo=timezone.utc)
                uk_tz = zoneinfo.ZoneInfo("Europe/London")
                real_time_ref = real_time_ref_utc.astimezone(uk_tz)
                
                start_time_hardware = heartbeat['Time'].iloc[0]
                start_time_dt = start_time_hardware.to_pydatetime()
                if start_time_dt.tzinfo is None:
                    start_time_dt = start_time_dt.replace(tzinfo=uk_tz)
                real_time_offset = real_time_ref - start_time_dt
                vprint(verbose, f"Calculated real-time offset: {real_time_offset}")
        except Exception as e:
            print(f"Error calculating real-time offset: {e}")
    
    # Create timestamp interpolation mapping (same as load_all_streams)
    timestamp_to_time = pd.Series()
    interpolate_time = None
    if not heartbeat.empty and 'Time' in heartbeat.columns and 'TimestampSeconds' in heartbeat.columns:
        heartbeat['Time'] = pd.to_datetime(heartbeat['Time'], errors='coerce')
        timestamp_to_time = pd.Series(data=heartbeat['Time'].values, index=heartbeat['TimestampSeconds'])
        
        def interpolate_time(seconds):
            """Interpolate timestamps from seconds, with safety checks"""
            if timestamp_to_time.empty:
                return pd.NaT
            int_seconds = int(seconds)
            fractional_seconds = seconds % 1
            if int_seconds in timestamp_to_time.index:
                base_time = timestamp_to_time.loc[int_seconds]
                return base_time + pd.to_timedelta(fractional_seconds, unit='s')
            return pd.NaT
        
        vprint(verbose, "Created timestamp interpolation mapping")
    
    # === LOAD EXPERIMENT EVENTS ===
    event_types = {
        'initiation_sequence': [],
        'end_initiation': [],
        'await_reward': [],
        'reset': [],
        'choose_random_sequence': [],
        'sample_reward_condition': []
    }
    
    experiment_events_dir = root / "ExperimentEvents"
    
    if not experiment_events_dir.exists():
        print("No ExperimentEvents directory found")
        return {f'combined_{event_type}_df': pd.DataFrame() for event_type in event_types.keys()}
    
    csv_files = list(experiment_events_dir.glob("*.csv"))
    vprint(verbose, f"Found {len(csv_files)} experiment event files")
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            ev_df = pd.read_csv(csv_file)
            vprint(verbose, f"Processing event file: {csv_file.name} with {len(ev_df)} rows")
            
            # Handle timestamp conversion (same logic as original notebook)
            if "Seconds" in ev_df.columns and interpolate_time is not None:
                ev_df = ev_df.sort_values("Seconds")
                ev_df["Time"] = ev_df["Seconds"].apply(interpolate_time)
                vprint(verbose, "Using Seconds column with interpolation")
            else:
                # Fallback: use seconds as relative time
                ev_df["Time"] = pd.to_datetime(ev_df["Seconds"], unit='s')
                vprint(verbose, "Using Seconds column as raw timestamp")
            
            # Apply real-time offset (CRITICAL for synchronization)
            if real_time_offset != pd.Timedelta(0):
                ev_df["Time"] = ev_df["Time"] + real_time_offset
                vprint(verbose, f"Applied real-time offset: {real_time_offset}")
            
            # Extract events
            if "Value" in ev_df.columns:
                vprint(verbose, f"Found Value column with values: {ev_df['Value'].unique()}")
                
                event_mappings = {
                    'EndInitiation': 'end_initiation',
                    'InitiationSequence': 'initiation_sequence', 
                    'Reset': 'reset',
                    'AwaitReward': 'await_reward',
                    'SampleRewardCondition': 'sample_reward_condition',
                    'ChooseRandomSequence': 'choose_random_sequence'
                }
                
                for event_value, event_key in event_mappings.items():
                    event_df = ev_df[ev_df["Value"] == event_value].copy()
                    if not event_df.empty:
                        vprint(verbose, f"Found {len(event_df)} {event_value} events")
                        event_df[event_value] = True
                        event_types[event_key].append(event_df[["Time", event_value]])
                        
        except Exception as e:
            print(f"Error processing event file {csv_file.name}: {e}")
    
    # Combine events into final DataFrames
    results = {}
    event_name_mapping = {
        'end_initiation': 'EndInitiation',
        'initiation_sequence': 'InitiationSequence',
        'reset': 'Reset', 
        'await_reward': 'AwaitReward',
        'sample_reward_condition': 'SampleRewardCondition',
        'choose_random_sequence': 'ChooseRandomSequence'
    }
    
    for event_key, frames_list in event_types.items():
        df_name = f'combined_{event_key}_df'
        column_name = event_name_mapping[event_key]
        
        if len(frames_list) > 0:
            combined_df = pd.concat(frames_list, ignore_index=True)
            combined_df.reset_index(drop=True, inplace=True)
            # Sort by time for proper chronological order
            combined_df = combined_df.sort_values('Time').reset_index(drop=True)
            results[df_name] = combined_df
            vprint(verbose, f"Combined {len(combined_df)} {column_name} events")
        else:
            results[df_name] = pd.DataFrame(columns=["Time", column_name])
            print(f"No {column_name} events found")
    
    print(f"Experiment events loading complete! All events synchronized with load_all_streams timing.")
    return results


def load_odor_mapping(root, *, data=None, verbose: bool = True, **kwargs):
    """
    Load odor mapping from session settings
    
    Parameters:
    -----------
    root : Path
        Experiment root directory
    data : dict, optional
        Data dictionary from load_all_streams() containing valve data
        If None, will load valve data internally
    
    Returns:
    --------
    dict containing odor mapping information
    """
    
    vprint(verbose, "Loading odor mapping from session settings...")
    
    # Get valve data
    if data is not None:
        olfactometer_valves_0 = data.get('olfactometer_valves_0', pd.DataFrame())
        olfactometer_valves_1 = data.get('olfactometer_valves_1', pd.DataFrame())
    else:
        # Load valve data if not provided
        try:
            olfactometer_reader = harp.create_reader('device_schemas/olfactometer.yml', epoch=harp.REFERENCE_EPOCH)
            olfactometer_valves_0 = load(olfactometer_reader.OdorValveState, root/"Olfactometer0")
            olfactometer_valves_1 = load(olfactometer_reader.OdorValveState, root/"Olfactometer1")
        except Exception as e:
            print(f"Could not load valve data: {e}")
            olfactometer_valves_0 = pd.DataFrame()
            olfactometer_valves_1 = pd.DataFrame()
    
    # Create valve data dictionary
    olfactometer_valves = {
        0: olfactometer_valves_0,
        1: olfactometer_valves_1,
    }
    
    try:
        # Load session settings (experiment-specific configuration)
        session_settings, session_schema = detect_settings.detect_settings(root)
        vprint(verbose, "Loaded session settings")
        
        # Extract valve configurations for each olfactometer
        olfactometer_commands = session_settings.metadata.iloc[0].olfactometerCommands
        olf_valves0 = [cmd.valvesOpenO0 for cmd in olfactometer_commands]
        olf_valves1 = [cmd.valvesOpenO1 for cmd in olfactometer_commands]
        
        vprint(verbose, f"Found {len(olf_valves0)} valve configurations for olfactometer 0")
        vprint(verbose, f"Found {len(olf_valves1)} valve configurations for olfactometer 1")
        
        # Create command index mapping (valve number -> command index)
        olf_command_idx = {}
        
        # Map olfactometer 0 valves (0-3) to command indices
        for val in range(4):
            try:
                cmd_idx = next(i for i, lst in enumerate(olf_valves0) if val in lst)
                olf_command_idx[f'0{val}'] = cmd_idx
            except StopIteration:
                print(f"Warning: Valve {val} not found in olfactometer 0 configuration")
        
        # Map olfactometer 1 valves (0-3) to command indices  
        for val in range(4):
            try:
                cmd_idx = next(i for i, lst in enumerate(olf_valves1) if val in lst)
                olf_command_idx[f'1{val}'] = cmd_idx
            except StopIteration:
                print(f"Warning: Valve {val} not found in olfactometer 1 configuration")
        
        vprint(verbose, f"Created valve-to-command mapping: {olf_command_idx}")
        
        # Create odor name mapping
        odour_to_olfactometer_map = [[] for _ in range(len(olfactometer_valves))]
        
        for valve_key, cmd_idx in olf_command_idx.items():
            olf_id = int(valve_key[0])  # Extract olfactometer ID (0 or 1)
            odor_name = olfactometer_commands[cmd_idx].name
            odour_to_olfactometer_map[olf_id].append(odor_name)
        
        vprint(verbose, f"Created odor mapping: {odour_to_olfactometer_map}")
        
        # Create reverse mapping: valve -> odor name
        valve_to_odor = {}
        for valve_key, cmd_idx in olf_command_idx.items():
            odor_name = olfactometer_commands[cmd_idx].name
            valve_to_odor[valve_key] = odor_name
        
        # Create olfactometer -> odor list mapping
        olfactometer_to_odors = {}
        for olf_id in range(len(olfactometer_valves)):
            olfactometer_to_odors[olf_id] = odour_to_olfactometer_map[olf_id]
        print("Odor mapping loaded successfully")

        return {
            'olfactometer_valves': olfactometer_valves,
            'session_settings': session_settings,
            'session_schema': session_schema,
            'olf_valves0': olf_valves0,
            'olf_valves1': olf_valves1,
            'olf_command_idx': olf_command_idx,
            'odour_to_olfactometer_map': odour_to_olfactometer_map,
            'valve_to_odor': valve_to_odor,
            'olfactometer_to_odors': olfactometer_to_odors
        }
        
    except Exception as e:
        print(f"Error loading odor mapping: {e}")
        return {
            'olfactometer_valves': olfactometer_valves,
            'session_settings': None,
            'session_schema': None,
            'olf_valves0': [],
            'olf_valves1': [],
            'olf_command_idx': {},
            'odour_to_olfactometer_map': [[], []],
            'valve_to_odor': {},
            'olfactometer_to_odors': {0: [], 1: []}
        }
    


# ================= Functions for Trial Analysis and Classification ========================


def detect_trials(data, events, root, verbose=True):
    """
    Trial Detection Function     
    Parameters:
    -----------
    data : dict
        Data dictionary containing poke information
    events : dict
        Events dictionary containing initiation sequences
    verbose : bool, default=True
        Whether to print detailed progress information
    
    Logic:
    ------
    1. For each poke, check if it + merged gaps <sample_offset_time reach ≥minimum_sampling_time total
    2. STOP as soon as we reach minimum_sampling_time (don't continue merging)
    3. If sequence fails to reach minimum_sampling_time before a ≥sample_offset_time gap, record as failed attempt
    4. Next poke after ≥sample_offset_time gap starts a new attempt
    """
    
    # Get experimental parameters automatically
    sample_offset_time, minimum_sampling_time, _ = get_experiment_parameters(root)
    # Convert to milliseconds for consistency with existing logic
    sample_offset_time_ms = sample_offset_time * 1000
    minimum_sampling_time_ms = minimum_sampling_time * 1000

    if verbose:
        print("TRIAL DETECTION")
        print("=" * 60)
        print(f"Parameters: minimum_sampling_time={minimum_sampling_time_ms}ms, sample_offset_time={sample_offset_time_ms}ms")
    
    initiation_events = events['combined_initiation_sequence_df'].copy()
    cue_pokes = data['digital_input_data']['DIPort0'].copy()
    
    trials = []
    initiated_sequences = []
    non_initiated_sequences = []
    
    for idx, initiation_row in initiation_events.iterrows():
        initiation_time = initiation_row['Time']
        
        # Find next initiation sequence
        if idx + 1 < len(initiation_events):
            next_initiation_time = initiation_events.iloc[idx + 1]['Time']
        else:
            next_initiation_time = cue_pokes.index[-1]
        
        if verbose:
            print(f"\nInitiationSequence {idx}: {initiation_time}")
        
        # Get all poke data between initiations
        period_pokes = cue_pokes[(cue_pokes.index > initiation_time) & 
                                (cue_pokes.index < next_initiation_time)]
        
        if period_pokes.empty:
            if verbose:
                print(f"  No pokes found")
            continue
        
        # Find all poke periods (start, end) pairs
        poke_periods = []
        current_start = None
        
        for timestamp, state in period_pokes.items():
            if state and current_start is None:
                current_start = timestamp
            elif not state and current_start is not None:
                poke_periods.append((current_start, timestamp))
                current_start = None
        
        # Handle poke extending to end
        if current_start is not None:
            poke_periods.append((current_start, period_pokes.index[-1]))
        
        if not poke_periods:
            if verbose:
                print(f"  No complete poke periods found")
            continue
        
        if verbose:
            print(f"  Found {len(poke_periods)} poke periods")
        
        # Process pokes sequentially, building continuous sequences
        trial_found = False
        attempt_num = 0
        i = 0
        
        while i < len(poke_periods) and not trial_found:
            attempt_num += 1
            sequence_start = poke_periods[i][0]
            
            if verbose:
                print(f"    Starting attempt {attempt_num} at {sequence_start}")
            
            # Build continuous sequence starting from poke i
            continuous_time = 0
            sequence_end = sequence_start
            j = i
            
            while j < len(poke_periods):
                poke_start, poke_end = poke_periods[j]
                
                if j == i:
                    # First poke in sequence
                    poke_duration = (poke_end - poke_start).total_seconds() * 1000
                    continuous_time = poke_duration
                    sequence_end = poke_end
                    
                    if verbose:
                        print(f"      Poke {j+1}: {poke_duration:.1f}ms (total: {continuous_time:.1f}ms)")
                else:
                    # Check gap to this poke
                    prev_end = poke_periods[j-1][1]
                    gap = (poke_start - prev_end).total_seconds() * 1000
                    
                    if gap >= sample_offset_time_ms:
                        # Gap too large - end this sequence
                        if verbose:
                            print(f"      Gap to poke {j+1}: {gap:.1f}ms (≥{sample_offset_time_ms}ms - sequence ends)")
                        break
                    else:
                        # Merge this poke
                        poke_duration = (poke_end - poke_start).total_seconds() * 1000
                        continuous_time += gap + poke_duration
                        sequence_end = poke_end
                        
                        if verbose:
                            print(f"      Poke {j+1}: gap {gap:.1f}ms + {poke_duration:.1f}ms (total: {continuous_time:.1f}ms)")
                
                # Check if we've reached minimum_sampling_time
                if continuous_time >= minimum_sampling_time_ms:
                    # SUCCESS! Stop here
                    if verbose:
                        print(f"      SUCCESS: {continuous_time:.1f}ms continuous poke (≥{minimum_sampling_time_ms}ms)")
                    
                    # Add to trials
                    trial_entry = {
                        'initiation_sequence_time': initiation_time,
                        'trial_start': sequence_start,
                        'trial_end': next_initiation_time,
                        'continuous_poke_time_ms': continuous_time,
                        'trial_id': len(trials),
                        'attempt_number': attempt_num
                    }
                    trials.append(trial_entry)
                    
                    # Add to initiated_sequences
                    initiated_sequence_entry = {
                        'initiation_sequence_time': initiation_time,
                        'sequence_start': sequence_start,
                        'sequence_end': next_initiation_time,
                        'continuous_poke_time_ms': continuous_time,
                        'trial_id': len(trials) - 1,
                        'attempt_number': attempt_num,
                        'timestamp': sequence_start
                    }
                    initiated_sequences.append(initiated_sequence_entry)
                    
                    trial_found = True
                    break
                
                j += 1
            
            if not trial_found:
                # This sequence failed
                if verbose:
                    print(f"      FAILED: {continuous_time:.1f}ms continuous poke (<{minimum_sampling_time_ms}ms)")
                
                # Add to non_initiated_sequences
                non_initiated_sequence_entry = {
                    'initiation_sequence_time': initiation_time,
                    'attempt_start': sequence_start,
                    'attempt_end': sequence_end,
                    'continuous_poke_time_ms': continuous_time,
                    'attempt_number': attempt_num,
                    'timestamp': sequence_start,
                    'failure_reason': 'insufficient_continuous_poke_time'
                }
                non_initiated_sequences.append(non_initiated_sequence_entry)
                
                # Move to next sequence start
                # Find next poke that's ≥sample_offset_time_ms away
                next_i = j  # j is where we stopped (gap was ≥sample_offset_time_ms or end of periods)
                if next_i >= len(poke_periods):
                    # No more pokes
                    break
                
                i = next_i
        
        if not trial_found and verbose:
            print(f"  No successful trial found for this initiation sequence")
    
    # Convert to DataFrames and sort by timestamp for chronological access
    results = {
        'trials': pd.DataFrame(trials),
        'initiated_sequences': pd.DataFrame(initiated_sequences).sort_values('timestamp') if initiated_sequences else pd.DataFrame(),
        'non_initiated_sequences': pd.DataFrame(non_initiated_sequences).sort_values('timestamp') if non_initiated_sequences else pd.DataFrame()
    }
    
    # ALWAYS display summary
    vprint(verbose, "\n" + "="*50)
    vprint(verbose, "DETECTION SUMMARY:")
    vprint(verbose, f"Trials: {len(results['trials'])}")
    vprint(verbose, f"Initiated sequences: {len(results['initiated_sequences'])}")
    vprint(verbose, f"Non-initiated sequences: {len(results['non_initiated_sequences'])}")
    vprint(verbose, "="*50)
    
    return results

def get_experiment_parameters(root):
    """
    Simple function to extract sampleOffsetTime and minimumSamplingTime
    
    Returns:
        tuple: (sampleOffsetTime, minimumSamplingTime)
    """
    session_settings, session_schema = detect_settings.detect_settings(root)
    
    # Get sampleOffsetTime from SessionSettings
    sample_offset_time = session_settings.metadata.iloc[0].metadata.sampleOffsetTime
    
    # Get minimumSamplingTime from Schema  
    minimum_sampling_time = session_schema['minimumSamplingTime']
    
    response_time = session_schema['responseTime']

    return sample_offset_time, minimum_sampling_time, response_time

def classify_trials(data, events, trial_counts, odor_map, stage, root, verbose=True):# Classify trials and get valve/poke times. Part of wrapper function
    """
    Same classification as classify_trial_outcomes_extensive, plus:
      - position_valve_times and position_poke_times per trial
      - summary printouts for poke/valve time ranges by position and by odor
    Response-time analysis is removed.
    """
    sample_offset_time, minimum_sampling_time, response_time = get_experiment_parameters(root)
    # Convert to milliseconds for consistency with existing logic
    sample_offset_time_ms = sample_offset_time * 1000
    minimum_sampling_time_ms = minimum_sampling_time * 1000
    response_time_sec = response_time 
    if response_time_sec is None:
        raise ValueError("Response time parameter cannot be extracted from Schema file. Check detect_settings function.")

    if verbose:
        print("=" * 80)
        print("CLASSIFYING TRIAL OUTCOMES WITH HIDDEN RULE AND VALVE/POKE TIME ANALYSIS")
        print("=" * 80)
        print(f"Sample offset time: {sample_offset_time_ms} ms")
        print(f"Minimum sampling time: {minimum_sampling_time_ms} ms")
        print(f"Response time window: {response_time_sec} s")

    # Hidden rule location from stage
    hidden_rule_location = None
    sequence_name = None
    if isinstance(stage, dict):
        sequence_name = stage.get('stage_name') or str(stage)
        if stage.get('hidden_rule_index') is not None:
            try:
                hidden_rule_location = int(stage['hidden_rule_index'])
            except Exception:
                hidden_rule_location = None
    if hidden_rule_location is None:
        sequence_name = sequence_name or str(stage)
        m = re.search(r'Location(\d+)', sequence_name)
        if m:
            hidden_rule_location = int(m.group(1))
    hidden_rule_position = hidden_rule_location + 1 if isinstance(hidden_rule_location, int) else None
    if verbose:
        if hidden_rule_location is not None:
            print(f"Hidden rule location extracted: Location{hidden_rule_location} (index {hidden_rule_location}, position {hidden_rule_position})")
        else:
            print(f"No Hidden Rule Location found in sequence name: {sequence_name}. Proceeding without Hidden Rule analysis.")

    # Base trial data
    initiated_trials = trial_counts['initiated_sequences'].copy()
    non_initiated_trials = trial_counts['non_initiated_sequences'].copy()

    # Events
    await_reward_times = events['combined_await_reward_df']['Time'].tolist() if 'combined_await_reward_df' in events else []

    # Supply port activities
    supply_port1_times = data['pulse_supply_1'].index.tolist() if not data['pulse_supply_1'].empty else []
    supply_port2_times = data['pulse_supply_2'].index.tolist() if not data['pulse_supply_2'].empty else []
    all_supply_port_times = sorted(supply_port1_times + supply_port2_times)

    # Reward port pokes
    port1_pokes = data['digital_input_data'].get('DIPort1', pd.Series(dtype=bool))
    port2_pokes = data['digital_input_data'].get('DIPort2', pd.Series(dtype=bool))

    # Nose-poke data (Port0) for poke-time analysis during odors
    poke_data = data['digital_input_data'].get('DIPort0', pd.Series(dtype=bool))

    poke_series_full = poke_data.astype(bool)
    poke_series_full = poke_series_full.sort_index()
    _rises = poke_series_full & ~poke_series_full.shift(1, fill_value=False)
    _falls = ~poke_series_full & poke_series_full.shift(1, fill_value=False)
    _starts = list(poke_series_full.index[_rises])
    _ends = list(poke_series_full.index[_falls])
    poke_intervals = []
    i = j = 0
    while i < len(_starts) and j < len(_ends):
        if _ends[j] <= _starts[i]:
            j += 1
            continue
        poke_intervals.append((_starts[i], _ends[j]))
        i += 1
        j += 1
    # If the series starts IN without a detected start edge, optionally prepend
    if poke_series_full.size and poke_series_full.iloc[0] and (not _starts or poke_series_full.index[0] < _starts[0]):
        # close it at the first fall after the beginning
        first_fall = next((t for t in _ends if t > poke_series_full.index[0]), None)
        if first_fall is not None:
            poke_intervals.insert(0, (poke_series_full.index[0], first_fall))


    # Build valve activation list
    olfactometer_valves = odor_map['olfactometer_valves']
    valve_to_odor = odor_map['valve_to_odor']

    all_valve_activations = []
    for olf_id, valve_data in olfactometer_valves.items():
        if valve_data.empty:
            continue
        for i, valve_col in enumerate(valve_data.columns):
            valve_key = f"{olf_id}{i}"
            if valve_key not in valve_to_odor:
                continue
            odor_name = valve_to_odor[valve_key]
            if odor_name.lower() == 'purge':
                continue

            valve_series = valve_data[valve_col]
            valve_activations = valve_series & ~valve_series.shift(1, fill_value=False)
            activation_times = valve_activations[valve_activations == True].index.tolist()
            valve_deactivations = ~valve_series & valve_series.shift(1, fill_value=False)
            deactivation_times = valve_deactivations[valve_deactivations == True].index.tolist()

            for activation_time in activation_times:
                next_deactivations = [t for t in deactivation_times if t > activation_time]
                deactivation_time = min(next_deactivations) if next_deactivations else valve_series.index[-1]
                all_valve_activations.append({
                    'start_time': activation_time,
                    'end_time': deactivation_time,
                    'odor_name': odor_name,
                    'valve_key': valve_key
                })

    all_valve_activations.sort(key=lambda x: x['start_time'])

    if verbose:
        print(f"Found {len(all_valve_activations)} total valve activations (excluding Purge)")
        print(f"Analyzing {len(initiated_trials)} initiated trials...")
        print(f"Found {len(await_reward_times)} AwaitReward events")
        print(f"Found {len(all_supply_port_times)} total supply port activities")

    # Result containers
    completed_sequences = []
    aborted_sequences = []
    aborted_sequences_hr = []
    completed_hr = []
    completed_hr_missed = []
    completed_rewarded = []
    completed_unrewarded = []
    completed_timeout = []
    completed_hr_rewarded = []
    completed_hr_unrewarded = []
    completed_hr_timeout = []
    completed_hr_missed_rewarded = []
    completed_hr_missed_unrewarded = []
    completed_hr_missed_timeout = []
    non_initiated_odor1_attempts = []
    initiated_trials = trial_counts['initiated_sequences'].copy()
    initiated_trials_list = []

    # Aggregators for summary prints (completed trials only)
    agg_position_poke_times = {pos: [] for pos in range(1, 6)}
    agg_position_valve_times = {pos: [] for pos in range(1, 6)}
    agg_odor_poke_times = defaultdict(list)
    agg_odor_valve_times = defaultdict(list)

    # Helpers
    def get_trial_valve_sequence(trial_start, trial_end):
        trial_valve_activations = []
        for valve_activation in all_valve_activations:
            valve_start = valve_activation['start_time']
            valve_end = valve_activation['end_time']
            if valve_start <= trial_end and valve_end >= trial_start:
                trial_valve_activations.append(valve_activation)
        trial_valve_activations.sort(key=lambda x: x['start_time'])
        odor_sequence = [activation['odor_name'] for activation in trial_valve_activations]
        return odor_sequence, trial_valve_activations


    hr_odor_set = None
    if hidden_rule_location is not None:
        try:
            _, schema_settings = detect_settings.detect_settings(root)
            odors = (schema_settings.get('hiddenRuleOdorsInferred') or [])
            if len(odors) < 2:
                raise ValueError("Hidden Rule Odor Identities could not be inferred from Schema.")
            hr_odor_set = set(map(str, odors))
            if verbose:
                print(f"Hidden Rule Odors inferred: {sorted(hr_odor_set)}")
        except Exception as e:
            raise ValueError(f"Hidden Rule Odor Identities could not be inferred from Schema: {e}")

    def check_hidden_rule(odor_sequence, idx):
        if idx is None or hr_odor_set is None:
            return False, False
        if idx < 0 or idx >= len(odor_sequence):
            return False, False
        odor_at_location = odor_sequence[idx]
        hit_hidden_rule = odor_at_location in hr_odor_set
        return True, hit_hidden_rule
    
    def window_poke_summary(window_start, window_end):
        if window_start is None or window_end is None or window_start >= window_end:
            return {'poke_time_ms': 0.0, 'poke_first_in': None, 'poke_odor_start': window_start}
        
        s_bool = poke_series_full
        prev = s_bool.loc[:window_start]
        in_at_start = bool(prev.iloc[-1]) if len(prev) else False
        w = s_bool.loc[window_start:window_end]
        
        if w.empty and not in_at_start:
            return {'poke_time_ms': 0.0, 'poke_first_in': None, 'poke_odor_start': window_start}
        
        rises = w & ~w.shift(1, fill_value=in_at_start)
        falls = ~w & w.shift(1, fill_value=in_at_start)
        intervals = []
        cur = window_start if in_at_start else None
        first_in = window_start if in_at_start else None
        
        for ts in w.index:
            if rises.get(ts, False) and cur is None:
                cur = ts
                if first_in is None:
                    first_in = ts
            if falls.get(ts, False) and cur is not None:
                intervals.append((cur, ts))
                cur = None
        
        if cur is not None:
            intervals.append((cur, window_end))
        
        if not intervals:
            return {'poke_time_ms': 0.0, 'poke_first_in': None, 'poke_odor_start': window_start}
        
        # Merge across gaps <= sample_offset_time_ms
        merged = [intervals[0]]
        for s2, e2 in intervals[1:]:
            ls, le = merged[-1]
            gap_ms = (s2 - le).total_seconds() * 1000.0
            if gap_ms <= sample_offset_time_ms:
                # Extend but cap at window_end
                merged[-1] = (ls, min(max(le, e2), window_end))
            else:
                merged.append((s2, e2))
        
        # Extract first block and cap at window_end
        first_block_start, first_block_end = merged[0]
        first_block_end_capped = min(first_block_end, window_end)
        first_block_ms = max(0.0, (first_block_end_capped - first_block_start).total_seconds() * 1000.0)
        
        return {
            'poke_time_ms': float(first_block_ms),
            'poke_first_in': first_in,
            'poke_odor_start': window_start
        }

    
    def _attempt_bout_from_poke_in(anchor_ts, cap_end=None):
        """
        Return (first_in_ts, bout_end_ts_capped, duration_ms) for the attempt whose valve starts at anchor_ts.
        - If anchor_ts falls inside an IN interval, start at that interval's start.
        - Else, use the first IN interval that starts at/after anchor_ts.
        - Merge backward across previous IN intervals while OUT gaps <= sample_offset_time_ms
        (to include pre-anchor pokes that are part of the same bout).
        - Merge forward across subsequent IN intervals while OUT gaps <= sample_offset_time_ms.
        - Cap the merged bout at cap_end if provided.
        """
        if anchor_ts is None or not poke_intervals:
            return None, None, 0.0

        starts_only = [s for s, _ in poke_intervals]

        # Find interval covering anchor or the first one after
        from bisect import bisect_left, bisect_right
        idx = bisect_right(starts_only, anchor_ts) - 1
        if 0 <= idx < len(poke_intervals) and poke_intervals[idx][0] <= anchor_ts < poke_intervals[idx][1]:
            k = idx
        else:
            k = bisect_left(starts_only, anchor_ts)
            if k >= len(poke_intervals):
                return None, None, 0.0

        # Start with the interval at k
        bout_start, bout_end = poke_intervals[k]

        # Backward merge: include prior intervals if the gap <= sample_offset_time_ms
        m = k
        while m - 1 >= 0:
            prev_start, prev_end = poke_intervals[m - 1]
            if cap_end is not None and prev_start < cap_end:
                # Don't merge intervals that start before cap_end
                break
            gap_ms = (bout_start - prev_end).total_seconds() * 1000.0
            if gap_ms <= sample_offset_time_ms:
                bout_start = prev_start
                m -= 1
            else:
                break

        # Forward merge: include next intervals if the gap <= sample_offset_time_ms (respect cap_end)
        n = k
        cur_end = bout_end
        while n + 1 < len(poke_intervals):
            next_start, next_end = poke_intervals[n + 1]
            if cap_end is not None and next_start >= cap_end:
                break
            gap_ms = (next_start - cur_end).total_seconds() * 1000.0
            if gap_ms <= sample_offset_time_ms:
                cur_end = max(cur_end, min(next_end, cap_end))
                n += 1
            else:
                break

        # Cap forward end at cap_end if provided
        bout_end_capped = cur_end
        if cap_end is not None and bout_end_capped is not None and bout_end_capped > cap_end:
            bout_end_capped = cap_end

        dur_ms = max(0.0, (bout_end_capped - bout_start).total_seconds() * 1000.0)
        return bout_start, bout_end_capped, float(dur_ms)

    def analyze_trial_valve_and_poke_times(trial_valve_events):
        position_locations = {}
        position_valve_times = {}
        position_poke_times = {}
        prior_presentations = []

        # Position 1: last individual activation of first odor
        if trial_valve_events:
            first_odor_valve = trial_valve_events[0]['valve_key']
            first_odor_activations = []
            for event in trial_valve_events:
                if event['valve_key'] == first_odor_valve:
                    first_odor_activations.append(event)
                else:
                    break
            if first_odor_activations:
                # Position 1 = LAST activation of first odor
                position_locations[1] = first_odor_activations[-1]
                # Earlier activations are prior_presentations (failed attempts)
                prior_presentations = [
                    {
                        'position': 1,
                        'odor_name': e['odor_name'],
                        'valve_start': e['start_time'],
                        'valve_end': e['end_time'],
                    }
                    for e in first_odor_activations[:-1]
                ]

        # Positions 2-5: Keep ONLY the LAST occurrence of each new odor
        # (Don't merge consecutive events with same valve_key, just track the last one)
        odor_to_pos = {}
        next_pos = 2
        
        for event in trial_valve_events[len(first_odor_activations if trial_valve_events else []):]:
            odor = event['odor_name']
            
            # If we haven't seen this odor yet, assign it a position
            if odor not in odor_to_pos and next_pos <= 5:
                odor_to_pos[odor] = next_pos
                next_pos += 1
            
            # If this odor has a position, track it (will be overwritten by later occurrences)
            if odor in odor_to_pos:
                position_locations[odor_to_pos[odor]] = event

        # Valve timing per position
        for position in range(1, 6):
            if position not in position_locations:
                continue
            loc = position_locations[position]
            valve_start = loc['start_time']
            valve_end = loc['end_time']
            valve_duration_ms = (valve_end - valve_start).total_seconds() * 1000
            entry = {
                'position': position,
                'odor_name': loc['odor_name'],
                'valve_start': valve_start,
                'valve_end': valve_end,
                'valve_duration_ms': valve_duration_ms
            }
            if position == 1:
                entry['prior_presentations'] = prior_presentations
            position_valve_times[position] = entry 

        # Poke-time analysis: use ONLY the LAST valve event per position
        poke_position_locations = dict(position_locations)

        s_bool = poke_data.astype(bool)

        # Compute consolidated poke time
        for position in range(1, 6):
            if position not in poke_position_locations:
                continue
            loc = poke_position_locations[position]
            odor_start = loc['start_time']
            odor_end = loc['end_time']

            # State at window start
            prev_slice = s_bool.loc[:odor_start]
            state_at_start = bool(prev_slice.iloc[-1]) if len(prev_slice) else False

            # Window slice
            w = s_bool.loc[odor_start:odor_end]
            if w.empty and not state_at_start:
                continue

            # Edges relative to start state
            rises = w & ~w.shift(1, fill_value=state_at_start)
            falls = ~w & w.shift(1, fill_value=state_at_start)

            # Build IN intervals within [odor_start, odor_end]
            intervals = []
            current_start = odor_start if state_at_start else None
            for ts in w.index:
                if rises.get(ts, False) and current_start is None:
                    current_start = ts
                if falls.get(ts, False) and current_start is not None:
                    intervals.append((current_start, ts))
                    current_start = None
            if current_start is not None:
                intervals.append((current_start, odor_end))  # clip at odor_end

            if not intervals:
                continue

            # Merge across gaps <= sample_offset_time_ms
            merged = [intervals[0]]
            for start, end in intervals[1:]:
                ls, le = merged[-1]
                gap_ms = (start - le).total_seconds() * 1000.0
                if gap_ms <= sample_offset_time_ms:
                    merged[-1] = (ls, max(le, end))
                else:
                    merged.append((start, end))

            first_block_ms = (merged[0][1] - merged[0][0]).total_seconds() * 1000.0
            consolidated_poke_time_ms = first_block_ms
            first_poke_in = merged[0][0] if merged else None

            if consolidated_poke_time_ms > 0:
                position_poke_times[position] = {
                    'position': position,
                    'odor_name': loc['odor_name'],
                    'poke_time_ms': consolidated_poke_time_ms,
                    'poke_odor_start': odor_start,
                    'poke_odor_end': odor_end,
                    'poke_first_in': first_poke_in,
                }

        return position_valve_times, position_poke_times

    # Process trials
    for _, trial in initiated_trials.iterrows():
        trial_start = trial['sequence_start']
        trial_end = trial['sequence_end']
        odor_sequence, valve_activations = get_trial_valve_sequence(trial_start, trial_end)

        odor_to_pos = {}
        next_pos = 1
        positions = []
        for ev in valve_activations:
            od = ev['odor_name']
            if od not in odor_to_pos and next_pos <= 5:
                odor_to_pos[od] = next_pos
                next_pos += 1
            positions.append(odor_to_pos.get(od))

        presentations = []
        for idx_in_trial, (ev, pos) in enumerate(zip(valve_activations, positions)):
            if not isinstance(pos, (int, np.integer)):
                continue
            vstart, vend = ev['start_time'], ev['end_time']
            vdur_ms = (vend - vstart).total_seconds() * 1000.0
            psum = window_poke_summary(vstart, vend)
            presentations.append({
                'index_in_trial': idx_in_trial,
                'position': int(pos),
                'odor_name': ev['odor_name'],
                'valve_start': vstart,
                'valve_end': vend,
                'valve_duration_ms': float(vdur_ms),
                'poke_time_ms': float(psum.get('poke_time_ms', 0.0)),
                'poke_first_in': psum.get('poke_first_in'),
            })

        # Last relevant odor index (ignore valve durations < sample_offset_time_ms)
        last_event_index = None
        for i in range(len(valve_activations) - 1, -1, -1):
            st, en = valve_activations[i]['start_time'], valve_activations[i]['end_time']
            if (en - st).total_seconds() * 1000.0 >= sample_offset_time_ms:
                last_event_index = i
                break


        position_valve_times, position_poke_times = analyze_trial_valve_and_poke_times(valve_activations)

        pos1_info = position_valve_times.get(1, {}) or {}
        last_pos1_start = pos1_info.get('valve_start')

        # Record earlier Position-1 presentations as non-initiated attempts with correct poke timing
        for attempt in pos1_info.get('prior_presentations', []) or []:
            a_start = attempt.get('valve_start')   # attempt valve start (for reference)
            # Cap at the last Pos1 valve START (trial starts at last odor 1 opening)
            last_pos1_valve_end = pos1_info.get('valve_end')  # The valve CLOSE time
            first_in, bout_end, dur_ms = _attempt_bout_from_poke_in(anchor_ts=a_start, cap_end=last_pos1_valve_end)
            non_initiated_odor1_attempts.append({
                'trial_id': trial['trial_id'] if 'trial_id' in trial else None,
                'attempt_start': a_start,
                'attempt_end': attempt.get('valve_end'),
                'odor_name': attempt.get('odor_name'),
                'attempt_first_poke_in': first_in,
                'attempt_poke_time_ms': dur_ms,
            })

        # Compute corrected trial start = first poke-in within last Pos1 window (existing local window logic)
        corrected_start = None
        pos1_poke = position_poke_times.get(1)
        if pos1_poke:
            corrected_start = pos1_poke.get('poke_first_in') or pos1_poke.get('poke_odor_start')

        
        trial_await_rewards = [t for t in await_reward_times if trial_start <= t <= trial_end]

        trial_dict = trial.to_dict()
        trial_dict['odor_sequence'] = odor_sequence
        trial_dict['num_odors'] = len(odor_sequence)
        trial_dict['last_odor'] = odor_sequence[-1] if odor_sequence else None
        trial_dict['hidden_rule_location'] = hidden_rule_location
        trial_dict['sequence_name'] = sequence_name
        trial_dict['position_valve_times'] = position_valve_times
        trial_dict['position_poke_times'] = position_poke_times
        trial_dict['presentations'] = presentations
        trial_dict['last_event_index'] = last_event_index
        if corrected_start is not None:
            trial_dict['sequence_start_corrected'] = corrected_start

        enough_odors, hit_hidden_rule = check_hidden_rule(odor_sequence, hidden_rule_location)
        trial_dict['enough_odors_for_hr'] = enough_odors
        trial_dict['hit_hidden_rule'] = hit_hidden_rule

        initiated_trials_list.append(trial_dict)
        if trial_await_rewards:
            # Aggregate ranges for completed trials
            # Valve times
            for pos, v in (position_valve_times or {}).items():
                if v and 'valve_duration_ms' in v:
                    agg_position_valve_times[pos].append(v['valve_duration_ms'])
                    odor_name = v.get('odor_name')
                    if odor_name:
                        agg_odor_valve_times[odor_name].append(v['valve_duration_ms'])
            # Poke times
            for pos, p in (position_poke_times or {}).items():
                if p and 'poke_time_ms' in p:
                    agg_position_poke_times[pos].append(p['poke_time_ms'])
                    odor_name = p.get('odor_name')
                    if odor_name:
                        agg_odor_poke_times[odor_name].append(p['poke_time_ms'])

            await_reward_time = min(trial_await_rewards)
            trial_dict['await_reward_time'] = await_reward_time

            if hit_hidden_rule:
                if len(odor_sequence) == hidden_rule_position:
                    completed_hr.append(trial_dict.copy())
                    hr_category = 'completed_hr'
                else:
                    completed_hr_missed.append(trial_dict.copy())
                    hr_category = 'completed_hr_missed'
            else:
                hr_category = 'completed_normal'

            supply1_after_await = [t for t in supply_port1_times if await_reward_time <= t <= trial_end]
            supply2_after_await = [t for t in supply_port2_times if await_reward_time <= t <= trial_end]

            if supply1_after_await or supply2_after_await:
                all_supply_times = []
                if supply1_after_await:
                    all_supply_times.extend([(t, 1, 'A') for t in supply1_after_await])
                if supply2_after_await:
                    all_supply_times.extend([(t, 2, 'B') for t in supply2_after_await])
                all_supply_times.sort(key=lambda x: x[0])
                first_supply_time, first_supply_port, first_supply_odor = all_supply_times[0]

                trial_dict['first_supply_time'] = first_supply_time
                trial_dict['first_supply_port'] = first_supply_port
                trial_dict['first_supply_odor_identity'] = first_supply_odor
                trial_dict['supply1_count'] = len(supply1_after_await)
                trial_dict['supply2_count'] = len(supply2_after_await)
                trial_dict['total_supply_count'] = len(supply1_after_await) + len(supply2_after_await)

                completed_rewarded.append(trial_dict.copy())
                if hr_category == 'completed_hr':
                    completed_hr_rewarded.append(trial_dict.copy())
                elif hr_category == 'completed_hr_missed':
                    completed_hr_missed_rewarded.append(trial_dict.copy())
            else:
                poke_window_end = await_reward_time + pd.Timedelta(seconds=response_time_sec)
                port1_pokes_in_window = []
                port2_pokes_in_window = []

                if not port1_pokes.empty:
                    port1_window = port1_pokes[await_reward_time:poke_window_end]
                    port1_starts = port1_window & ~port1_window.shift(1, fill_value=False)
                    port1_pokes_in_window = port1_starts[port1_starts == True].index.tolist()
                if not port2_pokes.empty:
                    port2_window = port2_pokes[await_reward_time:poke_window_end]
                    port2_starts = port2_window & ~port2_window.shift(1, fill_value=False)
                    port2_pokes_in_window = port2_starts[port2_starts == True].index.tolist()

                all_reward_pokes = []
                if port1_pokes_in_window:
                    all_reward_pokes.extend([(t, 1, 'A') for t in port1_pokes_in_window])
                if port2_pokes_in_window:
                    all_reward_pokes.extend([(t, 2, 'B') for t in port2_pokes_in_window])
                all_reward_pokes.sort(key=lambda x: x[0])

                trial_dict['poke_window_end'] = poke_window_end
                trial_dict['port1_pokes_count'] = len(port1_pokes_in_window)
                trial_dict['port2_pokes_count'] = len(port2_pokes_in_window)
                trial_dict['total_reward_pokes'] = len(all_reward_pokes)

                if all_reward_pokes:
                    first_poke_time, first_poke_port, first_poke_odor = all_reward_pokes[0]
                    trial_dict['first_reward_poke_time'] = first_poke_time
                    trial_dict['first_reward_poke_port'] = first_poke_port
                    trial_dict['first_reward_poke_odor_identity'] = first_poke_odor
                    completed_unrewarded.append(trial_dict.copy())
                    if hr_category == 'completed_hr':
                        completed_hr_unrewarded.append(trial_dict.copy())
                    elif hr_category == 'completed_hr_missed':
                        completed_hr_missed_unrewarded.append(trial_dict.copy())
                else:
                    completed_timeout.append(trial_dict.copy())
                    if hr_category == 'completed_hr':
                        completed_hr_timeout.append(trial_dict.copy())
                    elif hr_category == 'completed_hr_missed':
                        completed_hr_missed_timeout.append(trial_dict.copy())
            completed_sequences.append(trial_dict.copy())

        else:
            aborted_sequences.append(trial_dict.copy())
            if hit_hidden_rule:
                aborted_sequences_hr.append(trial_dict.copy())


    if isinstance(non_initiated_trials, pd.DataFrame) and not non_initiated_trials.empty:
        odor_names = []
        for _, row in non_initiated_trials.iterrows():
            min_time_diff = float('inf')
            closest_odor = None
            attempt_start = row.get('attempt_start') or row.get('sequence_start')
            attempt_end = row.get('attempt_end') or row.get('sequence_end')
            found_odor = None
            for olf_id, valve_data in odor_map['olfactometer_valves'].items():
                if valve_data.empty:
                    continue
                for i, valve_col in enumerate(valve_data.columns):
                    valve_key = f"{olf_id}{i}"
                    odor_name = odor_map['valve_to_odor'].get(valve_key)
                    if not odor_name or odor_name.lower() == 'purge':
                        continue
                    s = valve_data[valve_col]
                    rises = s & ~s.shift(1, fill_value=False)
                    starts = list(s.index[rises])
                    falls = ~s & s.shift(1, fill_value=False)
                    ends = list(s.index[falls])
                    for st, en in zip(starts, ends):
                        if st <= attempt_end and en >= attempt_start:
                            found_odor = odor_name
                            break
                        time_diff = min(abs((st - attempt_start).total_seconds()),
                                        abs((en - attempt_end).total_seconds()))
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            closest_odor = odor_name
                    if found_odor:
                        break
                if found_odor:
                    break
            if found_odor is None:
                found_odor = closest_odor  # fallback to closest
            odor_names.append(found_odor)
        non_initiated_trials = non_initiated_trials.copy()
        non_initiated_trials['odor_name'] = odor_names
    initiated_trials = pd.DataFrame(initiated_trials_list)
    # Build result with both singular and plural aliases for HR categories
    result = {
        'non_initiated_sequences': non_initiated_trials,
        'initiated_sequences': initiated_trials,
        'completed_sequences': pd.DataFrame(completed_sequences),
        'aborted_sequences': pd.DataFrame(aborted_sequences),
        'non_initiated_odor1_attempts': pd.DataFrame(non_initiated_odor1_attempts),

        'aborted_sequences_HR': pd.DataFrame(aborted_sequences_hr),
        'completed_sequences_HR': pd.DataFrame(completed_hr),
        'completed_sequences_HR_missed': pd.DataFrame(completed_hr_missed),

        'completed_sequence_rewarded': pd.DataFrame(completed_rewarded),
        'completed_sequence_unrewarded': pd.DataFrame(completed_unrewarded),
        'completed_sequence_reward_timeout': pd.DataFrame(completed_timeout),

        'completed_sequence_HR_rewarded': pd.DataFrame(completed_hr_rewarded),
        'completed_sequence_HR_unrewarded': pd.DataFrame(completed_hr_unrewarded),
        'completed_sequence_HR_reward_timeout': pd.DataFrame(completed_hr_timeout),
        'completed_sequence_HR_missed_rewarded': pd.DataFrame(completed_hr_missed_rewarded),
        'completed_sequence_HR_missed_unrewarded': pd.DataFrame(completed_hr_missed_unrewarded),
        'completed_sequence_HR_missed_reward_timeout': pd.DataFrame(completed_hr_missed_timeout),
    }

    if isinstance(result['non_initiated_sequences'], pd.DataFrame) and not result['non_initiated_sequences'].empty:
        df = result['non_initiated_sequences'].copy()
        if 'continuous_poke_time_ms' in df.columns:
            df['pos1_poke_time_ms'] = pd.to_numeric(df['continuous_poke_time_ms'], errors='coerce').fillna(0.0)
        result['non_initiated_sequences'] = df

    # Plural aliases to prevent KeyErrors in downstream code
    result['completed_sequences_HR_rewarded'] = result['completed_sequence_HR_rewarded']
    result['completed_sequences_HR_unrewarded'] = result['completed_sequence_HR_unrewarded']
    result['completed_sequences_HR_reward_timeout'] = result['completed_sequence_HR_reward_timeout']
    result['completed_sequences_HR_missed_rewarded'] = result['completed_sequence_HR_missed_rewarded']
    result['completed_sequences_HR_missed_unrewarded'] = result['completed_sequence_HR_missed_unrewarded']
    result['completed_sequences_HR_missed_reward_timeout'] = result['completed_sequence_HR_missed_reward_timeout']

    result['hidden_rule_position'] = hidden_rule_position
    result['hidden_rule_odors'] = sorted(list(hr_odor_set)) if hr_odor_set is not None else []

    if verbose:
        print(f"\nTRIAL CLASSIFICATION RESULTS WITH HIDDEN RULE AND VALVE/POKE TIME ANALYSIS:")
        print(f"Hidden Rule Location: Position {hidden_rule_position} (index {hidden_rule_location})\n")
        print(f"Hidden Rule Odors: {', '.join(result['hidden_rule_odors']) if result['hidden_rule_odors'] else 'None'}\n")

        # safe percent helper
        def _pct(n, d):
            try:
                d = float(d)
                return 0.0 if d == 0 else (float(n) / d * 100.0)
            except Exception:
                return 0.0

        base_non_init_df = result.get('non_initiated_sequences', pd.DataFrame())
        pos1_attempts_df = result.get('non_initiated_odor1_attempts', pd.DataFrame())

        base_non_init_count = 0 if base_non_init_df is None or base_non_init_df.empty else len(base_non_init_df)
        pos1_attempts_count = 0 if pos1_attempts_df is None or pos1_attempts_df.empty else len(pos1_attempts_df)

        total_non_init = base_non_init_count + pos1_attempts_count
        ini_n = len(initiated_trials)
        total_attempts = ini_n + total_non_init

        print(f"Total attempts: {total_attempts}")
        print(f"-- Non-initiated sequences (total): {total_non_init} ({_pct(total_non_init, total_attempts):.1f}%)")
        if pos1_attempts_count:
            print(f"    -- Position 1 attempts within trials {pos1_attempts_count} ({_pct(pos1_attempts_count, total_non_init):.1f}%)")
            print(f"    -- Baseline non-initiated sequences {base_non_init_count} ({_pct(base_non_init_count, total_non_init):.1f}%)")
        print(f"-- Initiated sequences (trials): {ini_n} ({_pct(ini_n, total_attempts):.1f}%)\n")

        print("INITIATED TRIALS BREAKDOWN:")
        comp_n = len(result['completed_sequences'])
        print(f"-- Completed sequences: {comp_n} ({_pct(comp_n, ini_n):.1f}%)")
        print(f"   -- Hidden Rule trials (HR): {len(result['completed_sequences_HR'])} ({_pct(len(result['completed_sequences_HR']), ini_n):.1f}%)")
        print(f"   -- Hidden Rule Missed (HR_missed): {len(result['completed_sequences_HR_missed'])} ({_pct(len(result['completed_sequences_HR_missed']), ini_n):.1f}%)")
        print(f"-- Aborted sequences: {len(result['aborted_sequences'])} ({_pct(len(result['aborted_sequences']), ini_n):.1f}%)")
        print(f"   -- Aborted Hidden Rule trials (HR): {len(result['aborted_sequences_HR'])} ({_pct(len(result['aborted_sequences_HR']), ini_n):.1f}%)\n")

        print("REWARD STATUS BREAKDOWN:")
        cs = comp_n
        if cs > 0:
            print(f"-- Rewarded: {len(result['completed_sequence_rewarded'])} ({_pct(len(result['completed_sequence_rewarded']), cs):.1f}%)")
            print(f"-- Unrewarded: {len(result['completed_sequence_unrewarded'])} ({_pct(len(result['completed_sequence_unrewarded']), cs):.1f}%)")
            print(f"-- Reward timeout: {len(result['completed_sequence_reward_timeout'])} ({_pct(len(result['completed_sequence_reward_timeout']), cs):.1f}%)\n")

        print("HIDDEN RULE SPECIFIC BREAKDOWN:")
        hr_total = len(result['completed_sequences_HR'])
        if hr_total > 0:
            print(f"-- HR Rewarded: {len(result['completed_sequence_HR_rewarded'])} ({_pct(len(result['completed_sequence_HR_rewarded']), hr_total):.1f}%)")
            print(f"-- HR Unrewarded: {len(result['completed_sequence_HR_unrewarded'])} ({_pct(len(result['completed_sequence_HR_unrewarded']), hr_total):.1f}%)")
            print(f"-- HR Timeout: {len(result['completed_sequence_HR_reward_timeout'])} ({_pct(len(result['completed_sequence_HR_reward_timeout']), hr_total):.1f}%)")

        hr_missed_total = len(result['completed_sequences_HR_missed'])
        if hr_missed_total > 0:
            print(f"Completed HR Missed trials: {hr_missed_total}")
            print(f"-- HR Missed Rewarded: {len(result['completed_sequence_HR_missed_rewarded'])} ({len(result['completed_sequence_HR_missed_rewarded'])/hr_missed_total*100:.1f}%)")
            print(f"-- HR Missed Unrewarded: {len(result['completed_sequence_HR_missed_unrewarded'])} ({len(result['completed_sequence_HR_missed_unrewarded'])/hr_missed_total*100:.1f}%)")
            print(f"-- HR Missed Timeout: {len(result['completed_sequence_HR_missed_reward_timeout'])} ({len(result['completed_sequence_HR_missed_reward_timeout'])/hr_missed_total*100:.1f}%)")
        print()

        # Additional requested summaries
        print("POKE TIME RANGES BY POSITION:")
        print("-" * 40)
        for pos in range(1, 6):
            times = agg_position_poke_times[pos]
            if times:
                min_v = min(times); max_v = max(times); avg_v = sum(times) / len(times)
                print(f"Position {pos}: {min_v:.1f} - {max_v:.1f}ms (avg: {avg_v:.1f}ms, n={len(times)})")
            else:
                print(f"Position {pos}: No data")

        print("\nVALVE TIME RANGES BY POSITION:")
        print("-" * 40)
        for pos in range(1, 6):
            times = agg_position_valve_times[pos]
            if times:
                min_v = min(times); max_v = max(times); avg_v = sum(times) / len(times)
                print(f"Position {pos}: {min_v:.1f} - {max_v:.1f}ms (avg: {avg_v:.1f}ms, n={len(times)})")
            else:
                print(f"Position {pos}: No data")

        print("\nPOKE TIME RANGES BY ODOR (ALL POSITIONS):")
        print("-" * 50)
        for odor_name in sorted(agg_odor_poke_times.keys()):
            times = agg_odor_poke_times[odor_name]
            if times:
                min_v = min(times); max_v = max(times); avg_v = sum(times) / len(times)
                print(f"{odor_name}: {min_v:.1f} - {max_v:.1f}ms (avg: {avg_v:.1f}ms, n={len(times)})")

        print("\nVALVE TIME RANGES BY ODOR (ALL POSITIONS):")
        print("-" * 50)
        for odor_name in sorted(agg_odor_valve_times.keys()):
            times = agg_odor_valve_times[odor_name]
            if times:
                min_v = min(times); max_v = max(times); avg_v = sum(times) / len(times)
                print(f"{odor_name}: {min_v:.1f} - {max_v:.1f}ms (avg: {avg_v:.1f}ms, n={len(times)})")

        print("\nNON-INITIATED TRIALS POKE TIMES:")
        print("-" * 40)
        if not result['non_initiated_sequences'].empty:
            base = result['non_initiated_sequences']
            pos1 = result['non_initiated_odor1_attempts']
            print(f"Baseline non-initiated: n={len(base)} avg={base['pos1_poke_time_ms'].mean():.1f} ms range={base['pos1_poke_time_ms'].min():.1f}-{base['pos1_poke_time_ms'].max():.1f} ms")
            if not pos1.empty:
                s = pd.to_numeric(pos1['attempt_poke_time_ms'], errors='coerce').dropna()
                print(f"Pos1 attempts: n={len(s)} avg={s.mean():.1f} ms range={s.min():.1f}-{s.max():.1f} ms")
            else:
                print("Pos1 attempts: n=0")

        # Verify totals
        total_classified = (len(result['completed_sequence_rewarded'])
                            + len(result['completed_sequence_unrewarded'])
                            + len(result['completed_sequence_reward_timeout'])
                            + len(result['aborted_sequences']))
        if total_classified == len(initiated_trials):
            print(f"\nClassification complete: all {len(initiated_trials)} trials classified")
        else:
            print(f"\nClassification mismatch: {total_classified} classified vs {len(initiated_trials)} total")

    return result

def analyze_response_times(data, trial_counts, events, odor_map, stage, root, verbose=True):# Analyze response times for all completed trials. Part of wrapper function
    """
    Analyze response times for all completed trials (clean version).
    Behavior and prints match analyze_response_times_all_trials_fixed.

    Returns a dict with:
      - rewarded_response_times
      - unrewarded_response_times
      - timeout_delayed_response_times
      - timeout_response_delay_times
      - all_response_times
      - failed_calculations
    """

    sample_offset_time, minimum_sampling_time, response_time = get_experiment_parameters(root)
    sample_offset_time_ms = sample_offset_time * 1000
    minimum_sampling_time_ms = minimum_sampling_time * 1000
    response_time_sec = response_time 
    if response_time_sec is None:
        raise ValueError("Response time parameter cannot be extracted from Schema file. Check detect_settings function.")

    if verbose:
        print("=" * 80)
        print("RESPONSE TIME ANALYSIS - ALL COMPLETED TRIALS")
        print("=" * 80)

    # Extract hidden rule location
    hidden_rule_location = None
    sequence_name = None
    if isinstance(stage, dict):
        sequence_name = stage.get('stage_name') or str(stage)
        if stage.get('hidden_rule_index') is not None:
            try:
                hidden_rule_location = int(stage['hidden_rule_index'])
            except Exception:
                hidden_rule_location = None
    if hidden_rule_location is None:
        sequence_name = sequence_name or str(stage)
        m = re.search(r'Location(\d+)', sequence_name)
        if m:
            hidden_rule_location = int(m.group(1))
    hidden_rule_position = hidden_rule_location + 1 if isinstance(hidden_rule_location, int) else None
    if verbose:
        if hidden_rule_location is not None:
            print(f"Hidden rule location extracted: Location{hidden_rule_location} (index {hidden_rule_location}, position {hidden_rule_position})")
        else:
            print(f"No Hidden Rule Location found in sequence name: {sequence_name}. Proceeding without Hidden Rule analysis.")

    # Get initiated trials and events (same as main function)
    initiated_trials = trial_counts['initiated_sequences']
    await_reward_times = events['combined_await_reward_df']['Time'].tolist() if 'combined_await_reward_df' in events else []

    # Get supply port activities for reward classification
    supply_port1_times = data['pulse_supply_1'].index.tolist() if not data['pulse_supply_1'].empty else []
    supply_port2_times = data['pulse_supply_2'].index.tolist() if not data['pulse_supply_2'].empty else []

    # Filter for completed trials
    completed_trials_all = []
    for _, trial in initiated_trials.iterrows():
        trial_start = trial['sequence_start']
        trial_end = trial['sequence_end']

        trial_await_rewards = [t for t in await_reward_times if trial_start <= t <= trial_end]
        if trial_await_rewards:
            trial_dict = trial.to_dict()
            trial_dict['await_reward_time'] = min(trial_await_rewards)
            completed_trials_all.append(trial_dict)

    if verbose:
        print(f"Total completed trials: {len(completed_trials_all)}\n")

    # Get poke and port data
    poke_data = data['digital_input_data']['DIPort0'].copy() if 'DIPort0' in data['digital_input_data'] else pd.Series(dtype=bool)
    port1_pokes = data['digital_input_data']['DIPort1'] if 'DIPort1' in data['digital_input_data'] else pd.Series(dtype=bool)
    port2_pokes = data['digital_input_data']['DIPort2'] if 'DIPort2' in data['digital_input_data'] else pd.Series(dtype=bool)

    # Build valve activation list
    olfactometer_valves = odor_map['olfactometer_valves']
    valve_to_odor = odor_map['valve_to_odor']

    all_valve_activations = []
    for olf_id, valve_data in olfactometer_valves.items():
        if valve_data.empty:
            continue
        for i, valve_col in enumerate(valve_data.columns):
            valve_key = f"{olf_id}{i}"
            if valve_key in valve_to_odor:
                odor_name = valve_to_odor[valve_key]
                if odor_name.lower() == 'purge':
                    continue

                valve_series = valve_data[valve_col]
                valve_activations = valve_series & ~valve_series.shift(1, fill_value=False)
                activation_times = valve_activations[valve_activations == True].index.tolist()
                valve_deactivations = ~valve_series & valve_series.shift(1, fill_value=False)
                deactivation_times = valve_deactivations[valve_deactivations == True].index.tolist()

                for activation_time in activation_times:
                    next_deactivations = [t for t in deactivation_times if t > activation_time]
                    deactivation_time = min(next_deactivations) if next_deactivations else valve_series.index[-1]

                    all_valve_activations.append({
                        'start_time': activation_time,
                        'end_time': deactivation_time,
                        'odor_name': odor_name,
                        'valve_key': valve_key
                    })

    all_valve_activations.sort(key=lambda x: x['start_time'])

    # Helpers
    def get_trial_valve_sequence(trial_start, trial_end):
        trial_valve_activations = []
        for valve_activation in all_valve_activations:
            valve_start = valve_activation['start_time']
            valve_end = valve_activation['end_time']
            if valve_start <= trial_end and valve_end >= trial_start:
                trial_valve_activations.append(valve_activation)
        trial_valve_activations.sort(key=lambda x: x['start_time'])
        odor_sequence = [activation['odor_name'] for activation in trial_valve_activations]
        return odor_sequence, trial_valve_activations

    hr_odor_set = None
    if hidden_rule_location is not None:
        try:
            _, schema_settings = detect_settings.detect_settings(root)
            odors = (schema_settings.get('hiddenRuleOdorsInferred') or [])
            if len(odors) < 2:
                raise ValueError("Hidden Rule Odor Identities could not be inferred from Schema.")
            hr_odor_set = set(map(str, odors))
            if verbose:
                print(f"Hidden Rule Odors inferred: {sorted(hr_odor_set)}")
        except Exception as e:
            raise ValueError(f"Hidden Rule Odor Identities could not be inferred from Schema: {e}")


    def check_hidden_rule(odor_sequence, idx):
        if idx is None or hr_odor_set is None:
            return False, False
        if idx < 0 or idx >= len(odor_sequence):
            return False, False
        odor_at_location = odor_sequence[idx]
        hit_hidden_rule = odor_at_location in hr_odor_set
        return True, hit_hidden_rule

    def find_next_trial_start(current_trial_end, all_trials):
        next_starts = [t['sequence_start'] for t in all_trials if t['sequence_start'] > current_trial_end]
        return min(next_starts) if next_starts else None

    # Analyze all completed trials
    rewarded_response_times = []
    unrewarded_response_times = []
    timeout_delayed_response_times = []
    timeout_response_delay_times = []
    failed_calculations = 0
    hr_rewarded_response_times = []
    per_trial_rows = []

    for trial_dict in completed_trials_all:
        trial_id = trial_dict.get('trial_id')
        trial_start = trial_dict['sequence_start']
        trial_end = trial_dict['sequence_end']
        await_reward_time = trial_dict['await_reward_time']

        # Get valve sequence
        odor_sequence, trial_valve_events = get_trial_valve_sequence(trial_start, trial_end)
        if not trial_valve_events:
            failed_calculations += 1
            per_trial_rows.append({
                'trial_id': trial_id,
                'response_time_ms': np.nan,
                'response_time_category': None,
            })
            continue

        # Check hidden rule
        _, hit_hidden_rule = check_hidden_rule(odor_sequence, hidden_rule_location)

        # Determine target odor position
        if hit_hidden_rule and len(odor_sequence) == hidden_rule_position:
            target_position = hidden_rule_location
        else:
            target_position = len(odor_sequence) - 1

        # Find target valve event
        target_valve_event = None
        if target_position == 0:
            # Position 1: last individual activation of first odor
            first_odor_valve = trial_valve_events[0]['valve_key']
            first_odor_activations = []
            for event in trial_valve_events:
                if event['valve_key'] == first_odor_valve:
                    first_odor_activations.append(event)
                else:
                    break
            if first_odor_activations:
                target_valve_event = first_odor_activations[-1]
        else:
            # Group consecutive events
            grouped_presentations = []
            current_valve = None
            current_start_time = None
            current_end_time = None
            current_odor_name = None

            for event in trial_valve_events:
                if event['valve_key'] != current_valve:
                    if current_valve is not None:
                        grouped_presentations.append({
                            'valve_key': current_valve,
                            'odor_name': current_odor_name,
                            'start_time': current_start_time,
                            'end_time': current_end_time
                        })
                    current_valve = event['valve_key']
                    current_odor_name = event['odor_name']
                    current_start_time = event['start_time']
                    current_end_time = event['end_time']
                else:
                    current_end_time = event['end_time']

            if current_valve is not None:
                grouped_presentations.append({
                    'valve_key': current_valve,
                    'odor_name': current_odor_name,
                    'start_time': current_start_time,
                    'end_time': current_end_time
                })

            # Map odor sequence positions to group indices
            odor_position_to_group = {}
            group_idx = 0
            current_group_valve = None

            for seq_pos, (odor_name, valve_event) in enumerate(zip(odor_sequence, trial_valve_events)):
                if valve_event['valve_key'] != current_group_valve:
                    current_group_valve = valve_event['valve_key']
                    if group_idx < len(grouped_presentations):
                        odor_position_to_group[seq_pos] = group_idx
                        group_idx += 1
                else:
                    if seq_pos > 0 and (seq_pos - 1) in odor_position_to_group:
                        odor_position_to_group[seq_pos] = odor_position_to_group[seq_pos - 1]

            if target_position in odor_position_to_group:
                group_index = odor_position_to_group[target_position]
                target_valve_event = grouped_presentations[group_index]

        if target_valve_event is None:
            failed_calculations += 1
            per_trial_rows.append({
                'trial_id': trial_id,
                'response_time_ms': np.nan,
                'response_time_category': None,
            })
            continue

        # Find last poke out in extended window around target odor
        odor_start = target_valve_event['start_time']
        odor_end = target_valve_event['end_time']
        search_end = max(await_reward_time, odor_end + pd.Timedelta(seconds=1))

        extended_poke_data = poke_data.loc[odor_start:search_end]
        if extended_poke_data.empty:
            failed_calculations += 1
            per_trial_rows.append({
                'trial_id': trial_id,
                'response_time_ms': np.nan,
                'response_time_category': None,
            })
            continue

        last_poke_out_time = None
        prev_state = poke_data.loc[:odor_start].iloc[-1] if len(poke_data.loc[:odor_start]) > 0 else False
        for timestamp, current_state in extended_poke_data.items():
            if prev_state and not current_state:
                last_poke_out_time = timestamp
            prev_state = current_state

        if last_poke_out_time is None:
            failed_calculations += 1
            per_trial_rows.append({
                'trial_id': trial_id,
                'response_time_ms': np.nan,
                'response_time_category': None,
            })
            continue

        # Reward window and search for reward pokes
        poke_window_end = await_reward_time + pd.Timedelta(seconds=response_time_sec)
        search_start = max(last_poke_out_time, await_reward_time)

        port1_pokes_in_window = []
        port2_pokes_in_window = []

        if not port1_pokes.empty:
            port1_window = port1_pokes[search_start:poke_window_end]
            port1_starts = port1_window & ~port1_window.shift(1, fill_value=False)
            port1_pokes_in_window = port1_starts[port1_starts == True].index.tolist()

        if not port2_pokes.empty:
            port2_window = port2_pokes[search_start:poke_window_end]
            port2_starts = port2_window & ~port2_window.shift(1, fill_value=False)
            port2_pokes_in_window = port2_starts[port2_starts == True].index.tolist()

        all_reward_pokes = port1_pokes_in_window + port2_pokes_in_window

        response_time_ms = None
        if all_reward_pokes:
            first_reward_poke = min(all_reward_pokes)
            response_time_ms = (first_reward_poke - last_poke_out_time).total_seconds() * 1000

        # Determine reward status (same as main classification)
        supply1_after_await = [t for t in supply_port1_times if await_reward_time <= t <= trial_end]
        supply2_after_await = [t for t in supply_port2_times if await_reward_time <= t <= trial_end]

        # NEW: authoritative per-trial categorization + row capture
        is_rewarded = bool(supply1_after_await or supply2_after_await)
        if is_rewarded:
            if response_time_ms is not None:
                rewarded_response_times.append(response_time_ms)
                # optional: HR subset if you track it
                if hit_hidden_rule and len(odor_sequence) == hidden_rule_position:
                    hr_rewarded_response_times.append(response_time_ms)
                per_trial_rows.append({
                    'trial_id': trial_id,
                    'response_time_ms': float(response_time_ms),
                    'response_time_category': 'rewarded',
                })
            else:
                failed_calculations += 1
                per_trial_rows.append({
                    'trial_id': trial_id,
                    'response_time_ms': np.nan,
                    'response_time_category': None,
                })
        else:
            # Check full window from await_reward for unrewarded vs timeout
            port1_full_window = []
            port2_full_window = []
            if not port1_pokes.empty:
                port1_window_full = port1_pokes[await_reward_time:poke_window_end]
                port1_starts_full = port1_window_full & ~port1_window_full.shift(1, fill_value=False)
                port1_full_window = port1_starts_full[port1_starts_full == True].index.tolist()
            if not port2_pokes.empty:
                port2_window_full = port2_pokes[await_reward_time:poke_window_end]
                port2_starts_full = port2_window_full & ~port2_window_full.shift(1, fill_value=False)
                port2_full_window = port2_starts_full[port2_starts_full == True].index.tolist()

            is_unrewarded = bool(port1_full_window or port2_full_window)
            if is_unrewarded:
                if response_time_ms is not None:
                    unrewarded_response_times.append(response_time_ms)
                    per_trial_rows.append({
                        'trial_id': trial_id,
                        'response_time_ms': float(response_time_ms),
                        'response_time_category': 'unrewarded',
                    })
                else:
                    failed_calculations += 1
                    per_trial_rows.append({
                        'trial_id': trial_id,
                        'response_time_ms': np.nan,
                        'response_time_category': None,
                    })
            else:
                # Timeout: look for delayed responses until next completed trial start
                next_trial_start = find_next_trial_start(trial_end, completed_trials_all)
                extended_search_end = next_trial_start if next_trial_start else (poke_data.index[-1] if not poke_data.empty else poke_window_end)

                delayed_search_start = poke_window_end

                delayed_port1_pokes = []
                delayed_port2_pokes = []
                if not port1_pokes.empty and delayed_search_start < extended_search_end:
                    w = port1_pokes[delayed_search_start:extended_search_end]
                    s = w & ~w.shift(1, fill_value=False)
                    delayed_port1_pokes = s[s == True].index.tolist()
                if not port2_pokes.empty and delayed_search_start < extended_search_end:
                    w = port2_pokes[delayed_search_start:extended_search_end]
                    s = w & ~w.shift(1, fill_value=False)
                    delayed_port2_pokes = s[s == True].index.tolist()

                delayed_reward_pokes = delayed_port1_pokes + delayed_port2_pokes
                if delayed_reward_pokes:
                    first_delayed = min(delayed_reward_pokes)
                    response_time_ms = (first_delayed - last_poke_out_time).total_seconds() * 1000
                    timeout_delayed_response_times.append(response_time_ms)
                    # also store delay beyond window if desired
                    timeout_response_delay_times.append((first_delayed - poke_window_end).total_seconds() * 1000.0)
                    per_trial_rows.append({
                        'trial_id': trial_id,
                        'response_time_ms': float(response_time_ms),
                        'response_time_category': 'timeout_delayed',
                    })
                else:
                    failed_calculations += 1
                    per_trial_rows.append({
                        'trial_id': trial_id,
                        'response_time_ms': np.nan,
                        'response_time_category': None,
                    })

    # Print results
    if verbose:
        print(f"RESPONSE TIME ANALYSIS RESULTS:")
        print(f"Total completed trials analyzed: {len(completed_trials_all)}")
        print(f"Failed response time calculations: {failed_calculations}")
        print(f"Successful response time calculations: {len(rewarded_response_times) + len(unrewarded_response_times) + len(timeout_delayed_response_times)}")
        print()

        print(f"REWARDED TRIALS:")
        if rewarded_response_times:
            print(f"  Count: {len(rewarded_response_times)}")
            print(f"  Range: {min(rewarded_response_times):.1f} - {max(rewarded_response_times):.1f}ms")
            print(f"  Average: {sum(rewarded_response_times) / len(rewarded_response_times):.1f}ms")
            print(f"  Median: {sorted(rewarded_response_times)[len(rewarded_response_times)//2]:.1f}ms")
        else:
            print(f"  No rewarded trials with response times")

        if hr_rewarded_response_times:
            print(f"\nHR REWARDED TRIALS (response times):")
            print(f"  Count: {len(hr_rewarded_response_times)}")
            print(f"  Range: {min(hr_rewarded_response_times):.1f} - {max(hr_rewarded_response_times):.1f}ms")
            print(f"  Average: {sum(hr_rewarded_response_times)/len(hr_rewarded_response_times):.1f}ms")
        else:
            print(f"\nHR REWARDED TRIALS (response times): none")

        print(f"\nUNREWARDED TRIALS:")
        if unrewarded_response_times:
            print(f"  Count: {len(unrewarded_response_times)}")
            print(f"  Range: {min(unrewarded_response_times):.1f} - {max(unrewarded_response_times):.1f}ms")
            print(f"  Average: {sum(unrewarded_response_times) / len(unrewarded_response_times):.1f}ms")
            print(f"  Median: {sorted(unrewarded_response_times)[len(unrewarded_response_times)//2]:.1f}ms")
        else:
            print(f"  No unrewarded trials with response times")

        print(f"\nTIMEOUT TRIALS WITH DELAYED RESPONSES:")
        if timeout_delayed_response_times:
            print(f"  Count: {len(timeout_delayed_response_times)}")
            print(f"  Response time (poke out to delayed poke):")
            print(f"    Range: {min(timeout_delayed_response_times):.1f} - {max(timeout_delayed_response_times):.1f}ms")
            print(f"    Average: {sum(timeout_delayed_response_times) / len(timeout_delayed_response_times):.1f}ms")
            print(f"    Median: {sorted(timeout_delayed_response_times)[len(timeout_delayed_response_times)//2]:.1f}ms")
            print(f"  Response delay time (window end to delayed poke):")
            print(f"    Range: {min(timeout_response_delay_times):.1f} - {max(timeout_response_delay_times):.1f}ms")
            print(f"    Average: {sum(timeout_response_delay_times) / len(timeout_response_delay_times):.1f}ms")
            print(f"    Median: {sorted(timeout_response_delay_times)[len(timeout_response_delay_times)//2]:.1f}ms")
        else:
            print(f"  No timeout trials with delayed responses")

        print(f"\nALL TRIALS WITH RESPONSE TIMES:")
        all_response_times = rewarded_response_times + unrewarded_response_times + timeout_delayed_response_times
        if all_response_times:
            print(f"  Count: {len(all_response_times)}")
            print(f"  Range: {min(all_response_times):.1f} - {max(all_response_times):.1f}ms")
            print(f"  Average: {sum(all_response_times) / len(all_response_times):.1f}ms")
            print(f"  Median: {sorted(all_response_times)[len(all_response_times)//2]:.1f}ms")

    all_response_times = rewarded_response_times + unrewarded_response_times + timeout_delayed_response_times

    # NEW: build per-trial DataFrame
    per_trial_df = pd.DataFrame(per_trial_rows)

    return {
        'rewarded_response_times': rewarded_response_times,
        'unrewarded_response_times': unrewarded_response_times,
        'timeout_delayed_response_times': timeout_delayed_response_times,
        'timeout_response_delay_times': timeout_response_delay_times,
        'all_response_times': all_response_times,
        'failed_calculations': failed_calculations,
        'per_trial': per_trial_df,  # NEW
    }

def abortion_classification(data, events, classification, odor_map, root, verbose=True): # Classify aborted trials with response times, poke times, FA etc. Part of wrapper function
    """
    Further classify aborted trials:
      - Compute valve and poke times per odor presentation (same rules as other trials)
      - Determine last relevant odor (last valve open with duration >= sample_offset_time_ms)
      - Compute poke time for that last odor (from poke-in that covers/starts at valve_start,
        merging gaps <= sample_offset_time_ms, ends at first large gap)
      - Abortion type:
          * reinitiation_abortion if last-odor poke >= minimum_sampling_time_ms
          * initiation_abortion otherwise
      - Abortion time = last cue-port poke-out within the trial window
      - False alarm (FA) detection window = (abortion_time, next cue-port poke after next InitiationSequence)
          * If any reward-port poke occurs in this window -> FA
            - FA_time_in: latency <= response_time
            - FA_time_out: latency <= 3 * response_time
            - FA_late: latency > 3 * response_time
          * Else nFA

    Returns:
      pd.DataFrame with detailed aborted trials:
        ['trial_id','sequence_start','sequence_end','odor_sequence',
         'position_valve_times','position_poke_times',
         'last_odor_position','last_odor_name','last_odor_valve_duration_ms',
         'last_odor_poke_time_ms','abortion_type',
         'abortion_time','fa_label','fa_time','fa_latency_ms']
    """

    DIP0 = data['digital_input_data'].get('DIPort0', pd.Series(dtype=bool)).astype(bool)
    DIP1 = data['digital_input_data'].get('DIPort1', pd.Series(dtype=bool)).astype(bool)
    DIP2 = data['digital_input_data'].get('DIPort2', pd.Series(dtype=bool)).astype(bool)
    
    # NOW use them
    dip1_rises = DIP1[DIP1 & ~DIP1.shift(1, fill_value=False)].index.tolist()
    dip2_rises = DIP2[DIP2 & ~DIP2.shift(1, fill_value=False)].index.tolist()
    reward_rises = sorted(dip1_rises + dip2_rises)

    # Parameters
    sample_offset_time, minimum_sampling_time, response_time = get_experiment_parameters(root)
    sample_offset_time_ms = float(sample_offset_time) * 1000.0
    minimum_sampling_time_ms = float(minimum_sampling_time) * 1000.0
    response_time_ms = float(response_time) * 1000.0

    # Inputs
    aborted_df = classification.get('aborted_sequences', pd.DataFrame())
    if not isinstance(aborted_df, pd.DataFrame) or aborted_df.empty:
        if verbose:
            print("abortion_classification: no aborted trials found.")
        return pd.DataFrame()

    DIP0 = data['digital_input_data'].get('DIPort0', pd.Series(dtype=bool)).astype(bool)  # cue port
    DIP1 = data['digital_input_data'].get('DIPort1', pd.Series(dtype=bool)).astype(bool)  # reward port 1
    DIP2 = data['digital_input_data'].get('DIPort2', pd.Series(dtype=bool)).astype(bool)  # reward port 2

    # Build global poke intervals for cue port
    def build_intervals(series_bool):
        rises = series_bool & ~series_bool.shift(1, fill_value=False)
        falls = ~series_bool & series_bool.shift(1, fill_value=False)
        starts = list(series_bool.index[rises])
        ends = list(series_bool.index[falls])
        intervals = []
        i = j = 0
        while i < len(starts) and j < len(ends):
            if ends[j] <= starts[i]:
                j += 1
                continue
            intervals.append((starts[i], ends[j]))
            i += 1
            j += 1
        return intervals

    cue_intervals = build_intervals(DIP0)

    # Reward-port rising edges (for FA)
    def rising_times(series_bool):
        rises = series_bool & ~series_bool.shift(1, fill_value=False)
        return list(series_bool.index[rises])

    reward_rises = sorted(rising_times(DIP1) + rising_times(DIP2))
    cue_rises = rising_times(DIP0)

    # Helper: bout from poke-in that covers/starts after anchor, merging gaps <= sample_offset_time_ms; no cap
    def bout_from_anchor(anchor_ts):
        if anchor_ts is None or not cue_intervals:
            return None, None, 0.0
        starts_only = [s for s, _ in cue_intervals]
        # interval covering anchor?
        idx = bisect_right(starts_only, anchor_ts) - 1
        within = None
        if 0 <= idx < len(cue_intervals):
            s0, e0 = cue_intervals[idx]
            if s0 <= anchor_ts < e0:
                within = idx
        if within is not None:
            k = within
        else:
            k = bisect_left(starts_only, anchor_ts)
            if k >= len(cue_intervals):
                return None, None, 0.0
        # merge forward across short gaps
        bout_start, cur_end = cue_intervals[k]
        m = k
        while m + 1 < len(cue_intervals):
            s2, e2 = cue_intervals[m + 1]
            gap_ms = (s2 - cur_end).total_seconds() * 1000.0
            if gap_ms <= sample_offset_time_ms:
                cur_end = max(cur_end, e2)
                m += 1
            else:
                break
        dur_ms = max(0.0, (cur_end - bout_start).total_seconds() * 1000.0)
        return bout_start, cur_end, float(dur_ms)

    # Build all valve activations (exclude Purge) with odor names
    olfactometer_valves = odor_map.get('olfactometer_valves', {})
    valve_to_odor = odor_map.get('valve_to_odor', {})

    def resolve_odor_name(olf_id, idx, col=None):
        # Try explicit mapping variants
        name = valve_to_odor.get((olf_id, idx))
        if name is None and col is not None:
            name = valve_to_odor.get(col)
        if name is None:
            name = valve_to_odor.get(f"{olf_id}{idx}")
        # Fallback to grid map
        if not isinstance(name, str):
            grid = odor_map.get('odour_to_olfactometer_map') or odor_map.get('odor_to_olfactometer_map')
            if isinstance(grid, (list, tuple)) and len(grid) > olf_id:
                row = grid[olf_id]
                if isinstance(row, (list, tuple)) and 0 <= idx < len(row):
                    name = row[idx]
        return name if isinstance(name, str) else None

    all_valve_activations = []
    for olf_id, df in olfactometer_valves.items():
        if df is None or getattr(df, 'empty', True):
            continue
        for i, col in enumerate(df.columns):
            odor_name = resolve_odor_name(olf_id, i, col=col)
            if not odor_name or odor_name.lower() == 'purge':
                continue
            s = df[col].astype(bool)
            rises = s & ~s.shift(1, fill_value=False)
            falls = ~s & s.shift(1, fill_value=False)
            starts = list(s.index[rises])
            ends = list(s.index[falls])
            j = 0
            for st in starts:
                while j < len(ends) and ends[j] <= st:
                    j += 1
                if j >= len(ends):
                    break
                all_valve_activations.append({
                    'start_time': starts[starts.index(st)],  # keep ref
                    'end_time': ends[j],
                    'odor_name': odor_name,
                    'olf_id': olf_id,
                    'col_index': i,
                })
                j += 1
    all_valve_activations.sort(key=lambda x: x['start_time'])

    # Helpers to extract trial events and per-odor poke times
    def trial_valve_events(t_start, t_end):
        evs = []
        for ev in all_valve_activations:
            if ev['end_time'] <= t_start:
                continue
            if ev['start_time'] >= t_end:
                break  # because sorted
            evs.append(ev)
        return evs

    def window_poke_summary(window_start, window_end):
        if window_start is None or window_end is None or window_start >= window_end:
            return {'poke_time_ms': 0.0, 'poke_first_in': None, 'poke_odor_start': window_start}
        s_bool = DIP0  # cue-port boolean series
        s_bool = s_bool.sort_index()
        prev = s_bool.loc[:window_start]
        in_at_start = bool(prev.iloc[-1]) if len(prev) else False
        w = s_bool.loc[window_start:window_end]
        if w.empty and not in_at_start:
            return {'poke_time_ms': 0.0, 'poke_first_in': None, 'poke_odor_start': window_start}
        rises = w & ~w.shift(1, fill_value=in_at_start)
        falls = ~w & w.shift(1, fill_value=in_at_start)
        intervals = []
        cur = window_start if in_at_start else None
        first_in = window_start if in_at_start else None
        for ts in w.index:
            if rises.get(ts, False) and cur is None:
                cur = ts
                if first_in is None:
                    first_in = ts
            if falls.get(ts, False) and cur is not None:
                intervals.append((cur, ts))
                cur = None
        if cur is not None:
            intervals.append((cur, window_end))
        if not intervals:
            return {'poke_time_ms': 0.0, 'poke_first_in': None, 'poke_odor_start': window_start}
        merged = [intervals[0]]
        for s2, e2 in intervals[1:]:
            ls, le = merged[-1]
            gap_ms = (s2 - le).total_seconds() * 1000.0
            if gap_ms <= sample_offset_time_ms:
                merged[-1] = (ls, max(le, e2))
            else:
                merged.append((s2, e2))
        first_block_ms = (merged[0][1] - merged[0][0]).total_seconds() * 1000.0
        return {'poke_time_ms': float(first_block_ms), 'poke_first_in': first_in, 'poke_odor_start': window_start}

    # InitiationSequence times (for FA end window)
    init_times = []
    ci_key = 'combined_initiation_sequence_df'
    if ci_key in events and isinstance(events[ci_key], pd.DataFrame) and not events[ci_key].empty:
        init_times = list(events[ci_key]['Time'])

    # Process aborted trials
    rows = []
    for _, tr in aborted_df.iterrows():
        t_start = tr.get('sequence_start') or tr.get('trial_start') or tr.get('start_time')
        t_end = tr.get('sequence_end') or tr.get('trial_end') or tr.get('end_time')
        trial_id = tr.get('trial_id', tr.name)
        if pd.isna(t_start) or pd.isna(t_end) or t_start is None or t_end is None:
            continue

        # Extract valve events and odor sequence
        evs = trial_valve_events(t_start, t_end)
        odor_sequence = [e['odor_name'] for e in evs]

        # Map first-seen odor to position 1..5
        odor_to_pos = {}
        next_pos = 1
        positions = []
        for e in evs:
            od = e['odor_name']
            if od not in odor_to_pos and next_pos <= 5:
                odor_to_pos[od] = next_pos
                next_pos += 1
            positions.append(odor_to_pos.get(od))

        # Collect ALL presentations with poke/valve timings
        presentations_all = []
        position_valve_times = {}
        position_poke_times = {}

        for idx_in_trial, (e, pos) in enumerate(zip(evs, positions)):
            if not isinstance(pos, (int, np.integer)):
                continue
            valve_start = e['start_time']
            valve_end = e['end_time']
            valve_dur_ms = (valve_end - valve_start).total_seconds() * 1000.0

            # Poke summary within valve window (merge gaps ≤ sample_offset_time_ms)
            psum = window_poke_summary(valve_start, valve_end)

            # Save full presentation record
            presentations_all.append({
                'index_in_trial': idx_in_trial,
                'position': int(pos),
                'odor_name': e['odor_name'],
                'valve_start': valve_start,
                'valve_end': valve_end,
                'valve_duration_ms': float(valve_dur_ms),
                'poke_time_ms': float(psum.get('poke_time_ms', 0.0)),
                'poke_first_in': psum.get('poke_first_in'),
            })

            # Keep last presentation per position for backward-compatibility
            position_valve_times[int(pos)] = {
                'position': int(pos),
                'odor_name': e['odor_name'],
                'valve_start': valve_start,
                'valve_end': valve_end,
                'valve_duration_ms': float(valve_dur_ms),
            }
            psum_pos = dict(psum)
            psum_pos['odor_name'] = e['odor_name']
            position_poke_times[int(pos)] = psum_pos

        # Last relevant odor (ignore < sample_offset_time_ms)
        last_idx = None
        for i in range(len(evs) - 1, -1, -1):
            dur_ms = (evs[i]['end_time'] - evs[i]['start_time']).total_seconds() * 1000.0
            if dur_ms >= sample_offset_time_ms:
                last_idx = i
                break

        # Map first-seen odor to position 1..5
        odor_to_pos = {}
        next_pos = 1
        positions = []
        for e in evs:
            od = e['odor_name']
            if od not in odor_to_pos and next_pos <= 5:
                odor_to_pos[od] = next_pos
                next_pos += 1
            positions.append(odor_to_pos.get(od))

        # position_valve_times and position_poke_times (last presentation per position within trial)
        position_valve_times = {}
        position_poke_times = {}
        for e, pos in zip(evs, positions):
            if not isinstance(pos, (int, np.integer)):
                continue
            valve_start = e['start_time']; valve_end = e['end_time']
            valve_dur_ms = (valve_end - valve_start).total_seconds() * 1000.0
            # keep the last presentation per position
            position_valve_times[pos] = {
                'position': pos,
                'odor_name': e['odor_name'],
                'valve_start': valve_start,
                'valve_end': valve_end,
                'valve_duration_ms': valve_dur_ms
            }
            psum = window_poke_summary(valve_start, valve_end)
            psum['odor_name'] = e['odor_name']
            position_poke_times[pos] = psum

        # Last relevant odor (ignore < sample_offset_time_ms)
        last_idx = None
        for i in range(len(evs) - 1, -1, -1):
            dur_ms = (evs[i]['end_time'] - evs[i]['start_time']).total_seconds() * 1000.0
            if dur_ms >= sample_offset_time_ms:
                last_idx = i
                break

        last_odor_name = None
        last_odor_pos = None
        last_valve_dur_ms = 0.0
        last_odor_poke_ms = 0.0

        if last_idx is not None:
            last_ev = evs[last_idx]
            last_odor_name = last_ev['odor_name']
            last_odor_pos = positions[last_idx]
            last_valve_dur_ms = (last_ev['end_time'] - last_ev['start_time']).total_seconds() * 1000.0

            # Authoritative: poke time strictly within [valve_start, valve_end]
            psum_last = window_poke_summary(last_ev['start_time'], last_ev['end_time'])
            last_odor_poke_ms = float(psum_last.get('poke_time_ms', 0.0))

        # Abortion type
        abortion_type = 'reinitiation_abortion' if last_odor_poke_ms >= minimum_sampling_time_ms else 'initiation_abortion'

        # Abortion time = last cue-port poke-out in [t_start, t_end]
        abortion_time = None
        # Intervals overlapping trial
        overlapping = [(max(s, t_start), min(e, t_end)) for (s, e) in cue_intervals if e > t_start and s < t_end]
        if overlapping:
            abortion_time = overlapping[-1][1]

        # FA window: from abortion_time to next cue-port poke after next InitiationSequence
        fa_label = 'nFA'
        fa_time = pd.NaT
        fa_latency_ms = np.nan
        fa_port = None 

        if abortion_time is not None:
            # Find next initiation after t_end
            next_init = None
            if init_times:
                idx = bisect_right(init_times, t_end)
                if idx < len(init_times):
                    next_init = init_times[idx]
            # Find first cue-port rising AFTER next_init
            fa_window_end = None
            if next_init is not None and cue_rises:
                k = bisect_right(cue_rises, next_init)
                if k < len(cue_rises):
                    fa_window_end = cue_rises[k]
            # If no end found, cap at session end (last timestamp we have)
            if fa_window_end is None:
                # fallback: last timestamp among known streams
                candidates = []
                for df in [DIP0, DIP1, DIP2]:
                    if not df.empty:
                        candidates.append(df.index[-1])
                fa_window_end = max(candidates) if candidates else abortion_time

            # Scan for first reward-port poke in (abortion_time, fa_window_end]
            if reward_rises:
                lo = bisect_right(reward_rises, abortion_time)
                hi = bisect_right(reward_rises, fa_window_end)
                if lo < hi:
                    fa_time = reward_rises[lo]
                    fa_latency_ms = (fa_time - abortion_time).total_seconds() * 1000.0
                    
                    # Determine which port the FA poke came from ← NEW
                    if fa_time in dip1_rises:
                        fa_port = 1
                    elif fa_time in dip2_rises:
                        fa_port = 2
                    
                    if fa_latency_ms <= response_time_ms:
                        fa_label = 'FA_time_in'
                    elif fa_latency_ms <= 3.0 * response_time_ms:
                        fa_label = 'FA_time_out'
                    else:
                        fa_label = 'FA_late'


        rows.append({
            'trial_id': trial_id,
            'sequence_start': t_start,
            'sequence_end': t_end,
            'odor_sequence': odor_sequence,
            'presentations': presentations_all,     
            'last_event_index': last_idx,            
            'position_valve_times': position_valve_times,
            'position_poke_times': position_poke_times,
            'last_odor_position': last_odor_pos,
            'last_odor_name': last_odor_name,
            'last_odor_valve_duration_ms': float(last_valve_dur_ms),
            'last_odor_poke_time_ms': float(last_odor_poke_ms),
            'abortion_type': abortion_type,
            'abortion_time': abortion_time,
            'fa_label': fa_label,
            'fa_time': fa_time,
            'fa_latency_ms': float(fa_latency_ms) if pd.notna(fa_latency_ms) else np.nan,
            'fa_port': fa_port, 
        })

    aborted_detailed = pd.DataFrame(rows)

    def _norm_fa(val):
        if pd.isna(val):
            return 'nFA'
        s = str(val).strip().lower()
        if s in ('fa_time_in', 'fa in', 'fa_in', 'in'):
            return 'FA_time_in'
        if s in ('fa_time_out', 'fa out', 'fa_out', 'out'):
            return 'FA_time_out'
        if s in ('fa_late', 'late'):
            return 'FA_late'
        return 'nFA'

    aborted_detailed['fa_label'] = aborted_detailed['fa_label'].apply(_norm_fa)


    if verbose and not aborted_detailed.empty:
        total = int(len(aborted_detailed))
        ini = int((aborted_detailed['abortion_type'] == 'initiation_abortion').sum())
        rei = int((aborted_detailed['abortion_type'] == 'reinitiation_abortion').sum())

        def pct(n, d):
            return (n / d * 100.0) if d else 0.0



        print("=" * 80)
        print("ABORTED TRIALS CLASSIFICATION SUMMARY")
        print("=" * 80)

        print(f"- Total Aborted Trials: {total}")
        print(f"  - Re-Initiation Abortions: {rei} ({pct(rei, total):.1f}%)")
        print(f"  - Initiation Abortions:    {ini} ({pct(ini, total):.1f}%)")

        # False Alarms summary
        fa_in_count  = int((aborted_detailed['fa_label'] == 'FA_time_in').sum())
        fa_out_count = int((aborted_detailed['fa_label'] == 'FA_time_out').sum())
        fa_late_count= int((aborted_detailed['fa_label'] == 'FA_late').sum())
        fa_total = fa_in_count + fa_out_count + fa_late_count
        nfa_count = total - fa_total

        print("\nFalse Alarms:")
        print(f"  - non-FA Abortions: {nfa_count}")
        print(f"  - False Alarm abortions: {fa_total} ({pct(fa_total, total):.1f}%)")
        if fa_total > 0:
            print(f"      - FA Time In (Within Response Time Window {response_time_ms}):  {fa_in_count} ({pct(fa_in_count, fa_total):.1f}%)")
            s_in = pd.to_numeric(
                aborted_detailed.loc[aborted_detailed['fa_label'] == 'FA_time_in', 'fa_latency_ms'],
                errors='coerce'
            ).dropna()
            if len(s_in):
                print(f"          - Response Time: avg={s_in.mean():.1f} ms, range: {s_in.min():.1f} - {s_in.max():.1f} ms")
            print(f"      - FA Time Out (Up to 3x Response Time Window {response_time}):  {fa_out_count} ({pct(fa_out_count, fa_total):.1f}%)")
            s_out = pd.to_numeric(
                aborted_detailed.loc[aborted_detailed['fa_label'] == 'FA_time_out', 'fa_latency_ms'],
                errors='coerce'
            ).dropna()
            if len(s_out):
                print(f"          - Response Time: avg={s_out.mean():.1f} ms, range: {s_out.min():.1f} - {s_out.max():.1f} ms")
            print(f"      - FA Late (After 3x Response Time up to next trial):{fa_late_count} ({pct(fa_late_count, fa_total):.1f}%)")
            s_late = pd.to_numeric(
                aborted_detailed.loc[aborted_detailed['fa_label'] == 'FA_late', 'fa_latency_ms'],
                errors='coerce'
            ).dropna()
            if len(s_late):
                print(f"          - Response Time: avg={s_late.mean():.1f} ms, range: {s_late.min():.1f} - {s_late.max():.1f} ms")

            hr_pos = int(classification.get('hidden_rule_position', 1))

            # Normalize FA labels
            def _norm_fa(val):
                if pd.isna(val):
                    return 'nFA'
                s = str(val).strip().lower()
                if s in ('fa_time_in', 'fa in', 'fa_in', 'in'):
                    return 'FA_time_in'
                if s in ('fa_time_out', 'fa out', 'fa_out', 'out'):
                    return 'FA_time_out'
                if s in ('fa_late', 'late'):
                    return 'FA_late'
                return 'nFA'
            aborted_detailed['fa_label'] = aborted_detailed['fa_label'].apply(_norm_fa)

            abortions_at_hr_pos = aborted_detailed[aborted_detailed['last_odor_position'] == hr_pos].copy()

            # Resolve HR-aborted trial IDs from classification (robust to key naming)
            hr_ab_df = None
            for k in ('aborted_sequences_HR', 'aborted_HR_sequences', 'aborted_hidden_rule_sequences'):
                if isinstance(classification.get(k), pd.DataFrame) and not classification[k].empty and 'trial_id' in classification[k]:
                    hr_ab_df = classification[k]
                    break
            if hr_ab_df is not None:
                hr_aborted_ids = set(hr_ab_df['trial_id'])
            elif 'hit_hidden_rule' in abortions_at_hr_pos.columns:
                hr_aborted_ids = set(abortions_at_hr_pos.loc[abortions_at_hr_pos['hit_hidden_rule'] == True, 'trial_id'])
            else:
                hr_aborted_ids = set()

            in_hr_trials = abortions_at_hr_pos[abortions_at_hr_pos['trial_id'].isin(hr_aborted_ids)].copy()
            non_hr_trials = abortions_at_hr_pos[~abortions_at_hr_pos['trial_id'].isin(hr_aborted_ids)].copy()

            # Helper to print FA breakdown
            def _print_fa_counts(df, indent="    "):
                order = ['nFA', 'FA_time_in', 'FA_time_out', 'FA_late']
                cnt = df['fa_label'].value_counts().reindex(order, fill_value=0)
                total = int(len(df))
                for lbl in order:
                    v = int(cnt.get(lbl, 0))
                    pct = (v / total * 100.0) if total else 0.0
                    print(f"{indent}{lbl}: {v} ({pct:.1f}%)")

            total_at_hr = int(len(abortions_at_hr_pos))
            print(f"\n  Abortions at Hidden Rule Position {hr_pos}: n={total_at_hr}")

            total_in_hr = int(len(in_hr_trials))
            print(f"    Of which in Hidden Rule Trials: n={total_in_hr}")
            if total_in_hr > 0:
                _print_fa_counts(in_hr_trials, indent="        ")

            total_non_hr = int(len(non_hr_trials))
            print(f"    Non-Hidden Rule Abortions at HR Location: n={total_non_hr}")
            if total_non_hr > 0:
                _print_fa_counts(non_hr_trials, indent="        ")

        # Helper for stats lines
        def stats_line(series, label):
            s = pd.to_numeric(series, errors='coerce').dropna()
            if s.empty:
                print(f"{label}: n=0")
            else:
                print(f"{label}: n={len(s)} | avg={s.mean():.1f} ms | range={s.min():.1f}-{s.max():.1f} ms")

        # Non-last odor poke times (>= minimum_sampling_time_ms), requires 'presentations'
        if 'presentations' in aborted_detailed.columns and 'last_event_index' in aborted_detailed.columns:
            pres_df = aborted_detailed[['trial_id', 'presentations', 'last_event_index']].explode('presentations')
            pres_df = pres_df.dropna(subset=['presentations']).copy()
            if not pres_df.empty:
                pres = pd.concat(
                    [pres_df.drop(columns=['presentations']),
                     pres_df['presentations'].apply(pd.Series)],
                    axis=1
                )
                # Exclude the last relevant odor per trial
                pres['is_last'] = pres['index_in_trial'] == pres['last_event_index']
                pres = pres[~pres['is_last']].copy()

                # Only pokes >= minimum_sampling_time_ms
                pres['poke_time_ms'] = pd.to_numeric(pres['poke_time_ms'], errors='coerce')
                pres_valid = pres[pres['poke_time_ms'] >= minimum_sampling_time_ms].copy()

                print("\nNon-last Odor Pokes:")
                stats_line(pres_valid['poke_time_ms'], "  - All non-last odors")

                # By position
                if 'position' in pres_valid.columns and not pres_valid.empty:
                    for pos, grp in pres_valid.groupby('position'):
                        stats_line(grp['poke_time_ms'], f"  - Position {int(pos)}")

                # By odor name/type
                if 'odor_name' in pres_valid.columns and not pres_valid.empty:
                    for odor, grp in pres_valid.groupby('odor_name'):
                        stats_line(grp['poke_time_ms'], f"  - Odor {odor}")
            else:
                print("\nNon-last Odor Pokes: n=0 (no presentations info)")
        else:
            print("\nNon-last odor pokes: presentations not attached; update abortion_classification to store 'presentations' and 'last_event_index'.")

        # Last-odor poke stats by abortion type
        print("\nLast Odor Poke Times:")
        stats_line(
            aborted_detailed.loc[aborted_detailed['abortion_type'] == 'reinitiation_abortion', 'last_odor_poke_time_ms'],
            "  - Re-Initiation Abortions"
        )
        stats_line(
            aborted_detailed.loc[aborted_detailed['abortion_type'] == 'initiation_abortion', 'last_odor_poke_time_ms'],
            "  - Initiation Abortions"
        )

        # Counts by last odor name
        print("\nCounts by last odor:")
        if 'last_odor_name' in aborted_detailed.columns:
            by_odor = (
                aborted_detailed
                .groupby(['last_odor_name', 'abortion_type'])
                .size()
                .unstack(fill_value=0)
                .rename(columns={'reinitiation_abortion': 'Re-initiation', 'initiation_abortion': 'Initiation'})
            )
            # Total per odor row
            totals = aborted_detailed.groupby('last_odor_name').size()
            for odor in totals.index:
                rei_c = int(by_odor.loc[odor].get('Re-initiation', 0))
                ini_c = int(by_odor.loc[odor].get('Initiation', 0))
                tot = int(totals.loc[odor])
                print(f"  - {odor}: {tot} abortions, Re-initiation {rei_c}, Initiation {ini_c}")
        else:
            print("  (missing last_odor_name)")

        # Counts by last position
        print("\nCounts by last position:")
        if 'last_odor_position' in aborted_detailed.columns:
            by_pos = (
                aborted_detailed
                .groupby(['last_odor_position', 'abortion_type'])
                .size()
                .unstack(fill_value=0)
                .rename(columns={'reinitiation_abortion': 'Re-initiation', 'initiation_abortion': 'Initiation'})
            )
            totals_pos = aborted_detailed.groupby('last_odor_position').size()
            for pos in sorted(totals_pos.index):
                rei_c = int(by_pos.loc[pos].get('Re-initiation', 0))
                ini_c = int(by_pos.loc[pos].get('Initiation', 0))
                tot = int(totals_pos.loc[pos])
                print(f"  - Position {int(pos)}: {tot} abortions, Re-initiation {rei_c}, Initiation {ini_c}")
        else:
            print("  (missing last_odor_position)")

    def build_abortion_index(df: pd.DataFrame):
        idx = {}
        if df is None or df.empty:
            return {
                'by_trial': {},
                'by_position': {},
                'by_odor': {},
                'by_type': {},
                'by_fa_label': {},
            }
        # Ensure trial_id exists as indexable key
        df2 = df.copy()
        # Some pipelines may have non-unique or NaN trial_id; drop NaN for dict keys
        df2 = df2.dropna(subset=['trial_id'])
        # by_trial -> full row as a dict for each trial_id
        try:
            by_trial = df2.set_index('trial_id', drop=False).apply(lambda r: r.to_dict(), axis=1).to_dict()
        except Exception:
            # fallback: iterate
            by_trial = {row['trial_id']: row.to_dict() for _, row in df2.iterrows()}

        # Helper to group trial IDs by a column
        def group_ids(col):
            m = {}
            if col in df2.columns:
                for k, g in df2.groupby(col):
                    # Keep order by sequence_start if present
                    trials = list(g.sort_values('sequence_start')['trial_id']) if 'sequence_start' in g else list(g['trial_id'])
                    m[k] = trials
            return m

        idx['by_trial'] = by_trial
        idx['by_position'] = group_ids('last_odor_position')
        idx['by_odor'] = group_ids('last_odor_name')
        idx['by_type'] = group_ids('abortion_type')
        idx['by_fa_label'] = group_ids('fa_label')
        return idx

    aborted_index = build_abortion_index(aborted_detailed)

    # Attach to classification dict for downstream use
    try:
        classification['aborted_sequences_detailed'] = aborted_detailed
        classification['aborted_index'] = aborted_index
    except Exception:
        pass

    return aborted_detailed

def classify_noninitiated_FA(noninit_df, DIP0, DIP1, DIP2, response_time, hr_odors=None):
    """Classify False Alarms in non-initiated trials"""
    
    results = []
    
    # Get port rises
    dip1_rises = DIP1[DIP1 & ~DIP1.shift(1, fill_value=False)].index.tolist()
    dip2_rises = DIP2[DIP2 & ~DIP2.shift(1, fill_value=False)].index.tolist()
    reward_rises = sorted(dip1_rises + dip2_rises)
    
    cue_rises = list(DIP0[DIP0 & ~DIP0.shift(1, fill_value=False)].index)
    response_time_ms = float(response_time) * 1000.0

    for _, row in noninit_df.iterrows():
        attempt_end = row.get('attempt_end')
        if pd.isna(attempt_end):
            continue
            
        # Find next cue port poke-in after attempt_end
        next_cue_in = None
        cue_after = [t for t in cue_rises if t > attempt_end]
        if cue_after:
            next_cue_in = cue_after[0]
        else:
            next_cue_in = max(DIP0.index) if not DIP0.empty else attempt_end

        # Scan for first reward-port poke in (attempt_end, next_cue_in]
        fa_label = 'nFA'
        fa_time = pd.NaT
        fa_latency_ms = np.nan
        fa_port = None  # ← NEW
        
        reward_after = [t for t in reward_rises if attempt_end < t <= next_cue_in]
        if reward_after:
            fa_time = reward_after[0]
            fa_latency_ms = (fa_time - attempt_end).total_seconds() * 1000.0
            
            # Determine which port ← NEW
            if fa_time in dip1_rises:
                fa_port = 1
            elif fa_time in dip2_rises:
                fa_port = 2
            
            if fa_latency_ms <= response_time_ms:
                fa_label = 'FA_time_in'
            elif fa_latency_ms <= 3.0 * response_time_ms:
                fa_label = 'FA_time_out'
            else:
                fa_label = 'FA_late'

        # HR status for position 1
        is_hr = False
        if hr_odors is not None:
            odor_name = row.get('odor_name')
            is_hr = odor_name in hr_odors

        results.append({
            **row.to_dict(),
            'fa_label': fa_label,
            'fa_time': fa_time,
            'fa_latency_ms': fa_latency_ms,
            'fa_port': fa_port,  # ← NEW
            'is_hr': is_hr
        })
        
    return pd.DataFrame(results)

def build_classification_index(classification: dict) -> dict: # Classification function for easier dictionary access later on
    """
    Build convenient lookup indices over classification outputs.
    Provides:
      - by_trial: trial_id -> full row dict (completed_with_RT preferred, else completed, else aborted_detailed)
      - categories.completed.*_ids: lists of trial_ids for major completed categories (and HR variants)
      - sets.*: quick sets of IDs for initiated, completed, aborted
      - aborted: re-exposes the aborted_index (by_position/by_odor/by_type/by_fa_label)
    """

    idx = {'by_trial': {}, 'categories': {'completed': {}}, 'sets': {}, 'aborted': {}}

    # Prefer completed_with_RT for richer rows
    comp_df = classification.get('completed_sequences_with_response_times')
    if not isinstance(comp_df, pd.DataFrame) or comp_df.empty:
        comp_df = classification.get('completed_sequences', pd.DataFrame())

    ab_det = classification.get('aborted_sequences_detailed')
    ab_df = ab_det if isinstance(ab_det, pd.DataFrame) else classification.get('aborted_sequences', pd.DataFrame())

    # by_trial: completed first (wins), then aborted to fill missing ones
    if isinstance(comp_df, pd.DataFrame) and not comp_df.empty and 'trial_id' in comp_df:
        for _, r in comp_df.iterrows():
            tid = r.get('trial_id')
            if pd.notna(tid):
                idx['by_trial'][tid] = r.to_dict()
    if isinstance(ab_df, pd.DataFrame) and not ab_df.empty and 'trial_id' in ab_df:
        for _, r in ab_df.iterrows():
            tid = r.get('trial_id')
            if pd.notna(tid) and tid not in idx['by_trial']:
                idx['by_trial'][tid] = r.to_dict()

    # Completed category ID lists
    def ids_from(name):
        df = classification.get(name, pd.DataFrame())
        return [] if not isinstance(df, pd.DataFrame) or df.empty or 'trial_id' not in df else list(df['trial_id'])

    c = idx['categories']['completed']
    c['rewarded_ids'] = ids_from('completed_sequence_rewarded')
    c['unrewarded_ids'] = ids_from('completed_sequence_unrewarded')
    c['timeout_ids'] = ids_from('completed_sequence_reward_timeout')

    c['hr_rewarded_ids'] = ids_from('completed_sequence_HR_rewarded')
    c['hr_unrewarded_ids'] = ids_from('completed_sequence_HR_unrewarded')
    c['hr_timeout_ids'] = ids_from('completed_sequence_HR_reward_timeout')

    c['hr_missed_rewarded_ids'] = ids_from('completed_sequence_HR_missed_rewarded')
    c['hr_missed_unrewarded_ids'] = ids_from('completed_sequence_HR_missed_unrewarded')
    c['hr_missed_timeout_ids'] = ids_from('completed_sequence_HR_missed_reward_timeout')

    # Sets for quick membership tests
    idx['sets']['initiated_ids'] = (
        set(classification['initiated_sequences']['trial_id']) 
        if isinstance(classification.get('initiated_sequences'), pd.DataFrame) 
        and 'trial_id' in classification['initiated_sequences'] else set()
    )
    idx['sets']['completed_ids'] = set(comp_df['trial_id']) if isinstance(comp_df, pd.DataFrame) and 'trial_id' in comp_df else set()
    idx['sets']['aborted_ids'] = (
        set(classification['aborted_sequences']['trial_id']) 
        if isinstance(classification.get('aborted_sequences'), pd.DataFrame) 
        and 'trial_id' in classification['aborted_sequences'] else set()
    )

    # Aborted sub-index (already built by abortion_classification)
    ab_index = classification.get('aborted_index')
    if isinstance(ab_index, dict):
        idx['aborted'] = ab_index
    else:
        # Minimal fallback
        idx['aborted'] = {'by_trial': {}, 'by_position': {}, 'by_odor': {}, 'by_type': {}, 'by_fa_label': {}}
        if isinstance(ab_df, pd.DataFrame) and not ab_df.empty:
            try:
                idx['aborted']['by_trial'] = ab_df.set_index('trial_id', drop=False).apply(lambda r: r.to_dict(), axis=1).to_dict()
            except Exception:
                idx['aborted']['by_trial'] = {r['trial_id']: r.to_dict() for _, r in ab_df.dropna(subset=['trial_id']).iterrows()}
            def group_ids(col):
                out = {}
                if col in ab_df.columns:
                    for k, g in ab_df.groupby(col):
                        out[k] = list(g.sort_values('sequence_start')['trial_id']) if 'sequence_start' in g else list(g['trial_id'])
                return out
            for col, key in [('last_odor_position','by_position'), ('last_odor_name','by_odor'), ('abortion_type','by_type'), ('fa_label','by_fa_label')]:
                idx['aborted'][key] = group_ids(col)

    return idx

def classify_and_analyze_with_response_times(data, events, trial_counts, odor_map, stage, root, verbose=True):# Wrapper function to fully classify all trials. 
    """
    Orchestrates classification + valve/poke timing + response-time augmentation.

    Returns:
      {
        'classification': <dict from classify_trial_outcomes_with_pokes_and_valves2>,
        'response_time_analysis': <dict from analyze_response_times>,
        'completed_sequences_with_response_times': <DataFrame of completed trials with RT columns>
      }
    """
    sample_offset_time, minimum_sampling_time, response_time = get_experiment_parameters(root)
    sample_offset_time_ms = sample_offset_time * 1000
    minimum_sampling_time_ms = minimum_sampling_time * 1000
    response_time_sec = response_time
    if response_time_sec is None:
        raise ValueError("Response time parameter cannot be extracted from Schema file. Check detect_settings function.")

    params = {
        'sample_offset_time_ms': sample_offset_time_ms,
        'minimum_sampling_time_ms': minimum_sampling_time_ms,
        'response_time_window_sec': response_time_sec
    }


    # 1) Run the stable classifier (valve/poke timing included)
    classification = classify_trials(
        data, events, trial_counts, odor_map, stage, root, verbose=verbose
    )

    # 2) Run the response-time summary analyzer (prints/aggregates like the notebook)
    rt_summary = analyze_response_times(
        data, trial_counts, events, odor_map, stage, root, verbose=verbose
    )

    # 3) Aborted trial details
    aborted_detailed = abortion_classification(
        data, events, classification, odor_map, root, verbose=verbose
    )
    classification['aborted_sequences_detailed'] = aborted_detailed

    # 3) Build fast lookup indices for downstream use
    classification['index'] = build_classification_index(classification)

    # 4) Hidden rule position from stage name/index 
    sequence_name = None
    hidden_rule_location = None
    if isinstance(stage, dict):
        sequence_name = stage.get('stage_name') or str(stage)
        if stage.get('hidden_rule_index') is not None:
            try:
                hidden_rule_location = int(stage['hidden_rule_index'])
            except Exception:
                hidden_rule_location = None
    if hidden_rule_location is None:
        sequence_name = sequence_name or str(stage)
        m = re.search(r'Location(\d+)', sequence_name)
        if m:
            hidden_rule_location = int(m.group(1))
    hidden_rule_pos = hidden_rule_location + 1 if isinstance(hidden_rule_location, int) else None
    if hidden_rule_location is not None:
        print(f"Hidden rule location extracted: Location{hidden_rule_location} (index {hidden_rule_location}, position {hidden_rule_pos})")
    else:
        print(f"No Hidden Rule Location found in sequence name: {sequence_name}. Proceeding without Hidden Rule analysis.")

# 5) Attach params and RT summary to classification
    classification['hidden_rule_position'] = hidden_rule_pos
    classification.update(params)
    classification['response_time_analysis'] = rt_summary
    
# 6) Build completed_sequences_with_response_times by merging analyzer per_trial (no recomputation)
    completed_df = classification.get('completed_sequences', pd.DataFrame()).copy()
    per_trial_df = rt_summary.get('per_trial')
    if isinstance(completed_df, pd.DataFrame) and not completed_df.empty and isinstance(per_trial_df, pd.DataFrame) and not per_trial_df.empty:
        if 'trial_id' in completed_df.columns and 'trial_id' in per_trial_df.columns:
            completed_with_rt = completed_df.merge(
                per_trial_df[['trial_id', 'response_time_ms', 'response_time_category']],
                on='trial_id',
                how='left',
                validate='one_to_one'
            )
        else:
            completed_with_rt = completed_df.copy()
            completed_with_rt['response_time_ms'] = np.nan
            completed_with_rt['response_time_category'] = np.nan
    else:
        completed_with_rt = completed_df
        if isinstance(completed_with_rt, pd.DataFrame) and not completed_with_rt.empty:
            # ensure RT columns exist
            if 'response_time_ms' not in completed_with_rt.columns:
                completed_with_rt['response_time_ms'] = np.nan
            if 'response_time_category' not in completed_with_rt.columns:
                completed_with_rt['response_time_category'] = np.nan

    classification['completed_sequences_with_response_times'] = completed_with_rt

    # 7) Build indices after everything is attached
    classification['index'] = build_classification_index(classification)

    # 8) Return wrapper payload
    return {
        'classification': classification,
        'response_time_analysis': rt_summary,
        'completed_sequences_with_response_times': completed_with_rt,
    }


# ========================== Functions for saving results ==========================
def _json_safe(obj):
    """Recursively convert objects to JSON-friendly types."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if np.isnan(f) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass
    try:
        import pandas as _pd
        if isinstance(obj, _pd.Timedelta):
            return obj.total_seconds()
    except Exception:
        pass
    if isinstance(obj, Path):
        return str(obj)
    return obj

def _find_parent_named(start: Path, prefix: str) -> Path | None:
    for p in [Path(start)] + list(Path(start).parents):
        if p.name.startswith(prefix):
            return p
    return None

def _find_rawdata_root(start: Path) -> Path | None:
    for p in [Path(start)] + list(Path(start).parents):
        if p.name == "rawdata":
            return p
    return None

def resolve_derivatives_output_dir(root) -> tuple[Path, dict]:
    root = Path(root).resolve()
    rawdata_dir = _find_rawdata_root(root)
    if rawdata_dir is None:
        raise ValueError(f"Could not find 'rawdata' in parents of: {root}")

    hypnose_dir = rawdata_dir.parent
    sub_dir = _find_parent_named(root, "sub-")
    ses_dir = _find_parent_named(root, "ses-")
    if sub_dir is None or ses_dir is None:
        raise ValueError(f"Could not resolve sub-/ses- from: {root}")

    out_dir = hypnose_dir / "derivatives" / sub_dir.name / ses_dir.name / "saved_analysis_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, {
        "hypnose_dir": str(hypnose_dir),
        "rawdata_dir": str(rawdata_dir),
        "sub_folder": sub_dir.name,
        "ses_folder": ses_dir.name,
    }

def _json_default(o):
    if isinstance(o, (pd.Timestamp, )):
        return o.isoformat()
    if hasattr(o, "isoformat"):
        try:
            return o.isoformat()
        except Exception:
            pass
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return None if np.isnan(f) else f
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)

def _normalize_df_for_io(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    JSON-encode object columns containing dict/list/tuple/set/ndarray.
    Returns (normalized_df, jsonified_columns).
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df, []
    df2 = df.copy()
    json_cols = []

    def _is_nullish(v):
        if v is None:
            return True
        try:
            if isinstance(v, (float, np.floating)):
                return math.isnan(float(v))
        except Exception:
            pass
        return False

    def _json_default_local(o):
        try:
            return _json_default(o)
        except NameError:
            return _json_safe(o)

    for col in df2.columns:
        if df2[col].dtype == "object":
            sample = df2[col].dropna().head(10).tolist()
            needs_json = any(isinstance(v, (dict, list, tuple, set, np.ndarray)) for v in sample)
            if needs_json:
                json_cols.append(col)
                df2[col] = df2[col].apply(
                    lambda v: (None if _is_nullish(v) else json.dumps(v, default=_json_default_local))
                )
    return df2, json_cols

def save_session_analysis_results(classification: dict, root, session_metadata: dict | None = None, data=None, events=None, verbose: bool = True) -> Path:
    out_dir, info = resolve_derivatives_output_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "session": _json_safe(session_metadata or {}),
        "paths": info,
        "tables": {},
        "artifacts": {},
        "notes": "DataFrames saved as CSV; object columns JSON-encoded. See *.schema.json.",
    }

    saved_any = False
    saved_names: set[str] = set()

    def _save_df(name: str, df) -> bool:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return False
        if name in saved_names:
            return True
        f_csv = out_dir / f"{name}.csv"
        f_schema = out_dir / f"{name}.schema.json"
        try:
            df_norm, json_cols = _normalize_df_for_io(df)
            df_norm.to_csv(f_csv, index=False)
            with open(f_schema, "w", encoding="utf-8") as sf:
                json.dump({"jsonified_columns": json_cols}, sf, indent=2)
            manifest["tables"][name] = f_csv.name
            saved_names.add(name)
            return True
        except Exception as e:
            vprint(verbose, f"[save] WARNING: failed writing {name}: {e}")
            return False

    # 1) Save all top-level DataFrames
    if isinstance(classification, dict):
        for key, val in classification.items():
            if _save_df(key, val):
                saved_any = True

    # 2) Explicit key tables (covers merged dicts)
    for k in [
        "initiated_sequences","non_initiated_sequences","non_initiated_odor1_attempts",
        "completed_sequences","completed_sequences_with_response_times",
        "completed_sequence_rewarded","completed_sequence_unrewarded","completed_sequence_reward_timeout",
        "completed_sequences_HR","completed_sequence_HR_rewarded","completed_sequence_HR_unrewarded","completed_sequence_HR_reward_timeout",
        "completed_sequences_HR_missed","completed_sequence_HR_missed_rewarded","completed_sequence_HR_missed_unrewarded","completed_sequence_HR_missed_reward_timeout",
        "aborted_sequences","aborted_sequences_HR","aborted_sequences_detailed", "non_initiated_FA",
    ]:
        df = classification.get(k) if isinstance(classification, dict) else None
        if _save_df(k, df):
            saved_any = True

    # 3) Extract run start and end times
    runs = manifest["session"].get("runs", [])
    london_tz = zoneinfo.ZoneInfo("Europe/London")

    for run in runs:
        # Extract start time from folder path
        run_start_str = run["root"].split("/")[-1]  # Extract the timestamp part (e.g., "2025-10-17T12-57-05")
        run_start = datetime.strptime(run_start_str, "%Y-%m-%dT%H-%M-%S").replace(tzinfo=zoneinfo.ZoneInfo("UTC"))
        run_start_london = run_start.astimezone(london_tz)
        run["start_time"] = run_start_london.isoformat()

        # Use precomputed end time if available
        stage_info = run.get("stage", {})
        precomputed_end_time = stage_info.get("run_end_time") if isinstance(stage_info, dict) else None
        
        if precomputed_end_time is not None:
            # Ensure precomputed_end_time is a datetime object
            if isinstance(precomputed_end_time, str):
                precomputed_end_time = datetime.fromisoformat(precomputed_end_time)

            # Convert to London time
            if precomputed_end_time.tzinfo is None:
                run_end_london = precomputed_end_time.replace(tzinfo=london_tz)
            else:
                run_end_london = precomputed_end_time.astimezone(london_tz)
            run["end_time"] = run_end_london.isoformat()
        else:
            # Fallback: try to extract from current data/events (existing logic)
            try:
                all_timestamps = []
                
                # Only try this fallback if we have data and events for this specific run
                if data is not None and events is not None:
                    # This fallback logic would need to filter by run, but it's complex
                    # Better to ensure the precomputed end time is always available
                    pass
                
                run["end_time"] = None
                if verbose:
                    print(f"Warning: No precomputed end time for run {run.get('run_id')}")
            except Exception as e:
                print(f"Error extracting end time for run {run.get('run_id')}: {e}")
                run["end_time"] = None

    # 4) Calculate gaps between runs
    for i in range(len(runs) - 1):
        run_end = runs[i].get("end_time")
        next_run_start = runs[i + 1].get("start_time")
        if run_end and next_run_start:
            run_end_dt = datetime.fromisoformat(run_end)
            next_run_start_dt = datetime.fromisoformat(next_run_start)
            gap = next_run_start_dt - run_end_dt
            runs[i]["gap_to_next_run"] = str(gap)
        else:
            runs[i]["gap_to_next_run"] = None

    manifest["session"]["runs"] = runs
    
    # 5) Indices
    indices_dir = out_dir / "indices"
    indices_dir.mkdir(parents=True, exist_ok=True)
    idx_payloads = {
        "index": classification.get("index", {}),
        "aborted_index": classification.get("aborted_index", classification.get("index", {}).get("aborted", {})),
    }
    for name, payload in idx_payloads.items():
        with open(indices_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(payload), f, indent=2)

    # 6) Response-time analysis artifacts
    rta = classification.get("response_time_analysis")
    if isinstance(rta, dict):
        try:
            with open(out_dir / "response_time_analysis.json", "w", encoding="utf-8") as f:
                json.dump(_json_safe(rta), f, indent=2)
        except Exception as e:
            vprint(verbose, f"[save] WARNING: failed writing response_time_analysis.json: {e}")
        per_trial = rta.get("per_trial")
        if isinstance(per_trial, pd.DataFrame) and not per_trial.empty:
            if _save_df("response_time_per_trial", per_trial):
                saved_any = True

    # 7) Manifest + summary
    try:
        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(manifest), f, indent=2)
    except Exception as e:
        vprint(verbose, f"[save] WARNING: failed writing manifest.json: {e}")
    counts = {}
    def _n(name):
        df = classification.get(name)
        return int(len(df)) if isinstance(df, pd.DataFrame) else 0
    for k in [
        "initiated_sequences","non_initiated_sequences","non_initiated_odor1_attempts",
        "completed_sequences","completed_sequences_with_response_times",
        "completed_sequence_rewarded","completed_sequence_unrewarded","completed_sequence_reward_timeout",
        "aborted_sequences","aborted_sequences_detailed",
    ]:
        counts[k] = _n(k)

    # Add combined non-initiated total (baseline + pos1 attempts)
    counts["non_initiated_total"] = (
        counts.get("non_initiated_sequences", 0)
        + counts.get("non_initiated_odor1_attempts", 0)
    )

    params = {
        "sample_offset_time_ms": classification.get("sample_offset_time_ms"),
        "minimum_sampling_time_ms": classification.get("minimum_sampling_time_ms"),
        "response_time_window_sec": classification.get("response_time_window_sec"),
        "hidden_rule_position": classification.get("hidden_rule_position"),
        "hidden_rule_odors": classification.get("hidden_rule_odors"),
    }
    summary = {
        "created_at": manifest["created_at"],
        "session": manifest["session"],
        "counts": counts,
        "params": params,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2)

    vprint(verbose, f"Saved analysis to: {out_dir} ({'some tables' if saved_any else 'no tables'})")
    return out_dir

# ========================== Functions for multiple session analysis ========================== 

def _concat_align(dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
    dfs = [d for d in dfs if isinstance(d, pd.DataFrame) and not d.empty]
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, axis=0, ignore_index=True, sort=False)

def _assign_global_trial_ids(classif: dict) -> dict:
    """
    Ensure unique trial_id across merged runs.
    Uses sequence_start + run_id + original trial_id for stable ordering.
    """
    comp = classif.get('completed_sequences', pd.DataFrame())
    abo = classif.get('aborted_sequences', pd.DataFrame())
    cols = ['trial_id', 'run_id', 'sequence_start']
    frames = []
    if isinstance(comp, pd.DataFrame) and not comp.empty:
        frames.append(comp[[c for c in cols if c in comp.columns]])
    if isinstance(abo, pd.DataFrame) and not abo.empty:
        frames.append(abo[[c for c in cols if c in abo.columns]])
    if not frames:
        return classif

    all_trials = _concat_align(frames).dropna(subset=['trial_id']).copy()
    # Fill missing run_id with 1 before ordering (shouldn't happen if we add run_id)
    if 'run_id' not in all_trials.columns:
        all_trials['run_id'] = 1
    all_trials = all_trials.sort_values(
        [c for c in ['sequence_start', 'run_id', 'trial_id'] if c in all_trials.columns]
    ).reset_index(drop=True)

    mapping = { (int(r), int(t)): i+1 for i, (r, t) in enumerate(zip(all_trials['run_id'], all_trials['trial_id'])) }

    def remap_df(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty or 'trial_id' not in df.columns:
            return df
        df = df.copy()
        if 'run_id' not in df.columns:
            df['run_id'] = 1
        df['trial_id'] = [mapping.get((int(r), int(t)), t) for r, t in zip(df['run_id'], df['trial_id'])]
        return df

    for k, v in list(classif.items()):
        if isinstance(v, pd.DataFrame) and 'trial_id' in v.columns:
            classif[k] = remap_df(v)

    return classif

def _coerce_int_like(s):
    try:
        return pd.to_numeric(s, errors='coerce').astype('Int64')
    except Exception:
        return s

def _with_run_id(df: pd.DataFrame, run_id: int) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if 'run_id' not in out.columns:
        out['run_id'] = run_id
    return out

def merge_classifications(run_results: list[dict], verbose: bool = True) -> dict:
    """
    Robust multi-run merger.
    - Accepts either:
        * the full return of classify_and_analyze_with_response_times (dict with 'classification', etc.), or
        * a classification dict itself (legacy)
    - Concatenates tables directly with run_id; no parent-child re-alignment.
    - Uses completed_sequences_with_response_times as the authoritative completed-trials table.
    - Aggregates response_time_analysis lists safely.
    - Rebuilds 'index' with build_classification_index on the merged dict.
    """

    if not run_results:
        raise ValueError("merge_classifications: no run results provided")

    # Normalize inputs to classification dicts and collect aux data
    per_run_cls = []
    per_run_rta = []
    per_run_params = []
    for ridx, r in enumerate(run_results, start=1):
        if r is None:
            continue
        # Either top-level dict with 'classification' or directly a classification dict
        if isinstance(r, dict) and 'classification' in r:
            cls = r['classification'] or {}
            rta = r.get('response_time_analysis') or cls.get('response_time_analysis') or {}
        else:
            cls = r or {}
            rta = cls.get('response_time_analysis') or {}
        per_run_cls.append((ridx, cls))
        per_run_rta.append((ridx, rta))

        # collect params
        per_run_params.append({
            'run_id': ridx,
            'sample_offset_time_ms': cls.get('sample_offset_time_ms'),
            'minimum_sampling_time_ms': cls.get('minimum_sampling_time_ms'),
            'response_time_window_sec': cls.get('response_time_window_sec'),
            'hidden_rule_position': cls.get('hidden_rule_position'),
        })

    # Keys we will merge if present
    # Keep completed_sequences_with_response_times as authoritative (has all columns of completed_sequences + RT cols)
    preferred_tables = [
        'initiated_sequences',
        'non_initiated_sequences',
        'non_initiated_odor1_attempts',
        'completed_sequences',  # keep it too for compatibility
        'completed_sequences_with_response_times',  # authoritative for RT-based summary
        'completed_sequence_rewarded',
        'completed_sequence_unrewarded',
        'completed_sequence_reward_timeout',
        'completed_sequences_HR',
        'completed_sequence_HR_rewarded',
        'completed_sequence_HR_unrewarded',
        'completed_sequence_HR_reward_timeout',
        'completed_sequences_HR_missed',
        'completed_sequence_HR_missed_rewarded',
        'completed_sequence_HR_missed_unrewarded',
        'completed_sequence_HR_missed_reward_timeout',
        'aborted_sequences',
        'aborted_sequences_HR',
        'aborted_sequences_detailed',
        'non_initiated_FA',
    ]

    def _normalize_trial_id(s):
        if pd.isna(s):
            return None
        try:
            return int(s)
        except (ValueError, TypeError):
            return s
    
    merged: dict = {}
    # Concatenate tables
    for key in preferred_tables:
        parts = []
        for ridx, cls in per_run_cls:
            df = cls.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df2 = _with_run_id(df, ridx)
                # Normalize known ID columns to int-like (but don't force if incompatible)
                if 'trial_id' in df2.columns:
                    df2['trial_id'] = df2['trial_id'].apply(_normalize_trial_id)
                parts.append(df2)
        merged[key] = pd.concat(parts, axis=0, ignore_index=True, sort=False) if parts else pd.DataFrame()

    # Invariant sanity: if RT table is missing, synthesize minimally
    if merged['completed_sequences_with_response_times'].empty and not merged['completed_sequences'].empty:
        tmp = merged['completed_sequences'].copy()
        if 'response_time_ms' not in tmp.columns:
            tmp['response_time_ms'] = np.nan
        if 'response_time_category' not in tmp.columns:
            tmp['response_time_category'] = np.nan
        merged['completed_sequences_with_response_times'] = tmp

    # Sort time-based tables by sequence_start if present (stabilizes summaries)
    for key in preferred_tables:
        df = merged.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty and 'sequence_start' in df.columns:
            merged[key] = df.sort_values(['sequence_start', 'run_id'] if 'run_id' in df.columns else ['sequence_start']).reset_index(drop=True)

    # Merge response_time_analysis lists
    rta_agg = defaultdict(list)
    for _, rta in per_run_rta:
        for k in ['rewarded_response_times', 'unrewarded_response_times',
                  'timeout_delayed_response_times', 'timeout_response_delay_times',
                  'all_response_times']:
            vals = rta.get(k, [])
            if isinstance(vals, (list, tuple, np.ndarray)):
                rta_agg[k].extend(list(vals))
    merged['response_time_analysis'] = dict(rta_agg)

    # Carry session/global params; warn if heterogeneous
    def _pick_param(name):
        vals = [p.get(name) for p in per_run_params if p.get(name) is not None]
        if not vals:
            return None
        same = all(v == vals[0] for v in vals)
        if not same and verbose:
            print(f"[merge_classifications] WARNING: parameter '{name}' differs across runs; using first value: {vals[0]} (found: {sorted(set(vals))})")
        return vals[0]

    merged['sample_offset_time_ms'] = _pick_param('sample_offset_time_ms')
    merged['minimum_sampling_time_ms'] = _pick_param('minimum_sampling_time_ms')
    merged['response_time_window_sec'] = _pick_param('response_time_window_sec')
    merged['hidden_rule_position'] = _pick_param('hidden_rule_position')

    hr_odors_all: list[str] = []
    for _, cls in per_run_cls:
        od = cls.get('hidden_rule_odors')
        if isinstance(od, (list, tuple)):
            hr_odors_all.extend([str(x) for x in od if isinstance(x, str) and x])
    if hr_odors_all:
        seen = set()
        uniq = []
        for x in hr_odors_all:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        merged['hidden_rule_odors'] = uniq
    else:
        merged.setdefault('hidden_rule_odors', [])

    # Per-run counts (sanity)
    runs_meta = []
    for ridx, cls in per_run_cls:
        def _n(k):
            df = cls.get(k)
            return int(len(df)) if isinstance(df, pd.DataFrame) else 0
        runs_meta.append({
            'run_id': ridx,
            'counts': {
                'initiated_sequences': _n('initiated_sequences'),
                'non_initiated_sequences': _n('non_initiated_sequences'),
                'non_initiated_odor1_attempts': _n('non_initiated_odor1_attempts'),
                'completed_sequences': _n('completed_sequences'),
                'completed_sequences_with_response_times': _n('completed_sequences_with_response_times'),
                'aborted_sequences': _n('aborted_sequences'),
            }
        })
    merged['runs'] = runs_meta

    # Rebuild index for convenience
    try:
        merged['index'] = build_classification_index(merged)
    except Exception:
        merged['index'] = {}

    if verbose:
        # Simple sanity: comp vs comp_rt lengths
        comp = merged.get('completed_sequences', pd.DataFrame())
        comp_rt = merged.get('completed_sequences_with_response_times', pd.DataFrame())
        if isinstance(comp, pd.DataFrame) and isinstance(comp_rt, pd.DataFrame) and not comp.empty:
            if len(comp) != len(comp_rt):
                print(f"[merge_classifications] NOTE: completed_sequences ({len(comp)}) != completed_sequences_with_response_times ({len(comp_rt)}). Using the RT table for RT summaries.")

        # Additional sanity: merged counts vs per-run sums
        try:
            total_non_ini = int(len(merged.get('non_initiated_sequences', [])))
            total_pos1 = int(len(merged.get('non_initiated_odor1_attempts', [])))
            total_initiated = int(len(merged.get('initiated_sequences', [])))
            sum_non_ini = sum(r['counts']['non_initiated_sequences'] for r in merged.get('runs', []))
            sum_pos1 = sum(r['counts']['non_initiated_odor1_attempts'] for r in merged.get('runs', []))
            sum_initiated = sum(r['counts']['initiated_sequences'] for r in merged.get('runs', []))
            if (total_non_ini != sum_non_ini) or (total_pos1 != sum_pos1) or (total_initiated != sum_initiated):
                print("[merge_classifications] WARNING: count mismatch after merge")
                print(f"  non_initiated total: merged={total_non_ini} vs per-run sum={sum_non_ini}")
                print(f"  pos1 total:          merged={total_pos1} vs per-run sum={sum_pos1}")
                print(f"  initiated total:     merged={total_initiated} vs per-run sum={sum_initiated}")
        except Exception:
            pass
    return merged


def print_merged_session_summary(merged_classification: dict, subjid=None, date=None, save=False, out_dir=None) -> None:
    """
    Summary for merged multi-run results.
    Uses completed_sequences_with_response_times as the authoritative completed table,
    avoiding re-merging that can drop RT rows.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):

        cls = merged_classification or {}

        # Parameters
        sample_offset_time_ms = cls.get("sample_offset_time_ms")
        minimum_sampling_time_ms = cls.get("minimum_sampling_time_ms")
        response_time_window_sec = cls.get("response_time_window_sec")
        hr_pos = cls.get("hidden_rule_position")
        hr_idx = (hr_pos - 1) if isinstance(hr_pos, (int, np.integer)) else None

        # Tables
        def get_df(key):
            df = cls.get(key)
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

        ini = get_df("initiated_sequences")
        non_ini = get_df("non_initiated_sequences")
        non_ini_pos1 = get_df("non_initiated_odor1_attempts")

        comp_rt = get_df("completed_sequences_with_response_times")  # authoritative
        comp = get_df("completed_sequences")
        if comp.empty and not comp_rt.empty:
            comp = comp_rt  # comp_rt has everything from comp + RT cols

        comp_rew = get_df("completed_sequence_rewarded")
        comp_unr = get_df("completed_sequence_unrewarded")
        comp_tmo = get_df("completed_sequence_reward_timeout")

        comp_hr = get_df("completed_sequences_HR")
        comp_hr_missed = get_df("completed_sequences_HR_missed")
        ab = get_df("aborted_sequences")
        ab_hr = get_df("aborted_sequences_HR")
        ab_det = get_df("aborted_sequences_detailed")

        # Helpers
        def pct(n, d):
            return (n / d * 100.0) if d else 0.0

        def fmt_ms(v):
            try:
                return f"{float(v):.1f}"
            except Exception:
                return "n/a"

        print("=" * 80, "\n")
        print("=" * 80)
        print(f"SUMMARY: TRIAL CLASSIFICATION AND POKE TIME ANALYSIS FOR SUBJECT [{subjid}] DATE [{date}]")
        print("=" * 80, "\n")
        print("=" * 80)
        if sample_offset_time_ms is not None:
            print(f"Sample offset time: {fmt_ms(sample_offset_time_ms)} ms")
        if minimum_sampling_time_ms is not None:
            print(f"Minimum sampling time: {fmt_ms(minimum_sampling_time_ms)} ms")
        if response_time_window_sec is not None:
            print(f"Response time window: {float(response_time_window_sec):.2f} s")

        # Attempts overview
        baseline_n = int(len(non_ini))
        pos1_n = int(len(non_ini_pos1))
        non_ini_total = baseline_n + pos1_n
        total_attempts = int(len(ini)) + non_ini_total
        print("\nTRIAL CLASSIFICATIONs:")
        print(f"Hidden Rule Location: Position {hr_pos if hr_pos is not None else 'None'} (index {hr_idx if hr_idx is not None else 'None'})\n")
        hr_odors = merged_classification.get('hidden_rule_odors') or []
        print(f"Hidden Rule Odors: {', '.join(hr_odors) if hr_odors else 'None'}\n")
        print(f"Total attempts: {total_attempts}")
        print(f"-- Non-initiated sequences (total): {non_ini_total} ({pct(non_ini_total, total_attempts):.1f}%)")
        print(f"    -- Position 1 attempts within trials {pos1_n} ({pct(pos1_n, non_ini_total):.1f}%)")
        print(f"    -- Baseline non-initiated sequences {baseline_n} ({pct(baseline_n, non_ini_total):.1f}%)")
        print(f"-- Initiated sequences (\033[1mtrials\033[0m]): {int(len(ini))} ({pct(len(ini), total_attempts):.1f}%)\n")

        # Initiated breakdown
        comp_n = int(len(comp))
        ab_n = int(len(ab))
        print("INITIATED TRIALS BREAKDOWN:")
        print(f"-- Completed sequences: {comp_n} ({pct(comp_n, len(ini)): .1f}%)")
        print(f"   -- Hidden Rule trials (HR): {int(len(comp_hr))} ({pct(len(comp_hr), comp_n):.1f}%)")
        print(f"   -- Hidden Rule Missed (HR_missed): {int(len(comp_hr_missed))} ({pct(len(comp_hr_missed), comp_n):.1f}%)")
        print(f"-- Aborted sequences: {ab_n} ({pct(ab_n, len(ini)): .1f}%)")
        print(f"   -- Aborted Hidden Rule trials (HR): {int(len(ab_hr))} ({pct(len(ab_hr), ab_n):.1f}%)\n")

        print(f"REWARDED TRIALS BREAKDOWN:")
        print(f"-- Rewarded: {int(len(comp_rew))} ({pct(len(comp_rew), comp_n):.1f}%)")
        print(f"-- Unrewarded: {int(len(comp_unr))} ({pct(len(comp_unr), comp_n):.1f}%)")
        print(f"-- Reward Timeout: {int(len(comp_tmo))} ({pct(len(comp_tmo), comp_n):.1f}%)\n")

        # Aggregate poke/valve time stats from completed trials (use comp, which has nested columns)
        def collect_pos_stats(df: pd.DataFrame):
            pos_poke = {i: [] for i in range(1, 6)}
            pos_valve = {i: [] for i in range(1, 6)}
            odor_poke = defaultdict(list)
            odor_valve = defaultdict(list)
            if df.empty:
                return pos_poke, pos_valve, odor_poke, odor_valve
            for _, r in df.iterrows():
                pps = r.get("position_poke_times") or {}
                vps = r.get("position_valve_times") or {}
                for pos in range(1, 6):
                    if pos in pps:
                        v = pps[pos].get("poke_time_ms")
                        if v is not None and not (isinstance(v, float) and np.isnan(v)):
                            pos_poke[pos].append(float(v))
                        od = pps[pos].get("odor_name")
                        if od is not None:
                            odor_poke[od].append(float(v) if v is not None else np.nan)
                    if pos in vps:
                        v = vps[pos].get("valve_duration_ms")
                        if v is not None and not (isinstance(v, float) and np.isnan(v)):
                            pos_valve[pos].append(float(v))
                        od = vps[pos].get("odor_name")
                        if od is not None:
                            odor_valve[od].append(float(v) if v is not None else np.nan)
            odor_poke = {k: [x for x in vals if not (isinstance(x, float) and np.isnan(x))] for k, vals in odor_poke.items()}
            odor_valve = {k: [x for x in vals if not (isinstance(x, float) and np.isnan(x))] for k, vals in odor_valve.items()}
            return pos_poke, pos_valve, odor_poke, odor_valve

        pos_poke, pos_valve, odor_poke, odor_valve = collect_pos_stats(comp)

        def print_range_block_pos(dct):
            print("----------------------------------------")
            for pos in range(1, 6):
                vals = dct.get(pos, [])
                if not vals:
                    continue
                a = np.asarray(vals, dtype=float)
                print(f"Position {pos}: {a.min():.1f} - {a.max():.1f}ms (avg: {a.mean():.1f}ms, n={a.size})")

        def print_range_block_odor(dct):
            print("--------------------------------------------------")
            for od in sorted(dct.keys()):
                vals = dct[od]
                if not vals:
                    continue
                a = np.asarray(vals, dtype=float)
                print(f"{od}: {a.min():.1f} - {a.max():.1f}ms (avg: {a.mean():.1f}ms, n={a.size})")

        print("POKE TIME RANGES BY POSITION:")
        print_range_block_pos(pos_poke)
        print("\nVALVE TIME RANGES BY POSITION:")
        print_range_block_pos(pos_valve)
        print("\nPOKE TIME RANGES BY ODOR (ALL POSITIONS):")
        print_range_block_odor(odor_poke)
        print("\nVALVE TIME RANGES BY ODOR (ALL POSITIONS):")
        print_range_block_odor(odor_valve)

        # Non-initiated poke times
        def _choose_poke_series(df: pd.DataFrame, prefer_cols: list[str]) -> pd.Series:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return pd.Series([], dtype=float)
            for c in prefer_cols:
                if c in df.columns:
                    return pd.to_numeric(df[c], errors="coerce").dropna()
            return pd.Series([], dtype=float)

        print("\nNON-INITIATED TRIALS POKE TIMES:")
        print("----------------------------------------")
        base_vals = _choose_poke_series(non_ini, ["continuous_poke_time_ms", "poke_time_ms", "poke_time", "poke_ms"])
        if base_vals.empty:
            print(f"Baseline non-initiated: n={baseline_n} (no valid poke times)")
        else:
            print(f"Baseline non-initiated: n={baseline_n} median={base_vals.median():.1f} ms range={base_vals.min():.1f}-{base_vals.max():.1f} ms")
        pos1_vals = _choose_poke_series(non_ini_pos1, ["pos1_poke_time_ms", "attempt_poke_time_ms", "poke_time_ms", "poke_time", "poke_ms"])
        if pos1_vals.empty:
            print(f"Pos1 attempts: n={pos1_n} (no valid poke times)")
        else:
            print(f"Pos1 attempts: n={pos1_n} median={pos1_vals.median():.1f} ms range={pos1_vals.min():.1f}-{pos1_vals.max():.1f} ms")

        # Response time analysis from comp_rt
        print("=" * 80)
        print("RESPONSE TIME ANALYSIS - ALL COMPLETED TRIALS")
        print("=" * 80)
        print(f"Total completed trials: {int(len(comp_rt))}\n")

        s_cat = comp_rt['response_time_category'] if 'response_time_category' in comp_rt.columns else pd.Series([], dtype='object')
        failed = int(s_cat.isna().sum()) if not comp_rt.empty else 0
        succeeded = int(len(comp_rt) - failed)
        print("RESPONSE TIME ANALYSIS RESULTS:")
        print(f"Total completed trials analyzed: {int(len(comp_rt))}")
        print(f"Failed response time calculations: {failed}")
        print(f"Successful response time calculations: {succeeded}\n")

        def rt_block(df, label):
            if df.empty or 'response_time_ms' not in df.columns:
                print(f"{label}:\n  No {label.lower()}")
                return
            s = pd.to_numeric(df['response_time_ms'], errors="coerce").dropna()
            if s.empty:
                print(f"{label}:\n  No {label.lower()}")
                return
            print(f"{label}:")
            print(f"  Count: {int(len(s))}")
            print(f"  Range: {s.min():.1f} - {s.max():.1f}ms")
            print(f"  Average: {s.mean():.1f}ms")
            print(f"  Median: {s.median():.1f}ms\n")

        def _cat(df: pd.DataFrame, cat: str) -> pd.DataFrame:
            if not isinstance(df, pd.DataFrame) or df.empty or 'response_time_category' not in df.columns:
                return pd.DataFrame()
            m = df['response_time_category'].astype('object') == cat
            return df[m]

        rew_rt = _cat(comp_rt, "rewarded")
        unr_rt = _cat(comp_rt, "unrewarded")
        tdel_rt = _cat(comp_rt, "timeout_delayed")

        rt_block(rew_rt, "REWARDED TRIALS")

        if not comp_hr.empty and not comp_rt.empty and 'trial_id' in comp_hr.columns and 'trial_id' in comp_rt.columns:
            hr_ids = set(comp_hr['trial_id'].dropna().tolist())
            hr_rew_rt = rew_rt[rew_rt['trial_id'].isin(hr_ids)] if not rew_rt.empty else pd.DataFrame()
            if hr_rew_rt.empty:
                print("HR REWARDED TRIALS (response times): none\n")
            else:
                rt_block(hr_rew_rt, "HR REWARDED TRIALS (response times)")
        else:
            print("HR REWARDED TRIALS (response times): none\n")

        rt_block(unr_rt, "UNREWARDED TRIALS")
        if tdel_rt.empty:
            print("REWARD TIMEOUT TRIALS:\n  No reward timeout trials\n")
        else:
            rt_block(tdel_rt, "REWARD TIMEOUT TRIALS")

        s_all = pd.to_numeric(comp_rt.get("response_time_ms"), errors="coerce").dropna() if not comp_rt.empty else pd.Series([], dtype=float)
        print("ALL TRIALS WITH RESPONSE TIMES:")
        if s_all.empty:
            print("  No trials with response times")
        else:
            print(f"  Count: {int(len(s_all))}")
            print(f"  Range: {s_all.min():.1f} - {s_all.max():.1f}ms")
            print(f"  Average: {s_all.mean():.1f}ms")
            print(f"  Median: {s_all.median():.1f}ms")

        # Aborted trials summary (same logic as before, but using concatenated tables)
        if not ab_det.empty:
            print("=" * 80)
            print("ABORTED TRIALS CLASSIFICATION SUMMARY")
            print("=" * 80)
            total = int(len(ab_det))
            ini_c = int((ab_det["abortion_type"] == "initiation_abortion").sum()) if 'abortion_type' in ab_det.columns else 0
            rei_c = int((ab_det["abortion_type"] == "reinitiation_abortion").sum()) if 'abortion_type' in ab_det.columns else 0
            print(f"- Total Aborted Trials: {total}")
            print(f"  - Re-Initiation Abortions: {rei_c} ({pct(rei_c, total):.1f}%)")
            print(f"  - Initiation Abortions:    {ini_c} ({pct(ini_c, total):.1f}%)\n")

            def _norm_fa(val):
                if pd.isna(val):
                    return 'nFA'
                s = str(val).strip().lower()
                if s in ('fa_time_in', 'fa in', 'fa_in', 'in'):
                    return 'FA_time_in'
                if s in ('fa_time_out', 'fa out', 'fa_out', 'out'):
                    return 'FA_time_out'
                if s in ('fa_late', 'late'):
                    return 'FA_late'
                return 'nFA'
            if 'fa_label' in ab_det.columns:
                ab_det = ab_det.copy()
                ab_det["fa_label"] = ab_det["fa_label"].apply(_norm_fa)

            fa_in = int((ab_det["fa_label"] == "FA_time_in").sum()) if "fa_label" in ab_det.columns else 0
            fa_out = int((ab_det["fa_label"] == "FA_time_out").sum()) if "fa_label" in ab_det.columns else 0
            fa_late = int((ab_det["fa_label"] == "FA_late").sum()) if "fa_label" in ab_det.columns else 0
            fa_total = fa_in + fa_out + fa_late
            nfa = total - fa_total

            print("False Alarms:")
            print(f"  - non-FA Abortions: {nfa} ({pct(nfa, total):.1f}%)")
            print(f"  - False Alarm abortions: {fa_total} ({pct(fa_total, total):.1f}%)")
            if fa_total > 0:
                print(f"      - FA Time In - Within Response Time Window ({float(response_time_window_sec) if response_time_window_sec is not None else 'n/a'} s):  {fa_in} ({pct(fa_in, fa_total):.1f}%)")
                s_in = pd.to_numeric(ab_det.loc[ab_det['fa_label'] == 'FA_time_in', 'fa_latency_ms'], errors='coerce').dropna() if 'fa_latency_ms' in ab_det.columns else pd.Series([], dtype=float)
                if len(s_in):
                    print(f"          - Response Time: median={s_in.median():.1f} ms, avg={s_in.mean():.1f} ms, range: {s_in.min():.1f} - {s_in.max():.1f} ms")
                if response_time_window_sec is not None:
                    lower_rt = response_time_window_sec
                    upper_rt = response_time_window_sec * 3
                    print(f"      - FA Time Out - Up to 3x Response Time Window ({int(lower_rt)}-{int(upper_rt)} s):  {fa_out} ({pct(fa_out, fa_total):.1f}%)")
                else:
                    print(f"      - FA Time Out: {fa_out} ({pct(fa_out, fa_total):.1f}%)")
                s_out = pd.to_numeric(ab_det.loc[ab_det['fa_label'] == 'FA_time_out', 'fa_latency_ms'], errors='coerce').dropna() if 'fa_latency_ms' in ab_det.columns else pd.Series([], dtype=float)
                if len(s_out):
                    print(f"          - Response Time: median={s_out.median():.1f} ms, avg={s_out.mean():.1f} ms, range: {s_out.min():.1f} - {s_out.max():.1f} ms")
                print(f"      - FA Late - After 3x Response Time up to next trial: {fa_late} ({pct(fa_late, fa_total):.1f}%)")
                s_late = pd.to_numeric(ab_det.loc[ab_det['fa_label'] == 'FA_late', 'fa_latency_ms'], errors='coerce').dropna() if 'fa_latency_ms' in ab_det.columns else pd.Series([], dtype=float)
                if len(s_late):
                    print(f"          - Response Time: median={s_late.median():.1f} ms, avg={s_late.mean():.1f} ms, range: {s_late.min():.1f} - {s_late.max():.1f} ms")

            # Abortions at Hidden Rule position: split into HR vs non-HR trials, with FA breakdown
            if hr_pos is not None and 'last_odor_position' in ab_det.columns:
                abortions_at_hr_pos = ab_det[ab_det['last_odor_position'] == hr_pos].copy()
                # Resolve HR-aborted trial IDs from merged classification
                hr_ab_df = cls.get('aborted_sequences_HR')
                if isinstance(hr_ab_df, pd.DataFrame) and not hr_ab_df.empty and 'trial_id' in hr_ab_df.columns:
                    hr_aborted_ids = set(hr_ab_df['trial_id'].dropna().tolist())
                else:
                    hr_aborted_ids = set()
                in_hr_trials = abortions_at_hr_pos[abortions_at_hr_pos['trial_id'].isin(hr_aborted_ids)].copy() if 'trial_id' in abortions_at_hr_pos.columns else pd.DataFrame()
                non_hr_trials = abortions_at_hr_pos[~abortions_at_hr_pos.get('trial_id', pd.Series([])).isin(hr_aborted_ids)].copy() if 'trial_id' in abortions_at_hr_pos.columns else abortions_at_hr_pos.copy()

                def _print_fa_counts(df, indent="        "):
                    order = ['nFA', 'FA_time_in', 'FA_time_out', 'FA_late']
                    labels = [
                        "Non-False Alarm",
                        "FA Time In (Within Response Time)",
                        "FA Time Out (Up to 3x Response Time)",
                        "FA Late (> 3x Response Time)"
                    ]
                    if df is None or df.empty or 'fa_label' not in df.columns:
                        return
                    cnt = df['fa_label'].value_counts().reindex(order, fill_value=0)
                    total_n = int(len(df))
                    for lbl, key in zip(labels, order):
                        v = int(cnt.get(key, 0))
                        p = (v / total_n * 100.0) if total_n else 0.0
                        print(f"{indent}- {lbl}: {v} ({p:.1f}%)")

                total_at_hr = int(len(abortions_at_hr_pos))
                print(f"\n  Abortions at Hidden Rule Position {hr_pos}: n={total_at_hr}")
                print(f"    Of which in Hidden Rule Trials: n={int(len(in_hr_trials))}")
                _print_fa_counts(in_hr_trials)
                print(f"    Non-Hidden Rule Abortions at HR Location: n={int(len(non_hr_trials))}")
                _print_fa_counts(non_hr_trials)

            # False Alarm classification for non-initiated trials (if present)
            fa_noninit_df = merged_classification.get('non_initiated_FA', pd.DataFrame())
            if isinstance(fa_noninit_df, pd.DataFrame) and not fa_noninit_df.empty: 
                print("\nFalse Alarm Classification for Non-Initiated Trials:")
                print(f"  Total Non-Initiated FA Trials: {int(len(fa_noninit_df))}")
                counts = fa_noninit_df['fa_label'].value_counts().reindex(['nFA','FA_time_in','FA_time_out','FA_late'], fill_value=0)
                total = int(len(fa_noninit_df))
                print(f"   - Non-False Alarm: {counts['nFA']} ({(counts['nFA']/total*100.0):.1f}%)")
                print(f"   - FA Time In (Within Response Time): {counts['FA_time_in']} ({(counts['FA_time_in']/total*100.0):.1f}%)")
                print(f"   - FA Time Out (Up to 3x Response Time): {counts['FA_time_out']} ({(counts['FA_time_out']/total*100.0):.1f}%)")
                print(f"   - FA Late (> 3x Response Time): {counts['FA_late']} ({(counts['FA_late']/total*100.0):.1f}%)")   


            # Helper for stats lines
            def _stats_line(series, label):
                s = pd.to_numeric(series, errors='coerce').dropna()
                if s.empty:
                    print(f"{label}: n=0")
                else:
                    print(f"{label}: n={len(s)} | median={s.median():.1f} ms | avg={s.mean():.1f} ms | range={s.min():.1f}-{s.max():.1f} ms")

            # Non-last Odor Pokes (exclude last_event_index per trial), only >= minimum_sampling_time_ms
            if {'presentations', 'last_event_index'}.issubset(ab_det.columns):
                pres_df = ab_det[['trial_id', 'presentations', 'last_event_index']].explode('presentations').dropna(subset=['presentations']).copy()
                if not pres_df.empty:
                    pres = pd.concat([pres_df.drop(columns=['presentations']), pres_df['presentations'].apply(pd.Series)], axis=1)
                    pres['is_last'] = pres['index_in_trial'] == pres['last_event_index']
                    pres = pres[~pres['is_last']].copy()
                    pres['poke_time_ms'] = pd.to_numeric(pres.get('poke_time_ms'), errors='coerce')
                    pres_valid = pres[(pres['poke_time_ms'] >= (minimum_sampling_time_ms or 0))].copy()

                    print("\nPoke Times for all Odors (Except aborted Odor):")
                    _stats_line(pres_valid['poke_time_ms'], "  - All Odors (except aborted)")

                    if 'position' in pres_valid.columns and not pres_valid.empty:
                        for pos, grp in pres_valid.groupby('position'):
                            _stats_line(grp['poke_time_ms'], f"  - Position {int(pos)}")

                    if 'odor_name' in pres_valid.columns and not pres_valid.empty:
                        for odor, grp in pres_valid.groupby('odor_name'):
                            _stats_line(grp['poke_time_ms'], f"  - Odor {odor}")
                else:
                    print("\nPoke Times for all Odors except aborted: n=0 (no presentations info)")
            else:
                print("\n Poke Times for all Odors except aborted: presentations not attached in aborted_sequences_detailed")

            # Last Odor Poke Times by abortion type
            if 'last_odor_poke_time_ms' in ab_det.columns and 'abortion_type' in ab_det.columns:
                print("\nAborted Odor Poke Times:")
                _stats_line(ab_det.loc[ab_det['abortion_type'] == 'reinitiation_abortion', 'last_odor_poke_time_ms'],
                            "  - Re-Initiation Abortions")
                _stats_line(ab_det.loc[ab_det['abortion_type'] == 'initiation_abortion', 'last_odor_poke_time_ms'],
                            "  - Initiation Abortions")

            # Counts by last odor
            if 'last_odor_name' in ab_det.columns and 'abortion_type' in ab_det.columns:
                print("\nCounts by last odor:")
                by_odor = (
                    ab_det
                    .groupby(['last_odor_name', 'abortion_type'])
                    .size()
                    .unstack(fill_value=0)
                    .rename(columns={'reinitiation_abortion': 'Re-initiation', 'initiation_abortion': 'Initiation'})
                )
                totals = ab_det.groupby('last_odor_name').size()
                for odor in totals.index:
                    rei_c = int(by_odor.loc[odor].get('Re-initiation', 0))
                    ini_c = int(by_odor.loc[odor].get('Initiation', 0))
                    tot = int(totals.loc[odor])
                    print(f"  - {odor}: {tot} abortions, Re-initiation {rei_c}, Initiation {ini_c}")

            # Counts by last position
            if 'last_odor_position' in ab_det.columns and 'abortion_type' in ab_det.columns:
                print("\nCounts by last position:")
                by_pos = (
                    ab_det
                    .groupby(['last_odor_position', 'abortion_type'])
                    .size()
                    .unstack(fill_value=0)
                    .rename(columns={'reinitiation_abortion': 'Re-initiation', 'initiation_abortion': 'Initiation'})
                )
                totals_pos = ab_det.groupby('last_odor_position').size()
                for pos in sorted(totals_pos.index):
                    rei_c = int(by_pos.loc[pos].get('Re-initiation', 0))
                    ini_c = int(by_pos.loc[pos].get('Initiation', 0))
                    tot = int(totals_pos.loc[pos])
                    print(f"  - Position {int(pos)}: {tot} abortions, Re-initiation {rei_c}, Initiation {ini_c}")

    print(buffer.getvalue())
    if save:
        # Determine output directory
        if out_dir is None:
            # Try to resolve from classification dict
            root = merged_classification.get("paths", {}).get("rawdata_dir", None)
            if root is None:
                root = merged_classification.get("rawdata_dir", None)
            if root is None:
                print("No output directory found for saving summary.")
                return
            out_dir = Path(root).parent / "derivatives" / f"sub-{subjid}" / f"ses-{date}" / "saved_analysis_results"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"merged_summary_{subjid}_{date}.txt"
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            f.write(buffer.getvalue())
        print(f"Saved merged session summary to {out_dir / fname}")
    

def analyze_session_multi_run_by_id_date(subject_id: str, date_str: str, *, verbose: bool = True, max_runs: int = 32, save: bool = True, print_summary: bool = True):
    """
    Analyze all experiment files for a subject on a given date, then merge and (optionally) save.
    Now builds data/events/odor_map/trial_counts/stage from the root returned by load_experiment().
    """

    subject_id = str(subject_id)
    date_str = str(date_str)

    def _maybe_silent(callable_, *args, **kwargs):
        if verbose:
            return callable_(*args, **kwargs)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return callable_(*args, **kwargs)
    
    # Discover unique experiment files for this subject/date
    session_roots: list[Path] = []
    visited: set[Path] = set()
    le = globals().get("load_experiment")
    if callable(le):
        for i in range(max_runs):
            try:
                root_i = _maybe_silent(le, subject_id, date_str, index=i)
                if not root_i:
                    break
                p = Path(root_i).resolve()
                if p in visited:
                    vprint(verbose, f"[analyze_session_multi_run] Duplicate experiment root at index {i}: {p}. Stopping discovery.")
                    break
                visited.add(p)
                session_roots.append(p)
            except (IndexError, FileNotFoundError) as e:
                vprint(verbose, f"[analyze_session_multi_run] Stopping at index {i}: {e}")
                break
    if not session_roots:
        raise RuntimeError(f"No experiment runs found for subject={subject_id} date={date_str}")

    # Sort oldest -> newest (or reverse=True for newest first)
    def _parse_ts(p: Path):
        from datetime import datetime
        try:
            return datetime.strptime(p.name, "%Y-%m-%dT%H-%M-%S")
        except Exception:
            return datetime.min
    try:
        session_roots.sort(key=_parse_ts)
    except Exception:
        session_roots.sort(key=lambda p: str(p))
    per_run = []
    merge_inputs = []
    roots: list[Optional[Path]] = []
    stages = []

    def extract_run_end_time(data, events):
        """Extract the latest timestamp from data and events for a single run"""
        all_timestamps = []
        
        # Collect timestamps from data streams
        for key, stream in data.items():
            if isinstance(stream, pd.DataFrame) and "Time" in stream.columns:
                timestamps = pd.to_datetime(stream["Time"], errors="coerce").dropna()
                all_timestamps.extend(timestamps)
            elif isinstance(stream, pd.Series) and hasattr(stream.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(stream.index):
                all_timestamps.extend(stream.index)
        
        # Collect timestamps from events
        for key, event in events.items():
            if isinstance(event, pd.DataFrame) and "Time" in event.columns:
                timestamps = pd.to_datetime(event["Time"], errors="coerce").dropna()
                all_timestamps.extend(timestamps)
        
        # Find the latest timestamp
        if all_timestamps:
            return max(all_timestamps)
        return None


    for i, root in enumerate(session_roots[:max_runs]):
        try:
            vprint(verbose, f"[analyze_session_multi_run] Loaded run index {i}: root={root}")

            # Detect stage (best-effort)
            try:
                import src.processing.detect_stage as detect_stage_module
                stage = detect_stage_module.detect_stage(root)
            except Exception:
                stage = {'stage_name': str(root)}

            # Run pipeline
            data = _maybe_silent(load_all_streams, root, verbose=verbose)
            events = _maybe_silent(load_experiment_events, root, verbose=verbose)
            run_end_time = extract_run_end_time(data, events)
            odor_map = _maybe_silent(load_odor_mapping, root, data=data, verbose=verbose)
            trial_counts = detect_trials(data, events, root, verbose=verbose)

            out = _maybe_silent(
                classify_and_analyze_with_response_times,
                data, events, trial_counts, odor_map, stage, root, verbose=verbose
            )

            if isinstance(stage, dict):
                stage['run_end_time'] = run_end_time
            else:
                stage = {'stage_name': str(stage), 'run_end_time': run_end_time}

            cls = out['classification'] if isinstance(out, dict) and 'classification' in out else out
            DIP0 = data['digital_input_data'].get('DIPort0', pd.Series(dtype=bool))
            DIP1 = data['digital_input_data'].get('DIPort1', pd.Series(dtype=bool))
            DIP2 = data['digital_input_data'].get('DIPort2', pd.Series(dtype=bool))
            hr_odors = cls.get('hidden_rule_odors', [])
            non_init = cls.get('non_initiated_sequences', pd.DataFrame())
            pos_1_attempt = cls.get('non_initiated_odor1_attempts', pd.DataFrame())
            fa_input_data = pd.concat([non_init, pos_1_attempt], ignore_index=True)
            response_time = cls.get('response_time_window_sec') 
            fa_noninit_df = classify_noninitiated_FA(fa_input_data, DIP0, DIP1, DIP2, response_time, hr_odors=hr_odors)
            cls['non_initiated_FA'] = fa_noninit_df
            out['classification'] = cls

            # Normalize outputs for merging
            if isinstance(out, dict) and 'classification' in out:
                cls = out['classification'] or {}
                merge_inputs.append(out)  # keep full dict so RT tables survive merge
            elif isinstance(out, dict):
                cls = out or {}
                merge_inputs.append(cls)
            else:
                cls = out or {}
                merge_inputs.append({'classification': cls})

            if not isinstance(cls, dict) or not cls:
                raise RuntimeError("Empty classification output")

            per_run.append(cls)
            roots.append(root)
            stages.append(stage)

        except Exception as e:
            vprint(verbose, f"[analyze_session_multi_run] Skipping run index {i} due to error: {e}")
            continue

    if not per_run:
        raise RuntimeError(f"No runs analyzed for subject={subject_id} date={date_str}")

    merged = merge_classifications(merge_inputs, verbose=verbose)
    merged['aborted_index'] = merged.get('index', {}).get('aborted', {})
    merged['non_initiated_FA'] = merged.get('non_initiated_FA', pd.DataFrame())

    save_dir = None
    save_err: Exception | None = None
    if save:
        first_root = roots[0] if roots and roots[0] is not None else None
        session_meta = {
            'multi_run': True,
            'subject_id': subject_id,
            'date': date_str,
            'runs': [
                {
                    'run_id': ridx + 1,
                    'root': str(r) if r is not None else None,
                    'stage': stages[ridx] if ridx < len(stages) else None
                }
                for ridx, r in enumerate(roots)
            ]
        }
        try:
            save_dir = save_session_analysis_results(merged, first_root, session_meta, data, events, verbose=verbose)
        except Exception as e:
            save_err = e
            vprint(verbose, f"[save] WARNING: {e}")

    if print_summary:
        print_merged_session_summary(merged, subjid=subject_id, date=date_str, save=save, out_dir=save_dir)
    
    if save:
        if save_dir:
            print(f"[save] Success: results saved to: {save_dir}")
        else:
            msg = f"[save] FAILED {save_err}" if save_err else "[save] FAILED: no output directory returned"
            print(msg)
    else:
        print(f"[save] Skipped: save=False")

    cls = merged
    return {
        "classification": cls,                      
        "merged_classification": cls,              
        "per_run_classifications": per_run,
        "roots": roots,
        "stages": stages,
        "save_dir": save_dir,

        # convenience shortcuts to common tables
        "completed_sequences_with_response_times": cls.get("completed_sequences_with_response_times", pd.DataFrame()),
        "completed_sequences": cls.get("completed_sequences", pd.DataFrame()),
        "completed_sequence_rewarded": cls.get("completed_sequence_rewarded", pd.DataFrame()),
        "completed_sequence_unrewarded": cls.get("completed_sequence_unrewarded", pd.DataFrame()),
        "completed_sequence_reward_timeout": cls.get("completed_sequence_reward_timeout", pd.DataFrame()),
        "aborted_sequences": cls.get("aborted_sequences", pd.DataFrame()),
    }


def build_position_pokes_table(classification: dict, *, threshold_ms: float | None = None) -> pd.DataFrame:
    """
    Flatten per-position poke info from completed_sequences into a tidy DataFrame.
    Columns: run_id, trial_id, position, odor, poke_time_ms, poke_first_in, valve_open_ts, valve_close_ts.
    If threshold_ms is provided, rows with poke_time_ms >= threshold_ms are dropped.
    """

    comp = classification.get("completed_sequences", pd.DataFrame())
    if not isinstance(comp, pd.DataFrame) or comp.empty:
        return pd.DataFrame(columns=[
            "run_id","trial_id","position","odor","poke_time_ms","poke_first_in","valve_open_ts","valve_close_ts"
        ])

    def _iter_items(ppt):
        if isinstance(ppt, Mapping):
            for k, v in ppt.items():
                if not isinstance(v, Mapping):
                    try:
                        v = dict(v)
                    except Exception:
                        continue
                pos = v.get("position")
                if pos is None:
                    try:
                        pos = int(k)
                    except Exception:
                        pos = k
                yield pos, v
        elif isinstance(ppt, (list, tuple)):
            for v in ppt:
                if isinstance(v, Mapping):
                    yield v.get("position"), v

    def _norm_valves(pvt):
        out = {}
        if isinstance(pvt, Mapping):
            items = list(pvt.items())
        elif isinstance(pvt, (list, tuple)):
            items = [(v.get("position"), v) for v in pvt if isinstance(v, Mapping)]
        else:
            items = []
        for k, v in items:
            if not isinstance(v, Mapping):
                try:
                    v = dict(v)
                except Exception:
                    v = {}
            pos = v.get("position")
            if pos is None:
                try:
                    pos = int(k)
                except Exception:
                    pos = k
            out[pos] = v
        return out

    rows = []
    for _, row in comp.iterrows():
        ppt = row.get("position_poke_times")
        if not isinstance(ppt, (dict, list, tuple)):
            continue
        pvt = row.get("position_valve_times") or {}
        valve_map = _norm_valves(pvt)

        run_id = row.get("run_id")
        trial_id = row.get("trial_id")
        try:
            run_id = int(run_id) if pd.notna(run_id) else None
        except Exception:
            pass
        try:
            trial_id = int(trial_id) if pd.notna(trial_id) else trial_id
        except Exception:
            pass

        for pos, info in _iter_items(ppt):
            if not isinstance(info, Mapping):
                try:
                    info = dict(info)
                except Exception:
                    continue
            poke_ms = pd.to_numeric(info.get("poke_time_ms"), errors="coerce")
            if pd.isna(poke_ms) or poke_ms <= 0:
                continue
            if threshold_ms is not None and float(poke_ms) >= float(threshold_ms):
                continue
            try:
                pos_norm = int(pos) if pos is not None else None
            except Exception:
                pos_norm = pos
            vt = valve_map.get(pos_norm, {})
            rows.append({
                "run_id": run_id,
                "trial_id": trial_id,
                "position": pos_norm,
                "odor": info.get("odor_name") or (vt or {}).get("odor_name"),
                "poke_time_ms": float(poke_ms),
                "poke_first_in": info.get("poke_first_in"),
                "valve_open_ts": (vt or {}).get("valve_open_ts"),
                "valve_close_ts": (vt or {}).get("valve_close_ts"),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        if "valve_open_ts" in out.columns:
            out["valve_open_ts"] = pd.to_datetime(out["valve_open_ts"], errors="coerce")
        if "poke_first_in" in out.columns:
            out["poke_first_in"] = pd.to_datetime(out["poke_first_in"], errors="coerce")
        out = out.sort_values(["run_id","trial_id","position","valve_open_ts"], kind="stable", na_position="last").reset_index(drop=True)
    return out


def _parse_date_input(dates_input):
    """
    Parse date input into a list of dates to analyze.
    
    - If dates_input is a list/iterable: return as-is (specific dates)
    - If dates_input is a tuple of 2 elements: treat as (start_date, end_date) range
      and discover all dates in that range from the filesystem
    - If dates_input is None: return None (analyze all dates)
    
    Returns: list of dates (int YYYYMMDD format) or None
    """
    if dates_input is None:
        return None
    
    # If it's a tuple with exactly 2 elements, treat as a range
    if isinstance(dates_input, tuple) and len(dates_input) == 2:
        start_date = int(dates_input[0])
        end_date = int(dates_input[1])
        
        # Convert to datetime for range operations
        start_dt = pd.to_datetime(str(start_date), format='%Y%m%d')
        end_dt = pd.to_datetime(str(end_date), format='%Y%m%d')
        
        # Generate all dates in range
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
        dates_list = [int(dt.strftime('%Y%m%d')) for dt in date_range]
        
        return dates_list
    
    # Otherwise treat as specific dates (list, set, etc.)
    return list(dates_input)

def batch_analyze_sessions(
    subjids=None,
    dates=None,
    *,
    save=True,
    verbose=False,
    print_summary=True,
    max_runs=200
):
    """
    Analyze all sessions for given subject(s) and/or date(s).
    - If subjids is None: analyze all subjects found in rawdata.
    - If dates is None: analyze all dates found for each subject.
    - If both are lists: analyze all combinations.
    - Handles missing subjects/dates gracefully.
    Returns a dict: {(subjid, date): result_dict}
    """
    base_path = Path(project_root) / "data" / "rawdata"
    results = {}

    # Discover subjects
    if subjids is None:
        subj_dirs = sorted(base_path.glob("sub-*_id-*"))
        subjids = [int(str(d.name).split('_')[0].replace('sub-', '')) for d in subj_dirs]
    else:
        subjids = [int(s) for s in subjids]

    dates_to_run_global = _parse_date_input(dates)

    for subjid in subjids:
        subj_str = f"sub-{str(subjid).zfill(3)}"
        subj_dirs = list(base_path.glob(f"{subj_str}_id-*"))
        if not subj_dirs:
            print(f"[batch_analyze_sessions] WARNING: Subject {subjid} not found.")
            continue
        subj_dir = subj_dirs[0]
        
        # Discover available dates for this subject
        session_dirs = sorted(subj_dir.glob("ses-*_date-*"))
        available_dates = [int(d.name.split('date-')[-1]) for d in session_dirs]
        
        # Determine which dates to run for this subject
        if dates_to_run_global is None:
            # Analyze all available dates
            dates_for_subject = available_dates
        else:
            # Use only dates that exist for this subject
            dates_for_subject = [dt for dt in dates_to_run_global if dt in available_dates]
            missing = [dt for dt in dates_to_run_global if dt not in available_dates]
            if missing and verbose:
                for dt in missing:
                    print(f"[batch_analyze_sessions] WARNING: Date {dt} not found for subject {subjid}.")
        
        for date in dates_for_subject:
            try:
                print(f"\n[batch_analyze_sessions] Analyzing subject {subjid}, date {date}...")
                res = analyze_session_multi_run_by_id_date(
                    subjid, date,
                    save=save,
                    verbose=verbose,
                    print_summary=print_summary,
                    max_runs=max_runs
                )
                results[(subjid, date)] = res
            except Exception as e:
                print(f"[batch_analyze_sessions] WARNING: Failed to analyze subject {subjid}, date {date}: {e}")
                continue
    
    # Return summary of analyzed sessions
    analyzed = {}
    for (subjid, date) in results.keys():
        analyzed.setdefault(subjid, []).append(date)
    
    print("\nAnalyzed session(s) for:")
    for subjid in sorted(analyzed):
        print(f"Subject {subjid}:")
        for date in sorted(analyzed[subjid]):
            print(f"    {date}")

    return results

# ========================= Further functions / miscillaneous =========================


def cut_video(subjid, date, start_time, end_time, index=None, fps=30):
    """
    Cut a video snippet for a subject and date, given start and end time (HH:MM:SS.s).
    Automatically finds the correct experiment folder whose video covers the requested time window.
    """
    from utils.classification_utils import load_all_streams
    base_path = Path(project_root) / "data" / "rawdata"
    subjid_str = f"sub-{str(subjid).zfill(3)}"
    date_str = str(date)
    subject_dirs = list(base_path.glob(f"{subjid_str}_id-*"))
    if not subject_dirs:
        raise FileNotFoundError(f"No subject directory found for {subjid_str}")
    subject_dir = subject_dirs[0]
    session_dirs = list(subject_dir.glob(f"ses-*_date-{date_str}"))
    if not session_dirs:
        all_sessions = list(subject_dir.glob("ses-*"))
        session_names = [d.name for d in all_sessions]
        raise FileNotFoundError(f"No session found for date {date_str} in {subject_dir}.\nAvailable sessions: {session_names}")
    session_dir = session_dirs[0]
    behav_dir = session_dir / "behav"
    if not behav_dir.exists():
        raise FileNotFoundError(f"No behav directory found in {session_dir}")
    experiment_dirs = [d for d in behav_dir.iterdir() if d.is_dir() and re.match(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', d.name)]
    if not experiment_dirs:
        all_dirs = [d.name for d in behav_dir.iterdir() if d.is_dir()]
        raise FileNotFoundError(f"No experiment directories found in {behav_dir}.\nAvailable directories: {all_dirs}")
    experiment_dirs.sort(key=lambda x: x.name)

    # Parse times
    start_dt = pd.to_datetime(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {start_time}")
    end_dt = pd.to_datetime(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {end_time}")

    # If index is given, use it directly
    if index is not None:
        if index >= len(experiment_dirs) or index < 0:
            raise IndexError(f"Index {index} out of range. Available indices: 0-{len(experiment_dirs)-1}")
        root = experiment_dirs[index]
        video_dir = root / "VideoData"
        streams = load_all_streams(root, verbose=False)
        video_meta = streams['video_data']
        frames_in_window = video_meta[(video_meta.index >= start_dt) & (video_meta.index <= end_dt)]
        if frames_in_window.empty:
            print("No frames found in the requested time window for the specified index.")
            return None
    else:
        # Search all experiment folders for matching frames
        root = None
        frames_in_window = None
        video_meta = None
        video_dir = None
        for exp_dir in experiment_dirs:
            streams = load_all_streams(exp_dir, verbose=False)
            vm = streams['video_data']
            fiw = vm[(vm.index >= start_dt) & (vm.index <= end_dt)]
            if not fiw.empty:
                root = exp_dir
                video_meta = vm
                frames_in_window = fiw
                video_dir = exp_dir / "VideoData"
                break
        if frames_in_window is None or frames_in_window.empty:
            print("No frames found in the requested time window in any experiment folder.")
            return None

    # Detect frame column
    frame_col = [c for c in frames_in_window.columns if 'frame' in c.lower()][0]
    frame_indices = frames_in_window[frame_col].tolist()
    
    # Filter out macOS resource fork files (files starting with ._)
    avi_files = sorted([f for f in video_dir.glob("*.avi") if not f.name.startswith("._")])
    
    if not avi_files:
        print("No valid AVI files found in video directory.")
        return None
    
    images = []
    for video_file in avi_files:
        print(f"Attempting to read from: {video_file}")
        cap = cv2.VideoCapture(str(video_file))
        
        if not cap.isOpened():
            print(f"Failed to open {video_file}")
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video has {total_frames} total frames, requesting frames: {frame_indices[0]} to {frame_indices[-1]}")
        
        images = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                print(f"Failed to read frame {idx}")
        
        cap.release()
        
        if images:
            print(f"Successfully read {len(images)} frames from {video_file}")
            break
        else:
            print(f"No frames could be read from {video_file}")
    
    if not images:
        print("No frames extracted from any video file.")
        return None
        
    # Output folder (derivatives/video_analysis)
    server_root = base_path.resolve().parent
    derivatives_dir = server_root / "derivatives" / subject_dir.name / session_dir.name / "video_analysis"
    derivatives_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = derivatives_dir / f"video_cut_{start_dt.strftime('%H-%M-%S-%f')}_{end_dt.strftime('%H-%M-%S-%f')}.mp4"
    clip = ImageSequenceClip(images, fps=fps)
    clip.write_videofile(str(out_mp4), codec="libx264", audio=False)
    print(f"Saved cut video to: {out_mp4}")
    return out_mp4


def plot_valve_and_poke_events(
    root,
    time_window=None,
    interactive=True,
    show=True,
    verbose=True,
):
    """
    Plot valve and poke events efficiently by:
      - Discovering experiment subfolders
      - If time_window is provided, selecting only subfolders whose heartbeat span overlaps the window
      - Loading only the required registers (valves, digital inputs, outputs, pulse supplies)
      - Applying the same real-time correction as load_all_streams (heartbeat + folder timestamp)
      - Concatenating and plotting

    time_window: tuple/list like (start_str, end_str or None) using 'HH:MM:SS[.ms]' in Europe/London.
                 If end is None, a 1-minute window starting at start is used.
    """
    import re
    import numpy as np
    import pandas as pd
    import harp
    from pathlib import Path
    from datetime import datetime, timezone
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import zoneinfo
    from glob import glob as _glob

    mpl.rcParams['timezone'] = 'Europe/London'
    uk_tz = zoneinfo.ZoneInfo("Europe/London")

    # --- Discover all experiment folders ---
    behav_dir = Path(root)
    if behav_dir.name != "behav":
        behav_dir = behav_dir.parent if behav_dir.parent.name == "behav" else behav_dir
    exp_dirs = [d for d in behav_dir.iterdir() if d.is_dir() and d.name.startswith("20") and "T" in d.name]
    exp_dirs.sort(key=lambda x: x.name)
    if verbose:
        print(f"Found {len(exp_dirs)} experiment files in {behav_dir}")

    if not exp_dirs:
        raise FileNotFoundError(f"No experiment directories found in: {behav_dir}")

    # Helpers to parse experiment folder timestamp and build a window
    def parse_exp_ts_to_uk(exp_dir: Path) -> datetime:
        # Name format 'YYYY-MM-DDTHH-MM-SS'
        m = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', exp_dir.name)
        if not m:
            # fallback: try parent
            m = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', exp_dir.parent.name)
        if not m:
            return None
        real_time_str = m.group(0)
        # Original pipeline: treat folder timestamp as UTC, then convert to Europe/London
        ref_utc = datetime.strptime(real_time_str, '%Y-%m-%dT%H-%M-%S').replace(tzinfo=timezone.utc)
        return ref_utc.astimezone(uk_tz)

    # Parse requested time window (Europe/London)
    # If user provided time strings only, anchor them to the date of the first exp_dir
    first_exp_dt_uk = parse_exp_ts_to_uk(exp_dirs[0])
    if first_exp_dt_uk is None:
        # fallback to today's date
        first_exp_dt_uk = datetime.now(uk_tz)
    if time_window is not None:
        start_str = time_window[0]
        end_str = time_window[1] if len(time_window) > 1 and time_window[1] else None

        date_str = first_exp_dt_uk.strftime('%Y-%m-%d')
        # Try with and without microseconds
        def _parse_hhmmss(s):
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                try:
                    return pd.to_datetime(f"{date_str} {s}", format=fmt)
                except Exception:
                    continue
            # generic parser
            return pd.to_datetime(f"{date_str} {s}", errors='coerce')
        start_dt = _parse_hhmmss(start_str)
        if pd.isna(start_dt):
            raise ValueError(f"Could not parse start time: {start_str}")
        if start_dt.tzinfo is None:
            start_dt = start_dt.tz_localize(uk_tz)
        if end_str:
            end_dt = _parse_hhmmss(end_str)
            if pd.isna(end_dt):
                raise ValueError(f"Could not parse end time: {end_str}")
            if end_dt.tzinfo is None:
                end_dt = end_dt.tz_localize(uk_tz)
        else:
            end_dt = start_dt + pd.Timedelta(minutes=1)
    else:
        start_dt = None
        end_dt = None

    # --- Readers and minimal loader (only needed registers) ---
    behavior_reader = harp.create_reader('device_schemas/behavior.yml', epoch=harp.REFERENCE_EPOCH)
    olf_reader = harp.create_reader('device_schemas/olfactometer.yml', epoch=harp.REFERENCE_EPOCH)

    registers = dict(
        odor_valve_state_0=("olfactometer_valves_0", olf_reader.OdorValveState, "Olfactometer0"),
        odor_valve_state_1=("olfactometer_valves_1", olf_reader.OdorValveState, "Olfactometer1"),
        digital_input=("digital_input_data", behavior_reader.DigitalInputState, "Behavior"),
        pulse_supply_1=("pulse_supply_1", behavior_reader.PulseSupplyPort1, "Behavior"),
        pulse_supply_2=("pulse_supply_2", behavior_reader.PulseSupplyPort2, "Behavior"),
        output_set=("output_set", behavior_reader.OutputSet, "Behavior"),
        output_clear=("output_clear", behavior_reader.OutputClear, "Behavior"),
        heartbeat=("heartbeat", behavior_reader.TimestampSeconds, "Behavior"),
    )

    def _safe_concat(dfs):
        dfs = [d for d in dfs if d is not None and isinstance(d, (pd.Series, pd.DataFrame)) and not d.empty]
        if not dfs:
            # Preserve type used downstream
            return pd.DataFrame()
        out = pd.concat(dfs, axis=0)
        try:
            out = out.sort_index()
        except Exception:
            pass
        return out

    def _apply_offset_and_localize(df, offset: pd.Timedelta):
        if df is None or (hasattr(df, "empty") and df.empty):
            return df
        idx = df.index
        # Ensure datetime index
        if not isinstance(idx, pd.DatetimeIndex):
            # If the reader produced a column 'Time', move it to index
            if 'Time' in df.columns:
                df = df.set_index('Time')
                idx = df.index
            else:
                # give up (unexpected)
                return df
        # Apply offset
        try:
            df.index = df.index + offset
        except Exception:
            pass
        # Ensure tz-aware Europe/London
        if df.index.tz is None:
            try:
                df.index = df.index.tz_localize(uk_tz)
            except Exception:
                # If index already tz-aware in another tz, try convert
                try:
                    df.index = df.index.tz_convert(uk_tz)
                except Exception:
                    pass
        return df

    def _slice(df, start, end):
        if df is None or (hasattr(df, "empty") and df.empty) or start is None or end is None:
            return df
        try:
            return df.loc[(df.index >= start) & (df.index <= end)]
        except Exception:
            return df

    # Per-file loader: enumerate all files for a register, skip empty/bad, concat the rest
    def _load_register_files(reg, folder: Path) -> pd.DataFrame | None:
        try:
            folder = Path(folder)
            if not folder.exists():
                return None
            pattern = f"{folder.joinpath(folder.name)}_{reg.register.address}_*.bin"
            files = sorted(_glob(pattern))
            if not files:
                return None
            chunks = []
            for f in files:
                try:
                    df = reg.read(f)
                    if df is None or (hasattr(df, "empty") and df.empty):
                        continue
                    chunks.append(df)
                except Exception:
                    # skip bad file
                    continue
            if not chunks:
                return None
            out = pd.concat(chunks, axis=0)
            try:
                out = out.sort_index()
            except Exception:
                pass
            return out
        except Exception:
            return None

    def _compute_real_time_offset(exp_dir: Path) -> tuple[pd.Timedelta, pd.Timestamp | None, pd.Timestamp | None]:
        """
        Compute the same real_time_offset used by load_all_streams.
        Also returns the heartbeat span (after offset) for quick overlap checks.
        """
        offset = pd.Timedelta(0)
        hb_start_uk = None
        hb_end_uk = None
        try:
            hb = load(registers['heartbeat'][1], exp_dir / registers['heartbeat'][2])
            if not hb.empty:
                hb = hb.reset_index()  # ensure 'Time' is a column
            # Folder timestamp
            m = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', exp_dir.name)
            if not m:
                m = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', exp_dir.parent.name)
            if m:
                real_time_str = m.group(0)
                real_time_ref_utc = datetime.strptime(real_time_str, '%Y-%m-%dT%H-%M-%S')
                real_time_ref_utc = real_time_ref_utc.replace(tzinfo=timezone.utc)
                real_time_ref = real_time_ref_utc.astimezone(uk_tz)
                if 'Time' in hb.columns and len(hb) > 0:
                    hb['Time'] = pd.to_datetime(hb['Time'], errors='coerce')
                    start_time_dt = hb['Time'].iloc[0].to_pydatetime() if isinstance(hb['Time'].iloc[0], pd.Timestamp) else hb['Time'].iloc[0]
                    if start_time_dt.tzinfo is None:
                        start_time_dt = start_time_dt.replace(tzinfo=uk_tz)
                    offset = real_time_ref - start_time_dt
            # Heartbeat span (after offset)
            if 'Time' in hb.columns and not hb.empty:
                times = pd.to_datetime(hb['Time'], errors='coerce') + offset
                if times.dt.tz is None:
                    times = times.dt.tz_localize(uk_tz)
                else:
                    times = times.dt.tz_convert(uk_tz)
                hb_start_uk = times.min()
                hb_end_uk = times.max()
        except Exception as e:
            if verbose:
                print(f"[WARN] Heartbeat timing failed for {exp_dir.name}: {e}")
        return offset, hb_start_uk, hb_end_uk

    # If a time_window is provided, preselect only exp_dirs that overlap it using heartbeat span
    candidate_exp_dirs = []
    exp_dir_offsets = {}           # exp_dir -> offset
    exp_dir_first_times = []       # for session cut detection, earliest per exp
    for exp_dir in exp_dirs:
        offset, hb_start, hb_end = _compute_real_time_offset(exp_dir)
        exp_dir_offsets[exp_dir] = offset
        # Collect earliest known time (for later session cut markers)
        if hb_start is not None:
            exp_dir_first_times.append(hb_start)

        if start_dt is None or end_dt is None:
            candidate_exp_dirs.append(exp_dir)
            continue
        # Include only if heartbeat span overlaps the requested window
        if hb_start is None or hb_end is None:
            # Unknown span; conservatively include
            candidate_exp_dirs.append(exp_dir)
        else:
            if not (hb_end < start_dt or hb_start > end_dt):
                candidate_exp_dirs.append(exp_dir)

    if verbose:
        print(f"Selected {len(candidate_exp_dirs)} experiment(s) for loading based on time window overlap.")

    # --- Load and align required streams for each selected experiment ---
    valves0_list = []
    valves1_list = []
    digital_list = []
    pulse1_list = []
    pulse2_list = []
    output_set_list = []
    output_clear_list = []
    endinit_list = []

    def _try_load(label: str, reg, folder: Path) -> pd.DataFrame | None:
        try:
            if not folder.exists():
                return None
            # Load all matching files per register, skip empty/bad, concat good parts
            df = _load_register_files(reg, folder)
            if df is None or (hasattr(df, 'empty') and df.empty):
                return None
            return df
        except Exception as e:
            if verbose:
                print(f"[WARN] {label} load failed in {folder}: {e}")
            return None

    for exp_dir in candidate_exp_dirs:
        offset = exp_dir_offsets.get(exp_dir, pd.Timedelta(0))

        loaded = {
            key: _try_load(label, reg, exp_dir / subfolder)
            for key, (label, reg, subfolder) in registers.items()
            if key != "heartbeat"
        }

        # Apply offset and tz to each loaded stream
        for k in list(loaded.keys()):
            loaded[k] = _apply_offset_and_localize(loaded[k], offset)
            # If a time window is specified, slice to it now
            if time_window is not None:
                loaded[k] = _slice(loaded[k], start_dt, end_dt)

        # Add to accumulators
        if loaded.get('odor_valve_state_0') is not None and not loaded['odor_valve_state_0'].empty:
            valves0_list.append(loaded['odor_valve_state_0'])
        if loaded.get('odor_valve_state_1') is not None and not loaded['odor_valve_state_1'].empty:
            valves1_list.append(loaded['odor_valve_state_1'])
        if loaded.get('digital_input') is not None and not loaded['digital_input'].empty:
            digital_list.append(loaded['digital_input'])
        if loaded.get('pulse_supply_1') is not None and not loaded['pulse_supply_1'].empty:
            pulse1_list.append(loaded['pulse_supply_1'])
        if loaded.get('pulse_supply_2') is not None and not loaded['pulse_supply_2'].empty:
            pulse2_list.append(loaded['pulse_supply_2'])
        if loaded.get('output_set') is not None and not loaded['output_set'].empty:
            output_set_list.append(loaded['output_set'])
        if loaded.get('output_clear') is not None and not loaded['output_clear'].empty:
            output_clear_list.append(loaded['output_clear'])

        # EndInitiation events via the synced event loader (already uses the same offset logic internally)
        try:
            events = load_experiment_events(exp_dir, verbose=False)
            endinit_df = events.get('combined_end_initiation_df', pd.DataFrame())
            if isinstance(endinit_df, pd.DataFrame) and not endinit_df.empty:
                # Ensure index on Time and tz-aware
                if endinit_df.index.name != "Time" and "Time" in endinit_df.columns:
                    endinit_df = endinit_df.set_index("Time")
                if endinit_df.index.tz is None:
                    endinit_df.index = endinit_df.index.tz_localize(uk_tz)
                endinit_df = endinit_df.sort_index()
                # Slice to window if provided
                if time_window is not None:
                    endinit_df = _slice(endinit_df, start_dt, end_dt)
                # Keep only the EndInitiation flag
                if 'EndInitiation' not in endinit_df.columns and not endinit_df.empty:
                    endinit_df['EndInitiation'] = True
                endinit_list.append(endinit_df[['EndInitiation']])
        except Exception as e:
            if verbose:
                print(f"[WARN] Failed to load EndInitiation for {exp_dir.name}: {e}")

    # --- Concatenate all streams ---
    valves_0 = _safe_concat(valves0_list)
    valves_1 = _safe_concat(valves1_list)
    digital_input_data = _safe_concat(digital_list)
    pulse_supply_1 = _safe_concat(pulse1_list)
    pulse_supply_2 = _safe_concat(pulse2_list)
    output_set_all = _safe_concat(output_set_list)
    output_clear_all = _safe_concat(output_clear_list)
    endinit_df = _safe_concat(endinit_list)

    # Localize remaining naive indices (defensive)
    for df in [valves_0, valves_1, digital_input_data, pulse_supply_1, pulse_supply_2, endinit_df]:
        if isinstance(df, (pd.DataFrame, pd.Series)) and not df.empty:
            try:
                if df.index.tz is None:
                    df.index = df.index.tz_localize(uk_tz)
            except Exception:
                pass

    # Create odour_led (same logic as load_all_streams)
    if isinstance(output_set_all, pd.DataFrame) and not output_set_all.empty and \
       isinstance(output_clear_all, pd.DataFrame) and not output_clear_all.empty:
        try:
            if 'DOPort0' in output_clear_all and 'DOPort0' in output_set_all:
                odour_led = concat_digi_events(output_clear_all['DOPort0'].astype(bool),
                                               output_set_all['DOPort0'].astype(bool))
            else:
                odour_led = pd.Series(dtype=bool)
                if verbose:
                    print("[WARN] DOPort0 not found in outputs; odour_led unavailable.")
        except Exception as e:
            odour_led = pd.Series(dtype=bool)
            if verbose:
                print(f"[WARN] Could not create odour_led: {e}")
    else:
        odour_led = pd.Series(dtype=bool)
        if verbose:
            print("Could not create odour_led (missing output data).")

    # --- Detect session cuts (>5 min gap) using earliest per-experiment time (from heartbeat spans) ---
    session_cuts = []
    all_start_times = [t for t in exp_dir_first_times if t is not None]
    all_start_times = sorted(all_start_times)
    for i in range(1, len(all_start_times)):
        gap = (all_start_times[i] - all_start_times[i-1]).total_seconds()
        if gap > 300:
            session_cuts.append(all_start_times[i])
    if verbose and session_cuts:
        print(f"Session cuts detected at: {[t.strftime('%H:%M:%S') for t in session_cuts]}")

    # If a time_window was provided and we didn't slice earlier because of missing offset, do it now
    if time_window is not None:
        def restrict(df):
            if isinstance(df, (pd.DataFrame, pd.Series)) and not df.empty:
                try:
                    return df.loc[(df.index >= start_dt) & (df.index <= end_dt)]
                except Exception:
                    return df
            return df
        valves_0 = restrict(valves_0)
        valves_1 = restrict(valves_1)
        digital_input_data = restrict(digital_input_data)
        pulse_supply_1 = restrict(pulse_supply_1)
        pulse_supply_2 = restrict(pulse_supply_2)
        odour_led = restrict(odour_led) if isinstance(odour_led, pd.Series) else odour_led
        endinit_df = restrict(endinit_df)

    # --- Odor mapping (names for legend) ---
    odor_map = load_odor_mapping(exp_dirs[0], verbose=False)
    odour_to_olfactometer_map = odor_map.get('odour_to_olfactometer_map', [["A","B","C","D"],["E","F","G","Purge"]])

    # --- Plotting helpers ---
    def extend_to_window_end(df, end_dt_local):
        """Extend the last value of a DataFrame or Series to end_dt if needed."""
        if df is None or (hasattr(df, 'empty') and df.empty) or end_dt_local is None:
            return df
        try:
            if df.index[-1] >= end_dt_local:
                return df
        except Exception:
            return df
        if isinstance(df, pd.DataFrame):
            last_row = df.iloc[[-1]].copy()
            last_row.index = [end_dt_local]
            return pd.concat([df, last_row]).sort_index()
        elif isinstance(df, pd.Series):
            last_val = df.iloc[-1]
            extension = pd.Series([last_val], index=[end_dt_local])
            return pd.concat([df, extension]).sort_index()
        return df

    if interactive:
        try:
            get_ipython().run_line_magic('matplotlib', 'ipympl')
        except Exception:
            plt.ion()
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    # Colors for valves
    valve_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    # --- Extend signals to end_dt for clean step lines in a clipped window ---
    if time_window is not None:
        for name in [
            'odour_led',
            'digital_input_data',
            'valves_0',
            'valves_1',
            'pulse_supply_1',
            'pulse_supply_2',
            'endinit_df',
        ]:
            obj = locals()[name]
            if obj is not None and hasattr(obj, "empty") and not obj.empty:
                obj_extended = extend_to_window_end(obj, end_dt)
                locals()[name] = obj_extended

    # Plot odour LED and pokes
    if isinstance(odour_led, pd.Series) and not odour_led.empty:
        plt.step(odour_led.index, odour_led.astype(float) * 0.8, where='post', c='black', linewidth=2, label='Odour LED', alpha=0.7)
    if isinstance(digital_input_data, pd.DataFrame) and not digital_input_data.empty and 'DIPort0' in digital_input_data:
        plt.step(digital_input_data.index, digital_input_data['DIPort0'].astype(float) * 0.6, where='post', c='darkgray', linewidth=1, label='Odour Pokes')

    # Plot individual valves from olfactometer 0 and EndInitiation events and reward delivery
    valve_offset = -0.2
    if isinstance(valves_0, pd.DataFrame) and not valves_0.empty:
        for i, valve_col in enumerate(valves_0.columns):
            valve_data = valves_0[valve_col]
            color = valve_colors[i % len(valve_colors)]
            plt.step(valve_data.index, valve_data.astype(float) * 0.8 + valve_offset, where='post',
                    c=color, linewidth=1.5, label=f'{odour_to_olfactometer_map[0][i] if i < len(odour_to_olfactometer_map[0]) else valve_col} (Olfac1)', alpha=0.8)
            if isinstance(endinit_df, pd.DataFrame) and not endinit_df.empty and 'EndInitiation' in endinit_df:
                plt.scatter(endinit_df.index, endinit_df['EndInitiation'].astype(float) * 0.5 + valve_offset, s=5, c='k')
            if isinstance(pulse_supply_1, pd.DataFrame) and not pulse_supply_1.empty:
                plt.scatter(pulse_supply_1.index, np.ones(len(pulse_supply_1)) * (0.5 + valve_offset), s=5, c='r')
            if isinstance(pulse_supply_2, pd.DataFrame) and not pulse_supply_2.empty:
                plt.scatter(pulse_supply_2.index, np.ones(len(pulse_supply_2)) * (0.5 + valve_offset), s=5, c='r')
            valve_offset -= 0.3

    # Plot individual valves from olfactometer 1 and EndInitiation events and reward delivery
    if isinstance(valves_1, pd.DataFrame) and not valves_1.empty:
        base = len(valves_0.columns) if isinstance(valves_0, pd.DataFrame) and not valves_0.empty else 0
        for i, valve_col in enumerate(valves_1.columns):
            valve_data = valves_1[valve_col]
            color = valve_colors[(i + base) % len(valve_colors)]
            plt.step(valve_data.index, valve_data.astype(float) * 0.8 + valve_offset, where='post',
                    c=color, linewidth=1.5, label=f'{odour_to_olfactometer_map[1][i] if i < len(odour_to_olfactometer_map[1]) else valve_col} (Olfac2)', alpha=0.8, linestyle='--')
            if isinstance(endinit_df, pd.DataFrame) and not endinit_df.empty and 'EndInitiation' in endinit_df:
                plt.scatter(endinit_df.index, endinit_df['EndInitiation'].astype(float) * 0.5 + valve_offset, s=5, c='k')
            if isinstance(pulse_supply_1, pd.DataFrame) and not pulse_supply_1.empty:
                plt.scatter(pulse_supply_1.index, np.ones(len(pulse_supply_1)) * (0.5 + valve_offset), s=5, c='r')
            if isinstance(pulse_supply_2, pd.DataFrame) and not pulse_supply_2.empty:
                plt.scatter(pulse_supply_2.index, np.ones(len(pulse_supply_2)) * (0.5 + valve_offset), s=5, c='r')
            valve_offset -= 0.3

    # Plot reward pokes and reward delivery
    if isinstance(digital_input_data, pd.DataFrame) and not digital_input_data.empty:
        for di_col in digital_input_data.columns:
            if di_col == 'DIPort1':
                DIPort_data = digital_input_data[di_col]
                plt.step(DIPort_data.index, DIPort_data.astype(float) * 0.8 + valve_offset, where='post', c='orange', linewidth=1.5, label='Reward Pokes A')
                if isinstance(pulse_supply_1, pd.DataFrame) and not pulse_supply_1.empty:
                    plt.scatter(pulse_supply_1.index, np.ones(len(pulse_supply_1)) * (0.5 + valve_offset), s=5, c='r')
                if isinstance(pulse_supply_2, pd.DataFrame) and not pulse_supply_2.empty:
                    plt.scatter(pulse_supply_2.index, np.ones(len(pulse_supply_2)) * (0.5 + valve_offset), s=5, c='r')
                if isinstance(endinit_df, pd.DataFrame) and not endinit_df.empty and 'EndInitiation' in endinit_df:
                    plt.scatter(endinit_df.index, endinit_df['EndInitiation'].astype(float) * 0.5 + valve_offset, s=5, c='k')
                valve_offset -= 0.3
            elif di_col == 'DIPort2':
                DIPort_data = digital_input_data[di_col]
                plt.step(DIPort_data.index, DIPort_data.astype(float) * 0.8 + valve_offset, where='post', c='cyan', linewidth=1.5, label='Reward Pokes B')
                if isinstance(pulse_supply_1, pd.DataFrame) and not pulse_supply_1.empty:
                    plt.scatter(pulse_supply_1.index, np.ones(len(pulse_supply_1)) * (0.5 + valve_offset), s=5, c='r')
                if isinstance(pulse_supply_2, pd.DataFrame) and not pulse_supply_2.empty:
                    plt.scatter(pulse_supply_2.index, np.ones(len(pulse_supply_2)) * (0.5 + valve_offset), s=5, c='r', label='Reward Delivery')
                if isinstance(endinit_df, pd.DataFrame) and not endinit_df.empty and 'EndInitiation' in endinit_df:
                    plt.scatter(endinit_df.index, endinit_df['EndInitiation'].astype(float) * 0.5 + valve_offset, s=5, c='k', label='Trial End')
                valve_offset -= 0.3

    # --- Indicate session cuts ---
    for cut_time in session_cuts:
        plt.axvline(cut_time, color='gray', linestyle=':', linewidth=2, alpha=0.7)
        plt.text(cut_time, plt.ylim()[1], 'Session Cut', color='gray', rotation=90, va='top', ha='right', fontsize=8)

    # --- Adaptive/fine x-axis scale ---
    class AdaptiveTimeFormatter(mdates.DateFormatter):
        def __call__(self, x, pos=0):
            dt = mdates.num2date(x)
            return dt.strftime('%H:%M:%S.%f')[:-4]  # Show HH:MM:SS.ms

    def update_ticks(event):
        ax = event.canvas.figure.axes[0]
        xlim = ax.get_xlim()
        dt_start = mdates.num2date(xlim[0])
        dt_end = mdates.num2date(xlim[1])
        span = (dt_end - dt_start).total_seconds()
        # Adaptive major/minor ticks
        if span < 2:
            ax.xaxis.set_major_locator(mdates.SecondLocator(interval=1))
            ax.xaxis.set_minor_locator(mdates.MicrosecondLocator(interval=100000))  # 100ms
        elif span < 10:
            ax.xaxis.set_major_locator(mdates.SecondLocator(interval=2))
            ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=1))
        elif span < 60:
            ax.xaxis.set_major_locator(mdates.SecondLocator(interval=10))
            ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=1))
        elif span < 600:
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
            ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=10))
        elif span < 3600:
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
        elif span < 6*3600:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=10))
        else:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(AdaptiveTimeFormatter('%H:%M:%S.%f'))
        event.canvas.draw_idle()

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=uk_tz))

    if interactive:
        plt.gcf().canvas.mpl_connect('draw_event', update_ticks)
        # Trigger once
        try:
            update_ticks(type('event', (object,), {'canvas': plt.gcf().canvas})())
        except Exception:
            pass

    plt.xlabel('Time (Europe/London)')
    if time_window is not None:
        plt.xlim(start_dt, end_dt)
    plt.title('Individual Olfactometer Valve States vs Odour Pokes and LED')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim(valve_offset - 0.3, 1.2)
    plt.yticks([])  # Removes y-axis tick marks and labels
    plt.tight_layout()

    if show:
        plt.show()

    return fig