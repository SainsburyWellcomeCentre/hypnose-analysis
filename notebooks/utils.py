import sys
import os
project_root = os.path.abspath("/Users/joschua/repos/harris_lab/hypnose/hypnose-analysis")
if project_root not in sys.path:
    sys.path.append(project_root)
import os
import json
from dotmap import DotMap
import pandas as pd
import numpy as np
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
from datetime import datetime, timezone
from collections import defaultdict
from bisect import bisect_left, bisect_right


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
    

def load_json(reader: SessionData, root: Path) -> pd.DataFrame:
    root = Path(root)
    pattern = f"{root.joinpath(root.name)}_*.{reader.extension}"
    print(pattern)
    data = [reader.read(Path(file)) for file in sorted(glob(pattern))]
    return pd.concat(data)


def load(reader: Reader, root: Path) -> pd.DataFrame:
    root = Path(root)
    pattern = f"{root.joinpath(root.name)}_{reader.register.address}_*.bin"
    data = [reader.read(file) for file in sorted(glob(pattern))]
    return pd.concat(data)


def load_video(reader: Video, root: Path) -> pd.DataFrame:
    root = Path(root)
    pattern = f"{root.joinpath(root.name)}_*.csv"
    data = [reader.read(Path(file)) for file in sorted(glob(pattern))]
    return pd.concat(data)


def concat_digi_events(series_low: pd.DataFrame, series_high: pd.DataFrame) -> pd.DataFrame:
    """Concatenate seperate high and low dataframes to produce on/off vector"""
    data_off = ~series_low[series_low==True]
    data_on = series_high[series_high==True]
    return pd.concat([data_off, data_on]).sort_index()


def load_csv(reader: Csv, root: Path) -> pd.DataFrame:
    root = Path(root)
    pattern = f"{root.joinpath(reader.pattern).joinpath(reader.pattern)}_*.{reader.extension}"
    print(pattern)
    print([file for file in glob(pattern)])
    data = pd.concat([reader.read(Path(file)) for file in glob(pattern)])
    return data


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
    
    base_path = Path('/Volumes/harris/hypnose/rawdata')
    
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


def load_all_streams(root, apply_corrections = True):
    """
    Load all behavioral data streams with proper timestamp synchronization
    """
    print(f"Loading data streams from: {root}")
    
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
        print("Loaded heartbeat data")
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
                print(f"Calculated real-time offset: {real_time_offset}")
        except Exception as e:
            print(f"Error calculating real-time offset: {e}")
    
    # Create timestamp interpolation mapping
    timestamp_to_time = pd.Series()
    if not heartbeat.empty and 'Time' in heartbeat.columns and 'TimestampSeconds' in heartbeat.columns:
        heartbeat['Time'] = pd.to_datetime(heartbeat['Time'], errors='coerce')
        timestamp_to_time = pd.Series(data=heartbeat['Time'].values, index=heartbeat['TimestampSeconds'])
        print("Created timestamp interpolation mapping")
    
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
        print("Loaded digital_input_data")
    except Exception as e:
        print(f"Failed to load digital_input_data: {e}")
        data['digital_input_data'] = pd.DataFrame()
    
    try:
        data['output_set'] = load(behavior_reader.OutputSet, root/"Behavior")
        print("Loaded output_set")
    except Exception as e:
        print(f"Failed to load output_set: {e}")
        data['output_set'] = pd.DataFrame()
    
    try:
        data['output_clear'] = load(behavior_reader.OutputClear, root/"Behavior")
        print("Loaded output_clear")
    except Exception as e:
        print(f"Failed to load output_clear: {e}")
        data['output_clear'] = pd.DataFrame()
    
    # Olfactometer valve data
    try:
        data['olfactometer_valves_0'] = load(olfactometer_reader.OdorValveState, root/"Olfactometer0")
        print("Loaded olfactometer_valves_0")
    except Exception as e:
        print(f"Failed to load olfactometer_valves_0: {e}")
        data['olfactometer_valves_0'] = pd.DataFrame()
    
    try:
        data['olfactometer_valves_1'] = load(olfactometer_reader.OdorValveState, root/"Olfactometer1")
        print("Loaded olfactometer_valves_1")
    except Exception as e:
        print(f"Failed to load olfactometer_valves_1: {e}")
        data['olfactometer_valves_1'] = pd.DataFrame()
    
    # End valve states (commented in original but included for completeness)
    try:
        data['olfactometer_end_0'] = load(olfactometer_reader.EndValveState, root/"Olfactometer0")
        print("Loaded olfactometer_end_0")
    except Exception as e:
        print(f"Failed to load olfactometer_end_0: {e}")
        data['olfactometer_end_0'] = pd.DataFrame()
    
    # Analog data
    try:
        data['analog_data'] = load(behavior_reader.AnalogData, root/"Behavior")
        print("Loaded analog_data")
    except Exception as e:
        print(f"Failed to load analog_data: {e}")
        data['analog_data'] = pd.DataFrame()
    
    # Flow meter data
    try:
        data['flow_meter'] = load(olfactometer_reader.Flowmeter, root/"Olfactometer0")
        print("Loaded flow_meter")
    except Exception as e:
        print(f"Failed to load flow_meter: {e}")
        data['flow_meter'] = pd.DataFrame()
    
    # Video data
    try:
        video_reader = Video()
        data['video_reader'] = video_reader
        data['video_data'] = load_video(video_reader, root/"VideoData")
        print("Loaded video_data")
    except Exception as e:
        print(f"Failed to load video_data: {e}")
        data['video_reader'] = None
        data['video_data'] = pd.DataFrame()
    
    # Pulse supply (reward delivery)
    try:
        data['pulse_supply_1'] = load(behavior_reader.PulseSupplyPort1, root/"Behavior")
        print("Loaded pulse_supply_1")
    except Exception as e:
        print(f"Failed to load pulse_supply_1: {e}")
        data['pulse_supply_1'] = pd.DataFrame()
    
    try:
        data['pulse_supply_2'] = load(behavior_reader.PulseSupplyPort2, root/"Behavior")
        print("Loaded pulse_supply_2")
    except Exception as e:
        print(f"Failed to load pulse_supply_2: {e}")
        data['pulse_supply_2'] = pd.DataFrame()
    
    # Create combined odour LED signal
    try:
        if not data['output_clear'].empty and not data['output_set'].empty:
            data['odour_led'] = concat_digi_events(data['output_clear']['DOPort0'], data['output_set']['DOPort0'])
            print("Created odour_led")
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
        print("\nApplying time corrections to all data streams...")
        
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
                            print(f"Applied correction to {stream_name}")
                        else:
                            print(f"Skipped {stream_name} (not datetime index)")
                            
                    elif isinstance(data[stream_name], pd.Series):
                        # Check if index is datetime-like
                        if hasattr(data[stream_name].index, 'dtype') and pd.api.types.is_datetime64_any_dtype(data[stream_name].index):
                            data[stream_name].index = data[stream_name].index + real_time_offset
                            print(f"Applied correction to {stream_name}")
                        else:
                            print(f"Skipped {stream_name} (not datetime index)")
                except Exception as e:
                    print(f"Failed to apply correction to {stream_name}: {e}")
    


    print(f"\nData loading complete! Loaded {len([k for k, v in data.items() if not (isinstance(v, pd.DataFrame) and v.empty) and not (isinstance(v, pd.Series) and v.empty)])} streams successfully.")
    
    return data

def load_experiment_events(root):
    """
    Load and process experiment events with automatic time synchronization
    matching load_all_streams() timing corrections
    """
    
    print("Loading experiment events...")
    
    # === LOAD TIMING DATA ===
    try:
        behavior_reader = harp.create_reader('device_schemas/behavior.yml', epoch=harp.REFERENCE_EPOCH)
        heartbeat = load(behavior_reader.TimestampSeconds, root/"Behavior")
        if not heartbeat.empty:
            heartbeat.reset_index(inplace=True)
        print("Loaded heartbeat data for timing synchronization")
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
                print(f"Calculated real-time offset: {real_time_offset}")
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
        
        print("Created timestamp interpolation mapping")
    
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
    print(f"Found {len(csv_files)} experiment event files")
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            ev_df = pd.read_csv(csv_file)
            print(f"Processing event file: {csv_file.name} with {len(ev_df)} rows")
            
            # Handle timestamp conversion (same logic as original notebook)
            if "Seconds" in ev_df.columns and interpolate_time is not None:
                ev_df = ev_df.sort_values("Seconds")
                ev_df["Time"] = ev_df["Seconds"].apply(interpolate_time)
                print("Using Seconds column with interpolation")
            else:
                # Fallback: use seconds as relative time
                ev_df["Time"] = pd.to_datetime(ev_df["Seconds"], unit='s')
                print("Using Seconds column as raw timestamp")
            
            # Apply real-time offset (CRITICAL for synchronization)
            if real_time_offset != pd.Timedelta(0):
                ev_df["Time"] = ev_df["Time"] + real_time_offset
                print(f"Applied real-time offset: {real_time_offset}")
            
            # Extract events
            if "Value" in ev_df.columns:
                print(f"Found Value column with values: {ev_df['Value'].unique()}")
                
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
                        print(f"Found {len(event_df)} {event_value} events")
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
            print(f"Combined {len(combined_df)} {column_name} events")
        else:
            results[df_name] = pd.DataFrame(columns=["Time", column_name])
            print(f"No {column_name} events found")
    
    print(f"Experiment events loading complete! All events synchronized with load_all_streams timing.")
    return results


def load_odor_mapping(root, data=None):
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
    
    print("Loading odor mapping from session settings...")
    
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
        print("Loaded session settings")
        
        # Extract valve configurations for each olfactometer
        olfactometer_commands = session_settings.metadata.iloc[0].olfactometerCommands
        olf_valves0 = [cmd.valvesOpenO0 for cmd in olfactometer_commands]
        olf_valves1 = [cmd.valvesOpenO1 for cmd in olfactometer_commands]
        
        print(f"Found {len(olf_valves0)} valve configurations for olfactometer 0")
        print(f"Found {len(olf_valves1)} valve configurations for olfactometer 1")
        
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
        
        print(f"Created valve-to-command mapping: {olf_command_idx}")
        
        # Create odor name mapping
        odour_to_olfactometer_map = [[] for _ in range(len(olfactometer_valves))]
        
        for valve_key, cmd_idx in olf_command_idx.items():
            olf_id = int(valve_key[0])  # Extract olfactometer ID (0 or 1)
            odor_name = olfactometer_commands[cmd_idx].name
            odour_to_olfactometer_map[olf_id].append(odor_name)
        
        print(f"Created odor mapping: {odour_to_olfactometer_map}")
        
        # Create reverse mapping: valve -> odor name
        valve_to_odor = {}
        for valve_key, cmd_idx in olf_command_idx.items():
            odor_name = olfactometer_commands[cmd_idx].name
            valve_to_odor[valve_key] = odor_name
        
        # Create olfactometer -> odor list mapping
        olfactometer_to_odors = {}
        for olf_id in range(len(olfactometer_valves)):
            olfactometer_to_odors[olf_id] = odour_to_olfactometer_map[olf_id]
        
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
        print("TRIAL DETECTION - METHOD 3: Simplified")
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
    print("\n" + "="*50)
    print("DETECTION SUMMARY:")
    print(f"Trials: {len(results['trials'])}")
    print(f"Initiated sequences: {len(results['initiated_sequences'])}")
    print(f"Non-initiated sequences: {len(results['non_initiated_sequences'])}")
    total_attempts = len(results['initiated_sequences']) + len(results['non_initiated_sequences'])
    if total_attempts > 0:
        success_rate = len(results['initiated_sequences']) / total_attempts * 100
        print(f"Success rate: {success_rate:.1f}%")
    print("="*50)
    
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
                cur_end = max(cur_end, next_end)
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
                position_locations[1] = first_odor_activations[-1]
                prior_presentations = [
                    {
                        'position': 1,
                        'odor_name': e['odor_name'],
                        'valve_start': e['start_time'],
                        'valve_end': e['end_time'],
                    }
                    for e in first_odor_activations[:-1]
                ]
            else:
                prior_presentations = []
        else:
            prior_presentations = []

        # Group consecutive events for positions 2-5
        grouped = []
        current_valve = None
        current_start_time = None
        current_end_time = None
        current_odor_name = None

        for event in trial_valve_events:
            if event['valve_key'] != current_valve:
                if current_valve is not None:
                    grouped.append({
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
            grouped.append({
                'valve_key': current_valve,
                'odor_name': current_odor_name,
                'start_time': current_start_time,
                'end_time': current_end_time
            })

        for i, presentation in enumerate(grouped[1:], 2):
            if i <= 5:
                position_locations[i] = presentation

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

        # Poke-time analysis positions
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

            consolidated_poke_time_ms = sum((e - s).total_seconds() * 1000.0 for s, e in merged)
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

        position_valve_times, position_poke_times = analyze_trial_valve_and_poke_times(valve_activations)

        pos1_info = position_valve_times.get(1, {}) or {}
        last_pos1_start = pos1_info.get('valve_start')

        # Record earlier Position-1 presentations as non-initiated attempts with correct poke timing
        for attempt in pos1_info.get('prior_presentations', []) or []:
            a_start = attempt.get('valve_start')   # attempt valve start (for reference)
            # Cap at the last Pos1 valve START (trial starts at last odor 1 opening)
            first_in, bout_end, dur_ms = _attempt_bout_from_poke_in(anchor_ts=a_start, cap_end=last_pos1_start)
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
        if corrected_start is not None:
            trial_dict['sequence_start_corrected'] = corrected_start

        enough_odors, hit_hidden_rule = check_hidden_rule(odor_sequence, hidden_rule_location)
        trial_dict['enough_odors_for_hr'] = enough_odors
        trial_dict['hit_hidden_rule'] = hit_hidden_rule

        if trial_await_rewards:
            completed_sequences.append(trial_dict.copy())
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
        else:
            aborted_sequences.append(trial_dict.copy())
            if hit_hidden_rule:
                aborted_sequences_hr.append(trial_dict.copy())

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

    if verbose:
        print(f"\nTRIAL CLASSIFICATION RESULTS WITH HIDDEN RULE AND VALVE/POKE TIME ANALYSIS:")
        print(f"Hidden Rule Location: Position {hidden_rule_position} (index {hidden_rule_location})\n")


        base_non_init_df = result.get('non_initiated_sequences', pd.DataFrame())
        pos1_attempts_df = result.get('non_initiated_odor1_attempts', pd.DataFrame())

        base_non_init_count = 0 if base_non_init_df is None or base_non_init_df.empty else len(base_non_init_df)
        pos1_attempts_count = 0 if pos1_attempts_df is None or pos1_attempts_df.empty else len(pos1_attempts_df)

        total_non_init = base_non_init_count + pos1_attempts_count
        total_attempts = len(initiated_trials) + total_non_init
        print(f"Total attempts: {total_attempts}")
        print(f"-- Non-initiated sequences (total): {total_non_init} ({total_non_init/total_attempts*100:.1f}%)")
        if pos1_attempts_count:
            print(f"    -- Position 1 attempts within trials {pos1_attempts_count} ({pos1_attempts_count/total_non_init*100:.1f}%)")
            print(f"    -- Baseline non-initiated sequences {base_non_init_count} ({base_non_init_count/total_non_init*100:.1f}%)")
        print(f"-- Initiated sequences (trials): {len(initiated_trials)} ({len(initiated_trials)/total_attempts*100:.1f}%)\n")

        print("INITIATED TRIALS BREAKDOWN:")
        print(f"-- Completed sequences: {len(result['completed_sequences'])} ({len(result['completed_sequences'])/len(initiated_trials)*100:.1f}%)")
        print(f"   -- Hidden Rule trials (HR): {len(result['completed_sequences_HR'])} ({len(result['completed_sequences_HR'])/len(initiated_trials)*100:.1f}%)")
        print(f"   -- Hidden Rule Missed (HR_missed): {len(result['completed_sequences_HR_missed'])} ({len(result['completed_sequences_HR_missed'])/len(initiated_trials)*100:.1f}%)")
        print(f"-- Aborted sequences: {len(result['aborted_sequences'])} ({len(result['aborted_sequences'])/len(initiated_trials)*100:.1f}%)")
        print(f"   -- Aborted Hidden Rule trials (HR): {len(result['aborted_sequences_HR'])} ({len(result['aborted_sequences_HR'])/len(initiated_trials)*100:.1f}%)\n")

        print("REWARD STATUS BREAKDOWN:")
        cs = len(result['completed_sequences'])
        if cs > 0:
            print(f"-- Rewarded: {len(result['completed_sequence_rewarded'])} ({len(result['completed_sequence_rewarded'])/cs*100:.1f}%)")
            print(f"-- Unrewarded: {len(result['completed_sequence_unrewarded'])} ({len(result['completed_sequence_unrewarded'])/cs*100:.1f}%)")
            print(f"-- Reward timeout: {len(result['completed_sequence_reward_timeout'])} ({len(result['completed_sequence_reward_timeout'])/cs*100:.1f}%)\n")

        print("HIDDEN RULE SPECIFIC BREAKDOWN:")
        hr_total = len(result['completed_sequences_HR'])
        if hr_total > 0:
            print(f"-- HR Rewarded: {len(result['completed_sequence_HR_rewarded'])} ({len(result['completed_sequence_HR_rewarded'])/hr_total*100:.1f}%)")
            print(f"-- HR Unrewarded: {len(result['completed_sequence_HR_unrewarded'])} ({len(result['completed_sequence_HR_unrewarded'])/hr_total*100:.1f}%)")
            print(f"-- HR Timeout: {len(result['completed_sequence_HR_reward_timeout'])} ({len(result['completed_sequence_HR_reward_timeout'])/hr_total*100:.1f}%)")

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

    for trial_dict in completed_trials_all:
        trial_start = trial_dict['sequence_start']
        trial_end = trial_dict['sequence_end']
        await_reward_time = trial_dict['await_reward_time']

        # Get valve sequence
        odor_sequence, trial_valve_events = get_trial_valve_sequence(trial_start, trial_end)
        if not trial_valve_events:
            failed_calculations += 1
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
            continue

        # Find last poke out in extended window around target odor
        odor_start = target_valve_event['start_time']
        odor_end = target_valve_event['end_time']
        search_end = max(await_reward_time, odor_end + pd.Timedelta(seconds=1))

        extended_poke_data = poke_data.loc[odor_start:search_end]
        if extended_poke_data.empty:
            failed_calculations += 1
            continue

        last_poke_out_time = None
        prev_state = poke_data.loc[:odor_start].iloc[-1] if len(poke_data.loc[:odor_start]) > 0 else False
        for timestamp, current_state in extended_poke_data.items():
            if prev_state and not current_state:
                last_poke_out_time = timestamp
            prev_state = current_state

        if last_poke_out_time is None:
            failed_calculations += 1
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

        if supply1_after_await or supply2_after_await:
            if response_time_ms is not None:
                rewarded_response_times.append(response_time_ms)
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

            if port1_full_window or port2_full_window:
                if response_time_ms is not None:
                    unrewarded_response_times.append(response_time_ms)
            else:
                # Timeout: look for delayed responses until next completed trial start
                next_trial_start = find_next_trial_start(trial_end, completed_trials_all)
                extended_search_end = next_trial_start if next_trial_start else (poke_data.index[-1] if not poke_data.empty else poke_window_end)

                delayed_search_start = poke_window_end

                delayed_port1_pokes = []
                delayed_port2_pokes = []
                if not port1_pokes.empty and delayed_search_start < extended_search_end:
                    delayed_port1_window = port1_pokes[delayed_search_start:extended_search_end]
                    delayed_port1_starts = delayed_port1_window & ~delayed_port1_window.shift(1, fill_value=False)
                    delayed_port1_pokes = delayed_port1_starts[delayed_port1_starts == True].index.tolist()
                if not port2_pokes.empty and delayed_search_start < extended_search_end:
                    delayed_port2_window = port2_pokes[delayed_search_start:extended_search_end]
                    delayed_port2_starts = delayed_port2_window & ~delayed_port2_window.shift(1, fill_value=False)
                    delayed_port2_pokes = delayed_port2_starts[delayed_port2_starts == True].index.tolist()

                delayed_reward_pokes = delayed_port1_pokes + delayed_port2_pokes
                if delayed_reward_pokes:
                    first_delayed_poke = min(delayed_reward_pokes)
                    delayed_response_time_ms = (first_delayed_poke - last_poke_out_time).total_seconds() * 1000
                    response_delay_time_ms = (first_delayed_poke - poke_window_end).total_seconds() * 1000

                    timeout_delayed_response_times.append(delayed_response_time_ms)
                    timeout_response_delay_times.append(response_delay_time_ms)

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
    return {
        'rewarded_response_times': rewarded_response_times,
        'unrewarded_response_times': unrewarded_response_times,
        'timeout_delayed_response_times': timeout_delayed_response_times,
        'timeout_response_delay_times': timeout_response_delay_times,
        'all_response_times': all_response_times,
        'failed_calculations': failed_calculations
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
        # consolidated poke time within [window_start, window_end] (merge gaps <= sample_offset_time_ms)
        if window_start is None or window_end is None or window_start >= window_end:
            return {'poke_time_ms': 0.0, 'poke_first_in': None, 'poke_odor_start': window_start}
        # State at window start
        prev = DIP0.loc[:window_start]
        in_at_start = bool(prev.iloc[-1]) if len(prev) else False
        w = DIP0.loc[window_start:window_end]
        if w.empty and not in_at_start:
            return {'poke_time_ms': 0.0, 'poke_first_in': None, 'poke_odor_start': window_start}
        rises = w & ~w.shift(1, fill_value=in_at_start)
        falls = ~w & w.shift(1, fill_value=in_at_start)
        intervals = []
        cur = window_start if in_at_start else None
        first_in = None
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
        # merge across short gaps
        merged = [intervals[0]]
        for s2, e2 in intervals[1:]:
            ls, le = merged[-1]
            gap_ms = (s2 - le).total_seconds() * 1000.0
            if gap_ms <= sample_offset_time_ms:
                merged[-1] = (ls, max(le, e2))
            else:
                merged.append((s2, e2))
        total_ms = sum((e - s).total_seconds() * 1000.0 for s, e in merged)
        return {'poke_time_ms': float(total_ms), 'poke_first_in': first_in, 'poke_odor_start': window_start}

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

            # Poke time for last odor: from poke-in that covers/starts after valve_start, merge small gaps, end at first large gap
            _first_in, _end, dur_ms = bout_from_anchor(last_ev['start_time'])
            last_odor_poke_ms = dur_ms

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
        })

    aborted_detailed = pd.DataFrame(rows)

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
        nfa_count = int((aborted_detailed['fa_label'] == 'non-FA').sum())
        fa_in_count = int((aborted_detailed['fa_label'] == 'FA_time_in').sum())
        fa_out_count = int((aborted_detailed['fa_label'] == 'FA_time_out').sum())
        fa_late_count = int((aborted_detailed['fa_label'] == 'FA_late').sum())
        fa_total = fa_in_count + fa_out_count + fa_late_count

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
    # Convert to milliseconds for consistency with existing logic
    sample_offset_time_ms = sample_offset_time * 1000
    minimum_sampling_time_ms = minimum_sampling_time * 1000
    response_time_sec = response_time
    if response_time_sec is None:
        raise ValueError("Response time parameter cannot be extracted from Schema file. Check detect_settings function.")

    # 1) Run the stable classifier (valve/poke timing included)
    classification = classify_trials(
        data, events, trial_counts, odor_map, stage, root, verbose=verbose
    )

    # 2) Run the response-time summary analyzer (prints/aggregates like the notebook)
    rt_summary = analyze_response_times(
        data, trial_counts, events, odor_map, stage, root, verbose=verbose
    )

    aborted_detailed = abortion_classification(
        data, events, classification, odor_map, root, verbose=True
    )

    classification['aborted_sequences_detailed'] = aborted_detailed 

    # 4) Build fast lookup indices for downstream use
    classification['index'] = build_classification_index(classification)

    # 3) Augment completed trials with per-trial response times (same logic as analyzer,
    #    but reusing position_valve_times from the classifier to avoid recomputation)
    completed_df = classification['completed_sequences'].copy()
    if completed_df.empty:
        classification['completed_sequences_with_response_times'] = completed_df
        return {
            'classification': classification,
            'response_time_analysis': rt_summary,
            'completed_sequences_with_response_times': completed_df
        }

    # Parse hidden rule location (index)
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



    # Precompute data used below
    await_reward_times = events['combined_await_reward_df']['Time'].tolist() if 'combined_await_reward_df' in events else []
    supply_port1_times = data['pulse_supply_1'].index.tolist() if not data['pulse_supply_1'].empty else []
    supply_port2_times = data['pulse_supply_2'].index.tolist() if not data['pulse_supply_2'].empty else []
    poke_data = data['digital_input_data'].get('DIPort0', pd.Series(dtype=bool))
    port1_pokes = data['digital_input_data'].get('DIPort1', pd.Series(dtype=bool))
    port2_pokes = data['digital_input_data'].get('DIPort2', pd.Series(dtype=bool))

    # Build helper: next completed trial start lookup
    completed_trials_list = completed_df[['trial_id', 'sequence_start', 'sequence_end']].to_dict('records')
    def find_next_completed_start(current_end):
        next_starts = [t['sequence_start'] for t in completed_trials_list if t['sequence_start'] > current_end]
        return min(next_starts) if next_starts else None

    # Compute per-trial RTs
    rt_values = []
    rt_categories = []

    for _, row in completed_df.iterrows():
        # Recompute await_reward_time for this trial (not guaranteed present in completed_sequences)
        trial_start = row['sequence_start']
        trial_end = row['sequence_end']
        trial_awaits = [t for t in await_reward_times if trial_start <= t <= trial_end]
        if not trial_awaits:
            rt_values.append(None)
            rt_categories.append(None)
            continue
        await_reward_time = min(trial_awaits)
        poke_window_end = await_reward_time + pd.Timedelta(seconds=response_time_sec)

        # Determine target odor position (0-based) matching the analyzer’s rule
        num_odors = int(row.get('num_odors', 0) or 0)
        hit_hr = bool(row.get('hit_hidden_rule', False))
        if hidden_rule_location is not None and hit_hr and num_odors == hidden_rule_pos:
            target_pos_idx = hidden_rule_location  # zero-based
        else:
            target_pos_idx = max(0, num_odors - 1)

        # Use per-position valve windows from classifier output
        pos_valves = row.get('position_valve_times', {}) or {}
        pos_key = target_pos_idx + 1  # stored keys are 1..5
        pos_info = pos_valves.get(pos_key)
        if not pos_info:
            rt_values.append(None)
            rt_categories.append(None)
            continue

        odor_start = pos_info['valve_start']
        odor_end = pos_info['valve_end']
        search_end = max(await_reward_time, odor_end + pd.Timedelta(seconds=1))

        # Find last poke out during [odor_start, search_end]
        last_poke_out_time = None
        if not poke_data.empty:
            segment = poke_data.loc[odor_start:search_end]
            prev_state = poke_data.loc[:odor_start].iloc[-1] if len(poke_data.loc[:odor_start]) > 0 else False
            for ts, st in segment.items():
                if prev_state and not st:
                    last_poke_out_time = ts
                prev_state = st

        if last_poke_out_time is None:
            rt_values.append(None)
            rt_categories.append(None)
            continue

        # Reward window pokes (reward ports)
        search_start = max(last_poke_out_time, await_reward_time)
        port1_hits = []
        port2_hits = []
        if not port1_pokes.empty and search_start < poke_window_end:
            w = port1_pokes[search_start:poke_window_end]
            starts = w & ~w.shift(1, fill_value=False)
            port1_hits = starts[starts == True].index.tolist()
        if not port2_pokes.empty and search_start < poke_window_end:
            w = port2_pokes[search_start:poke_window_end]
            starts = w & ~w.shift(1, fill_value=False)
            port2_hits = starts[starts == True].index.tolist()

        all_reward_pokes = sorted(port1_hits + port2_hits)
        response_time_ms = None
        if all_reward_pokes:
            first_reward_poke = all_reward_pokes[0]
            response_time_ms = (first_reward_poke - last_poke_out_time).total_seconds() * 1000

        # Determine reward status (like classifier)
        supply1_after_await = [t for t in supply_port1_times if await_reward_time <= t <= trial_end]
        supply2_after_await = [t for t in supply_port2_times if await_reward_time <= t <= trial_end]

        if supply1_after_await or supply2_after_await:
            # Rewarded trial
            rt_values.append(response_time_ms)
            rt_categories.append('rewarded' if response_time_ms is not None else None)
        else:
            # Unrewarded vs timeout
            # Any poke in the full response window from await_reward?
            full_p1 = []
            full_p2 = []
            if not port1_pokes.empty:
                w = port1_pokes[await_reward_time:poke_window_end]
                s = w & ~w.shift(1, fill_value=False)
                full_p1 = s[s == True].index.tolist()
            if not port2_pokes.empty:
                w = port2_pokes[await_reward_time:poke_window_end]
                s = w & ~w.shift(1, fill_value=False)
                full_p2 = s[s == True].index.tolist()

            if full_p1 or full_p2:
                # Unrewarded
                rt_values.append(response_time_ms)
                rt_categories.append('unrewarded' if response_time_ms is not None else None)
            else:
                # Timeout: look for delayed response until next completed trial start
                next_start = find_next_completed_start(trial_end)
                extended_end = next_start if next_start else (poke_data.index[-1] if not poke_data.empty else poke_window_end)

                delayed_hits = []
                if not port1_pokes.empty and poke_window_end < extended_end:
                    w = port1_pokes[poke_window_end:extended_end]
                    s = w & ~w.shift(1, fill_value=False)
                    delayed_hits.extend(s[s == True].index.tolist())
                if not port2_pokes.empty and poke_window_end < extended_end:
                    w = port2_pokes[poke_window_end:extended_end]
                    s = w & ~w.shift(1, fill_value=False)
                    delayed_hits.extend(s[s == True].index.tolist())

                if delayed_hits:
                    first_delayed = min(delayed_hits)
                    response_time_ms = (first_delayed - last_poke_out_time).total_seconds() * 1000
                    rt_values.append(response_time_ms)
                    rt_categories.append('timeout_delayed')
                else:
                    rt_values.append(None)
                    rt_categories.append('timeout_no_response')

    completed_df['response_time_ms'] = rt_values
    completed_df['response_time_category'] = rt_categories

    # Attach augmented DataFrame back into classification dict for convenience
    classification['completed_sequences_with_response_times'] = completed_df

    return {
        'classification': classification,
        'response_time_analysis': rt_summary,
        'completed_sequences_with_response_times': completed_df
    }


# ========================== Functions for saving results ==========================
