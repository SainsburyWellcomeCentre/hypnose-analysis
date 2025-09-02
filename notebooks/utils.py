import sys
import os
project_root = os.path.abspath("/Users/joschua/repos/harris_lab/hypnose/hypnose-analysis")
if project_root not in sys.path:
    sys.path.append(project_root)
import os
import json
from dotmap import DotMap
import pandas as pd
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



def classify_trial_outcomes(data, events, trial_counts): # will be combined with the analyse_trial_valve_sequences into one comprehensive classification function
    """
    Classify trials into hierarchical categories based on completion and reward status:
    
    1. completed_sequence: trials with AwaitReward event
       - completed_sequence_rewarded: supply port activity detected
       - completed_sequence_unrewarded: poke in Port1/Port2 within 2.5s, no supply port
       - completed_sequence_reward_timeout: no poke in Port1/Port2 within 2.5s
    2. aborted_sequence: trials without AwaitReward event
    
    Returns:
        dict: Contains DataFrames for each trial category
    """
    print("=" * 60)
    print("CLASSIFYING TRIAL OUTCOMES")
    print("=" * 60)
    
    # Get base trial data
    initiated_trials = trial_counts['initiated_sequences'].copy()
    
    # Get event times
    await_reward_times = events['combined_await_reward_df']['Time'].tolist() if 'combined_await_reward_df' in events else []
    
    # Get supply port activities from pulse supply data (same as analyze_reward_events)
    supply_port1_times = []
    supply_port2_times = []
    
    if not data['pulse_supply_1'].empty:
        supply_port1_times = data['pulse_supply_1'].index.tolist()
    
    if not data['pulse_supply_2'].empty:
        supply_port2_times = data['pulse_supply_2'].index.tolist()
    
    all_supply_port_times = sorted(supply_port1_times + supply_port2_times)
    
    # Get reward port poke data
    port1_pokes = data['digital_input_data']['DIPort1'] if 'DIPort1' in data['digital_input_data'] else pd.Series(dtype=bool)
    port2_pokes = data['digital_input_data']['DIPort2'] if 'DIPort2' in data['digital_input_data'] else pd.Series(dtype=bool)
    
    # Initialize result containers
    completed_sequences = []
    aborted_sequences = []
    completed_rewarded = []
    completed_unrewarded = []
    completed_timeout = []
    
    print(f"Analyzing {len(initiated_trials)} initiated trials...")
    print(f"   Found {len(await_reward_times)} AwaitReward events")
    print(f"   Found {len(supply_port1_times)} supply port 1 activities")
    print(f"   Found {len(supply_port2_times)} supply port 2 activities")
    print(f"   Found {len(all_supply_port_times)} total supply port activities")
    
    for _, trial in initiated_trials.iterrows():
        trial_start = trial['sequence_start']
        trial_end = trial['sequence_end']
        trial_id = trial['trial_id']
        
        # Check if AwaitReward occurs within this trial
        trial_await_rewards = [
            t for t in await_reward_times 
            if trial_start <= t <= trial_end
        ]
        
        if trial_await_rewards:
            # This is a completed sequence
            completed_sequences.append(trial.to_dict())
            
            # Get the first AwaitReward time in this trial
            await_reward_time = min(trial_await_rewards)
            
            # Check for supply port activity after AwaitReward
            supply1_after_await = [
                t for t in supply_port1_times 
                if await_reward_time <= t <= trial_end
            ]
            supply2_after_await = [
                t for t in supply_port2_times 
                if await_reward_time <= t <= trial_end
            ]
            
            if supply1_after_await or supply2_after_await:
                # Rewarded trial
                trial_dict = trial.to_dict()
                trial_dict['await_reward_time'] = await_reward_time
                
                # Determine which port was rewarded first and set odor identity
                all_supply_times = []
                if supply1_after_await:
                    all_supply_times.extend([(t, 1, 'A') for t in supply1_after_await])
                if supply2_after_await:
                    all_supply_times.extend([(t, 2, 'B') for t in supply2_after_await])
                
                all_supply_times.sort(key=lambda x: x[0])  # Sort by time
                
                first_supply_time, first_supply_port, first_supply_odor = all_supply_times[0]
                
                trial_dict['first_supply_time'] = first_supply_time
                trial_dict['first_supply_port'] = first_supply_port
                trial_dict['first_supply_odor_identity'] = first_supply_odor
                trial_dict['supply1_count'] = len(supply1_after_await)
                trial_dict['supply2_count'] = len(supply2_after_await)
                trial_dict['total_supply_count'] = len(supply1_after_await) + len(supply2_after_await)
                
                completed_rewarded.append(trial_dict)
            else:
                # No supply port activity - check for reward port pokes within 2.5s
                poke_window_end = await_reward_time + pd.Timedelta(seconds=2.5)
                #poke_window_end = min(poke_window_end, trial_end)  # Don't exceed trial end; possibly not needed as 2.5s should be fixed.
                
                # Find poke events in Port1 and Port2 within the window
                port1_pokes_in_window = []
                port2_pokes_in_window = []
                
                # Check Port1 pokes
                if not port1_pokes.empty:
                    port1_window = port1_pokes[await_reward_time:poke_window_end]
                    # Find poke starts (transitions from False to True)
                    port1_starts = port1_window & ~port1_window.shift(1, fill_value=False)
                    port1_pokes_in_window = port1_starts[port1_starts == True].index.tolist()
                
                # Check Port2 pokes
                if not port2_pokes.empty:
                    port2_window = port2_pokes[await_reward_time:poke_window_end]
                    # Find poke starts (transitions from False to True)
                    port2_starts = port2_window & ~port2_window.shift(1, fill_value=False)
                    port2_pokes_in_window = port2_starts[port2_starts == True].index.tolist()
                
                # Create combined list with port identity and odor mapping
                all_reward_pokes = []
                if port1_pokes_in_window:
                    all_reward_pokes.extend([(t, 1, 'A') for t in port1_pokes_in_window])
                if port2_pokes_in_window:
                    all_reward_pokes.extend([(t, 2, 'B') for t in port2_pokes_in_window])
                
                all_reward_pokes.sort(key=lambda x: x[0])  # Sort by time
                
                trial_dict = trial.to_dict()
                trial_dict['await_reward_time'] = await_reward_time
                trial_dict['poke_window_end'] = poke_window_end
                trial_dict['port1_pokes_count'] = len(port1_pokes_in_window)
                trial_dict['port2_pokes_count'] = len(port2_pokes_in_window)
                trial_dict['total_reward_pokes'] = len(all_reward_pokes)
                
                if all_reward_pokes:
                    # Unrewarded trial (poked but no reward)
                    first_poke_time, first_poke_port, first_poke_odor = all_reward_pokes[0]
                    trial_dict['first_reward_poke_time'] = first_poke_time
                    trial_dict['first_reward_poke_port'] = first_poke_port
                    trial_dict['first_reward_poke_odor_identity'] = first_poke_odor
                    completed_unrewarded.append(trial_dict)
                else:
                    # Timeout trial (no poke within 2.5s)
                    completed_timeout.append(trial_dict)
        else:
            # This is an aborted sequence (no AwaitReward)
            aborted_sequences.append(trial.to_dict())
    
    # Create DataFrames
    result = {
        'completed_sequences': pd.DataFrame(completed_sequences),
        'aborted_sequences': pd.DataFrame(aborted_sequences),
        'completed_sequence_rewarded': pd.DataFrame(completed_rewarded),
        'completed_sequence_unrewarded': pd.DataFrame(completed_unrewarded),
        'completed_sequence_reward_timeout': pd.DataFrame(completed_timeout)
    }
    
    # Print summary statistics
    print(f"\nTRIAL CLASSIFICATION RESULTS:")
    print(f"   Total initiated trials: {len(initiated_trials)}")
    print(f"   -- Completed sequences: {len(result['completed_sequences'])} ({len(result['completed_sequences'])/len(initiated_trials)*100:.1f}%)")
    print(f"       -- Rewarded: {len(result['completed_sequence_rewarded'])} ({len(result['completed_sequence_rewarded'])/len(initiated_trials)*100:.1f}%)")
    print(f"       -- Unrewarded: {len(result['completed_sequence_unrewarded'])} ({len(result['completed_sequence_unrewarded'])/len(initiated_trials)*100:.1f}%)")
    print(f"       -- Reward timeout: {len(result['completed_sequence_reward_timeout'])} ({len(result['completed_sequence_reward_timeout'])/len(initiated_trials)*100:.1f}%)")
    print(f"   -- Aborted sequences: {len(result['aborted_sequences'])} ({len(result['aborted_sequences'])/len(initiated_trials)*100:.1f}%)")
    
    # Print odor identity breakdown for rewarded trials
    if len(result['completed_sequence_rewarded']) > 0:
        rewarded_df = result['completed_sequence_rewarded']
        odor_a_rewarded = len(rewarded_df[rewarded_df['first_supply_odor_identity'] == 'A'])
        odor_b_rewarded = len(rewarded_df[rewarded_df['first_supply_odor_identity'] == 'B'])
        print(f"       Rewarded breakdown: Odor A (Port 1): {odor_a_rewarded}, Odor B (Port 2): {odor_b_rewarded}")
    
    # Print odor identity breakdown for unrewarded trials
    if len(result['completed_sequence_unrewarded']) > 0:
        unrewarded_df = result['completed_sequence_unrewarded']
        odor_a_unrewarded = len(unrewarded_df[unrewarded_df['first_reward_poke_odor_identity'] == 'A'])
        odor_b_unrewarded = len(unrewarded_df[unrewarded_df['first_reward_poke_odor_identity'] == 'B'])
        print(f"       Unrewarded breakdown: Odor A (Port 1): {odor_a_unrewarded}, Odor B (Port 2): {odor_b_unrewarded}")
    
    # Verify totals
    total_classified = (len(result['completed_sequence_rewarded']) + 
                       len(result['completed_sequence_unrewarded']) + 
                       len(result['completed_sequence_reward_timeout']) + 
                       len(result['aborted_sequences']))
    
    if total_classified == len(initiated_trials):
        print(f"Classification complete: all {len(initiated_trials)} trials classified")
    else:
        print(f"Classification mismatch: {total_classified} classified vs {len(initiated_trials)} total")
    
    return result


def analyze_trial_valve_sequences(data, trial_outcomes, odor_map, verbose=True): # will be combined with the classify_trial_outcomes into one comprehensive classification function
    """
    Analyze valve opening sequences during completed trials
    
    Parameters:
    -----------
    data : dict
        Data dictionary containing olfactometer valve data
    trial_outcomes : dict
        Trial outcomes from classify_trial_outcomes function
    odor_map : dict
        Odor mapping information
    verbose : bool
        Whether to print detailed information
    
    Returns:
    --------
    dict: Analysis results containing odor counts and sequences
    """
    if verbose:
        print("=" * 60)
        print("ANALYZING VALVE SEQUENCES DURING COMPLETED TRIALS")
        print("=" * 60)
    
    # Get completed sequences
    completed_sequences = trial_outcomes['completed_sequences']
    
    if completed_sequences.empty:
        print("No completed sequences found")
        return {}
    
    # Get valve data and mapping
    olfactometer_valves = odor_map['olfactometer_valves']
    valve_to_odor = odor_map['valve_to_odor']
    
    # Build comprehensive valve activation list
    all_valve_activations = []
    for olf_id, valve_data in olfactometer_valves.items():
        if valve_data.empty:
            continue
        for i, valve_col in enumerate(valve_data.columns):
            valve_key = f"{olf_id}{i}"
            if valve_key in valve_to_odor:
                odor_name = valve_to_odor[valve_key]
                # Skip purge valves
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
    
    # Sort valve activations by time
    all_valve_activations.sort(key=lambda x: x['start_time'])
    
    if verbose:
        print(f"Found {len(all_valve_activations)} total valve activations (excluding Purge)")
        print(f"Analyzing {len(completed_sequences)} completed trials...")
    
    # Initialize counters
    odor_count_distribution = {i: 0 for i in range(1, 7)}  # 1-6 odors
    last_odor_counts = {}
    trial_sequences = []
    
    for _, trial in completed_sequences.iterrows():
        trial_start = trial['sequence_start']
        trial_end = trial['sequence_end']
        trial_id = trial['trial_id']
        
        # Find valve activations that occur during or overlap with this trial
        trial_valve_activations = []
        
        for valve_activation in all_valve_activations:
            valve_start = valve_activation['start_time']
            valve_end = valve_activation['end_time']
            
            # Check if valve activation overlaps with trial period
            # Include if: valve starts before trial end AND valve ends after trial start
            if valve_start <= trial_end and valve_end >= trial_start:
                trial_valve_activations.append(valve_activation)
        
        # Sort trial valve activations by start time
        trial_valve_activations.sort(key=lambda x: x['start_time'])
        
        # Extract odor sequence (Purge already excluded from all_valve_activations)
        odor_sequence = [activation['odor_name'] for activation in trial_valve_activations]
        
        # Count odors
        num_odors = len(odor_sequence)
        
        # Limit to maximum of 6 odors for counting
        num_odors_capped = min(num_odors, 6)
        if num_odors_capped > 0:
            odor_count_distribution[num_odors_capped] += 1
        
        # Get last odor (will not be Purge since Purge is excluded)
        last_odor = None
        if odor_sequence:
            last_odor = odor_sequence[-1]
            if last_odor not in last_odor_counts:
                last_odor_counts[last_odor] = 0
            last_odor_counts[last_odor] += 1
        
        # Store trial sequence info
        trial_sequences.append({
            'trial_id': trial_id,
            'trial_start': trial_start,
            'trial_end': trial_end,
            'odor_sequence': odor_sequence,
            'num_odors': num_odors,
            'last_odor': last_odor,
            'valve_activations': trial_valve_activations
        })
        
        if verbose and len(trial_sequences) <= 10:  # Show first 10 trials as examples
            print(f"\nTrial {trial_id}:")
            print(f"  Duration: {(trial_end - trial_start).total_seconds():.1f}s")
            print(f"  Odor sequence: {odor_sequence}")
            print(f"  Number of odors: {num_odors}")
            print(f"  Last odor: {last_odor}")
    
    # Print summary statistics
    print(f"\n" + "="*40)
    print("ODOR COUNT DISTRIBUTION:")
    print("="*40)
    total_trials = sum(odor_count_distribution.values())
    for num_odors in range(1, 7):
        count = odor_count_distribution[num_odors]
        percentage = (count / total_trials * 100) if total_trials > 0 else 0
        print(f"  {num_odors} odor{'s' if num_odors > 1 else ''}: {count} trials ({percentage:.1f}%)")
    
    print(f"\n" + "="*40)
    print("LAST ODOR DISTRIBUTION:")
    print("="*40)
    for odor_name, count in sorted(last_odor_counts.items()):
        percentage = (count / total_trials * 100) if total_trials > 0 else 0
        print(f"  {odor_name}: {count} trials ({percentage:.1f}%)")
    
    # Additional statistics
    trials_with_odors = sum(1 for seq in trial_sequences if seq['num_odors'] > 0)
    trials_without_odors = len(trial_sequences) - trials_with_odors
    
    print(f"\n" + "="*40)
    print("ADDITIONAL STATISTICS:")
    print("="*40)
    print(f"  Total completed trials: {len(trial_sequences)}")
    print(f"  Trials with odor delivery: {trials_with_odors}")
    print(f"  Trials without odor delivery: {trials_without_odors}")
    
    if trials_without_odors > 0:
        print(f"  Warning: {trials_without_odors} completed trials had no odor delivery")
    
    return {
        'trial_sequences': trial_sequences,
        'odor_count_distribution': odor_count_distribution,
        'last_odor_counts': last_odor_counts,
        'total_trials': total_trials,
        'trials_with_odors': trials_with_odors,
        'trials_without_odors': trials_without_odors
    }




def classify_trial_outcomes_extensive(data, events, trial_counts, odor_map, stage, verbose=True):
    """
    Classify trials into hierarchical categories based on completion, reward status, and hidden rule detection:
    
    1. Non-initiated sequences (from trial_counts)
    2. Initiated sequences (trials) subdivided into:
       - aborted_sequence: no AwaitReward event
         - aborted_sequence_HR: hidden rule odor (A/B) at LocationX
       - completed_sequence: has AwaitReward event
         - completed_sequence_HR: completed after LocationX odors (hit hidden rule)
         - completed_sequence_HR_missed: completed after >LocationX odors (missed hidden rule)
         
    Each completed category further subdivided into: rewarded, unrewarded, reward_timeout
    
    Returns:
        dict: Contains DataFrames for each trial category with hidden rule analysis
    """
    if verbose:
        print("=" * 80)
        print("CLASSIFYING TRIAL OUTCOMES WITH HIDDEN RULE ANALYSIS")
        print("=" * 80)
    
    # Extract hidden rule location from stage parameter (which contains the sequence name)
    hidden_rule_location = None
    sequence_name = str(stage)
    
    import re
    location_match = re.search(r'Location(\d+)', sequence_name)
    if location_match:
        hidden_rule_location = int(location_match.group(1))
        if verbose:
            print(f"Sequence name: {sequence_name}")
            print(f"Hidden rule location extracted: Location{hidden_rule_location} (index {hidden_rule_location}, position {hidden_rule_location + 1})")
    else:
        if verbose:
            print(f"Warning: No LocationX found in sequence name '{sequence_name}', hidden rule analysis will be skipped")
        # Fall back to original function
        return classify_trial_outcomes(data, events, trial_counts)
    
    # Get base trial data
    initiated_trials = trial_counts['initiated_sequences'].copy()
    non_initiated_trials = trial_counts['non_initiated_sequences'].copy()
    
    # Get event times
    await_reward_times = events['combined_await_reward_df']['Time'].tolist() if 'combined_await_reward_df' in events else []
    
    # Get supply port activities from pulse supply data
    supply_port1_times = []
    supply_port2_times = []
    
    if not data['pulse_supply_1'].empty:
        supply_port1_times = data['pulse_supply_1'].index.tolist()
    
    if not data['pulse_supply_2'].empty:
        supply_port2_times = data['pulse_supply_2'].index.tolist()
    
    all_supply_port_times = sorted(supply_port1_times + supply_port2_times)
    
    # Get reward port poke data
    port1_pokes = data['digital_input_data']['DIPort1'] if 'DIPort1' in data['digital_input_data'] else pd.Series(dtype=bool)
    port2_pokes = data['digital_input_data']['DIPort2'] if 'DIPort2' in data['digital_input_data'] else pd.Series(dtype=bool)
    
    # Build valve activation list (same as analyze_trial_valve_sequences)
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
                # Skip purge valves
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
    
    # Sort valve activations by time
    all_valve_activations.sort(key=lambda x: x['start_time'])
    
    if verbose:
        print(f"Found {len(all_valve_activations)} total valve activations (excluding Purge)")
        print(f"Analyzing {len(initiated_trials)} initiated trials...")
        print(f"Found {len(await_reward_times)} AwaitReward events")
        print(f"Found {len(all_supply_port_times)} total supply port activities")
    
    # Initialize result containers
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
    
    # Helper function to get valve sequence for a trial
    def get_trial_valve_sequence(trial_start, trial_end):
        """Get chronological valve sequence for a trial period"""
        trial_valve_activations = []
        
        for valve_activation in all_valve_activations:
            valve_start = valve_activation['start_time']
            valve_end = valve_activation['end_time']
            
            # Check if valve activation overlaps with trial period
            if valve_start <= trial_end and valve_end >= trial_start:
                trial_valve_activations.append(valve_activation)
        
        # Sort by start time and extract odor sequence
        trial_valve_activations.sort(key=lambda x: x['start_time'])
        odor_sequence = [activation['odor_name'] for activation in trial_valve_activations]
        
        return odor_sequence, trial_valve_activations
    
    # Helper function to check hidden rule
    def check_hidden_rule(odor_sequence, hidden_rule_location):
        """Check if hidden rule applies to this sequence"""
        if len(odor_sequence) <= hidden_rule_location:
            return False, False  # not_enough_odors, hit_hidden_rule
        
        odor_at_location = odor_sequence[hidden_rule_location]
        hit_hidden_rule = odor_at_location in ['OdorA', 'OdorB']
        
        return True, hit_hidden_rule  # enough_odors, hit_hidden_rule
    
    # Process initiated trials
    for _, trial in initiated_trials.iterrows():
        trial_start = trial['sequence_start']
        trial_end = trial['sequence_end']
        trial_id = trial['trial_id']
        
        # Get valve sequence for this trial
        odor_sequence, valve_activations = get_trial_valve_sequence(trial_start, trial_end)
        
        # Check if AwaitReward occurs within this trial
        trial_await_rewards = [
            t for t in await_reward_times 
            if trial_start <= t <= trial_end
        ]
        
        # Add basic trial info
        trial_dict = trial.to_dict()
        trial_dict['odor_sequence'] = odor_sequence
        trial_dict['num_odors'] = len(odor_sequence)
        trial_dict['last_odor'] = odor_sequence[-1] if odor_sequence else None
        trial_dict['hidden_rule_location'] = hidden_rule_location
        trial_dict['sequence_name'] = sequence_name
        
        # Check hidden rule
        enough_odors, hit_hidden_rule = check_hidden_rule(odor_sequence, hidden_rule_location)
        trial_dict['enough_odors_for_hr'] = enough_odors
        trial_dict['hit_hidden_rule'] = hit_hidden_rule
        
        if trial_await_rewards:
            # This is a completed sequence
            completed_sequences.append(trial_dict.copy())
            
            # Get the first AwaitReward time in this trial
            await_reward_time = min(trial_await_rewards)
            trial_dict['await_reward_time'] = await_reward_time
            
            # Determine if hidden rule was followed
            if hit_hidden_rule:
                # Check if completed at exactly the hidden rule location
                if len(odor_sequence) == hidden_rule_location + 1:
                    # Completed after hidden rule (correct behavior)
                    completed_hr.append(trial_dict.copy())
                    hr_category = 'completed_hr'
                else:
                    # Missed hidden rule (continued past it)
                    completed_hr_missed.append(trial_dict.copy())
                    hr_category = 'completed_hr_missed'
            else:
                # No hidden rule involvement (normal completion)
                hr_category = 'completed_normal'
            
            # Now check reward status
            supply1_after_await = [
                t for t in supply_port1_times 
                if await_reward_time <= t <= trial_end
            ]
            supply2_after_await = [
                t for t in supply_port2_times 
                if await_reward_time <= t <= trial_end
            ]
            
            if supply1_after_await or supply2_after_await:
                # Rewarded trial
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
                
                # Add to HR-specific categories
                if hr_category == 'completed_hr':
                    completed_hr_rewarded.append(trial_dict.copy())
                elif hr_category == 'completed_hr_missed':
                    completed_hr_missed_rewarded.append(trial_dict.copy())
                    
            else:
                # No supply port activity - check for reward port pokes within 2.5s
                poke_window_end = await_reward_time + pd.Timedelta(seconds=2.5)
                
                # Find poke events in Port1 and Port2 within the window
                port1_pokes_in_window = []
                port2_pokes_in_window = []
                
                # Check Port1 pokes
                if not port1_pokes.empty:
                    port1_window = port1_pokes[await_reward_time:poke_window_end]
                    port1_starts = port1_window & ~port1_window.shift(1, fill_value=False)
                    port1_pokes_in_window = port1_starts[port1_starts == True].index.tolist()
                
                # Check Port2 pokes
                if not port2_pokes.empty:
                    port2_window = port2_pokes[await_reward_time:poke_window_end]
                    port2_starts = port2_window & ~port2_window.shift(1, fill_value=False)
                    port2_pokes_in_window = port2_starts[port2_starts == True].index.tolist()
                
                # Create combined list with port identity and odor mapping
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
                    # Unrewarded trial (poked but no reward)
                    first_poke_time, first_poke_port, first_poke_odor = all_reward_pokes[0]
                    trial_dict['first_reward_poke_time'] = first_poke_time
                    trial_dict['first_reward_poke_port'] = first_poke_port
                    trial_dict['first_reward_poke_odor_identity'] = first_poke_odor
                    
                    completed_unrewarded.append(trial_dict.copy())
                    
                    # Add to HR-specific categories
                    if hr_category == 'completed_hr':
                        completed_hr_unrewarded.append(trial_dict.copy())
                    elif hr_category == 'completed_hr_missed':
                        completed_hr_missed_unrewarded.append(trial_dict.copy())
                        
                else:
                    # Timeout trial (no poke within 2.5s)
                    completed_timeout.append(trial_dict.copy())
                    
                    # Add to HR-specific categories
                    if hr_category == 'completed_hr':
                        completed_hr_timeout.append(trial_dict.copy())
                    elif hr_category == 'completed_hr_missed':
                        completed_hr_missed_timeout.append(trial_dict.copy())
        else:
            # This is an aborted sequence (no AwaitReward)
            aborted_sequences.append(trial_dict.copy())
            
            # Check if it was a hidden rule trial
            if hit_hidden_rule:
                aborted_sequences_hr.append(trial_dict.copy())
    
    # Create DataFrames
    result = {
        # Base categories
        'non_initiated_sequences': non_initiated_trials,
        'initiated_sequences': initiated_trials,
        'completed_sequences': pd.DataFrame(completed_sequences),
        'aborted_sequences': pd.DataFrame(aborted_sequences),
        
        # Hidden rule categories
        'aborted_sequences_HR': pd.DataFrame(aborted_sequences_hr),
        'completed_sequences_HR': pd.DataFrame(completed_hr),
        'completed_sequences_HR_missed': pd.DataFrame(completed_hr_missed),
        
        # Reward status categories (original)
        'completed_sequence_rewarded': pd.DataFrame(completed_rewarded),
        'completed_sequence_unrewarded': pd.DataFrame(completed_unrewarded),
        'completed_sequence_reward_timeout': pd.DataFrame(completed_timeout),
        
        # Hidden rule + reward status categories
        'completed_sequence_HR_rewarded': pd.DataFrame(completed_hr_rewarded),
        'completed_sequence_HR_unrewarded': pd.DataFrame(completed_hr_unrewarded),
        'completed_sequence_HR_reward_timeout': pd.DataFrame(completed_hr_timeout),
        'completed_sequence_HR_missed_rewarded': pd.DataFrame(completed_hr_missed_rewarded),
        'completed_sequence_HR_missed_unrewarded': pd.DataFrame(completed_hr_missed_unrewarded),
        'completed_sequence_HR_missed_reward_timeout': pd.DataFrame(completed_hr_missed_timeout),
    }
    
    # Print comprehensive summary statistics
    if verbose:
        print(f"\nTRIAL CLASSIFICATION RESULTS WITH HIDDEN RULE ANALYSIS:")
        print(f"Hidden Rule Location: Position {hidden_rule_location + 1} (index {hidden_rule_location})")
        print()
        
        total_attempts = len(initiated_trials) + len(non_initiated_trials)
        print(f"Total attempts: {total_attempts}")
        print(f"-- Non-initiated sequences: {len(non_initiated_trials)} ({len(non_initiated_trials)/total_attempts*100:.1f}%)")
        print(f"-- Initiated sequences (trials): {len(initiated_trials)} ({len(initiated_trials)/total_attempts*100:.1f}%)")
        print()
        
        print(f"INITIATED TRIALS BREAKDOWN:")
        print(f"Total initiated trials: {len(initiated_trials)}")
        print(f"-- Completed sequences: {len(result['completed_sequences'])} ({len(result['completed_sequences'])/len(initiated_trials)*100:.1f}%)")
        print(f"   -- Hidden Rule trials (HR): {len(result['completed_sequences_HR'])} ({len(result['completed_sequences_HR'])/len(initiated_trials)*100:.1f}%)")
        print(f"   -- Hidden Rule Missed (HR_missed): {len(result['completed_sequences_HR_missed'])} ({len(result['completed_sequences_HR_missed'])/len(initiated_trials)*100:.1f}%)")
        print(f"-- Aborted sequences: {len(result['aborted_sequences'])} ({len(result['aborted_sequences'])/len(initiated_trials)*100:.1f}%)")
        print(f"   -- Aborted Hidden Rule trials (HR): {len(result['aborted_sequences_HR'])} ({len(result['aborted_sequences_HR'])/len(initiated_trials)*100:.1f}%)")
        print()
        
        print(f"REWARD STATUS BREAKDOWN:")
        print(f"All completed trials: {len(result['completed_sequences'])}")
        if len(result['completed_sequences']) > 0:
            print(f"-- Rewarded: {len(result['completed_sequence_rewarded'])} ({len(result['completed_sequence_rewarded'])/len(result['completed_sequences'])*100:.1f}%)")
            print(f"-- Unrewarded: {len(result['completed_sequence_unrewarded'])} ({len(result['completed_sequence_unrewarded'])/len(result['completed_sequences'])*100:.1f}%)")
            print(f"-- Reward timeout: {len(result['completed_sequence_reward_timeout'])} ({len(result['completed_sequence_reward_timeout'])/len(result['completed_sequences'])*100:.1f}%)")
        print()
        
        print(f"HIDDEN RULE SPECIFIC BREAKDOWN:")
        hr_total = len(result['completed_sequences_HR'])
        if hr_total > 0:
            print(f"Completed HR trials: {hr_total}")
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
        
        # Verify totals
        total_classified = (len(result['completed_sequence_rewarded']) + 
                           len(result['completed_sequence_unrewarded']) + 
                           len(result['completed_sequence_reward_timeout']) + 
                           len(result['aborted_sequences']))
        
        if total_classified == len(initiated_trials):
            print(f"Classification complete: all {len(initiated_trials)} trials classified")
        else:
            print(f"Classification mismatch: {total_classified} classified vs {len(initiated_trials)} total")
    
    return result





def classify_trial_outcomes_with_pokes_and_valves(data, events, trial_counts, odor_map, stage, verbose=True):#Working version for valves, pokes, and response time. 
    """
    Classify trials into hierarchical categories based on completion, reward status, and hidden rule detection,
    with integrated poke time and response time analysis:
    
    1. Non-initiated sequences (from trial_counts)
    2. Initiated sequences (trials) subdivided into:
       - aborted_sequence: no AwaitReward event
         - aborted_sequence_HR: hidden rule odor (A/B) at LocationX
       - completed_sequence: has AwaitReward event
         - completed_sequence_HR: completed after LocationX odors (hit hidden rule)
         - completed_sequence_HR_missed: completed after >LocationX odors (missed hidden rule)
         
    Each completed category further subdivided into: rewarded, unrewarded, reward_timeout
    
    NEW: Adds poke time analysis for each position/odor and response time measurement
    Uses sequential valve grouping for positions 2-5, last individual activation for position 1
    
    Returns:
        dict: Contains DataFrames for each trial category with hidden rule analysis and poke/response times
    """
    if verbose:
        print("=" * 80)
        print("CLASSIFYING TRIAL OUTCOMES WITH HIDDEN RULE AND POKE/RESPONSE TIME ANALYSIS")
        print("=" * 80)
    
    # Extract hidden rule location from stage parameter (which contains the sequence name)
    hidden_rule_location = None
    sequence_name = str(stage)
    location_match = re.search(r'Location(\d+)', sequence_name)
    if location_match:
        hidden_rule_location = int(location_match.group(1))
        if verbose:
            print(f"Sequence name: {sequence_name}")
            print(f"Hidden rule location extracted: Location{hidden_rule_location} (index {hidden_rule_location}, position {hidden_rule_location + 1})")
    else:
        if verbose:
            print(f"Warning: No LocationX found in sequence name '{sequence_name}', hidden rule analysis will be skipped")
        # Fall back to original function
        return classify_trial_outcomes(data, events, trial_counts)
    
    # Get base trial data
    initiated_trials = trial_counts['initiated_sequences'].copy()
    non_initiated_trials = trial_counts['non_initiated_sequences'].copy()
    
    # Get event times
    await_reward_times = events['combined_await_reward_df']['Time'].tolist() if 'combined_await_reward_df' in events else []
    
    # Get supply port activities from pulse supply data
    supply_port1_times = []
    supply_port2_times = []
    
    if not data['pulse_supply_1'].empty:
        supply_port1_times = data['pulse_supply_1'].index.tolist()
    
    if not data['pulse_supply_2'].empty:
        supply_port2_times = data['pulse_supply_2'].index.tolist()
    
    all_supply_port_times = sorted(supply_port1_times + supply_port2_times)
    
    # Get reward port poke data
    port1_pokes = data['digital_input_data']['DIPort1'] if 'DIPort1' in data['digital_input_data'] else pd.Series(dtype=bool)
    port2_pokes = data['digital_input_data']['DIPort2'] if 'DIPort2' in data['digital_input_data'] else pd.Series(dtype=bool)
    
    # Get poke data for poke time analysis
    poke_data = data['digital_input_data']['DIPort0'].copy()
    
    # Build valve activation list (same as analyze_trial_valve_sequences)
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
                # Skip purge valves
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
    
    # Sort valve activations by time
    all_valve_activations.sort(key=lambda x: x['start_time'])
    
    # Parameters for poke analysis
    poke_gap_threshold_ms = 200
    minimum_poke_threshold_ms = 350
    
    if verbose:
        print(f"Found {len(all_valve_activations)} total valve activations (excluding Purge)")
        print(f"Analyzing {len(initiated_trials)} initiated trials...")
        print(f"Found {len(await_reward_times)} AwaitReward events")
        print(f"Found {len(all_supply_port_times)} total supply port activities")
    
    # Initialize result containers
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
    
    # Helper function to get valve sequence for a trial
    def get_trial_valve_sequence(trial_start, trial_end):
        """Get chronological valve sequence for a trial period"""
        trial_valve_activations = []
        
        for valve_activation in all_valve_activations:
            valve_start = valve_activation['start_time']
            valve_end = valve_activation['end_time']
            
            # Check if valve activation overlaps with trial period
            if valve_start <= trial_end and valve_end >= trial_start:
                trial_valve_activations.append(valve_activation)
        
        # Sort by start time and extract odor sequence
        trial_valve_activations.sort(key=lambda x: x['start_time'])
        odor_sequence = [activation['odor_name'] for activation in trial_valve_activations]
        
        return odor_sequence, trial_valve_activations
    
    # Helper function to check hidden rule
    def check_hidden_rule(odor_sequence, hidden_rule_location):
        """Check if hidden rule applies to this sequence"""
        if len(odor_sequence) <= hidden_rule_location:
            return False, False  # not_enough_odors, hit_hidden_rule
        
        odor_at_location = odor_sequence[hidden_rule_location]
        hit_hidden_rule = odor_at_location in ['OdorA', 'OdorB']
        
        return True, hit_hidden_rule  # enough_odors, hit_hidden_rule
    
    # Helper function to get valve timing and poke analysis for each position
    def analyze_trial_valve_and_poke_times(trial_valve_events):
        """Analyze valve timing and poke times for each position in a trial"""
        position_locations = {}
        position_valve_times = {}
        position_poke_times = {}
        
        # VALVE TIMING ANALYSIS
        # Position 1: Last individual activation of first odor
        # Positions 2-5: Group consecutive events, take first activation and last deactivation of each group
        
        # Handle position 1: Find LAST individual activation of first odor
        if trial_valve_events:
            first_odor_valve = trial_valve_events[0]['valve_key']
            
            # Find all individual activations of the first odor at the beginning
            first_odor_activations = []
            for event in trial_valve_events:
                if event['valve_key'] == first_odor_valve:
                    first_odor_activations.append(event)
                else:
                    break  # Stop when we hit a different valve
            
            if first_odor_activations:
                # Use the LAST individual activation for position 1
                position_locations[1] = first_odor_activations[-1]
        
        # Handle positions 2-5: Group consecutive events
        grouped_valve_presentations = []
        current_valve = None
        current_start_time = None
        current_end_time = None
        current_odor_name = None
        
        for event in trial_valve_events:
            if event['valve_key'] != current_valve:
                # Different valve - save previous group if exists
                if current_valve is not None:
                    grouped_valve_presentations.append({
                        'valve_key': current_valve,
                        'odor_name': current_odor_name,
                        'start_time': current_start_time,
                        'end_time': current_end_time
                    })
                
                # Start new group
                current_valve = event['valve_key']
                current_odor_name = event['odor_name']
                current_start_time = event['start_time']
                current_end_time = event['end_time']
            else:
                # Same valve - extend current group to latest end time
                current_end_time = event['end_time']
        
        # Don't forget the last group
        if current_valve is not None:
            grouped_valve_presentations.append({
                'valve_key': current_valve,
                'odor_name': current_odor_name,
                'start_time': current_start_time,
                'end_time': current_end_time
            })
        
        # Assign positions 2-5 based on grouped presentations
        for i, presentation in enumerate(grouped_valve_presentations[1:], 2):  # Start from position 2
            if i <= 5:
                position_locations[i] = presentation
        
        # VALVE TIMING: Calculate valve times for each position
        for position in range(1, 6):
            if position not in position_locations:
                continue
                
            location = position_locations[position]
            valve_start = location['start_time']
            valve_end = location['end_time']
            valve_duration_ms = (valve_end - valve_start).total_seconds() * 1000
            
            position_valve_times[position] = {
                'position': position,
                'odor_name': location['odor_name'],
                'valve_start': valve_start,
                'valve_end': valve_end,
                'valve_duration_ms': valve_duration_ms
            }
        
        # POKE TIME ANALYSIS: For poke analysis, use individual activations for position 1
        poke_position_locations = {}
        
        # Position 1 poke analysis: Use LAST individual activation (same as valve timing)
        if 1 in position_locations:
            poke_position_locations[1] = position_locations[1]
        
        # Positions 2-5 poke analysis: Use first individual activation of each group
        current_valve = None
        for event in trial_valve_events:
            if event['valve_key'] != current_valve:
                # This is the first activation of a new valve group
                position = None
                for pos, loc in position_locations.items():
                    if pos >= 2 and loc['valve_key'] == event['valve_key']:
                        position = pos
                        break
                
                if position and position <= 5:
                    poke_position_locations[position] = event
                
                current_valve = event['valve_key']
        
        # Analyze poke times for each position using individual activations
        for position in range(1, 6):
            if position not in poke_position_locations:
                continue
                
            location = poke_position_locations[position]
            odor_start = location['start_time']
            odor_end = location['end_time']
            
            # Get poke data during this odor presentation period
            odor_poke_data = poke_data.loc[odor_start:odor_end]
            
            if odor_poke_data.empty:
                continue
            
            # Check if already poking when valve opens
            valve_start_poke_status = False
            if len(poke_data.loc[:odor_start]) > 0:
                valve_start_poke_status = poke_data.loc[:odor_start].iloc[-1]
            
            # Find all poke transitions during odor period
            poke_transitions = []
            prev_state = valve_start_poke_status
            
            for timestamp, current_state in odor_poke_data.items():
                if current_state != prev_state:
                    poke_transitions.append({
                        'time': timestamp,
                        'state': current_state,  # True = poke in, False = poke out
                        'offset_ms': (timestamp - odor_start).total_seconds() * 1000
                    })
                    prev_state = current_state
            
            # Calculate consolidated poke time (same logic as analyze_poke_time_during_odors)
            consolidated_poke_time_ms = 0
            
            if valve_start_poke_status:
                # Already poking at valve start - start from valve onset
                current_poke_start = odor_start
                
                # Process transitions
                for i, transition in enumerate(poke_transitions):
                    if not transition['state']:  # This is a poke OUT
                        # End current poke period
                        poke_duration = (transition['time'] - current_poke_start).total_seconds() * 1000
                        consolidated_poke_time_ms += poke_duration
                        
                        # Check if we've reached threshold
                        if consolidated_poke_time_ms >= minimum_poke_threshold_ms:
                            break
                        
                        # Look for next poke IN
                        next_poke_in = None
                        for j in range(i + 1, len(poke_transitions)):
                            if poke_transitions[j]['state']:  # This is a poke IN
                                next_poke_in = poke_transitions[j]
                                break
                        
                        if next_poke_in:
                            gap_duration = (next_poke_in['time'] - transition['time']).total_seconds() * 1000
                            if gap_duration <= poke_gap_threshold_ms:
                                # Add gap and continue
                                consolidated_poke_time_ms += gap_duration
                                current_poke_start = next_poke_in['time']
                            else:
                                # Gap too long, stop
                                break
                        else:
                            # No more poke ins
                            break
                    
                # Handle case where poke continues to end of odor
                if len(poke_transitions) == 0 or (len(poke_transitions) > 0 and poke_transitions[-1]['state']):
                    # Still poking at end
                    remaining_duration = (odor_end - current_poke_start).total_seconds() * 1000
                    consolidated_poke_time_ms += remaining_duration
            
            else:
                # Not poking at valve start - find first poke IN
                first_poke_in = None
                for transition in poke_transitions:
                    if transition['state']:  # This is a poke IN
                        first_poke_in = transition
                        break
                
                if first_poke_in:
                    current_poke_start = first_poke_in['time']
                    start_index = poke_transitions.index(first_poke_in)
                    
                    # Process transitions from first poke in
                    for i in range(start_index + 1, len(poke_transitions)):
                        transition = poke_transitions[i]
                        
                        if not transition['state']:  # This is a poke OUT
                            # End current poke period
                            poke_duration = (transition['time'] - current_poke_start).total_seconds() * 1000
                            consolidated_poke_time_ms += poke_duration
                            
                            # Check if we've reached threshold
                            if consolidated_poke_time_ms >= minimum_poke_threshold_ms:
                                break
                            
                            # Look for next poke IN
                            next_poke_in = None
                            for j in range(i + 1, len(poke_transitions)):
                                if poke_transitions[j]['state']:  # This is a poke IN
                                    next_poke_in = poke_transitions[j]
                                    break
                            
                            if next_poke_in:
                                gap_duration = (next_poke_in['time'] - transition['time']).total_seconds() * 1000
                                if gap_duration <= poke_gap_threshold_ms:
                                    # Add gap and continue
                                    consolidated_poke_time_ms += gap_duration
                                    current_poke_start = next_poke_in['time']
                                    # Skip to the poke in we just processed
                                    while i < len(poke_transitions) - 1 and poke_transitions[i + 1] != next_poke_in:
                                        i += 1
                                else:
                                    # Gap too long, stop
                                    break
                            else:
                                # No more poke ins
                                break
                    
                    # Handle case where first poke continues to end
                    if len(poke_transitions) == 1 or (len(poke_transitions) > start_index and all(t['state'] for t in poke_transitions[start_index:])):
                        # Poke continues to end of odor
                        remaining_duration = (odor_end - current_poke_start).total_seconds() * 1000
                        consolidated_poke_time_ms += remaining_duration
            
            # Record the poke result
            if consolidated_poke_time_ms > 0:
                position_poke_times[position] = {
                    'position': position,
                    'odor_name': location['odor_name'],
                    'poke_time_ms': consolidated_poke_time_ms,
                    'poke_odor_start': odor_start,
                    'poke_odor_end': odor_end
                }
        
        return position_valve_times, position_poke_times
    
    # Helper function to calculate response time
    def calculate_response_time(trial_valve_events, await_reward_time):
        """Calculate response time from last poke out to first reward port poke"""
        if not trial_valve_events:
            return None, None
        
        # Find the last odor presentation
        last_odor_event = trial_valve_events[-1]
        last_odor_start = last_odor_event['start_time']
        last_odor_end = last_odor_event['end_time']
        
        # Get poke data during last odor
        last_odor_poke_data = poke_data.loc[last_odor_start:last_odor_end]
        
        if last_odor_poke_data.empty:
            return None, None
        
        # Find the last poke out during the last odor
        last_poke_out_time = None
        prev_state = poke_data.loc[:last_odor_start].iloc[-1] if len(poke_data.loc[:last_odor_start]) > 0 else False
        
        for timestamp, current_state in last_odor_poke_data.items():
            if prev_state and not current_state:  # Transition from poke in to poke out
                last_poke_out_time = timestamp
            prev_state = current_state
        
        if last_poke_out_time is None:
            return None, None
        
        # Find first reward port poke after await reward
        poke_window_end = await_reward_time + pd.Timedelta(seconds=2.5)
        
        # Check Port1 and Port2 pokes
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
        
        # Combine and find first
        all_reward_pokes = port1_pokes_in_window + port2_pokes_in_window
        
        if not all_reward_pokes:
            return last_poke_out_time, None
        
        first_reward_poke_time = min(all_reward_pokes)
        response_time_ms = (first_reward_poke_time - last_poke_out_time).total_seconds() * 1000
        
        return last_poke_out_time, response_time_ms
    
    # Process initiated trials
    for _, trial in initiated_trials.iterrows():
        trial_start = trial['sequence_start']
        trial_end = trial['sequence_end']
        trial_id = trial['trial_id']
        
        # Get valve sequence for this trial
        odor_sequence, valve_activations = get_trial_valve_sequence(trial_start, trial_end)
        
        # Analyze valve timing and poke times for this trial
        position_valve_times, position_poke_times = analyze_trial_valve_and_poke_times(valve_activations)
        
        # Check if AwaitReward occurs within this trial
        trial_await_rewards = [
            t for t in await_reward_times 
            if trial_start <= t <= trial_end
        ]
        
        # Add basic trial info
        trial_dict = trial.to_dict()
        trial_dict['odor_sequence'] = odor_sequence
        trial_dict['num_odors'] = len(odor_sequence)
        trial_dict['last_odor'] = odor_sequence[-1] if odor_sequence else None
        trial_dict['hidden_rule_location'] = hidden_rule_location
        trial_dict['sequence_name'] = sequence_name
        trial_dict['position_valve_times'] = position_valve_times
        trial_dict['position_poke_times'] = position_poke_times
        
        # Check hidden rule
        enough_odors, hit_hidden_rule = check_hidden_rule(odor_sequence, hidden_rule_location)
        trial_dict['enough_odors_for_hr'] = enough_odors
        trial_dict['hit_hidden_rule'] = hit_hidden_rule
        
        if trial_await_rewards:
            # This is a completed sequence
            completed_sequences.append(trial_dict.copy())
            
            # Get the first AwaitReward time in this trial
            await_reward_time = min(trial_await_rewards)
            trial_dict['await_reward_time'] = await_reward_time
            
            # Calculate response time for completed trials
            last_poke_out_time, response_time_ms = calculate_response_time(valve_activations, await_reward_time)
            trial_dict['last_poke_out_time'] = last_poke_out_time
            trial_dict['response_time_ms'] = response_time_ms
            
            # Determine if hidden rule was followed
            if hit_hidden_rule:
                # Check if completed at exactly the hidden rule location
                if len(odor_sequence) == hidden_rule_location + 1:
                    # Completed after hidden rule (correct behavior)
                    completed_hr.append(trial_dict.copy())
                    hr_category = 'completed_hr'
                else:
                    # Missed hidden rule (continued past it)
                    completed_hr_missed.append(trial_dict.copy())
                    hr_category = 'completed_hr_missed'
            else:
                # No hidden rule involvement (normal completion)
                hr_category = 'completed_normal'
            
            # Now check reward status
            supply1_after_await = [
                t for t in supply_port1_times 
                if await_reward_time <= t <= trial_end
            ]
            supply2_after_await = [
                t for t in supply_port2_times 
                if await_reward_time <= t <= trial_end
            ]
            
            if supply1_after_await or supply2_after_await:
                # Rewarded trial
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
                
                # Add to HR-specific categories
                if hr_category == 'completed_hr':
                    completed_hr_rewarded.append(trial_dict.copy())
                elif hr_category == 'completed_hr_missed':
                    completed_hr_missed_rewarded.append(trial_dict.copy())
                    
            else:
                # No supply port activity - check for reward port pokes within 2.5s
                poke_window_end = await_reward_time + pd.Timedelta(seconds=2.5)
                
                # Find poke events in Port1 and Port2 within the window
                port1_pokes_in_window = []
                port2_pokes_in_window = []
                
                # Check Port1 pokes
                if not port1_pokes.empty:
                    port1_window = port1_pokes[await_reward_time:poke_window_end]
                    port1_starts = port1_window & ~port1_window.shift(1, fill_value=False)
                    port1_pokes_in_window = port1_starts[port1_starts == True].index.tolist()
                
                # Check Port2 pokes
                if not port2_pokes.empty:
                    port2_window = port2_pokes[await_reward_time:poke_window_end]
                    port2_starts = port2_window & ~port2_window.shift(1, fill_value=False)
                    port2_pokes_in_window = port2_starts[port2_starts == True].index.tolist()
                
                # Create combined list with port identity and odor mapping
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
                    # Unrewarded trial (poked but no reward)
                    first_poke_time, first_poke_port, first_poke_odor = all_reward_pokes[0]
                    trial_dict['first_reward_poke_time'] = first_poke_time
                    trial_dict['first_reward_poke_port'] = first_poke_port
                    trial_dict['first_reward_poke_odor_identity'] = first_poke_odor
                    completed_unrewarded.append(trial_dict.copy())
                    
                    # Add to HR-specific categories
                    if hr_category == 'completed_hr':
                        completed_hr_unrewarded.append(trial_dict.copy())
                    elif hr_category == 'completed_hr_missed':
                        completed_hr_missed_unrewarded.append(trial_dict.copy())
                        
                else:
                    # Timeout trial (no poke within 2.5s)
                    completed_timeout.append(trial_dict.copy())
                    
                    # Add to HR-specific categories
                    if hr_category == 'completed_hr':
                        completed_hr_timeout.append(trial_dict.copy())
                    elif hr_category == 'completed_hr_missed':
                        completed_hr_missed_timeout.append(trial_dict.copy())
        else:
            # This is an aborted sequence (no AwaitReward)
            aborted_sequences.append(trial_dict.copy())
            
            # Check if it was a hidden rule trial
            if hit_hidden_rule:
                aborted_sequences_hr.append(trial_dict.copy())
    
    # Create DataFrames
    result = {
        # Base categories
        'non_initiated_sequences': non_initiated_trials,
        'initiated_sequences': initiated_trials,
        'completed_sequences': pd.DataFrame(completed_sequences),
        'aborted_sequences': pd.DataFrame(aborted_sequences),
        
        # Hidden rule categories
        'aborted_sequences_HR': pd.DataFrame(aborted_sequences_hr),
        'completed_sequences_HR': pd.DataFrame(completed_hr),
        'completed_sequences_HR_missed': pd.DataFrame(completed_hr_missed),
        
        # Reward status categories (original)
        'completed_sequence_rewarded': pd.DataFrame(completed_rewarded),
        'completed_sequence_unrewarded': pd.DataFrame(completed_unrewarded),
        'completed_sequence_reward_timeout': pd.DataFrame(completed_timeout),
        
        # Hidden rule + reward status categories
        'completed_sequence_HR_rewarded': pd.DataFrame(completed_hr_rewarded),
        'completed_sequence_HR_unrewarded': pd.DataFrame(completed_hr_unrewarded),
        'completed_sequence_HR_reward_timeout': pd.DataFrame(completed_hr_timeout),
        'completed_sequence_HR_missed_rewarded': pd.DataFrame(completed_hr_missed_rewarded),
        'completed_sequence_HR_missed_unrewarded': pd.DataFrame(completed_hr_missed_unrewarded),
        'completed_sequence_HR_missed_reward_timeout': pd.DataFrame(completed_hr_missed_timeout),
    }
    
    # Print comprehensive summary statistics
    if verbose:
        print(f"\nTRIAL CLASSIFICATION RESULTS WITH HIDDEN RULE AND VALVE/POKE TIME ANALYSIS:")
        print(f"Hidden Rule Location: Position {hidden_rule_location + 1} (index {hidden_rule_location})")
        print()
        
        total_attempts = len(initiated_trials) + len(non_initiated_trials)
        print(f"Total attempts: {total_attempts}")
        print(f"-- Non-initiated sequences: {len(non_initiated_trials)} ({len(non_initiated_trials)/total_attempts*100:.1f}%)")
        print(f"-- Initiated sequences (trials): {len(initiated_trials)} ({len(initiated_trials)/total_attempts*100:.1f}%)")
        print()
        
        print(f"INITIATED TRIALS BREAKDOWN:")
        print(f"Total initiated trials: {len(initiated_trials)}")
        print(f"-- Completed sequences: {len(result['completed_sequences'])} ({len(result['completed_sequences'])/len(initiated_trials)*100:.1f}%)")
        print(f"   -- Hidden Rule trials (HR): {len(result['completed_sequences_HR'])} ({len(result['completed_sequences_HR'])/len(initiated_trials)*100:.1f}%)")
        print(f"   -- Hidden Rule Missed (HR_missed): {len(result['completed_sequences_HR_missed'])} ({len(result['completed_sequences_HR_missed'])/len(initiated_trials)*100:.1f}%)")
        print(f"-- Aborted sequences: {len(result['aborted_sequences'])} ({len(result['aborted_sequences'])/len(initiated_trials)*100:.1f}%)")
        print(f"   -- Aborted Hidden Rule trials (HR): {len(result['aborted_sequences_HR'])} ({len(result['aborted_sequences_HR'])/len(initiated_trials)*100:.1f}%)")
        print()
        
        print(f"REWARD STATUS BREAKDOWN:")
        print(f"All completed trials: {len(result['completed_sequences'])}")
        if len(result['completed_sequences']) > 0:
            print(f"-- Rewarded: {len(result['completed_sequence_rewarded'])} ({len(result['completed_sequence_rewarded'])/len(result['completed_sequences'])*100:.1f}%)")
            print(f"-- Unrewarded: {len(result['completed_sequence_unrewarded'])} ({len(result['completed_sequence_unrewarded'])/len(result['completed_sequences'])*100:.1f}%)")
            print(f"-- Reward timeout: {len(result['completed_sequence_reward_timeout'])} ({len(result['completed_sequence_reward_timeout'])/len(result['completed_sequences'])*100:.1f}%)")
        print()
        
        print(f"HIDDEN RULE SPECIFIC BREAKDOWN:")
        hr_total = len(result['completed_sequences_HR'])
        if hr_total > 0:
            print(f"Completed HR trials: {hr_total}")
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
        
        # Print valve and poke time summary
        print(f"VALVE AND POKE TIME ANALYSIS:")
        completed_with_valve_times = [trial for trial in completed_sequences if trial.get('position_valve_times')]
        completed_with_poke_times = [trial for trial in completed_sequences if trial.get('position_poke_times')]
        completed_with_response = [trial for trial in completed_sequences if trial.get('response_time_ms') is not None]
        
        print(f"-- Completed trials with valve time data: {len(completed_with_valve_times)}/{len(completed_sequences)}")
        print(f"-- Completed trials with poke time data: {len(completed_with_poke_times)}/{len(completed_sequences)}")
        print(f"-- Completed trials with response time data: {len(completed_with_response)}/{len(completed_sequences)}")
        
        # Verify totals
        total_classified = (len(result['completed_sequence_rewarded']) + 
                           len(result['completed_sequence_unrewarded']) + 
                           len(result['completed_sequence_reward_timeout']) + 
                           len(result['aborted_sequences']))
        
        if total_classified == len(initiated_trials):
            print(f"Classification complete: all {len(initiated_trials)} trials classified")
        else:
            print(f"Classification mismatch: {total_classified} classified vs {len(initiated_trials)} total")
    
    return result






def classify_trials(data, events, trial_counts, odor_map, stage, root, verbose=True):#working version to classify trials and get valve/poke times. Part of wrapper function
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
        print("CLASSIFYING TRIAL OUTCOMES (NO RESPONSE-TIME) WITH HIDDEN RULE AND VALVE/POKE TIME ANALYSIS")
        print("=" * 80)
        print(f"Sample offset time: {sample_offset_time_ms} ms")
        print(f"Minimum sampling time: {minimum_sampling_time_ms} ms")
        print(f"Response time window: {response_time_sec} s")

    # Hidden rule location from stage
    hidden_rule_location = None
    sequence_name = str(stage)
    location_match = re.search(r'Location(\d+)', sequence_name)
    if location_match:
        hidden_rule_location = int(location_match.group(1))
        if verbose:
            print(f"Sequence name: {sequence_name}")
            print(f"Hidden rule location extracted: Location{hidden_rule_location} (index {hidden_rule_location}, position {hidden_rule_location + 1})")
    else:
        if verbose:
            print(f"Warning: No LocationX found in sequence name '{sequence_name}', hidden rule analysis will be skipped")
        return classify_trial_outcomes(data, events, trial_counts)

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

    # Poke-time analysis parameters
    poke_gap_threshold_ms = sample_offset_time_ms
    minimum_poke_threshold_ms = minimum_sampling_time_ms

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

    def check_hidden_rule(odor_sequence, idx):
        if len(odor_sequence) <= idx:
            return False, False
        odor_at_location = odor_sequence[idx]
        hit_hidden_rule = odor_at_location in ['OdorA', 'OdorB']
        return True, hit_hidden_rule

    def analyze_trial_valve_and_poke_times(trial_valve_events):
        position_locations = {}
        position_valve_times = {}
        position_poke_times = {}

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
            position_valve_times[position] = {
                'position': position,
                'odor_name': loc['odor_name'],
                'valve_start': valve_start,
                'valve_end': valve_end,
                'valve_duration_ms': valve_duration_ms
            }

        # Poke-time analysis positions
        poke_position_locations = {}
        if 1 in position_locations:
            poke_position_locations[1] = position_locations[1]

        current_valve = None
        for event in trial_valve_events:
            if event['valve_key'] != current_valve:
                position = None
                for pos, loc in position_locations.items():
                    if pos >= 2 and loc['valve_key'] == event['valve_key']:
                        position = pos
                        break
                if position and position <= 5:
                    poke_position_locations[position] = event
                current_valve = event['valve_key']

        # Compute consolidated poke time
        for position in range(1, 6):
            if position not in poke_position_locations:
                continue
            loc = poke_position_locations[position]
            odor_start = loc['start_time']
            odor_end = loc['end_time']
            odor_poke_data = poke_data.loc[odor_start:odor_end]
            if odor_poke_data.empty:
                continue

            valve_start_poke_status = False
            if len(poke_data.loc[:odor_start]) > 0:
                valve_start_poke_status = poke_data.loc[:odor_start].iloc[-1]

            poke_transitions = []
            prev_state = valve_start_poke_status
            for timestamp, current_state in odor_poke_data.items():
                if current_state != prev_state:
                    poke_transitions.append({
                        'time': timestamp,
                        'state': current_state,
                        'offset_ms': (timestamp - odor_start).total_seconds() * 1000
                    })
                    prev_state = current_state

            consolidated_poke_time_ms = 0

            if valve_start_poke_status:
                current_poke_start = odor_start
                for i, transition in enumerate(poke_transitions):
                    if not transition['state']:
                        consolidated_poke_time_ms += (transition['time'] - current_poke_start).total_seconds() * 1000
                        if consolidated_poke_time_ms >= minimum_poke_threshold_ms:
                            break
                        next_poke_in = None
                        for j in range(i + 1, len(poke_transitions)):
                            if poke_transitions[j]['state']:
                                next_poke_in = poke_transitions[j]
                                break
                        if next_poke_in:
                            gap = (next_poke_in['time'] - transition['time']).total_seconds() * 1000
                            if gap <= poke_gap_threshold_ms:
                                consolidated_poke_time_ms += gap
                                current_poke_start = next_poke_in['time']
                            else:
                                break
                        else:
                            break
                if len(poke_transitions) == 0 or (len(poke_transitions) > 0 and poke_transitions[-1]['state']):
                    consolidated_poke_time_ms += (odor_end - current_poke_start).total_seconds() * 1000
            else:
                first_poke_in = None
                for t in poke_transitions:
                    if t['state']:
                        first_poke_in = t
                        break
                if first_poke_in:
                    current_poke_start = first_poke_in['time']
                    start_index = poke_transitions.index(first_poke_in)
                    i = start_index
                    while i < len(poke_transitions):
                        transition = poke_transitions[i]
                        if not transition['state']:
                            consolidated_poke_time_ms += (transition['time'] - current_poke_start).total_seconds() * 1000
                            if consolidated_poke_time_ms >= minimum_poke_threshold_ms:
                                break
                            next_poke_in = None
                            for j in range(i + 1, len(poke_transitions)):
                                if poke_transitions[j]['state']:
                                    next_poke_in = poke_transitions[j]
                                    break
                            if next_poke_in:
                                gap = (next_poke_in['time'] - transition['time']).total_seconds() * 1000
                                if gap <= poke_gap_threshold_ms:
                                    consolidated_poke_time_ms += gap
                                    current_poke_start = next_poke_in['time']
                                    i = poke_transitions.index(next_poke_in)
                                else:
                                    break
                            else:
                                break
                        i += 1
                    if len(poke_transitions) == 1 or (len(poke_transitions) > start_index and all(t['state'] for t in poke_transitions[start_index:])):
                        consolidated_poke_time_ms += (odor_end - current_poke_start).total_seconds() * 1000

            if consolidated_poke_time_ms > 0:
                position_poke_times[position] = {
                    'position': position,
                    'odor_name': loc['odor_name'],
                    'poke_time_ms': consolidated_poke_time_ms,
                    'poke_odor_start': odor_start,
                    'poke_odor_end': odor_end
                }

        return position_valve_times, position_poke_times

    # Process trials
    for _, trial in initiated_trials.iterrows():
        trial_start = trial['sequence_start']
        trial_end = trial['sequence_end']
        odor_sequence, valve_activations = get_trial_valve_sequence(trial_start, trial_end)

        position_valve_times, position_poke_times = analyze_trial_valve_and_poke_times(valve_activations)

        trial_await_rewards = [t for t in await_reward_times if trial_start <= t <= trial_end]

        trial_dict = trial.to_dict()
        trial_dict['odor_sequence'] = odor_sequence
        trial_dict['num_odors'] = len(odor_sequence)
        trial_dict['last_odor'] = odor_sequence[-1] if odor_sequence else None
        trial_dict['hidden_rule_location'] = hidden_rule_location
        trial_dict['sequence_name'] = sequence_name
        trial_dict['position_valve_times'] = position_valve_times
        trial_dict['position_poke_times'] = position_poke_times

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
                if len(odor_sequence) == hidden_rule_location + 1:
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

    # Plural aliases to prevent KeyErrors in downstream code
    result['completed_sequences_HR_rewarded'] = result['completed_sequence_HR_rewarded']
    result['completed_sequences_HR_unrewarded'] = result['completed_sequence_HR_unrewarded']
    result['completed_sequences_HR_reward_timeout'] = result['completed_sequence_HR_reward_timeout']
    result['completed_sequences_HR_missed_rewarded'] = result['completed_sequence_HR_missed_rewarded']
    result['completed_sequences_HR_missed_unrewarded'] = result['completed_sequence_HR_missed_unrewarded']
    result['completed_sequences_HR_missed_reward_timeout'] = result['completed_sequence_HR_missed_reward_timeout']

    if verbose:
        print(f"\nTRIAL CLASSIFICATION RESULTS WITH HIDDEN RULE AND VALVE/POKE TIME ANALYSIS (NO RESPONSE-TIME):")
        print(f"Hidden Rule Location: Position {hidden_rule_location + 1} (index {hidden_rule_location})\n")

        total_attempts = len(initiated_trials) + len(non_initiated_trials)
        print(f"Total attempts: {total_attempts}")
        print(f"-- Non-initiated sequences: {len(non_initiated_trials)} ({len(non_initiated_trials)/total_attempts*100:.1f}%)")
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




def analyze_response_times(data, trial_counts, events, odor_map, stage, root, verbose=True):#working version to analyze response times for all completed trials. Part of wrapper function
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
        print("RESPONSE TIME ANALYSIS - ALL COMPLETED TRIALS (FIXED)")
        print("=" * 80)

    # Extract hidden rule location
    hidden_rule_location = None
    sequence_name = str(stage)
    location_match = re.search(r'Location(\d+)', sequence_name)
    if location_match:
        hidden_rule_location = int(location_match.group(1))
        if verbose:
            print(f"Hidden rule location: Position {hidden_rule_location + 1} (index {hidden_rule_location})")
    else:
        if verbose:
            print("No hidden rule location found")
        return None

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

    def check_hidden_rule(odor_sequence, idx):
        if len(odor_sequence) <= idx:
            return False, False
        odor_at_location = odor_sequence[idx]
        hit_hidden_rule = odor_at_location in ['OdorA', 'OdorB']
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
        if hit_hidden_rule and len(odor_sequence) == hidden_rule_location + 1:
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
                    delayed_port2_starts = port2_window & ~port2_window.shift(1, fill_value=False) if False else None  # placeholder to avoid NameError
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



def classify_and_analyze_with_response_times(data, events, trial_counts, odor_map, stage, root, verbose=True):
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
    sequence_name = str(stage)
    hidden_rule_location = None
    m = re.search(r'Location(\d+)', sequence_name)
    if m:
        hidden_rule_location = int(m.group(1))

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
        if hidden_rule_location is not None and hit_hr and num_odors == hidden_rule_location + 1:
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



