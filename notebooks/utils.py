import sys
import os
project_root = os.path.abspath("/Users/joschua/repos/harris_lab/hypnose/hypnose_analysis")
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
import harp
import datetime
from datetime import timezone
import zoneinfo
from src.processing import detect_settings
from datetime import datetime, timezone



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

    session_dirs = list(subject_dir.glob(f"ses-*_date-{date_str}"))
    
    if not session_dirs:
        raise FileNotFoundError(f"No session found for date {date_str} in {subject_dir}")
    
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
        raise FileNotFoundError(f"No experiment directories found in {behav_dir}")
    
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