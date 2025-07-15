from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import numpy as np
import datetime
import zoneinfo
import re
from typing import Dict, Any, Optional
from functools import reduce
import harp
import src.utils as utils

class BaseLoader(ABC):
    """Abstract base class for all data loaders"""
    
    @abstractmethod
    def load(self, path: Path) -> Dict[str, Any]:
        """Load data from path and return standardized format"""
        pass
    
    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate loaded data structure"""
        pass

class BehaviorDataLoader(BaseLoader):
    """Loader for behavioral hardware data (digital inputs, rewards)"""
    
    def load(self, path: Path) -> Dict[str, Any]:
        """Load behavioral data from Behavior directory"""
        behavior_path = path / "Behavior"
        
        if not behavior_path.exists():
            return self._empty_behavior_data(path)
        
        behavior_reader = harp.reader.create_reader(
            'device_schemas/behavior.yml', 
            epoch=harp.io.REFERENCE_EPOCH
        )
        
        try:
            # Load all behavioral data streams
            digital_input_data = self._safe_load(
                behavior_reader.DigitalInputState, behavior_path,
                default_columns=['Time', 'DIPort0', 'DIPort1', 'DIPort2']
            )
            
            pulse_supply_1 = self._safe_load(
                behavior_reader.PulseSupplyPort1, behavior_path,
                default_columns=['Time']
            )
            
            pulse_supply_2 = self._safe_load(
                behavior_reader.PulseSupplyPort2, behavior_path,
                default_columns=['Time']
            )
            
            heartbeat = self._safe_load(
                behavior_reader.TimestampSeconds, behavior_path,
                default_columns=['Time', 'TimestampSeconds']
            )
            
            # Reset indices for non-empty dataframes
            for df in [digital_input_data, pulse_supply_1, pulse_supply_2, heartbeat]:
                if not df.empty:
                    df.reset_index(inplace=True)
            
            return {
                'digital_input_data': digital_input_data,
                'pulse_supply_1': pulse_supply_1,
                'pulse_supply_2': pulse_supply_2,
                'heartbeat': heartbeat,
                'source_path': path,
                'loader_type': 'behavior'
            }
            
        except Exception as e:
            print(f"Error loading behavioral data: {e}")
            return self._empty_behavior_data(path)
    
    def _safe_load(self, reader_func, path: Path, default_columns: list) -> pd.DataFrame:
        """Safely load data with fallback to empty DataFrame"""
        try:
            return utils.load(reader_func, path)
        except ValueError:
            print(f"No data found for {reader_func.__name__}")
            return pd.DataFrame(columns=default_columns)
        except Exception as e:
            print(f"Error loading {reader_func.__name__}: {e}")
            return pd.DataFrame(columns=default_columns)
    
    def _empty_behavior_data(self, path: Path) -> Dict[str, Any]:
        """Return empty behavioral data structure"""
        return {
            'digital_input_data': pd.DataFrame(columns=['Time', 'DIPort0', 'DIPort1', 'DIPort2']),
            'pulse_supply_1': pd.DataFrame(columns=['Time']),
            'pulse_supply_2': pd.DataFrame(columns=['Time']),
            'heartbeat': pd.DataFrame(columns=['Time', 'TimestampSeconds']),
            'source_path': path,
            'loader_type': 'behavior'
        }
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate behavioral data structure"""
        required_keys = ['digital_input_data', 'pulse_supply_1', 'pulse_supply_2', 'heartbeat']
        return all(key in data for key in required_keys)

class OlfactometerDataLoader(BaseLoader):
    """Loader for olfactometer valve data"""
    
    def load(self, path: Path) -> Dict[str, Any]:
        """Load olfactometer data from Olfactometer0 and Olfactometer1 directories"""
        
        olfactometer_reader = harp.reader.create_reader(
            'device_schemas/olfactometer.yml', 
            epoch=harp.io.REFERENCE_EPOCH
        )
        
        try:
            # Load olfactometer 0 data
            olfactometer_valves_0 = self._safe_load(
                olfactometer_reader.OdorValveState, path / "Olfactometer0",
                default_columns=['Time', 'Valve0', 'Valve1', 'Valve2', 'Valve3']
            )
            
            # Load olfactometer 1 data
            olfactometer_valves_1 = self._safe_load(
                olfactometer_reader.OdorValveState, path / "Olfactometer1",
                default_columns=['Time', 'Valve0', 'Valve1', 'Valve2', 'Valve3']
            )
            
            # Reset indices for non-empty dataframes
            for df in [olfactometer_valves_0, olfactometer_valves_1]:
                if not df.empty:
                    df.reset_index(inplace=True)
            
            return {
                'olfactometer_valves_0': olfactometer_valves_0,
                'olfactometer_valves_1': olfactometer_valves_1,
                'source_path': path,
                'loader_type': 'olfactometer'
            }
            
        except Exception as e:
            print(f"Error loading olfactometer data: {e}")
            return self._empty_olfactometer_data(path)
    
    def _safe_load(self, reader_func, path: Path, default_columns: list) -> pd.DataFrame:
        """Safely load data with fallback to empty DataFrame"""
        try:
            return utils.load(reader_func, path)
        except ValueError:
            print(f"No data found for {path}")
            return pd.DataFrame(columns=default_columns)
        except Exception as e:
            print(f"Error loading from {path}: {e}")
            return pd.DataFrame(columns=default_columns)
    
    def _empty_olfactometer_data(self, path: Path) -> Dict[str, Any]:
        """Return empty olfactometer data structure"""
        return {
            'olfactometer_valves_0': pd.DataFrame(columns=['Time', 'Valve0', 'Valve1', 'Valve2', 'Valve3']),
            'olfactometer_valves_1': pd.DataFrame(columns=['Time', 'Valve0', 'Valve1', 'Valve2', 'Valve3']),
            'source_path': path,
            'loader_type': 'olfactometer'
        }
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate olfactometer data structure"""
        required_keys = ['olfactometer_valves_0', 'olfactometer_valves_1']
        return all(key in data for key in required_keys)

class ExperimentEventsLoader(BaseLoader):
    """Loader for experiment events (EndInitiation, InitiationSequence, AwaitReward, Reset)"""
    
    def load(self, path: Path) -> Dict[str, Any]:
        """Load experiment events from ExperimentEvents directory"""
        experiment_events_dir = path / "ExperimentEvents"
        
        if not experiment_events_dir.exists():
            print("No ExperimentEvents directory found")
            return self._empty_events_data(path)
        
        try:
            end_initiation_frames = []
            initiation_sequence_frames = []
            await_reward_frames = []
            reset_frames = []
            
            csv_files = list(experiment_events_dir.glob("*.csv"))
            print(f"Found {len(csv_files)} experiment event files")
            
            for csv_file in csv_files:
                try:
                    ev_df = pd.read_csv(csv_file)
                    print(f"Processing event file: {csv_file.name} with {len(ev_df)} rows")
                    
                    if "Value" in ev_df.columns:
                        print(f"Found Value column with values: {ev_df['Value'].unique()}")
                        
                        # EndInitiation events
                        eii_df = ev_df[ev_df["Value"] == "EndInitiation"].copy()
                        if not eii_df.empty:
                            eii_df.rename(columns={'Seconds': 'Time'}, inplace=True)
                            print(f"Found {len(eii_df)} EndInitiation events")
                            end_initiation_frames.append(eii_df)
                        
                        # InitiationSequence events
                        is_df = ev_df[ev_df["Value"] == "InitiationSequence"].copy()
                        if not is_df.empty:
                            is_df.rename(columns={'Seconds': 'Time'}, inplace=True)
                            print(f"Found {len(is_df)} InitiationSequence events")
                            initiation_sequence_frames.append(is_df)
                        
                        # AwaitReward events
                        ar_df = ev_df[ev_df["Value"] == "AwaitReward"].copy()
                        if not ar_df.empty:
                            ar_df.rename(columns={'Seconds': 'Time'}, inplace=True)
                            print(f"Found {len(ar_df)} AwaitReward events")
                            await_reward_frames.append(ar_df)
                        
                        # Reset events
                        reset_df = ev_df[ev_df["Value"] == "Reset"].copy()
                        if not reset_df.empty:
                            reset_df.rename(columns={'Seconds': 'Time'}, inplace=True)
                            print(f"Found {len(reset_df)} Reset events")
                            reset_frames.append(reset_df)
                            
                except Exception as e:
                    print(f"Error processing event file {csv_file.name}: {e}")
            
            return {
                'end_initiation_frames': end_initiation_frames,
                'initiation_sequence_frames': initiation_sequence_frames,
                'await_reward_frames': await_reward_frames,
                'reset_frames': reset_frames,
                'source_path': path,
                'loader_type': 'experiment_events'
            }
            
        except Exception as e:
            print(f"Error loading experiment events: {e}")
            return self._empty_events_data(path)
    
    def _empty_events_data(self, path: Path) -> Dict[str, Any]:
        """Return empty events data structure"""
        return {
            'end_initiation_frames': [],
            'initiation_sequence_frames': [],
            'await_reward_frames': [],
            'reset_frames': [],
            'source_path': path,
            'loader_type': 'experiment_events'
        }
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate experiment events data structure"""
        required_keys = ['end_initiation_frames', 'initiation_sequence_frames', 'await_reward_frames', 'reset_frames']
        return all(key in data for key in required_keys)

class MetadataLoader(BaseLoader):
    """Loader for session metadata and settings"""
    
    def load(self, path: Path) -> Dict[str, Any]:
        """Load metadata from session path"""
        settings_file = path / "SessionSettings"
        
        if not settings_file.exists():
            print(f"SessionSettings not found: {settings_file}")
            return self._empty_metadata(path)
            
        try:
            metadata_reader = utils.SessionData()
            session_settings = utils.load_json(metadata_reader, settings_file)
            
            return {
                'metadata': session_settings,
                'source_path': path,
                'loader_type': 'metadata'
            }
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return self._empty_metadata(path)
    
    def _empty_metadata(self, path: Path) -> Dict[str, Any]:
        """Return empty metadata structure"""
        return {
            'metadata': None,
            'source_path': path,
            'loader_type': 'metadata'
        }
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate metadata structure"""
        return 'metadata' in data and data['metadata'] is not None

class SessionDataLoader:
    """Main session data loader that coordinates all data types and processing"""
    
    def __init__(self):
        self.loaders = {
            'behavior': BehaviorDataLoader(),
            'olfactometer': OlfactometerDataLoader(),
            'experiment_events': ExperimentEventsLoader(),
            'metadata': MetadataLoader(),
        }
    
    def load_session(self, session_path: Path, 
                    data_types: Optional[list] = None) -> Dict[str, Any]:
        """
        Load and process all session data keeping raw timestamps
        
        Parameters:
        -----------
        session_path : Path
            Path to session directory
        data_types : list, optional
            List of data types to load. If None, loads all available.
            
        Returns:
        --------
        dict
            Combined and processed session data
        """
        session_path = Path(session_path)
        
        if data_types is None:
            data_types = list(self.loaders.keys())
        
        # Load raw data
        raw_data = {}
        for data_type in data_types:
            if data_type not in self.loaders:
                print(f"Warning: Unknown data type '{data_type}', skipping...")
                continue
                
            try:
                loader = self.loaders[data_type]
                data = loader.load(session_path)
                
                if loader.validate(data):
                    raw_data[data_type] = data
                else:
                    print(f"Warning: Validation failed for {data_type} in {session_path}")
                    
            except Exception as e:
                print(f"Error loading {data_type} from {session_path}: {e}")
                continue
        
        # Process data without timestamp conversion
        processed_data = self._process_session_data(raw_data, session_path)
        
        return processed_data
    
    def _process_session_data(self, raw_data: Dict[str, Any], session_path: Path) -> Dict[str, Any]:
        """Process raw data keeping original timestamps"""
        
        # Extract behavioral data
        behavior_data = raw_data.get('behavior', {})
        heartbeat = behavior_data.get('heartbeat', pd.DataFrame())
        
        # Process experiment events without timestamp conversion
        processed_events = self._process_experiment_events(
            raw_data.get('experiment_events', {})
        )
        
        # Calculate basic session metrics
        session_metrics = self._calculate_basic_metrics(behavior_data, heartbeat)
        
        # Combine all processed data
        return {
            'session_path': session_path,
            'session_id': session_path.name,
            'raw_data': raw_data,
            'processed_events': processed_events,
            'session_metrics': session_metrics
        }
    
    def _process_experiment_events(self, events_data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Process experiment events keeping raw timestamps"""
        
        event_types = {
            'end_initiation': events_data.get('end_initiation_frames', []),
            'initiation_sequence': events_data.get('initiation_sequence_frames', []),
            'await_reward': events_data.get('await_reward_frames', []),
            'reset': events_data.get('reset_frames', [])
        }
        
        processed_events = {}
        
        for event_type, frames in event_types.items():
            if not frames:
                processed_events[event_type] = pd.DataFrame(columns=["Time", event_type])
                continue
            
            processed_frames = []
            for frame in frames:
                frame_copy = frame.copy()
                
                # Keep original Time field without conversion
                if "Time" in frame_copy.columns:
                    frame_copy[event_type] = True
                    processed_frames.append(frame_copy[["Time", event_type]])
            
            if processed_frames:
                combined_df = pd.concat(processed_frames, ignore_index=True)
                print(f"Combined {len(combined_df)} {event_type} events")
                processed_events[event_type] = combined_df
            else:
                processed_events[event_type] = pd.DataFrame(columns=["Time", event_type])
        
        return processed_events
    
    def _calculate_basic_metrics(self, behavior_data: Dict[str, Any], heartbeat: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic session metrics"""
        metrics = {}
        
        # Session duration from heartbeat
        if not heartbeat.empty and 'Time' in heartbeat.columns:
            metrics['session_start'] = heartbeat['Time'].min()
            metrics['session_end'] = heartbeat['Time'].max()
            metrics['session_duration'] = metrics['session_end'] - metrics['session_start']
        else:
            metrics['session_start'] = None
            metrics['session_end'] = None
            metrics['session_duration'] = None
        
        # Digital input activity counts
        digital_input = behavior_data.get('digital_input_data', pd.DataFrame())
        if not digital_input.empty:
            for port in ['DIPort0', 'DIPort1', 'DIPort2']:
                if port in digital_input.columns:
                    metrics[f'{port}_activations'] = digital_input[port].sum()
                else:
                    metrics[f'{port}_activations'] = 0
        
        # Reward counts
        pulse_1 = behavior_data.get('pulse_supply_1', pd.DataFrame())
        pulse_2 = behavior_data.get('pulse_supply_2', pd.DataFrame())
        metrics['reward_1_count'] = len(pulse_1) if not pulse_1.empty else 0
        metrics['reward_2_count'] = len(pulse_2) if not pulse_2.empty else 0
        
        return metrics

# Convenience functions for backward compatibility
def load_session_data(session_path: Path, data_types: Optional[list] = None) -> Dict[str, Any]:
    """Load complete session data - main entry point"""
    loader = SessionDataLoader()
    return loader.load_session(session_path, data_types)

def load_behavior_data(session_path: Path) -> Dict[str, Any]:
    """Load just behavioral data"""
    loader = BehaviorDataLoader()
    return loader.load(session_path)

def load_olfactometer_data(session_path: Path) -> Dict[str, Any]:
    """Load just olfactometer data"""
    loader = OlfactometerDataLoader()
    return loader.load(session_path)