#### python

import re
import argparse
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import zoneinfo
import src.utils as utils  # Changed from relative to absolute import
from src.processing.detect_stage import detect_stage
from src.processing.detect_settings import detect_settings
from src.processing.process_olfactometer_valves import process_olfactometer_valves
import harp
import yaml
from functools import reduce

class RewardAnalyser:
    """
    Class-based analyzer that detects the session stage and runs 
    either the reward-analyser-stage1 or reward-analyser-stage2to8 logic.
    """

    def __init__(self, session_settings):
        self.session_settings = session_settings

    def _reward_analyser_stage1(self, data_path, reward_a=8.0, reward_b=8.0):
        """
        Stage1: Analyzes rewards and session length (reward-analyser-stage1).
        """
        root = Path(data_path)
        behavior_reader = harp.reader.create_reader('device_schemas/behavior.yml', epoch=harp.io.REFERENCE_EPOCH)
        olfactometer_reader = harp.reader.create_reader('device_schemas/olfactometer.yml', epoch=harp.io.REFERENCE_EPOCH)

        # Load data streams
        digital_input_data = utils.load(behavior_reader.DigitalInputState, root/"Behavior")
        pulse_supply_1 = utils.load(behavior_reader.PulseSupplyPort1, root/"Behavior")
        pulse_supply_2 = utils.load(behavior_reader.PulseSupplyPort2, root/"Behavior")
        heartbeat = utils.load(behavior_reader.TimestampSeconds, root/"Behavior")

        # Convert time index to column
        for df in [heartbeat, digital_input_data, pulse_supply_1, pulse_supply_2]:
            df.reset_index(inplace=True)

        # Derive real-time offset
        real_time_str = root.as_posix().split('/')[-1]
        real_time_ref_utc = datetime.datetime.strptime(
            real_time_str, '%Y-%m-%dT%H-%M-%S'
        ).replace(tzinfo=datetime.timezone.utc)
        uk_tz = zoneinfo.ZoneInfo("Europe/London")
        real_time_ref = real_time_ref_utc.astimezone(uk_tz)

        start_time_hardware = heartbeat['Time'].iloc[0]
        start_time_dt = start_time_hardware.to_pydatetime()
        if start_time_dt.tzinfo is None:
            start_time_dt = start_time_dt.replace(tzinfo=uk_tz)
        real_time_offset = real_time_ref - start_time_dt

        # Shift data
        digital_input_data_abs = digital_input_data.copy()
        pulse_supply_1_abs = pulse_supply_1.copy()
        pulse_supply_2_abs = pulse_supply_2.copy()
        for df_abs in [digital_input_data_abs, pulse_supply_1_abs, pulse_supply_2_abs]:
            df_abs['Time'] = df_abs['Time'] + real_time_offset

        # Count reward events
        r1_reward = pulse_supply_1_abs[['Time']].copy()
        r2_reward = pulse_supply_2_abs[['Time']].copy()
        num_r1_rewards = r1_reward.shape[0]
        num_r2_rewards = r2_reward.shape[0]
        total_vol_r1 = num_r1_rewards * reward_a
        total_vol_r2 = num_r2_rewards * reward_b
        total_delivered = total_vol_r1 + total_vol_r2

        # Session length
        start_time_sec = heartbeat['TimestampSeconds'].iloc[0]
        end_time_sec = heartbeat['TimestampSeconds'].iloc[-1]
        session_duration_sec = end_time_sec - start_time_sec
        h = int(session_duration_sec // 3600)
        m = int((session_duration_sec % 3600) // 60)
        s = int(session_duration_sec % 60)

        print(f"Rewards R1: {num_r1_rewards} (Total Volume: {total_vol_r1} µL)")
        print(f"Rewards R2: {num_r2_rewards} (Total Volume: {total_vol_r2} µL)")
        print(f"Overall Volume: {total_delivered} µL")
        print(f"Session Duration: {h}h {m}m {s}s")

    def _reward_analyser_stage2to8(self, data_path, reward_a=8.0, reward_b=8.0):
        """
        Stage2-8: Full session data analysis (reward-analyser-stage2to8).
        With improved error handling for missing or empty data.
        """
        root = Path(data_path)
        
        # Get session data and analyze decision accuracy
        session_data = self._get_session_data(root)
        
        # Extract reward and timing information
        num_r1_rewards = session_data.get('num_r1_rewards', 0)
        num_r2_rewards = session_data.get('num_r2_rewards', 0)
        total_vol_r1 = num_r1_rewards * reward_a
        total_vol_r2 = num_r2_rewards * reward_b
        total_delivered = total_vol_r1 + total_vol_r2
        session_duration_sec = session_data.get('session_duration_sec', 0)
        
        # Format time in a readable way
        h = int(session_duration_sec // 3600) if session_duration_sec else 0
        m = int((session_duration_sec % 3600) // 60) if session_duration_sec else 0
        s = int(session_duration_sec % 60) if session_duration_sec else 0
        
        # Print reward and session information
        print(f"Number of Reward A (r1) delivered: {num_r1_rewards} (Total Volume: {total_vol_r1} µL)")
        print(f"Number of Reward B (r2) delivered: {num_r2_rewards} (Total Volume: {total_vol_r2} µL)")
        print(f"Overall total volume delivered: {total_delivered} µL\n")
        print(f"Session Duration: {h}h {m}m {s}s\n")
        
        # Print decision accuracy if available
        accuracy_summary = session_data.get('accuracy_summary')
        if accuracy_summary and accuracy_summary.get('r1_respond', 0) + accuracy_summary.get('r2_respond', 0) > 0:
            print("Decision Accuracy (using EndInitiation from experiment events):")
            print(f"  R1 Trials: {accuracy_summary['r1_respond']}, Correct: {accuracy_summary['r1_correct']}, Accuracy: {accuracy_summary['r1_accuracy']:.2f}%")
            print(f"  R2 Trials: {accuracy_summary['r2_respond']}, Correct: {accuracy_summary['r2_correct']}, Accuracy: {accuracy_summary['r2_accuracy']:.2f}%")
            print(f"  Overall Accuracy: {accuracy_summary['overall_accuracy']:.2f}%")

    def _get_session_data(self, root):
        """
        Core function to load and process session data.
        Used by both _reward_analyser_stage2to8 and get_decision_accuracy.
        
        Parameters:
        -----------
        root : Path
            Path to the session directory
            
        Returns:
        --------
        dict
            Dictionary containing processed session data including
            rewards, timing, and decision accuracy.
        """
        session_data = {}
        
        # Create readers
        behavior_reader = harp.reader.create_reader('device_schemas/behavior.yml', epoch=harp.io.REFERENCE_EPOCH)
        olfactometer_reader = harp.reader.create_reader('device_schemas/olfactometer.yml', epoch=harp.io.REFERENCE_EPOCH)
        
        try:
            # Data loading with safe fallbacks for missing data
            try:
                digital_input_data = utils.load(behavior_reader.DigitalInputState, root/"Behavior")
            except ValueError:  # No objects to concatenate - no data
                digital_input_data = pd.DataFrame(columns=['Time', 'DIPort1', 'DIPort2'])
                print("No digital input data found.")
            except Exception as e:
                print(f"Error loading digital input data: {e}")
                digital_input_data = pd.DataFrame(columns=['Time', 'DIPort1', 'DIPort2'])
        
            try:
                pulse_supply_1 = utils.load(behavior_reader.PulseSupplyPort1, root/"Behavior")
            except ValueError:  # No objects to concatenate - no rewards delivered
                pulse_supply_1 = pd.DataFrame(columns=['Time'])
                print("No Reward A (PulseSupplyPort1) data found - possibly no rewards delivered.")
            except Exception as e:
                print(f"Error loading Reward A data: {e}")
                pulse_supply_1 = pd.DataFrame(columns=['Time'])
        
            try:
                pulse_supply_2 = utils.load(behavior_reader.PulseSupplyPort2, root/"Behavior")
            except ValueError:  # No objects to concatenate - no rewards delivered
                pulse_supply_2 = pd.DataFrame(columns=['Time'])
                print("No Reward B (PulseSupplyPort2) data found - possibly no rewards delivered.")
            except Exception as e:
                print(f"Error loading Reward B data: {e}")
                pulse_supply_2 = pd.DataFrame(columns=['Time'])
        
            try:
                olfactometer_valves_0 = utils.load(olfactometer_reader.OdorValveState, root/"Olfactometer0")
            except ValueError:
                olfactometer_valves_0 = pd.DataFrame(columns=['Time', 'Valve0', 'Valve1', 'Valve2', 'Valve3'])
                print("No olfactometer valve state data found.")
            except Exception as e:
                print(f"Error loading olfactometer valve state data: {e}")
                olfactometer_valves_0 = pd.DataFrame(columns=['Time', 'Valve0', 'Valve1', 'Valve2', 'Valve3'])
        
            try:
                olfactometer_valves_1 = utils.load(olfactometer_reader.OdorValveState, root/"Olfactometer1")
            except ValueError:
                olfactometer_valves_1 = pd.DataFrame(columns=['Time', 'Valve0', 'Valve1', 'Valve2', 'Valve3'])
                print("No data for Olfactometer1 found.")
            except Exception as e:
                print(f"Error loading Olfactometer1 data: {e}")
                olfactometer_valves_1 = pd.DataFrame(columns=['Time', 'Valve0', 'Valve1', 'Valve2', 'Valve3'])
        
            try:
                heartbeat = utils.load(behavior_reader.TimestampSeconds, root/"Behavior")
            except Exception as e:
                print(f"Error loading timestamp data: {e}")
                heartbeat = pd.DataFrame(columns=['Time', 'TimestampSeconds'])
            
            # Reset indices for non-empty dataframes
            for df in [digital_input_data, pulse_supply_1, pulse_supply_2, 
                    olfactometer_valves_0, olfactometer_valves_1, heartbeat]:
                if not df.empty:
                    df.reset_index(inplace=True)
            
            # Calculate real-time offset only if we have heartbeat data
            real_time_offset = pd.Timedelta(0)
            if not heartbeat.empty and 'Time' in heartbeat.columns and len(heartbeat) > 0:
                try:
                    real_time_str = root.name
                    match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', real_time_str)
                    if match:
                        real_time_str = match.group(0)
                    else:
                        # Try parent directory
                        real_time_str = root.parent.name
                        match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', real_time_str)
                        if match:
                            real_time_str = match.group(0)
                    
                    if re.match(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', real_time_str):
                        real_time_ref_utc = datetime.datetime.strptime(real_time_str, '%Y-%m-%dT%H-%M-%S')
                        real_time_ref_utc = real_time_ref_utc.replace(tzinfo=datetime.timezone.utc)
                        uk_tz = zoneinfo.ZoneInfo("Europe/London")
                        real_time_ref = real_time_ref_utc.astimezone(uk_tz)
                        
                        start_time_hardware = heartbeat['Time'].iloc[0]
                        start_time_dt = start_time_hardware.to_pydatetime()
                        if start_time_dt.tzinfo is None:
                            start_time_dt = start_time_dt.replace(tzinfo=uk_tz)
                        real_time_offset = real_time_ref - start_time_dt
                except Exception as e:
                    print(f"Error calculating real-time offset: {e}")
            
            # Create absolute time versions with checks for empty DataFrames
            digital_input_data_abs = digital_input_data.copy() if not digital_input_data.empty else pd.DataFrame()
            pulse_supply_1_abs = pulse_supply_1.copy() if not pulse_supply_1.empty else pd.DataFrame()
            pulse_supply_2_abs = pulse_supply_2.copy() if not pulse_supply_2.empty else pd.DataFrame()
            olfactometer_valves_0_abs = olfactometer_valves_0.copy() if not olfactometer_valves_0.empty else pd.DataFrame()
            olfactometer_valves_1_abs = olfactometer_valves_1.copy() if not olfactometer_valves_1.empty else pd.DataFrame()
            
            # Apply time offset to non-empty DataFrames
            for df_abs in [digital_input_data_abs, pulse_supply_1_abs, pulse_supply_2_abs,
                        olfactometer_valves_0_abs, olfactometer_valves_1_abs]:
                if not df_abs.empty and 'Time' in df_abs.columns:
                    df_abs['Time'] = df_abs['Time'] + real_time_offset
            
            # Map heartbeat times if we have data
            timestamp_to_time = pd.Series()
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
            
            # Store data about rewards
            session_data['num_r1_rewards'] = pulse_supply_1_abs.shape[0] if not pulse_supply_1_abs.empty else 0
            session_data['num_r2_rewards'] = pulse_supply_2_abs.shape[0] if not pulse_supply_2_abs.empty else 0
            
            # Store session duration
            if not heartbeat.empty and 'TimestampSeconds' in heartbeat.columns and len(heartbeat) > 1:
                start_time_sec = heartbeat['TimestampSeconds'].iloc[0]
                end_time_sec = heartbeat['TimestampSeconds'].iloc[-1]
                session_data['session_duration_sec'] = end_time_sec - start_time_sec
            else:
                session_data['session_duration_sec'] = 0
            
            # Detect stage 
            stage = detect_stage(root)

            # Process experiment events for EndInitiation and InitiationSequence
            end_initiation_frames = []
            start_initiation_frames = []
            experiment_events_dir = root / "ExperimentEvents"
            
            if experiment_events_dir.exists():
                csv_files = list(experiment_events_dir.glob("*.csv"))
                print(f"Found {len(csv_files)} experiment event files")
                
                for csv_file in csv_files:
                    try:
                        ev_df = pd.read_csv(csv_file)
                        print(f"Processing event file: {csv_file.name} with {len(ev_df)} rows")
                        
                        # Use Seconds field if available, otherwise use Time field
                        if "Seconds" in ev_df.columns and not timestamp_to_time.empty:
                            ev_df = ev_df.sort_values("Seconds").reset_index(drop=True)
                            ev_df["Time"] = ev_df["Seconds"].apply(interpolate_time)
                            print(f"Using Seconds column for interpolation")
                        else:
                            ev_df["Time"] = pd.to_datetime(ev_df["Time"], errors="coerce")
                            print(f"Using Time column directly")
                        
                        if "Time" in ev_df.columns:
                            ev_df["Time"] = ev_df["Time"] + real_time_offset
                            
                            if "Value" in ev_df.columns:
                                print(f"Found Value column with values: {ev_df['Value'].unique()}")
                                eii_df = ev_df[ev_df["Value"] == "EndInitiation"].copy()
                                if not eii_df.empty:
                                    print(f"Found {len(eii_df)} EndInitiation events")
                                    eii_df["EndInitiation"] = True
                                    end_initiation_frames.append(eii_df[["Time", "EndInitiation"]])

                                is_df = ev_df[ev_df["Value"] == "InitiationSequence"].copy()
                                if not is_df.empty:
                                    print(f"Found {len(is_df)} InitiationSequence events")
                                    is_df["InitiationSequence"] = True
                                    start_initiation_frames.append(is_df[["Time", "InitiationSequence"]])
                                # TODO: Continue here
                    except Exception as e:
                        print(f"Error processing event file {csv_file.name}: {e}")
            else:
                print("No ExperimentEvents directory found")
            
            # Safely combine EndInitiation frames
            if len(end_initiation_frames) > 0:
                combined_end_initiation_df = pd.concat(end_initiation_frames, ignore_index=True)
                print(f"Combined {len(combined_end_initiation_df)} EndInitiation events")
            else:
                combined_end_initiation_df = pd.DataFrame(columns=["Time", "EndInitiation"])
                print("No EndInitiation events found - cannot identify trial endings")
            
            # Safely combine InitiationSequence frames
            if len(start_initiation_frames) > 0:
                combined_start_initiation_df = pd.concat(start_initiation_frames, ignore_index=True)
                print(f"Combined {len(combined_start_initiation_df)} InitiationSequence events")
            else:
                combined_start_initiation_df = pd.DataFrame(columns=["Time", "InitiationSequence"])
                print("No InitiationSequence events found - cannot identify trial endings")
            
            # Now process events to calculate decision accuracy
            event_frames = []
            
            # Add odour poke events if available
            if not digital_input_data_abs.empty and 'DIPort0' in digital_input_data_abs.columns:
                try:
                    odour_poke_df = digital_input_data_abs[digital_input_data_abs['DIPort0'] == True].copy()
                    if not odour_poke_df.empty:
                        odour_poke_df = odour_poke_df[['Time']].copy()
                        odour_poke_df['odour_poke'] = True
                        event_frames.append(odour_poke_df)
                        print(f"Added {len(odour_poke_df)} odour poke events")

                        # Find OFF events
                        df = digital_input_data_abs.sort_values(by='Time').reset_index(drop=True)
                        poke_prev = df['DIPort0'].shift(1) # Shift Valve0 to compare previous row
                        poke_now = df['DIPort0']

                        # Detect where it was ON in previous row and now is OFF
                        off_after_on_mask = (poke_prev == True) & (poke_now == False)
                        odour_poke_off_df = df.loc[off_after_on_mask, ['Time']].copy()
                        odour_poke_off_df['odour_poke_off'] = True
                        event_frames.append(odour_poke_off_df)
                        print(f"Added {len(odour_poke_off_df)} odour poke OFF events")
                except Exception as e:
                    print(f"Error processing odour poke events: {e}")

            # Add r1 poke events if available
            if not digital_input_data_abs.empty and 'DIPort1' in digital_input_data_abs.columns:
                try:
                    r1_poke_df = digital_input_data_abs[digital_input_data_abs['DIPort1'] == True].copy()
                    if not r1_poke_df.empty:
                        r1_poke_df = r1_poke_df[['Time']].copy()
                        r1_poke_df['r1_poke'] = True
                        event_frames.append(r1_poke_df)
                        print(f"Added {len(r1_poke_df)} r1 poke events")
                except Exception as e:
                    print(f"Error processing r1 poke events: {e}")
            
            # Add r2 poke events if available
            if not digital_input_data_abs.empty and 'DIPort2' in digital_input_data_abs.columns:
                try:
                    r2_poke_df = digital_input_data_abs[digital_input_data_abs['DIPort2'] == True].copy()
                    if not r2_poke_df.empty:
                        r2_poke_df = r2_poke_df[['Time']].copy()
                        r2_poke_df['r2_poke'] = True
                        event_frames.append(r2_poke_df)
                        print(f"Added {len(r2_poke_df)} r2 poke events")
                except Exception as e:
                    print(f"Error processing r2 poke events: {e}")
            
            # Add reward delivery events if available
            if not pulse_supply_1_abs.empty and 'PulseSupplyPort1' in pulse_supply_1_abs.columns:
                try:
                    r1_df = pulse_supply_1_abs.copy()
                    if not r1_df.empty:
                        r1_df = r1_df[['Time']].copy()
                        r1_df['PulseSupplyPort1'] = True
                        event_frames.append(r1_df)
                        print(f"Added {len(r1_df)} r1 delivery events")
                except Exception as e:
                    print(f"Error processing r1 delivery events: {e}") 

            if not pulse_supply_2_abs.empty and 'PulseSupplyPort2' in pulse_supply_2_abs.columns:
                try:
                    r2_df = pulse_supply_2_abs.copy()
                    if not r2_df.empty:
                        r2_df = r2_df[['Time']].copy()
                        r2_df['PulseSupplyPort2'] = True
                        event_frames.append(r2_df)
                        print(f"Added {len(r2_df)} r2 delivery events")
                except Exception as e:
                    print(f"Error processing r2 delivery events: {e}") 

            # Get session settings
            session_settings, session_schema = detect_settings(root)
            
            # Add olfactometer valve events if available 
            olfactometer_valves = {
                0: olfactometer_valves_0_abs,
                1: olfactometer_valves_1_abs,
            }
    
            olf_valves0 = [cmd.valvesOpenO0 for cmd in session_settings.metadata.iloc[0].olfactometerCommands]
            olf_valves1 = [cmd.valvesOpenO1 for cmd in session_settings.metadata.iloc[0].olfactometerCommands]
            
            olf_command_idx = {
                f'0{val}': next(i for i, lst in enumerate(olf_valves0) if val in lst)
                for val in range(4)
            } | {
                f'1{val}': next(i for i, lst in enumerate(olf_valves1) if val in lst)
                for val in range(4)
            }                

            odour_to_olfactometer_map = {}
            for cmd, idx in olf_command_idx.items():
                olf_id = int(cmd[0])
                valve_id = int(cmd[1])
                olfactometer_df = olfactometer_valves[olf_id]

                if session_settings.metadata.iloc[0].olfactometerCommands[idx].name == 'Purge':
                    continue

                elif session_settings.metadata.iloc[0].olfactometerCommands[idx].name == 'OdorA':
                    on_label = 'r1_olf_valve'
                    off_label = 'r1_olf_valve_off'
                
                elif session_settings.metadata.iloc[0].olfactometerCommands[idx].name == 'OdorB':
                    on_label = 'r2_olf_valve'
                    off_label = 'r2_olf_valve_off'

                elif session_settings.metadata.iloc[0].olfactometerCommands[idx].name == 'OdorC':
                    on_label = 'odourC_olf_valve'
                    off_label = 'odourC_olf_valve_off'
                
                elif session_settings.metadata.iloc[0].olfactometerCommands[idx].name == 'OdorD':
                    on_label = 'odourD_olf_valve'
                    off_label = 'odourD_olf_valve_off'
                
                elif session_settings.metadata.iloc[0].olfactometerCommands[idx].name == 'OdorE':
                    on_label = 'odourE_olf_valve'
                    off_label = 'odourE_olf_valve_off'
                
                elif session_settings.metadata.iloc[0].olfactometerCommands[idx].name == 'OdorF':
                    on_label = 'odourF_olf_valve'
                    off_label = 'odourF_olf_valve_off'
                
                elif session_settings.metadata.iloc[0].olfactometerCommands[idx].name == 'OdorG':
                    on_label = 'odourG_olf_valve'
                    off_label = 'odourG_olf_valve_off'
                
                else:
                    continue

                odour_to_olfactometer_map[on_label] = olf_id + 1  

                event_frames = process_olfactometer_valves(olfactometer_df, f'Valve{valve_id}', on_label, off_label, event_frames)    

            # Add EndInitiation events if available
            if not combined_end_initiation_df.empty and 'EndInitiation' in combined_end_initiation_df.columns:
                event_frames.append(combined_end_initiation_df)
            
            # Add InitiationSequence events if available
            if not combined_start_initiation_df.empty and 'InitiationSequence' in combined_start_initiation_df.columns:
                event_frames.append(combined_start_initiation_df)
            
            # Only proceed if we have data to analyze
            if event_frames:
                try:
                    # Filter out empty dataframes
                    dfs_to_merge = [df for df in event_frames if not df.empty]

                    # Merge all on 'Time' (outer join keeps all timestamps, merging by shared times)
                    all_events_df = reduce(lambda left, right: pd.merge(left, right, on='Time', how='outer'), dfs_to_merge)
                    # all_events_df = pd.concat(event_frames, ignore_index=True)
                    print(f"Combined {len(all_events_df)} total events")
                    
                    # Explicitly add missing columns with default values to prevent errors
                    if stage > 7:
                        required_columns = ['Time', 'r1_poke', 'r2_poke', 'r1_olf_valve', 'r2_olf_valve', \
                                            'odourC_olf_valve', 'odourD_olf_valve', 'odourE_olf_valve', 'odourF_olf_valve', \
                                                'odourG_olf_valve', 'r1_olf_valve_off', 'r2_olf_valve_off', 'odourC_olf_valve_off', \
                                                    'odourD_olf_valve_off', 'odourE_olf_valve_off', 'odourF_olf_valve_off', \
                                                        'odourG_olf_valve_off', 'EndInitiation', 'InitiationSequence', 'odour_poke', 'odour_poke_off', \
                                                            'PulseSupplyPort1', 'PulseSupplyPort2']
                    else:
                        required_columns = ['Time', 'r1_poke', 'r2_poke', 'r1_olf_valve', 'r2_olf_valve', 'r1_olf_valve_off', \
                                            'r2_olf_valve_off', 'EndInitiation', 'InitiationSequence', 'odour_poke', 'odour_poke_off']
                    
                    for col in required_columns:
                        if col not in all_events_df.columns and col != 'Time':
                            all_events_df[col] = False
                        if col != "Time":
                            all_events_df[col] = all_events_df[col].fillna(False) # This might drop the first InitiationSequence event
                    all_events_df = all_events_df.dropna(subset=["Time"])

                    # all_events_df = all_events_df.fillna(False).infer_objects(copy=False)
                    all_events_df.sort_values('Time', inplace=True)
                    all_events_df.rename(columns={'Time': 'timestamp'}, inplace=True)
                    all_events_df.reset_index(drop=True, inplace=True)

                    # Check if we have any valid trial data
                    has_start_initiation = 'InitiationSequence' in all_events_df.columns and any(all_events_df['InitiationSequence'])
                    has_end_initiation = 'EndInitiation' in all_events_df.columns and any(all_events_df['EndInitiation'])
                    
                    has_valve_events = (('r1_olf_valve' in all_events_df.columns and any(all_events_df['r1_olf_valve'])) or
                                        ('r2_olf_valve' in all_events_df.columns and any(all_events_df['r2_olf_valve'])) or 
                                        ('odourC_olf_valve' in all_events_df.columns and any(all_events_df['odourC_olf_valve'])) or 
                                        ('odourD_olf_valve' in all_events_df.columns and any(all_events_df['odourD_olf_valve'])) or 
                                        ('odourE_olf_valve' in all_events_df.columns and any(all_events_df['odourE_olf_valve'])) or 
                                        ('odourF_olf_valve' in all_events_df.columns and any(all_events_df['odourF_olf_valve'])) or 
                                        ('odourG_olf_valve' in all_events_df.columns and any(all_events_df['odourG_olf_valve'])) or 
                                        ('r1_olf_valve_off' in all_events_df.columns and any(all_events_df['r1_olf_valve_off'])) or 
                                        ('r2_olf_valve_off' in all_events_df.columns and any(all_events_df['r2_olf_valve_off'])) or 
                                        ('odourC_olf_valve_off' in all_events_df.columns and any(all_events_df['odourC_olf_valve_off'])) or 
                                        ('odourD_olf_valve_off' in all_events_df.columns and any(all_events_df['odourD_olf_valve_off'])) or 
                                        ('odourE_olf_valve_off' in all_events_df.columns and any(all_events_df['odourE_olf_valve_off'])) or 
                                        ('odourF_olf_valve_off' in all_events_df.columns and any(all_events_df['odourF_olf_valve_off'])) or 
                                        ('odourG_olf_valve_off' in all_events_df.columns and any(all_events_df['odourG_olf_valve_off'])) )
                    has_poke_events = (('r1_poke' in all_events_df.columns and any(all_events_df['r1_poke'])) or
                                      ('r2_poke' in all_events_df.columns and any(all_events_df['r2_poke'])) or
                                      ('odour_poke' in all_events_df.columns and any(all_events_df['odour_poke'])) or
                                      ('odour_poke_off' in all_events_df.columns and any(all_events_df['odour_poke_off'])))
                    
                    print(f"Data check: EndInitiation events: {has_end_initiation}, " 
                          f"InitiationSequence events: {has_start_initiation}, " 
                          f"Valve events: {has_valve_events}, Poke events: {has_poke_events}")
                    
                    if has_start_initiation and has_end_initiation and has_valve_events and has_poke_events:
                        # Calculate decision accuracy
                        session_data['accuracy_summary'] = calculate_overall_decision_accuracy(all_events_df)

                        # Calculate response time
                        session_data['response_time'] = calculate_overall_response_time(all_events_df)

                        # Calculate false alarm rate and bias (freerun and sequences)
                        if stage > 7:  
                            session_data['false_alarm'] = calculate_overall_false_alarm(all_events_df, odour_poke_df, odour_poke_off_df, session_schema)
                            session_data['false_alarm_bias'] = calculate_overall_false_alarm_bias(all_events_df, odour_poke_df, odour_poke_off_df, session_schema, odour_to_olfactometer_map)
                        
                        # Calculate sequence completion (sequences)
                        if stage >= 9:  
                            session_data['sequence_completion'] = calculate_overall_sequence_completion(all_events_df, odour_poke_df, odour_poke_off_df, session_schema)
                            
                        # Calculate decision sensitivity (freerun)
                        if stage >= 8.2 and stage < 9:  # TODO: update for later sequence stages
                            session_data['sensitivity'] = calculate_overall_decision_sensitivity(all_events_df)

                    else:
                        if not has_end_initiation:
                            print("No EndInitiation events found - cannot identify trial endings")
                        if not has_valve_events:
                            print("No valve activation events found - cannot identify trial types")
                        if not has_poke_events:
                            print("No poke events found - cannot identify animal responses")
                        
                        # Add empty accuracy data
                        session_data['accuracy_summary'] = {
                            'r1_respond': 0, 'r1_correct': 0, 'r1_accuracy': 0,
                            'r2_respond': 0, 'r2_correct': 0, 'r2_accuracy': 0,
                            'overall_accuracy': 0
                        }
                        # Add empty response time data
                        session_data['response_time'] = {'rt': [],
                            'r1_correct_rt': [], 'r1_incorrect_rt': [],
                            'r1_avg_correct_rt': [], 'r1_avg_incorrect_rt': [], 'r1_avg_rt': [],
                            'r2_correct_rt': [], 'r2_incorrect_rt': [],
                            'r2_avg_correct_rt': [], 'r2_avg_incorrect_rt': [], 'r2_avg_rt': [],
                            'hit_rt': [], 'false_alarm_rt': [], 'trial_id': np.array([]).reshape(-1, 2)
                        }
                        # Add empty false alarm data
                        session_data['false_alarm'] = {
                            'C_pokes': 0, 'C_trials': 0,
                            'D_pokes': 0, 'D_trials': 0,
                            'E_pokes': 0, 'E_trials': 0,
                            'F_pokes': 0, 'F_trials': 0,
                            'G_pokes': 0, 'G_trials': 0,
                            'C_false_alarm': 0, 'D_false_alarm': 0,
                            'E_false_alarm': 0, 'F_false_alarm': 0,
                            'G_false_alarm': 0, 'overall_false_alarm': 0
                        }
                        # Add empty false alarm bias data 
                        session_data['false_alarm_bias'] = {
                            'odour_interval_pokes': 0,
                            'odour_interval_trials': 0,
                            'odour_interval_false_alarm': 0,
                            'interval_pokes': 0,
                            'interval_trials': 0,
                            'interval_false_alarm': 0,
                            'odour_same_olf_pokes': 0, 
                            'odour_same_olf_trials': 0, 
                            'odour_same_olf_false_alarm': 0, 
                            'odour_diff_olf_pokes': 0, 
                            'odour_diff_olf_trials': 0, 
                            'odour_same_olf_false_alarm': 0, 
                            'same_olf_pokes': 0, 
                            'same_olf_trials': 0, 
                            'same_olf_false_alarm': 0, 
                            'diff_olf_pokes': 0, 
                            'diff_olf_trials': 0, 
                            'diff_olf_false_alarm': 0,
                            'odour_false_alarm_trials': 0,
                            'odour_same_olf_rew_pairing': 0,
                            'odour_diff_olf_rew_pairing': 0,
                            'same_olf_rew_pairing': 0,
                            'diff_olf_rew_pairing': 0,
                            'same_olf_rew_false_alarm': 0,
                            'diff_olf_rew_false_alarm': 0
                        }
                        # Add empty sequence completion data 
                        session_data['sequence_completion'] = {
                            'complete_sequences': 0,
                            'incomplete_sequences': 0,
                            'commited_sequences': 0,
                            'uncommited_sequences': 0,
                            'completion_ratio': 0,
                            'commitment_ratio': 0 
                        }
                        # Add empty sensitivity data
                        session_data['sensitivity'] = {'r1_total': 0,
                            'r1_respond': 0,
                            'r1_sensitivity': 0,
                            'r2_total': 0,
                            'r2_respond': 0,
                            'r2_sensitivity': 0,
                            'overall_sensitivity': 0
                        }
                
                except Exception as e:
                    print(f"Error processing event data: {e}")
                    # Add empty accuracy data
                    session_data['accuracy_summary'] = {
                        'r1_respond': 0, 'r1_correct': 0, 'r1_accuracy': 0,
                        'r2_respond': 0, 'r2_correct': 0, 'r2_accuracy': 0,
                        'overall_accuracy': 0
                    }
                    # Add empty response time data
                    session_data['response_time'] = {'rt': [],
                        'r1_correct_rt': [], 'r1_incorrect_rt': [],
                        'r1_avg_correct_rt': [], 'r1_avg_incorrect_rt': [], 'r1_avg_rt': [],
                        'r2_correct_rt': [], 'r2_incorrect_rt': [],
                        'r2_avg_correct_rt': [], 'r2_avg_incorrect_rt': [], 'r2_avg_rt': [],
                        'hit_rt': [], 'false_alarm_rt': [], 'trial_id': np.array([]).reshape(-1, 2)
                    }
                    # Add empty false alarm data
                    session_data['false_alarm'] = {
                        'C_pokes': 0, 'C_trials': 0,
                        'D_pokes': 0, 'D_trials': 0,
                        'E_pokes': 0, 'E_trials': 0,
                        'F_pokes': 0, 'F_trials': 0,
                        'G_pokes': 0, 'G_trials': 0,
                        'C_false_alarm': 0, 'D_false_alarm': 0,
                        'E_false_alarm': 0, 'F_false_alarm': 0,
                        'G_false_alarm': 0, 'overall_false_alarm': 0
                    }
                    # Add empty false alarm bias data 
                    session_data['false_alarm_bias'] = {
                        'odour_interval_pokes': 0,
                        'odour_interval_trials': 0,
                        'odour_interval_false_alarm': 0,
                        'interval_pokes': 0,
                        'interval_trials': 0,
                        'interval_false_alarm': 0,
                        'odour_same_olf_pokes': 0, 
                        'odour_same_olf_trials': 0, 
                        'odour_same_olf_false_alarm': 0, 
                        'odour_diff_olf_pokes': 0, 
                        'odour_diff_olf_trials': 0, 
                        'odour_same_olf_false_alarm': 0, 
                        'same_olf_pokes': 0, 
                        'same_olf_trials': 0, 
                        'same_olf_false_alarm': 0, 
                        'diff_olf_pokes': 0, 
                        'diff_olf_trials': 0, 
                        'diff_olf_false_alarm': 0,
                        'odour_false_alarm_trials': 0,
                        'odour_same_olf_rew_pairing': 0,
                        'odour_diff_olf_rew_pairing': 0,
                        'same_olf_rew_pairing': 0,
                        'diff_olf_rew_pairing': 0,
                        'same_olf_rew_false_alarm': 0,
                        'diff_olf_rew_false_alarm': 0
                    }
                    # Add empty sequence completion data 
                    session_data['sequence_completion'] = {
                        'complete_sequences': 0,
                        'incomplete_sequences': 0,
                        'commited_sequences': 0,
                        'uncommited_sequences': 0,
                        'completion_ratio': 0,
                        'commitment_ratio': 0 
                    }
                    # Add empty sensitivity data
                    session_data['sensitivity'] = {'r1_total': 0,
                        'r1_respond': 0,
                        'r1_sensitivity': 0,
                        'r2_total': 0,
                        'r2_respond': 0,
                        'r2_sensitivity': 0,
                        'overall_sensitivity': 0
                    }
            else:
                print("No events available for decision accuracy, response time, false alarm rate and specificity calculation")
                # Add empty accuracy data
                session_data['accuracy_summary'] = {
                    'r1_respond': 0, 'r1_correct': 0, 'r1_accuracy': 0,
                    'r2_respond': 0, 'r2_correct': 0, 'r2_accuracy': 0,
                    'overall_accuracy': 0
                }
                # Add empty response time data
                session_data['response_time'] = {'rt': [],
                    'r1_correct_rt': [], 'r1_incorrect_rt': [],
                    'r1_avg_correct_rt': [], 'r1_avg_incorrect_rt': [], 'r1_avg_rt': [],
                    'r2_correct_rt': [], 'r2_incorrect_rt': [],
                    'r2_avg_correct_rt': [], 'r2_avg_incorrect_rt': [], 'r2_avg_rt': [],
                    'hit_rt': [], 'false_alarm_rt': [], 'trial_id': np.array([]).reshape(-1, 2)
                }
                # Add empty false alarm data
                session_data['false_alarm'] = {
                    'C_pokes': 0, 'C_trials': 0,
                    'D_pokes': 0, 'D_trials': 0,
                    'E_pokes': 0, 'E_trials': 0,
                    'F_pokes': 0, 'F_trials': 0,
                    'G_pokes': 0, 'G_trials': 0,
                    'C_false_alarm': 0, 'D_false_alarm': 0,
                    'E_false_alarm': 0, 'F_false_alarm': 0,
                    'G_false_alarm': 0, 'overall_false_alarm': 0
                }
                # Add empty false alarm bias data 
                session_data['false_alarm_bias'] = {
                    'odour_interval_pokes': 0,
                    'odour_interval_trials': 0,
                    'odour_interval_false_alarm': 0,
                    'interval_pokes': 0,
                    'interval_trials': 0,
                    'interval_false_alarm': 0,
                    'odour_same_olf_pokes': 0, 
                    'odour_same_olf_trials': 0, 
                    'odour_same_olf_false_alarm': 0, 
                    'odour_diff_olf_pokes': 0, 
                    'odour_diff_olf_trials': 0, 
                    'odour_same_olf_false_alarm': 0, 
                    'same_olf_pokes': 0, 
                    'same_olf_trials': 0, 
                    'same_olf_false_alarm': 0, 
                    'diff_olf_pokes': 0, 
                    'diff_olf_trials': 0, 
                    'diff_olf_false_alarm': 0,
                    'odour_false_alarm_trials': 0,
                    'odour_same_olf_rew_pairing': 0,
                    'odour_diff_olf_rew_pairing': 0,
                    'same_olf_rew_pairing': 0,
                    'diff_olf_rew_pairing': 0,
                    'same_olf_rew_false_alarm': 0,
                    'diff_olf_rew_false_alarm': 0
                }
                # Add empty sequence completion data 
                session_data['sequence_completion'] = {
                    'complete_sequences': 0,
                    'incomplete_sequences': 0,
                    'commited_sequences': 0,
                    'uncommited_sequences': 0,
                    'completion_ratio': 0,
                    'commitment_ratio': 0 
                }
                # Add empty sensitivity data
                session_data['sensitivity'] = {'r1_total': 0,
                    'r1_respond': 0,
                    'r1_sensitivity': 0,
                    'r2_total': 0,
                    'r2_respond': 0,
                    'r2_sensitivity': 0,
                    'overall_sensitivity': 0
                }
        
        except Exception as e:
            print(f"Error during session data processing: {e}")
            # Default session data with zeros
            session_data = {
                'num_r1_rewards': 0,
                'num_r2_rewards': 0,
                'session_duration_sec': 0,
                'accuracy_summary': {
                    'r1_respond': 0, 'r1_correct': 0, 'r1_accuracy': 0,
                    'r2_respond': 0, 'r2_correct': 0, 'r2_accuracy': 0,
                    'overall_accuracy': 0
                },
                'response_time': {'rt': [],
                    'r1_correct_rt': [], 'r1_incorrect_rt': [],
                    'r1_avg_correct_rt': [], 'r1_avg_incorrect_rt': [], 'r1_avg_rt': [],
                    'r2_correct_rt': [], 'r2_incorrect_rt': [],
                    'r2_avg_correct_rt': [], 'r2_avg_incorrect_rt': [], 'r2_avg_rt': [],
                    'hit_rt': [], 'false_alarm_rt': [], 'trial_id': np.array([]).reshape(-1, 2)
                },
                'false_alarm': {               
                    'C_pokes': 0, 'C_trials': 0,
                    'D_pokes': 0, 'D_trials': 0,
                    'E_pokes': 0, 'E_trials': 0,
                    'F_pokes': 0, 'F_trials': 0,
                    'G_pokes': 0, 'G_trials': 0,
                    'C_false_alarm': 0, 'D_false_alarm': 0,
                    'E_false_alarm': 0, 'F_false_alarm': 0,
                    'G_false_alarm': 0, 'overall_false_alarm': 0
                },
                'false_alarm_bias': {
                    'odour_interval_pokes': 0,
                    'odour_interval_trials': 0,
                    'odour_interval_false_alarm': 0,
                    'interval_pokes': 0,
                    'interval_trials': 0,
                    'interval_false_alarm': 0, 
                    'odour_same_olf_pokes': 0, 
                    'odour_same_olf_trials': 0, 
                    'odour_same_olf_false_alarm': 0, 
                    'odour_diff_olf_pokes': 0, 
                    'odour_diff_olf_trials': 0, 
                    'odour_same_olf_false_alarm': 0, 
                    'same_olf_pokes': 0, 
                    'same_olf_trials': 0, 
                    'same_olf_false_alarm': 0, 
                    'diff_olf_pokes': 0, 
                    'diff_olf_trials': 0, 
                    'diff_olf_false_alarm': 0,
                    'odour_false_alarm_trials': 0,
                    'odour_same_olf_rew_pairing': 0,
                    'odour_diff_olf_rew_pairing': 0,
                    'same_olf_rew_pairing': 0,
                    'diff_olf_rew_pairing': 0,
                    'same_olf_rew_false_alarm': 0,
                    'diff_olf_rew_false_alarm': 0
                },
                'sequence_completion': {
                    'complete_sequences': 0,
                    'incomplete_sequences': 0,
                    'commited_sequences': 0,
                    'uncommited_sequences': 0,
                    'completion_ratio': 0,
                    'commitment_ratio': 0 
                },
                'sensitivity': {'r1_total': 0,
                    'r1_respond': 0,
                    'r1_sensitivity': 0,
                    'r2_total': 0,
                    'r2_respond': 0,
                    'r2_sensitivity': 0,
                    'overall_sensitivity': 0
                }
            }
        
        return session_data

    @staticmethod
    def get_decision_accuracy(data_path):
        """
        Static method to calculate decision accuracy for a single session.
        
        Parameters:
        -----------
        data_path : str or Path
            Path to session data directory
            
        Returns:
        --------
        dict
            Dictionary with accuracy metrics or None if calculation fails
        """
        root = Path(data_path)
        
        # Process the given directory directly
        print(f"Processing decision accuracy for: {root}")
        
        # Create a temporary instance to access the _get_session_data method
        temp_instance = RewardAnalyser.__new__(RewardAnalyser)
        session_data = temp_instance._get_session_data(root)
        
        # Return just the accuracy summary
        return session_data.get('accuracy_summary', {
            'r1_respond': 0, 'r1_correct': 0, 'r1_accuracy': 0,
            'r2_respond': 0, 'r2_correct': 0, 'r2_accuracy': 0,
            'overall_accuracy': 0
        })

    @staticmethod
    def get_false_alarm(data_path):
        """
        Static method to calculate false alarms for each trial in a single session.
        
        Parameters:
        -----------
        data_path : str or Path
            Path to session data directory
            
        Returns:
        --------
        dict
            Dictionary with false alarm metrics or None if calculation fails
        """
        root = Path(data_path)
        
        # Process the given directory directly
        print(f"Processing false alarms for: {root}")
        
        # Create a temporary instance to access the _get_session_data method
        temp_instance = RewardAnalyser.__new__(RewardAnalyser)
        session_data = temp_instance._get_session_data(root)

        return session_data.get('false_alarm', {'C_pokes': 0, 'C_trials': 0,
                                                'D_pokes': 0, 'D_trials': 0, 'E_pokes': 0, 'E_trials': 0,
                                                'F_pokes': 0, 'F_trials': 0, 'G_pokes': 0, 'G_trials': 0,
                                                'C_false_alarm': 0,
                                                'D_false_alarm': 0, 'E_false_alarm': 0,
                                                'F_false_alarm': 0, 'G_false_alarm': 0, 
                                                'overall_false_alarm': 0
                                            })
    
    @staticmethod
    def get_false_alarm_bias(data_path):
        """
        Static method to calculate false alarm bias for non-rewarded trials in a single session.
        
        Parameters:
        -----------
        data_path : str or Path
            Path to session data directory
            
        Returns:
        --------
        dict
            Dictionary with false alarm bias metrics or None if calculation fails
        """
        root = Path(data_path)
        
        # Process the given directory directly
        print(f"Processing false alarm bias for: {root}")
        
        # Create a temporary instance to access the _get_session_data method
        temp_instance = RewardAnalyser.__new__(RewardAnalyser)
        session_data = temp_instance._get_session_data(root)

        return session_data.get('false_alarm_bias', {'odour_interval_pokes': 0,
                                                    'odour_interval_trials': 0,
                                                    'odour_interval_false_alarm': 0,
                                                    'interval_pokes': 0,
                                                    'interval_trials': 0,
                                                    'interval_false_alarm': 0,
                                                    'odour_same_olf_pokes': 0, 
                                                    'odour_same_olf_trials': 0, 
                                                    'odour_same_olf_false_alarm': 0, 
                                                    'odour_diff_olf_pokes': 0, 
                                                    'odour_diff_olf_trials': 0, 
                                                    'odour_same_olf_false_alarm': 0, 
                                                    'same_olf_pokes': 0, 
                                                    'same_olf_trials': 0, 
                                                    'same_olf_false_alarm': 0, 
                                                    'diff_olf_pokes': 0, 
                                                    'diff_olf_trials': 0, 
                                                    'diff_olf_false_alarm': 0,
                                                    'odour_false_alarm_trials': 0,
                                                    'odour_same_olf_rew_pairing': 0,
                                                    'odour_diff_olf_rew_pairing': 0,
                                                    'same_olf_rew_pairing': 0,
                                                    'diff_olf_rew_pairing': 0,
                                                    'same_olf_rew_false_alarm': 0,
                                                    'diff_olf_rew_false_alarm': 0 
                                                })
    
    @staticmethod
    def get_decision_sensitivity(data_path):
        """
        Static method to calculate decision sensitivity for a single session.
        
        Parameters:
        -----------
        data_path : str or Path
            Path to session data directory
            
        Returns:
        --------
        dict
            Dictionary with sensitivity metrics or None if calculation fails
        """
        root = Path(data_path)
        
        # Process the given directory directly
        print(f"Processing decision sensitivity for: {root}")
        
        # Create a temporary instance to access the _get_session_data method
        temp_instance = RewardAnalyser.__new__(RewardAnalyser)
        session_data = temp_instance._get_session_data(root)
        
        # Return just the sensitivity summary
        return session_data.get('sensitivity', {'r1_total': 0, 'r2_total': 0,
                            'r1_respond': 0, 'r1_sensitivity': 0,
                            'r2_respond': 0, 'r2_sensitivity': 0,
                            'overall_sensitivity': 0
            })

    @staticmethod
    def get_response_time(data_path):
        """
        Static method to calculate response time for each trial in a single session.
        
        Parameters:
        -----------
        data_path : str or Path
            Path to session data directory
            
        Returns:
        --------
        dict
            Dictionary with response time metrics or None if calculation fails
        """
        root = Path(data_path)
        
        # Process the given directory directly
        print(f"Processing response time for: {root}")
        
        # Create a temporary instance to access the _get_session_data method
        temp_instance = RewardAnalyser.__new__(RewardAnalyser)
        session_data = temp_instance._get_session_data(root)
        
        # Return just the response time summary
        return session_data.get('response_time', {'rt': [],
                        'r1_correct_rt': [], 'r1_incorrect_rt': [],
                        'r1_avg_correct_rt': [], 'r1_avg_incorrect_rt': [], 'r1_avg_rt': [],
                        'r2_correct_rt': [], 'r2_incorrect_rt': [],
                        'r2_avg_correct_rt': [], 'r2_avg_incorrect_rt': [], 'r2_avg_rt': [],
                        'hit_rt': [], 'false_alarm_rt': [], 'trial_id': np.array([]).reshape(-1, 2)
                    })

    @staticmethod
    def get_sequence_completion(data_path):
        """
        Static method to calculate sequence completion for each trial in a single session.
        
        Parameters:
        -----------
        data_path : str or Path
            Path to session data directory
            
        Returns:
        --------
        dict
            Dictionary with sequence completion metrics or None if calculation fails
        """
        root = Path(data_path)
        
        # Process the given directory directly
        print(f"Processing false alarms for: {root}")
        
        # Create a temporary instance to access the _get_session_data method
        temp_instance = RewardAnalyser.__new__(RewardAnalyser)
        session_data = temp_instance._get_session_data(root)
              
        return session_data.get('sequence_completion', {'complete_sequences': 0, 'incomplete_sequences': 0, 'commited_sequences': 0, 'uncommited_sequences': 0, 'completion_ratio': 0, 'commitment_ratio': 0})
    
    def _detect_stage(self):
        """
        Extracts the stage from metadata if available.
        Handles nested structure of sequences in metadata.
        """
        stage_found = None
        sequences = self.session_settings.iloc[0]['metadata'].sequences
        
        # Handle the nested list structure
        if isinstance(sequences, list):
            # Iterate through outer list
            for seq_group in sequences:
                if isinstance(seq_group, list):
                    # Iterate through inner list
                    for seq in seq_group:
                        if isinstance(seq, dict) and 'name' in seq:
                            print(f"Found sequence name: {seq['name']}")
                            match = re.search(r'_Stage(\d+)', seq['name'], re.IGNORECASE)
                            if match:
                                if 'FreeRun' in seq['name']:
                                    stage_number = int(match.group(1))
                                    stage_found = 8 + 0.1 * stage_number
                                    # stage_found = int(match.group(1)) + 7
                                elif 'Doubles' in seq['name']:
                                    stage_found = 9
                                elif 'Triples' in seq['name']:
                                    stage_found = 10
                                elif 'Quadruple' in seq['name']:
                                    stage_found = 11
                                else:
                                    stage_found = match.group(1)
                                return stage_found
                            else:
                                if 'Doubles' in seq['name']:
                                    stage_found = 9
                elif isinstance(seq_group, dict) and 'name' in seq_group:
                    # Handle case where outer list contains dicts directly
                    print(f"Found sequence name: {seq_group['name']}")
                    match = re.search(r'_Stage(\d+)', seq_group['name'], re.IGNORECASE)
                    if match:
                        if 'FreeRun' in seq_group['name']:
                            stage_number = int(match.group(1))
                            stage_found = 8 + 0.1 * stage_number
                            # stage_found = int(match.group(1)) + 7
                        elif 'Doubles' in seq_group['name']:
                            stage_found = 9
                        elif 'Triples' in seq_group['name']:
                            stage_found = 10
                        elif 'Quadruple' in seq_group['name']:
                            stage_found = 11
                        else:
                            stage_found = match.group(1)
                        return stage_found
                    else:
                        if 'Doubles' in seq_group['name']:
                            stage_found = 9
                            return stage_found
            
        print(f"Final stage detected: {stage_found}")
        return stage_found if stage_found else "Unknown"

    def run(self, data_path, reward_a=8.0, reward_b=8.0):
        """
        Run the appropriate reward analysis based on stage detection.
        """
        stage = self._detect_stage()       
        if stage == "1":
            print("Running stage 1 analyzer\n")
            self._reward_analyser_stage1(data_path, reward_a, reward_b)
        else:
            print(f"Running stage {stage} analyzer (stage 2-8)\n")
            self._reward_analyser_stage2to8(data_path, reward_a, reward_b)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine Stage1 & Stage2-8 Reward Analysis.")
    parser.add_argument("data_path", help="Path to the root folder containing behavior data.")
    parser.add_argument("--reward_a", type=float, default=8.0, help="Volume (µL) per Reward A.")
    parser.add_argument("--reward_b", type=float, default=8.0, help="Volume (µL) per Reward B.")
    args = parser.parse_args()

    print("Please instantiate RewardAnalyser with your session_settings and call .run(data_path, reward_a, reward_b).")

# Add module-level function to expose the static method
def get_decision_accuracy(data_path):
    """
    Calculate decision accuracy for a single session.
    
    Parameters:
    -----------
    data_path : str or Path
        Path to session data directory
        
    Returns:
    --------
    dict
        Dictionary with accuracy metrics or None if calculation fails
    """
    return RewardAnalyser.get_decision_accuracy(data_path)

def get_response_time(data_path):
    """
    Calculate response time for each trial.
    
    Parameters:
    -----------
    data_path : str or Path
        Path to session data directory
        
    Returns:
    --------
    dict
        Dictionary with response time metrics or None if calculation fails
    """
    return RewardAnalyser.get_response_time(data_path)

def get_false_alarm(data_path):
    """
    Calculate false alarms for each trial in a single session.
    
    Parameters:
    -----------
    data_path : str or Path
        Path to session data directory
        
    Returns:
    --------
    dict
        Dictionary with false alarm metrics or None if calculation fails
    """
    return RewardAnalyser.get_false_alarm(data_path)

def get_false_alarm_bias(data_path):
    """
    Calculate false alarm bias for each trial in a single session.
    
    Parameters:
    -----------
    data_path : str or Path
        Path to session data directory
        
    Returns:
    --------
    dict
        Dictionary with false alarm bias metrics or None if calculation fails
    """
    return RewardAnalyser.get_false_alarm_bias(data_path)

def get_sequence_completion(data_path):
    """
    Calculate sequence completion for each trial in a single session.
    
    Parameters:
    -----------
    data_path : str or Path
        Path to session data directory
        
    Returns:
    --------
    dict
        Dictionary with sequence completion metrics or None if calculation fails
    """
    return RewardAnalyser.get_sequence_completion(data_path)

def get_decision_sensitivity(data_path):
    """
    Calculate decision sensitivity for a single session.
    
    Parameters:
    -----------
    data_path : str or Path
        Path to session data directory
        
    Returns:
    --------
    dict
        Dictionary with sensitivity metrics or None if calculation fails
    """
    return RewardAnalyser.get_decision_sensitivity(data_path)

def get_decision_specificity(data_path):
    """
    Calculate decision specificity for a single session.
    
    Parameters:
    -----------
    data_path : str or Path
        Path to session data directory
        
    Returns:
    --------
    dict
        Dictionary with specificity metrics or None if calculation fails
    """
    return RewardAnalyser.get_decision_specificity(data_path)

def calculate_overall_decision_accuracy(events_df):
    """
    Calculate decision accuracy for r1/r2 trials.
    
    Parameters:
    -----------
    events_df : pandas.DataFrame
        DataFrame containing trial events with columns: 'timestamp', 'r1_poke', 'r2_poke', 'r1_olf_valve', 'r2_olf_valve', 
                                            'odourC_olf_valve', 'odourD_olf_valve', 'odourE_olf_valve', 'odourF_olf_valve', 
                                                'odourG_olf_valve', 'r1_olf_valve_off', 'r2_olf_valve_off', 'odourC_olf_valve_off', 
                                                    'odourD_olf_valve_off', 'odourE_olf_valve_off', 'odourF_olf_valve_off', 
                                                        'odourG_olf_valve_off', 'EndInitiation', 'odour_poke', 'odour_poke_off',
                                                        'PulseSupplyPort1', 'PulseSupplyPort2'
        
    Returns:
    --------
    dict
        Dictionary containing accuracy metrics for r1, r2, and overall trials
    """
    # Ensure events are in chronological order
    events_df = events_df.sort_values('timestamp').reset_index(drop=True)
    
    # Initialize counters
    r1_correct = 0
    r1_respond = 0
    r2_correct = 0
    r2_respond = 0
    
    # Find all trial end points
    end_initiation_indices = events_df.index[events_df['EndInitiation'] == True].tolist()

    # Process each trial
    for e in range(len(end_initiation_indices)):
        end_idx = end_initiation_indices[e]
        if e == len(end_initiation_indices) - 1:
            next_end_idx = len(events_df)
        else:
            next_end_idx = end_initiation_indices[e + 1]
        
        # Determine trial type (r1 or r2) by finding the most recent valve activation
        closest_valve_idx = None
        trial_type = None
        for i in range(end_idx - 1, -1, -1):
            if events_df.loc[i, 'r1_olf_valve']:
                closest_valve_idx = i
                trial_type = 'r1'
                break
            elif events_df.loc[i, 'r2_olf_valve']:
                closest_valve_idx = i
                trial_type = 'r2'
                break
            elif ('odourC_olf_valve' in events_df and events_df.loc[i, 'odourC_olf_valve']) or \
                    ('odourD_olf_valve' in events_df and events_df.loc[i, 'odourD_olf_valve']) or \
                    ('odourE_olf_valve' in events_df and events_df.loc[i, 'odourE_olf_valve']) or \
                    ('odourF_olf_valve' in events_df and events_df.loc[i, 'odourF_olf_valve']) or \
                    ('odourG_olf_valve' in events_df and events_df.loc[i, 'odourG_olf_valve']):
                closest_valve_idx = i
                trial_type = 'nonR'
                break
                
        # Skip if no valve activation found before this trial end or trial is non-rewarded
        if trial_type == 'nonR' or closest_valve_idx is None:
            continue
            
        # Find the first poke after trial end
        for j in range(end_idx + 1, next_end_idx):
            if events_df.loc[j, ['r1_poke', 'r2_poke']].any():
                # Count response trials
                if trial_type == 'r1':
                    r1_respond += 1
                elif trial_type == 'r2':
                    r2_respond += 1

                # Correct if poke matches trial type and reward is delivered
                if trial_type == 'r1' and events_df.loc[j, 'r1_poke']:
                    for k in range(j + 1, next_end_idx):
                        if events_df.loc[k, 'PulseSupplyPort1']:
                            r1_correct += 1
                            break 
                elif trial_type == 'r2' and events_df.loc[j, 'r2_poke']:
                    for k in range(j + 1, next_end_idx):
                        if events_df.loc[k, 'PulseSupplyPort2']:
                            r2_correct += 1
                            break 
                break

    # Calculate accuracy percentages with safety checks for division by zero
    r1_accuracy = (r1_correct / r1_respond * 100) if r1_respond > 0 else 0
    r2_accuracy = (r2_correct / r2_respond * 100) if r2_respond > 0 else 0
    overall_accuracy = ((r1_correct + r2_correct) / (r1_respond + r2_respond) * 100) if (r1_respond + r2_respond) > 0 else 0
    
    # Return detailed accuracy metrics
    return {
        'r1_respond': r1_respond,
        'r1_correct': r1_correct,
        'r1_accuracy': r1_accuracy,
        'r2_respond': r2_respond,
        'r2_correct': r2_correct,
        'r2_accuracy': r2_accuracy,
        'overall_accuracy': overall_accuracy
    }

def calculate_overall_response_time(events_df):
    """
    Calculate response time for r1/r2 trials.
    
    Parameters:
    -----------
    events_df : pandas.DataFrame
        DataFrame containing trial events with columns: 'timestamp', 'EndInitiation',
        'r1_olf_valve', 'r2_olf_valve', 'r1_poke', 'r2_poke'
        
    Returns:
    --------
    dict
        Dictionary containing response time metrics for r1, r2, and overall trials
    """
    # Ensure events are in chronological order
    events_df = events_df.sort_values('timestamp').reset_index(drop=True)

    # Initialize counters
    r1_correct = 0
    r1_total = 0
    r2_correct = 0
    r2_total = 0
    
    # Find all trial end points
    end_initiation_indices = events_df.index[events_df['EndInitiation'] == True].tolist()
    trial_id = np.zeros((len(end_initiation_indices), 2))

    # Process each trial
    rt = []
    r1_correct_rt = []
    r1_incorrect_rt = []
    r2_correct_rt = []
    r2_incorrect_rt = []
    for e, end_idx in enumerate(end_initiation_indices):
        # Determine trial type (r1 or r2 or nonR) by finding the most recent valve activation
        closest_valve_idx = None
        trial_type = None
        for i in range(end_idx - 1, -1, -1):  
            if events_df.loc[i, 'r1_olf_valve']:
                closest_valve_idx = i
                trial_type = 'r1'
                trial_id[e,0] = 1
                break
            elif events_df.loc[i, 'r2_olf_valve']:
                closest_valve_idx = i
                trial_type = 'r2'
                trial_id[e,0] = 2
                break
            elif ('odourC_olf_valve' in events_df and events_df.loc[i, 'odourC_olf_valve']) or \
                    ('odourD_olf_valve' in events_df and events_df.loc[i, 'odourD_olf_valve']) or \
                    ('odourE_olf_valve' in events_df and events_df.loc[i, 'odourE_olf_valve']) or \
                    ('odourF_olf_valve' in events_df and events_df.loc[i, 'odourF_olf_valve']) or \
                    ('odourG_olf_valve' in events_df and events_df.loc[i, 'odourG_olf_valve']):
                closest_valve_idx = i
                trial_type = 'nonR'
                break

        # Skip if no valve activation found before this trial end
        if trial_type == 'nonR' or closest_valve_idx is None:
            continue
        else:
            valve_time = events_df.loc[closest_valve_idx, 'timestamp']

        # Count trial by type
        if trial_type == 'r1':
            r1_total += 1
        elif trial_type == 'r2':
            r2_total += 1
            
        # Find the odour port offset 
        closest_offset_idx = None
        for k in range(end_idx - 1, -1, -1): 
            if events_df.loc[k, 'odour_poke_off']:
                closest_offset_idx = k
                break
        
        if closest_offset_idx is None:
            continue
        else:
            offset_time = events_df.loc[closest_offset_idx, 'timestamp']

        # Find the first poke after trial end
        closest_poke_idx = None
        for j in range(end_idx + 1, len(events_df)):
            if events_df.loc[j, 'r1_poke'] or events_df.loc[j, 'r2_poke']:
                
                # Calculate response time for each trial 
                poke_time = events_df.loc[j, 'timestamp']
                response_time = (poke_time - offset_time).total_seconds()
                rt.append(response_time)

                # Correct if poke matches trial type
                if trial_type == 'r1' and events_df.loc[j, 'r1_poke']:
                    closest_poke_idx = j
                    r1_correct += 1
                    r1_correct_rt.append(response_time)
                    trial_id[e,1] = 1
                elif trial_type == 'r1' and events_df.loc[j, 'r2_poke']:
                    r1_incorrect_rt.append(response_time)
                    trial_id[e,1] = 2
                elif trial_type == 'r2' and events_df.loc[j, 'r2_poke']:
                    closest_poke_idx = j
                    r2_correct += 1
                    r2_correct_rt.append(response_time)
                    trial_id[e,1] = 2
                elif trial_type == 'r2' and events_df.loc[j, 'r1_poke']:
                    r2_incorrect_rt.append(response_time)
                    trial_id[e,1] = 1
                break
        
        if closest_poke_idx is None: # TODO
            continue

    # Calculate overall response time 
    r1_avg_correct_rt = np.mean(r1_correct_rt)
    r1_avg_incorrect_rt = np.mean(r1_incorrect_rt)
    r1_avg_rt = np.mean(np.concatenate([r1_correct_rt, r1_incorrect_rt]))

    r2_avg_correct_rt = np.mean(r2_correct_rt)
    r2_avg_incorrect_rt = np.mean(r2_incorrect_rt)
    r2_avg_rt = np.mean(np.concatenate([r2_correct_rt, r2_incorrect_rt]))

    hit_rt = np.mean(np.concatenate([r1_correct_rt, r2_correct_rt]))
    false_alarm_rt = np.mean(np.concatenate([r1_incorrect_rt, r2_incorrect_rt]))
    
    # Return detailed response time metrics
    return {
        'rt': rt, 
        'r1_correct_rt': r1_correct_rt,
        'r1_incorrect_rt': r1_incorrect_rt,
        'r1_avg_correct_rt': r1_avg_correct_rt,
        'r1_avg_incorrect_rt': r1_avg_incorrect_rt,
        'r1_avg_rt': r1_avg_rt,
        'r2_correct_rt': r2_correct_rt,
        'r2_incorrect_rt': r2_incorrect_rt,
        'r2_avg_correct_rt': r2_avg_correct_rt,
        'r2_avg_incorrect_rt': r2_avg_incorrect_rt,
        'r2_avg_rt': r2_avg_rt,
        'hit_rt': hit_rt,
        'false_alarm_rt': false_alarm_rt,
        'trial_id': trial_id
    }

def calculate_overall_false_alarm(events_df, odour_poke_df, odour_poke_off_df, session_schema): 
    """
    Calculate false alarm rate for non-rewarded trials. Odours are considered if
    the poke is at least as long as the minimum sampling time. 
    
    Parameters:
    -----------
    events_df : pandas.DataFrame
        DataFrame containing trial events with columns: 'timestamp', 'r1_poke', 'r2_poke', 'r1_olf_valve', 'r2_olf_valve', 
                                            'odourC_olf_valve', 'odourD_olf_valve', 'odourE_olf_valve', 'odourF_olf_valve', 
                                                'odourG_olf_valve', 'r1_olf_valve_off', 'r2_olf_valve_off', 'odourC_olf_valve_off', 
                                                    'odourD_olf_valve_off', 'odourE_olf_valve_off', 'odourF_olf_valve_off', 
                                                        'odourG_olf_valve_off', 'EndInitiation', 'odour_poke', 'odour_poke_off',
                                                        'PulseSupplyPort1', 'PulseSupplyPort2'
        
    Returns:
    --------
    dict
        Dictionary containing false alarm metrics 
    """
    # Define useful variables
    olf_valve_cols = ['odourC_olf_valve', 'odourD_olf_valve', 'odourE_olf_valve', 'odourF_olf_valve', 'odourG_olf_valve']
    rew_valve_cols = ['r1_olf_valve', 'r2_olf_valve']
    minimumSamplingTime = session_schema['minimumSamplingTime']
    sampleOffsetTime = session_schema['sampleOffsetTime']
    
    # Collect all poke onset and offset events
    odour_poke_events_df = get_odour_poke_df(odour_poke_df, odour_poke_off_df)

    # Initialize counters
    C_total = C1_poke = C2_poke = 0
    D_total = D1_poke = D2_poke = 0
    E_total = E1_poke = E2_poke = 0
    F_total = F1_poke = F2_poke = 0
    G_total = G1_poke = G2_poke = 0
    
    # Find all trial end points
    end_initiation_indices = events_df.index[events_df['EndInitiation'] == True].tolist()

    # Process each sequence
    for e in range(len(end_initiation_indices)):
        end_idx = end_initiation_indices[e]
        if e == len(end_initiation_indices) - 1:
            next_end_idx = len(events_df)
        else:
            next_end_idx = end_initiation_indices[e + 1]

        if e == 0:
            prev_end_idx = 0
        else:
            prev_end_idx = end_initiation_indices[e - 1]
        
        for i in range(prev_end_idx, end_idx):
            valid_odour = False
            if events_df.loc[i, olf_valve_cols].any():
                # Determine if an odour is valid based on sampling
                valid_odour = is_odour_valid(i, events_df, odour_poke_events_df, sampleOffsetTime, minimumSamplingTime)
                if not valid_odour: 
                    continue

            # Process valid odours only
            if valid_odour: 
                k = i 
                odour = next((od for od in olf_valve_cols if events_df.loc[k, od]), None)

                # Determine trial type  
                if odour == 'odourC_olf_valve': C_total += 1
                elif odour == 'odourD_olf_valve': D_total += 1
                elif odour == 'odourE_olf_valve': E_total += 1
                elif odour == 'odourF_olf_valve': F_total += 1
                elif odour == 'odourG_olf_valve': G_total += 1 

                # Determine false alarms or correct rejections
                for l in range(k + 1, next_end_idx):
                    if events_df.loc[l, ['r1_poke', 'r2_poke']].any():  # false alarm
                        poke = 'r1_poke' if events_df.loc[l, 'r1_poke'] else 'r2_poke'
                        if odour == 'odourC_olf_valve':
                            C1_poke += (poke == 'r1_poke')
                            C2_poke += (poke == 'r2_poke')
                        elif odour == 'odourD_olf_valve':
                            D1_poke += (poke == 'r1_poke')
                            D2_poke += (poke == 'r2_poke')
                        elif odour == 'odourE_olf_valve':
                            E1_poke += (poke == 'r1_poke')
                            E2_poke += (poke == 'r2_poke')
                        elif odour == 'odourF_olf_valve':
                            F1_poke += (poke == 'r1_poke')
                            F2_poke += (poke == 'r2_poke')
                        elif odour == 'odourG_olf_valve':
                            G1_poke += (poke == 'r1_poke')
                            G2_poke += (poke == 'r2_poke')

                        k = l + 1
                        break
                        
                    elif events_df.loc[l, olf_valve_cols + rew_valve_cols].any():  # correct rejection
                        k = l 
                        break
                else:
                    k += 1

    # Calculate false alarm rate with safety checks for division by zero
    C_false_alarm = ((C1_poke + C2_poke) / C_total * 100) if C_total else 0
    D_false_alarm = ((D1_poke + D2_poke) / D_total * 100) if D_total else 0
    E_false_alarm = ((E1_poke + E2_poke) / E_total * 100) if E_total else 0
    F_false_alarm = ((F1_poke + F2_poke) / F_total * 100) if F_total else 0
    G_false_alarm = ((G1_poke + G2_poke) / G_total * 100) if G_total else 0

    nonR_pokes = C1_poke + C2_poke + D1_poke + D2_poke + E1_poke + E2_poke + F1_poke + F2_poke + G1_poke + G2_poke
    nonR_trials = C_total + D_total + E_total + F_total + G_total
    overall_false_alarm = (nonR_pokes / nonR_trials * 100) if nonR_trials > 0 else 0

    return {
        'C_pokes': C1_poke + C2_poke,
        'C_trials': C_total,
        'D_pokes': D1_poke + D2_poke,
        'D_trials': D_total,
        'E_pokes': E1_poke + E2_poke,
        'E_trials': E_total,
        'F_pokes': F1_poke + F2_poke,
        'F_trials': F_total,
        'G_pokes': G1_poke + G2_poke,
        'G_trials': G_total,
        'C_false_alarm': C_false_alarm,
        'D_false_alarm': D_false_alarm,
        'E_false_alarm': E_false_alarm,
        'F_false_alarm': F_false_alarm,
        'G_false_alarm': G_false_alarm,
        'overall_false_alarm': overall_false_alarm
    }

def calculate_overall_sequence_completion(events_df, odour_poke_df, odour_poke_off_df, session_schema):
    """
    Calculate sequence completion i.e. how many times a sequences is completed vs how many
    times a sequence is initiated from odour 1. 
    Calculate sequence commitment i.e. how many times the mouse attempts a preliminary reward
    before succesfully completing a sequence (calculated only if a full sequence must be completed 
    before the next one can start).
    
    Parameters:
    -----------
    events_df : pandas.DataFrame
        DataFrame containing trial events with columns: 'timestamp', 'r1_poke', 'r2_poke', 'r1_olf_valve', 'r2_olf_valve', 
                                            'odourC_olf_valve', 'odourD_olf_valve', 'odourE_olf_valve', 'odourF_olf_valve', 
                                                'odourG_olf_valve', 'r1_olf_valve_off', 'r2_olf_valve_off', 'odourC_olf_valve_off', 
                                                    'odourD_olf_valve_off', 'odourE_olf_valve_off', 'odourF_olf_valve_off', 
                                                        'odourG_olf_valve_off', 'EndInitiation', 'odour_poke', 'odour_poke_off', 
                                                        'PulseSupplyPort1', 'PulseSupplyPort2'
        
    Returns:
    --------
    dict
        Dictionary containing sequence completion and commitment metrics 
    """
    # TODO: Add the case when completionRequiresEngagement == False?

    # Define useful variables
    olf_valve_cols = ['r1_olf_valve', 'r2_olf_valve', 'odourC_olf_valve', 'odourD_olf_valve', 'odourE_olf_valve', 'odourF_olf_valve', 'odourG_olf_valve']
    rew_valve_cols = ['r1_olf_valve', 'r2_olf_valve']
    minimumSamplingTime = session_schema['minimumSamplingTime']
    sampleOffsetTime = session_schema['sampleOffsetTime']
    
    # Collect all poke onset and offset events
    odour_poke_events_df = get_odour_poke_df(odour_poke_df, odour_poke_off_df)

    # Find all trial end points (trial = sequence)
    end_initiation_indices = events_df.index[events_df['EndInitiation'] == True].tolist()

    # Calculate number of valid reward attempts NOTE this might be different to num_rewards from pulse supply ports
    valid_reward_attempts = 0    # following EndInitiation
    for e in range(len(end_initiation_indices)):
        end_idx = end_initiation_indices[e]
        if e == len(end_initiation_indices) - 1:
            next_end_idx = len(events_df)
        else:
            next_end_idx = end_initiation_indices[e + 1]
        
        i = end_idx
        while i < next_end_idx:
            if events_df.loc[i, ['r1_poke', 'r2_poke']].any():
                valid_reward_attempts += 1
                break
            else:
                i += 1

    # Calculate sequence completion and commitment ratios
    if valid_reward_attempts == len(end_initiation_indices):
        # 1. Trial end is marked by full sequence completion 
        # 2. Reward must be collected before the next sequence can start
        # sequence completion = (# of full sequence completions) / (# of initiations after rollback)
        # sequence commitment = (# of full sequence completions) / (# of reward sampling attempts)

        # Initialize counters
        complete_sequences = [0 for _ in range(len(end_initiation_indices))]
        incomplete_sequences = [0 for _ in range(len(end_initiation_indices))]
        commited_sequences = len(end_initiation_indices)
        uncommited_sequences = 0
        num_continuous_odours = [[] for _ in range(len(end_initiation_indices))]        
        first_odour = [None for _ in range(len(end_initiation_indices))]   

        # Keep a record of the first odour in each sequence
        for e in range(len(end_initiation_indices)):
            end_idx = end_initiation_indices[e]
            if e == 0:
                prev_end_idx = 0
            else:
                prev_end_idx = end_initiation_indices[e - 1]
            
            for i in range(prev_end_idx, end_idx):
                active_valve = [col for col in olf_valve_cols if events_df.at[i, col]]
                if active_valve:
                    first_odour[e] = active_valve[0]
                    break
        try:
            for e in range(len(end_initiation_indices)):            
                end_idx = end_initiation_indices[e]
                if e == len(end_initiation_indices) - 1:
                    next_end_idx = len(events_df)
                else:
                    next_end_idx = end_initiation_indices[e + 1]

                if e == 0:
                    prev_end_idx = 0
                else:
                    prev_end_idx = end_initiation_indices[e - 1]
                
                # Process each odour
                next_odours = list(set(olf_valve_cols) - set(first_odour[e]) - set(rew_valve_cols))
                valid_odour_counter = 0  # count number of consecutive odours
                odour_counter = 0
                
                for i in range(prev_end_idx, end_idx):
                    valid_odour = False
                    if events_df.loc[i, olf_valve_cols].any():
                        odour_counter += 1

                        # Determine if an odour is valid based on sampling
                        valid_odour = is_odour_valid(i, events_df, odour_poke_events_df, sampleOffsetTime, minimumSamplingTime)
                        
                        if not valid_odour: 
                            continue

                    # Process valid odours only
                    if valid_odour: 
                        for k in range(i, end_idx): 
                            if events_df.loc[k, first_odour[e]]:
                                valid_odour_counter = 0
                                num_continuous_odours[e].append(valid_odour_counter)
                                incomplete_sequences[e] += 1
                                break

                            elif events_df.loc[k, rew_valve_cols].any():
                                valid_odour_counter += 1
                                if num_continuous_odours[e]:
                                    num_continuous_odours[e].pop() # remove false inclusion of the last first odour
                                num_continuous_odours[e].append(valid_odour_counter)
                                complete_sequences[e] += 1
                                incomplete_sequences[e] -= 1 # remove false inclusion of the last first odour
                                break 

                            elif events_df.loc[k, next_odours].any():
                                valid_odour_counter += 1
                                if num_continuous_odours[e]:
                                    num_continuous_odours[e].pop() # remove false inclusion of the last first odour
                                num_continuous_odours[e].append(valid_odour_counter)
                                break 

                        # Find commited vs uncommited (reward attempt before completion) sequences
                        for l in range(k + 1, end_idx):
                            if events_df.loc[l, olf_valve_cols].any():
                                break
                            elif events_df.loc[l, ['r1_poke', 'r2_poke']].any():
                                uncommited_sequences += 1
                                break
            
            num_continuous_odours[e] = [x + 1 for x in num_continuous_odours[e]] 
            assert len(num_continuous_odours[e]) == complete_sequences[e] + incomplete_sequences[e], 'Weird number of sequences'
        
        except Exception as e:
            print(f"Error during odour processing for sequence completion: {e}")

        try:
            # Get all complete and incomplete sequences from the session
            complete_sequences = sum(trial for trial in complete_sequences)
            incomplete_sequences = sum(trial for trial in incomplete_sequences)

            # Calculate sequence completion ratio with safety checks for division by zero
            completion_ratio = complete_sequences / (complete_sequences + incomplete_sequences) * 100 if (complete_sequences + incomplete_sequences) > 0 else 0
            
            # Calculate sequence commitment ratio with safety checks for division by zero 
            commitment_ratio = commited_sequences / (commited_sequences + uncommited_sequences) * 100 if (commited_sequences + uncommited_sequences) > 0 else 0
        except Exception as e:
            print(f"Error during sequence completion and commitment calculation: {e}")
    
    else:
        # 1. Trial end is marked by partial sequence completion if sampling time is greater than the 
        # minimum for at least one odour 
        # 2. Reward does not need to be collected before the next sequence can start
        # sequence completion = num of trials with rewarded odour presentation / all trials
        # sequence commitment = N/A

        # Initialize counters
        complete_sequences = 0
        incomplete_sequences = 0
    
        # Process each trial (trial = all odours until poke out)
        num_continuous_odours = [[] for _ in range(complete_sequences)]        

        # Process each trial
        for e, end_idx in enumerate(end_initiation_indices[:-1]):
            # Determine trial type (r1, r2, or non-rewarded odour) by finding the most recent valve activation
            closest_valve_idx = None
            trial_type = None
            for i in range(end_idx - 1, -1, -1):
                if events_df.loc[i, 'r1_olf_valve']:
                    closest_valve_idx = i
                    trial_type = 'r1'
                    break
                elif events_df.loc[i, 'r2_olf_valve']:
                    closest_valve_idx = i
                    trial_type = 'r2'
                    break 
                elif events_df.loc[i, 'odourC_olf_valve']:
                    closest_valve_idx = i
                    trial_type = 'C'
                    break
                elif events_df.loc[i, 'odourD_olf_valve']:
                    closest_valve_idx = i
                    trial_type = 'D'
                    break
                elif events_df.loc[i, 'odourE_olf_valve']:
                    closest_valve_idx = i
                    trial_type = 'E'
                    break
                elif events_df.loc[i, 'odourF_olf_valve']:
                    closest_valve_idx = i
                    trial_type = 'F'
                    break
                elif events_df.loc[i, 'odourG_olf_valve']:
                    closest_valve_idx = i
                    trial_type = 'G'
                    break

            # Skip if no valve activation found before this trial end
            if closest_valve_idx is None:
                continue
                
            # Count trial by type
            if trial_type == 'r1' or trial_type == 'r2':
                complete_sequences += 1
            else:
                incomplete_sequences += 1

        # Calculate sequence completion ratio with safety checks for division by zero
        completion_ratio = complete_sequences / (complete_sequences + incomplete_sequences) * 100 if (complete_sequences + incomplete_sequences) > 0 else 0

        commited_sequences = 0
        uncommited_sequences = 0
        commitment_ratio = 0

    return {
        'complete_sequences': complete_sequences,
        'incomplete_sequences': incomplete_sequences,
        'commited_sequences': commited_sequences,
        'uncommited_sequences': uncommited_sequences,
        'completion_ratio': completion_ratio,
        'commitment_ratio': commitment_ratio        
    }

def calculate_overall_decision_sensitivity(events_df):
    """
    Calculate decision sensitivity for rewarded trials in freerun sessions.
    Sensitivity = TP / (TP + FN) = response_trials (A+B) / total_trials (A+B) 
    
    Parameters:
    -----------
    events_df : pandas.DataFrame
        DataFrame containing trial events with columns: 'timestamp', 'r1_poke', 'r2_poke', 'r1_olf_valve', 'r2_olf_valve', 
                                            'odourC_olf_valve', 'odourD_olf_valve', 'odourE_olf_valve', 'odourF_olf_valve', 
                                                'odourG_olf_valve', 'r1_olf_valve_off', 'r2_olf_valve_off', 'odourC_olf_valve_off', 
                                                    'odourD_olf_valve_off', 'odourE_olf_valve_off', 'odourF_olf_valve_off', 
                                                        'odourG_olf_valve_off', 'EndInitiation', 'odour_poke', 'odour_poke_off',
                                                        'PulseSupplyPort1', 'PulseSupplyPort2'
        
    Returns:
    --------
    dict
        Dictionary containing sensitivity metrics for r1 and r2 trials
    """
    
    # Ensure events are in chronological order
    events_df = events_df.sort_values('timestamp').reset_index(drop=True)
    
    # Initialize counters
    r1_respond = 0
    r1_total = 0
    r2_respond = 0
    r2_total = 0
    
    # Find all trial end points
    end_initiation_indices = events_df.index[events_df['EndInitiation'] == True].tolist()

    # Process each trial
    for e in range(len(end_initiation_indices)):
        end_idx = end_initiation_indices[e]
        if e == len(end_initiation_indices) - 1:
            next_end_idx = len(events_df)
        else:
            next_end_idx = end_initiation_indices[e + 1]
        
        # Determine trial type (r1 or r2 or nonR) by finding the most recent valve activation
        closest_valve_idx = None
        trial_type = None
        for i in range(end_idx - 1, -1, -1):
            if events_df.loc[i, 'r1_olf_valve']:
                closest_valve_idx = i
                trial_type = 'r1'
                break
            elif events_df.loc[i, 'r2_olf_valve']:
                closest_valve_idx = i
                trial_type = 'r2'
                break
            elif ('odourC_olf_valve' in events_df and events_df.loc[i, 'odourC_olf_valve']) or \
                    ('odourD_olf_valve' in events_df and events_df.loc[i, 'odourD_olf_valve']) or \
                    ('odourE_olf_valve' in events_df and events_df.loc[i, 'odourE_olf_valve']) or \
                    ('odourF_olf_valve' in events_df and events_df.loc[i, 'odourF_olf_valve']) or \
                    ('odourG_olf_valve' in events_df and events_df.loc[i, 'odourG_olf_valve']):
                closest_valve_idx = i
                trial_type = 'nonR'
                break
                
        # Skip if no valve activation found before this trial end or trial is non-rewarded
        if trial_type == 'nonR' or closest_valve_idx is None:
            continue

        # Count trial by type
        if trial_type == 'r1':
            r1_total += 1
        elif trial_type == 'r2':
            r2_total += 1
          
        # Determine if there was a reward poke or a new odour poke after trial end 
        for j in range(end_idx + 1, next_end_idx):
            if events_df.loc[j, ['r1_poke', 'r2_poke']].any():
                # Count response trials
                if trial_type == 'r1':
                    r1_respond += 1
                    break 
                elif trial_type == 'r2':
                    r2_respond += 1
                    break 

    # Calculate sensitivity percentages with safety checks for division by zero
    r1_sensitivity = (r1_respond / r1_total * 100) if r1_total > 0 else 0
    r2_sensitivity = (r2_respond / r2_total * 100) if r2_total > 0 else 0
    overall_sensitivity = ((r1_respond + r2_respond) / (r1_total + r2_total) * 100) if (r1_total + r2_total) > 0 else 0
    
    # Return detailed sensitivity metrics
    return {
        'r1_total': r1_total,
        'r1_respond': r1_respond,
        'r1_sensitivity': r1_sensitivity,
        'r2_total': r2_total,
        'r2_respond': r2_respond,
        'r2_sensitivity': r2_sensitivity,
        'overall_sensitivity': overall_sensitivity
    }

# # TODO:
# def calculate_overall_decision_specificity(events_df):
#     """
#     Calculate decision specificity for non-rewarded trials in freerun sessions.
#     Specificity = TN / (TN + FP) = non_response_trials (C+D+E+F+G) / total_trials (C+D+E+F+G)
    
#     Parameters:
#     -----------
#     events_df : pandas.DataFrame
#         DataFrame containing trial events with columns: 'timestamp', 'r1_poke', 'r2_poke', 'r1_olf_valve', 'r2_olf_valve', 
#                                             'odourC_olf_valve', 'odourD_olf_valve', 'odourE_olf_valve', 'odourF_olf_valve', 
#                                                 'odourG_olf_valve', 'r1_olf_valve_off', 'r2_olf_valve_off', 'odourC_olf_valve_off', 
#                                                     'odourD_olf_valve_off', 'odourE_olf_valve_off', 'odourF_olf_valve_off', 
#                                                         'odourG_olf_valve_off', 'EndInitiation', 'odour_poke', 'odour_poke_off'
        
#     Returns:
#     --------
#     dict
#         Dictionary containing specificity metrics for non-rewarded trials
#     """
#     return 

def calculate_overall_false_alarm_bias(events_df, odour_poke_df, odour_poke_off_df, session_schema, odour_to_olfactometer_map):
    """
    Calculate false alarm bias for non-rewarded trials.
    Time bias: % FA after # odours since reward
    Olfactometer bias: % FA where previous odour was from same or different olfactometer
    Reward side bias: % FA on reward port linked to rewarded odour from that olfactometer
    
    Parameters:
    -----------
    events_df : pandas.DataFrame
        DataFrame containing trial events with columns: 'timestamp', 'r1_poke', 'r2_poke', 'r1_olf_valve', 'r2_olf_valve', 
                                            'odourC_olf_valve', 'odourD_olf_valve', 'odourE_olf_valve', 'odourF_olf_valve', 
                                                'odourG_olf_valve', 'r1_olf_valve_off', 'r2_olf_valve_off', 'odourC_olf_valve_off', 
                                                    'odourD_olf_valve_off', 'odourE_olf_valve_off', 'odourF_olf_valve_off', 
                                                        'odourG_olf_valve_off', 'EndInitiation', 'odour_poke', 'odour_poke_off',
                                                        'PulseSupplyPort1', 'PulseSupplyPort2'
        
    Returns:
    --------
    dict
        Dictionary containing false alarm bias metrics 
    """
    # Define useful variables
    olf_valve_cols = ['odourC_olf_valve', 'odourD_olf_valve', 'odourE_olf_valve', 'odourF_olf_valve', 'odourG_olf_valve']
    rew_valve_cols = ['r1_olf_valve', 'r2_olf_valve']
    all_valve_cols = list(set(olf_valve_cols) | set(rew_valve_cols)) 
    minimumSamplingTime = session_schema['minimumSamplingTime']
    sampleOffsetTime = session_schema['sampleOffsetTime']
    
    # Collect all poke onset and offset events
    odour_poke_events_df = get_odour_poke_df(odour_poke_df, odour_poke_off_df)

    # Find all trial end points
    end_initiation_indices = events_df.index[events_df['EndInitiation'] == True].tolist()

    # Initialize counters
    binary_poke = []
    poke = []
    trial_type = []
    intervals = []
    olfactometer = []
    num_inter_reward_poke_odours = 0 

    # Process each sequence
    for e in range(len(end_initiation_indices)):
        end_idx = end_initiation_indices[e]
        if e == len(end_initiation_indices) - 1:
            next_end_idx = len(events_df)
        else:
            next_end_idx = end_initiation_indices[e + 1]

        if e == 0:
            prev_end_idx = 0
        else:
            prev_end_idx = end_initiation_indices[e - 1]

        # Process each odour
        for i in range(prev_end_idx, end_idx):
            valid_odour = False
            
            if events_df.loc[i, all_valve_cols].any():
                # Determine if an odour is valid based on sampling
                valid_odour = is_odour_valid(i, events_df, odour_poke_events_df, sampleOffsetTime, minimumSamplingTime)
                
                if not valid_odour: 
                    continue
                
            # Process valid odours only
            if valid_odour: 
                odour = next((od for od in all_valve_cols if events_df.loc[i, od]), None)
                
                # Find which olfactometer this odour came from
                olfactometer.append(odour_to_olfactometer_map.get(odour, -1)) 

                for j in range(i + 1, next_end_idx):
                    if events_df.loc[j, 'r1_poke'] or events_df.loc[j, 'r2_poke']:
                        binary_poke.append(1)
                        poke.append(1 if events_df.loc[j, 'r1_poke'] else 2)

                        if odour in rew_valve_cols:
                            num_inter_reward_poke_odours = 0
                            break
                        else:
                            num_inter_reward_poke_odours += 1
                            break
                    elif events_df.loc[j, all_valve_cols].any():
                        binary_poke.append(0)
                        poke.append(0)
                        num_inter_reward_poke_odours += 1
                        break 

                # Determine trial type  
                if odour == 'r1_olf_valve': trial_type.append('r1') 
                elif odour == 'r2_olf_valve': trial_type.append('r2') 
                elif odour == 'odourC_olf_valve': trial_type.append('C') 
                elif odour == 'odourD_olf_valve': trial_type.append('D') 
                elif odour == 'odourE_olf_valve': trial_type.append('E') 
                elif odour == 'odourF_olf_valve': trial_type.append('F') 
                elif odour == 'odourG_olf_valve': trial_type.append('G')  
        
                intervals.append(num_inter_reward_poke_odours)

    # Calculate false alarm rate for each trial type and interval
    binary_poke = np.array(binary_poke)
    trial_type = np.array(trial_type)
    olfactometer = np.array(olfactometer)
    intervals = np.array(intervals)
    poke = np.array(poke)

    nonR_odours = ['C', 'D', 'E', 'F', 'G']
    odour_interval_pokes = {odour: {} for odour in nonR_odours}
    odour_interval_trials = {odour: {} for odour in nonR_odours}
    odour_interval_false_alarm = {odour: {} for odour in nonR_odours}

    for odour in nonR_odours:
        for i in np.unique(intervals):
            odour_interval_pokes[odour][i] = len(np.where((binary_poke == 1) & (trial_type == odour) & (intervals == i))[0])
            odour_interval_trials[odour][i] = len(np.where((trial_type == odour) & (intervals == i))[0])
            odour_interval_false_alarm[odour][i] = (odour_interval_pokes[odour][i] / odour_interval_trials[odour][i] * 100) if odour_interval_trials[odour][i] else 0
    
    # Calculate overall false alarm rate for each interval
    interval_pokes = {}
    interval_trials = {}
    interval_false_alarm = {}
    for i in np.unique(intervals):
        interval_pokes[i] = np.sum([odour_interval_pokes[odour][i] for odour in nonR_odours])
        interval_trials[i] = np.sum([odour_interval_trials[odour][i] for odour in nonR_odours])
        interval_false_alarm[i] = (interval_pokes[i] / interval_trials[i] * 100) if interval_trials[i] else 0
   
    # Calculate false alarm rate for each odour depending on preceding olfactometer (olfactometer bias)
    odour_same_olf_pokes = {}
    odour_same_olf_trials = {}
    odour_same_olf_false_alarm = {}
    odour_diff_olf_pokes = {}
    odour_diff_olf_trials = {}
    odour_diff_olf_false_alarm = {}
    odour_false_alarm_trials = {}
    
    for odour in nonR_odours:
        olfactometer_change = np.abs(np.diff(olfactometer))
        odour_false_alarm_trials[odour] = len(np.where((binary_poke[1:] == 1) & (trial_type[1:] == odour))[0])
        
        odour_same_olf_pokes[odour] = len(np.where((binary_poke[1:] == 1) & (trial_type[1:] == odour) & (olfactometer_change == 0))[0])
        odour_same_olf_trials[odour] = len(np.where((trial_type[1:] == odour) & (olfactometer_change == 0))[0])
        odour_same_olf_false_alarm[odour] = (odour_same_olf_pokes[odour] / odour_false_alarm_trials[odour] * 100) if odour_false_alarm_trials[odour] else 0

        odour_diff_olf_pokes[odour] = len(np.where((binary_poke[1:] == 1) & (trial_type[1:] == odour) & (olfactometer_change == 1))[0])
        odour_diff_olf_trials[odour] = len(np.where((trial_type[1:] == odour) & (olfactometer_change == 1))[0])
        odour_diff_olf_false_alarm[odour] = (odour_diff_olf_pokes[odour] / odour_false_alarm_trials[odour] * 100) if odour_false_alarm_trials[odour] else 0

    all_olf_pokes = np.sum([odour_false_alarm_trials[odour] for odour in nonR_odours])

    same_olf_pokes = np.sum([odour_same_olf_pokes[odour] for odour in nonR_odours])
    same_olf_trials = np.sum([odour_same_olf_trials[odour] for odour in nonR_odours])
    same_olf_false_alarm = (same_olf_pokes / all_olf_pokes * 100) if all_olf_pokes else 0

    diff_olf_pokes = np.sum([odour_diff_olf_pokes[odour] for odour in nonR_odours])
    diff_olf_trials = np.sum([odour_diff_olf_trials[odour] for odour in nonR_odours])
    diff_olf_false_alarm = (diff_olf_pokes / all_olf_pokes * 100) if all_olf_pokes else 0
    
    # Calculate false alarm rate depending on olfactometer associated with reward side (reward side bias)
    rew_valve_cols = np.array(rew_valve_cols)
    rew_olf_map = np.where(poke == 0, 0, np.vectorize(odour_to_olfactometer_map.get)(rew_valve_cols[poke - 1]))

    odour_same_olf_rew_pairing = {}
    odour_diff_olf_rew_pairing = {}
    for odour in nonR_odours:
        odour_same_olf_rew_pairing[odour] = len(np.where((binary_poke == 1) & (trial_type == odour) & (rew_olf_map == olfactometer))[0])
        odour_diff_olf_rew_pairing[odour] = len(np.where((binary_poke == 1) & (trial_type == odour) & (rew_olf_map != olfactometer))[0])

    same_olf_rew_pairing = np.sum([odour_same_olf_rew_pairing[odour] for odour in nonR_odours])
    diff_olf_rew_pairing = np.sum([odour_diff_olf_rew_pairing[odour] for odour in nonR_odours])

    same_olf_rew_false_alarm = (same_olf_rew_pairing / all_olf_pokes * 100) if all_olf_pokes else 0
    diff_olf_rew_false_alarm = (diff_olf_rew_pairing / all_olf_pokes * 100) if all_olf_pokes else 0
    
    return {
        'odour_interval_pokes': odour_interval_pokes,
        'odour_interval_trials': odour_interval_trials,
        'odour_interval_false_alarm': odour_interval_false_alarm,
        'interval_pokes': interval_pokes,
        'interval_trials': interval_trials,
        'interval_false_alarm': interval_false_alarm, 
        'odour_same_olf_pokes': odour_same_olf_pokes, 
        'odour_same_olf_trials': odour_same_olf_trials, 
        'odour_same_olf_false_alarm': odour_same_olf_false_alarm, 
        'odour_diff_olf_pokes': odour_diff_olf_pokes, 
        'odour_diff_olf_trials': odour_diff_olf_trials, 
        'odour_diff_olf_false_alarm': odour_diff_olf_false_alarm, 
        'same_olf_pokes': same_olf_pokes, 
        'same_olf_trials': same_olf_trials, 
        'same_olf_false_alarm': same_olf_false_alarm, 
        'diff_olf_pokes': diff_olf_pokes, 
        'diff_olf_trials': diff_olf_trials, 
        'diff_olf_false_alarm': diff_olf_false_alarm,
        'odour_false_alarm_trials': odour_false_alarm_trials,
        'odour_same_olf_rew_pairing': odour_same_olf_rew_pairing,
        'odour_diff_olf_rew_pairing': odour_diff_olf_rew_pairing,
        'same_olf_rew_pairing': same_olf_rew_pairing,
        'diff_olf_rew_pairing': diff_olf_rew_pairing,
        'same_olf_rew_false_alarm': same_olf_rew_false_alarm,
        'diff_olf_rew_false_alarm': diff_olf_rew_false_alarm
    }


def is_odour_valid(idx, events_df, odour_poke_events_df, sampleOffsetTime, minimumSamplingTime):
    # Odour valve onset
    olf_valve_on_ts = events_df.loc[idx, 'timestamp']
    
    # Find last poke onset before odour valve onset
    poke_onset_candidates = odour_poke_events_df[
        (odour_poke_events_df['timestamp'] < olf_valve_on_ts) &
        (odour_poke_events_df['timestamp'].isin(events_df.loc[events_df['odour_poke'], 'timestamp']))
    ]
    poke_onset = poke_onset_candidates.index[-1]
    poke_onset_ts = odour_poke_events_df.loc[poke_onset, 'timestamp']
    
    # Find first poke offset after odour valve onset - ignore short offsets
    poke_offset = poke_onset + 1
    poke_offset_ts = odour_poke_events_df.loc[poke_offset, 'timestamp']

    next_poke_onset = poke_onset + 2
    while (next_poke_onset + 1 < len(odour_poke_events_df) and
        (odour_poke_events_df.loc[next_poke_onset, 'timestamp'] - poke_offset_ts).total_seconds() < sampleOffsetTime):
        poke_offset_ts = odour_poke_events_df.loc[next_poke_onset + 1, 'timestamp']
        next_poke_onset += 2
    
    # Check if offset before odour valve onset
    if poke_offset_ts < olf_valve_on_ts:
        return False

    # Filter based on sampling time
    if (poke_offset_ts - poke_onset_ts).total_seconds() < minimumSamplingTime:
        return False

    return True


def get_odour_poke_df(odour_poke_df, odour_poke_off_df):

    # Collect all poke onset and offset events
    odour_poke_events_df = [odour_poke_df, odour_poke_off_df]
    odour_poke_events_df = reduce(lambda left, right: pd.merge(left, right, on='Time', how='outer'), odour_poke_events_df)
                    
    for col in odour_poke_events_df:
        if col not in odour_poke_events_df.columns and col != 'Time':
            odour_poke_events_df[col] = False
        if col != "Time":
            odour_poke_events_df[col] = odour_poke_events_df[col].fillna(False) # This might drop the first InitiationSequence event
    odour_poke_events_df = odour_poke_events_df.dropna(subset=["Time"])

    odour_poke_events_df.sort_values('Time', inplace=True)
    odour_poke_events_df.rename(columns={'Time': 'timestamp'}, inplace=True)
    odour_poke_events_df.reset_index(drop=True, inplace=True)

    return odour_poke_events_df