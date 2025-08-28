#### python

import re
import argparse
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import zoneinfo
import src.utils as utils  # Changed from relative to absolute import
import harp

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
        if accuracy_summary and accuracy_summary.get('r1_total', 0) + accuracy_summary.get('r2_total', 0) > 0:
            print("Decision Accuracy (using EndInitiation from experiment events):")
            print(f"  R1 Trials: {accuracy_summary['r1_total']}, Correct: {accuracy_summary['r1_correct']}, Accuracy: {accuracy_summary['r1_accuracy']:.2f}%")
            print(f"  R2 Trials: {accuracy_summary['r2_total']}, Correct: {accuracy_summary['r2_correct']}, Accuracy: {accuracy_summary['r2_accuracy']:.2f}%")
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
            
            # Process experiment events for EndInitiation
            end_initiation_frames = []
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
                        poke_off = df.loc[off_after_on_mask, ['Time']].copy()
                        poke_off['odour_poke_off'] = True
                        event_frames.append(poke_off)
                        print(f"Added {len(poke_off)} odour poke OFF events")
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
            
            # Add r1 olfactometer valve events if available
            if not olfactometer_valves_0_abs.empty and 'Valve0' in olfactometer_valves_0_abs.columns:
                try:
                    r1_olf_df = olfactometer_valves_0_abs[olfactometer_valves_0_abs['Valve0'] == True].copy()
                    if not r1_olf_df.empty:
                        r1_olf_df = r1_olf_df[['Time']].copy()
                        r1_olf_df['r1_olf_valve'] = True
                        event_frames.append(r1_olf_df)
                        print(f"Added {len(r1_olf_df)} r1 valve events")

                        # Find OFF events
                        df = olfactometer_valves_0_abs.sort_values(by='Time').reset_index(drop=True)
                        valve_prev = df['Valve0'].shift(1) # Shift Valve0 to compare previous row
                        valve_now = df['Valve0']

                        # Detect where it was ON in previous row and now is OFF
                        off_after_on_mask = (valve_prev == True) & (valve_now == False)
                        r1_off = df.loc[off_after_on_mask, ['Time']].copy()
                        r1_off['r1_olf_valve_off'] = True
                        event_frames.append(r1_off)
                        print(f"Added {len(r1_off)} r1 valve OFF events")
                except Exception as e:
                    print(f"Error processing r1 valve events: {e}")
            
            # Add r2 olfactometer valve events if available
            if not olfactometer_valves_0_abs.empty and 'Valve1' in olfactometer_valves_0_abs.columns:
                try:
                    r2_olf_df = olfactometer_valves_0_abs[olfactometer_valves_0_abs['Valve1'] == True].copy()
                    if not r2_olf_df.empty:
                        r2_olf_df = r2_olf_df[['Time']].copy()
                        r2_olf_df['r2_olf_valve'] = True
                        event_frames.append(r2_olf_df)
                        print(f"Added {len(r2_olf_df)} r2 valve events")

                        # Find OFF events
                        df = olfactometer_valves_0_abs.sort_values(by='Time').reset_index(drop=True)
                        valve_prev = df['Valve1'].shift(1) # Shift Valve1 to compare previous row
                        valve_now = df['Valve1']

                        # Detect where it was ON in previous row and now is OFF
                        off_after_on_mask = (valve_prev == True) & (valve_now == False)
                        r2_off = df.loc[off_after_on_mask, ['Time']].copy()
                        r2_off['r2_olf_valve_off'] = True
                        event_frames.append(r2_off)
                        print(f"Added {len(r2_off)} r2 valve OFF events")
                except Exception as e:
                    print(f"Error processing r2 valve events: {e}")
            
            # Add odour C olfactometer valve events if available
            if not olfactometer_valves_0_abs.empty and 'Valve2' in olfactometer_valves_0_abs.columns:
                try:
                    odourC = olfactometer_valves_0_abs[olfactometer_valves_0_abs['Valve2'] == True].copy()
                    if not odourC.empty:
                        odourC = odourC[['Time']].copy()
                        odourC['odourC_olf_valve'] = True
                        event_frames.append(odourC)
                        print(f"Added {len(odourC)} odour C valve events")

                        # Find OFF events
                        df = olfactometer_valves_0_abs.sort_values(by='Time').reset_index(drop=True)
                        valve_prev = df['Valve2'].shift(1) # Shift Valve2 to compare previous row
                        valve_now = df['Valve2']

                        # Detect where it was ON in previous row and now is OFF
                        off_after_on_mask = (valve_prev == True) & (valve_now == False)
                        odourC_off = df.loc[off_after_on_mask, ['Time']].copy()
                        odourC_off['odourC_olf_valve_off'] = True
                        event_frames.append(odourC_off)
                        print(f"Added {len(odourC_off)} odour C valve OFF events")
                except Exception as e:
                    print(f"Error processing odour C valve events: {e}")
            
            # Add odour D olfactometer valve events if available
            if not olfactometer_valves_0_abs.empty and 'Valve3' in olfactometer_valves_0_abs.columns:
                try:
                    odourD = olfactometer_valves_0_abs[olfactometer_valves_0_abs['Valve3'] == True].copy()
                    if not odourD.empty:
                        odourD = odourD[['Time']].copy()
                        odourD['odourD_olf_valve'] = True
                        event_frames.append(odourD)
                        print(f"Added {len(odourD)} odour D valve events")

                        # Find OFF events
                        df = olfactometer_valves_0_abs.sort_values(by='Time').reset_index(drop=True)
                        valve_prev = df['Valve3'].shift(1) # Shift Valve3 to compare previous row
                        valve_now = df['Valve3']

                        # Detect where it was ON in previous row and now is OFF
                        off_after_on_mask = (valve_prev == True) & (valve_now == False)
                        odourD_off = df.loc[off_after_on_mask, ['Time']].copy()
                        odourD_off['odourD_olf_valve_off'] = True
                        event_frames.append(odourD_off)
                        print(f"Added {len(odourD_off)} odour D valve OFF events")
                except Exception as e:
                    print(f"Error processing odour D valve events: {e}")
            
            # Add odour E olfactometer valve events if available
            if not olfactometer_valves_1_abs.empty and 'Valve0' in olfactometer_valves_1_abs.columns:
                try:
                    odourE = olfactometer_valves_1_abs[olfactometer_valves_1_abs['Valve0'] == True].copy()
                    if not odourE.empty:
                        odourE = odourE[['Time']].copy()
                        odourE['odourE_olf_valve'] = True
                        event_frames.append(odourE)
                        print(f"Added {len(odourE)} odour E valve events")

                        # Find OFF events
                        df = olfactometer_valves_1_abs.sort_values(by='Time').reset_index(drop=True)
                        valve_prev = df['Valve0'].shift(1) # Shift Valve0 to compare previous row
                        valve_now = df['Valve0']

                        # Detect where it was ON in previous row and now is OFF
                        off_after_on_mask = (valve_prev == True) & (valve_now == False)
                        odourE_off = df.loc[off_after_on_mask, ['Time']].copy()
                        odourE_off['odourE_olf_valve_off'] = True
                        event_frames.append(odourE_off)
                        print(f"Added {len(odourE_off)} odour E valve OFF events")
                except Exception as e:
                    print(f"Error processing odour E valve events: {e}")

            # Add odour F olfactometer valve events if available
            if not olfactometer_valves_1_abs.empty and 'Valve1' in olfactometer_valves_1_abs.columns:
                try:
                    odourF = olfactometer_valves_1_abs[olfactometer_valves_1_abs['Valve1'] == True].copy()
                    if not odourF.empty:
                        odourF = odourF[['Time']].copy()
                        odourF['odourF_olf_valve'] = True
                        event_frames.append(odourF)
                        print(f"Added {len(odourF)} odour F valve events")

                        # Find OFF events
                        df = olfactometer_valves_1_abs.sort_values(by='Time').reset_index(drop=True)
                        valve1_prev = df['Valve1'].shift(1) # Shift Valve1 to compare previous row
                        valve1_now = df['Valve1']

                        # Detect where it was ON in previous row and now is OFF
                        off_after_on_mask = (valve1_prev == True) & (valve1_now == False)
                        odourF_off = df.loc[off_after_on_mask, ['Time']].copy()
                        odourF_off['odourF_olf_valve_off'] = True
                        event_frames.append(odourF_off)
                        print(f"Added {len(odourF_off)} odour F valve OFF events")

                except Exception as e:
                    print(f"Error processing odour F valve events: {e}")
            
            # Add odour G olfactometer valve events if available
            if not olfactometer_valves_1_abs.empty and 'Valve2' in olfactometer_valves_1_abs.columns:
                try:
                    odourG = olfactometer_valves_1_abs[olfactometer_valves_1_abs['Valve2'] == True].copy()
                    if not odourG.empty:
                        odourG = odourG[['Time']].copy()
                        odourG['odourG_olf_valve'] = True
                        event_frames.append(odourG)
                        print(f"Added {len(odourG)} odour G valve events")

                        # Find OFF events
                        df = olfactometer_valves_1_abs.sort_values(by='Time').reset_index(drop=True)
                        valve_prev = df['Valve2'].shift(1) # Shift Valve2 to compare previous row
                        valve_now = df['Valve2']

                        # Detect where it was ON in previous row and now is OFF
                        off_after_on_mask = (valve_prev == True) & (valve_now == False)
                        odourG_off = df.loc[off_after_on_mask, ['Time']].copy()
                        odourG_off['odourG_olf_valve_off'] = True
                        event_frames.append(odourG_off)
                        print(f"Added {len(odourG_off)} odour G valve OFF events")
                except Exception as e:
                    print(f"Error processing odour G valve events: {e}")
            
            # Add EndInitiation events if available
            if not combined_end_initiation_df.empty and 'EndInitiation' in combined_end_initiation_df.columns:
                event_frames.append(combined_end_initiation_df)
            
            # Only proceed if we have data to analyze
            if event_frames:
                try:
                    all_events_df = pd.concat(event_frames, ignore_index=True)
                    print(f"Combined {len(all_events_df)} total events")
                    
                    # Explicitly add missing columns with default values to prevent errors
                    if int(detect_stage(root)) > 7:
                        required_columns = ['timestamp', 'r1_poke', 'r2_poke', 'r1_olf_valve', 'r2_olf_valve', \
                                            'odourC_olf_valve', 'odourD_olf_valve', 'odourE_olf_valve', 'odourF_olf_valve', \
                                                'odourG_olf_valve', 'r1_olf_valve_off', 'r2_olf_valve_off', 'odourC_olf_valve_off', \
                                                    'odourD_olf_valve_off', 'odourE_olf_valve_off', 'odourF_olf_valve_off', \
                                                        'odourG_olf_valve_off', 'EndInitiation', 'odour_poke', 'odour_poke_off']
                    else:
                        required_columns = ['timestamp', 'r1_poke', 'r2_poke', 'r1_olf_valve', 'r2_olf_valve', 'r1_olf_valve_off', \
                                            'r2_olf_valve_off', 'EndInitiation', 'odour_poke', 'odour_poke_off']
                    for col in required_columns:
                        if col not in all_events_df.columns and col != 'timestamp':
                            all_events_df[col] = False
                    
                    # Handle the warning about downcasting properly
                    all_events_df = all_events_df.fillna(False).infer_objects(copy=False)
                    all_events_df.sort_values('Time', inplace=True)
                    all_events_df.rename(columns={'Time': 'timestamp'}, inplace=True)
                    all_events_df.reset_index(drop=True, inplace=True)

                    # Check if we have any valid trial data
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
                          f"Valve events: {has_valve_events}, Poke events: {has_poke_events}")
                    
                    if has_end_initiation and has_valve_events and has_poke_events:
                        # Calculate decision accuracy
                        session_data['accuracy_summary'] = calculate_overall_decision_accuracy(all_events_df)

                        # Calculate decision specificity 
                        if int(detect_stage(root)) > 7:
                            session_data['specificity_summary'] = calculate_overall_decision_specificity_freerun(all_events_df)

                        # TODO: doubles 

                        # Calculate response time
                        session_data['response_time'] = calculate_overall_response_time(all_events_df)
                        
                    else:
                        if not has_end_initiation:
                            print("No EndInitiation events found - cannot identify trial endings")
                        if not has_valve_events:
                            print("No valve activation events found - cannot identify trial types")
                        if not has_poke_events:
                            print("No poke events found - cannot identify animal responses")
                        
                        # Add empty accuracy data
                        session_data['accuracy_summary'] = {
                            'r1_total': 0, 'r1_correct': 0, 'r1_accuracy': 0,
                            'r2_total': 0, 'r2_correct': 0, 'r2_accuracy': 0,
                            'overall_accuracy': 0
                        }
                        # Add empty response time data
                        session_data['response_time'] = {'rt': 0,
                            'r1_correct_rt': 0, 'r1_incorrect_rt': 0,
                            'r1_avg_correct_rt': 0, 'r1_avg_incorrect_rt': 0, 'r1_avg_rt': 0,
                            'r2_correct_rt': 0, 'r2_incorrect_rt': 0,
                            'r2_avg_correct_rt': 0, 'r2_avg_incorrect_rt': 0, 'r2_avg_rt': 0,
                            'hit_rt': 0, 'false_alarm_rt': 0, 'trial_id': 0
                        }
                        # Add empty decision specificity data
                        session_data['specificity_summary'] = {
                            'C_pokes': 0, 'C_trials': 0,
                            'D_pokes': 0, 'D_trials': 0,
                            'E_pokes': 0, 'E_trials': 0,
                            'F_pokes': 0, 'F_trials': 0,
                            'G_pokes': 0, 'G_trials': 0,
                            'C_false_alarm': 0, 'D_false_alarm': 0,
                            'E_false_alarm': 0, 'F_false_alarm': 0,
                            'G_false_alarm': 0, 'overall_false_alarm': 0
                        }
                
                except Exception as e:
                    print(f"Error processing event data: {e}")
                    # Add empty accuracy data
                    session_data['accuracy_summary'] = {
                        'r1_total': 0, 'r1_correct': 0, 'r1_accuracy': 0,
                        'r2_total': 0, 'r2_correct': 0, 'r2_accuracy': 0,
                        'overall_accuracy': 0
                    }
                    # Add empty response time data
                    session_data['response_time'] = {'rt': 0,
                        'r1_correct_rt': 0, 'r1_incorrect_rt': 0,
                        'r1_avg_correct_rt': 0, 'r1_avg_incorrect_rt': 0, 'r1_avg_rt': 0,
                        'r2_correct_rt': 0, 'r2_incorrect_rt': 0,
                        'r2_avg_correct_rt': 0, 'r2_avg_incorrect_rt': 0, 'r2_avg_rt': 0,
                        'hit_rt': 0, 'false_alarm_rt': 0, 'trial_id': 0
                    }
                    # Add empty decision specificity data
                    session_data['specificity_summary'] = {
                        'C_pokes': 0, 'C_trials': 0,
                        'D_pokes': 0, 'D_trials': 0,
                        'E_pokes': 0, 'E_trials': 0,
                        'F_pokes': 0, 'F_trials': 0,
                        'G_pokes': 0, 'G_trials': 0,
                        'C_false_alarm': 0, 'D_false_alarm': 0,
                        'E_false_alarm': 0, 'F_false_alarm': 0,
                        'G_false_alarm': 0, 'overall_false_alarm': 0
                    }
            else:
                print("No events available for decision accuracy calculation")
                # Add empty accuracy data
                session_data['accuracy_summary'] = {
                    'r1_total': 0, 'r1_correct': 0, 'r1_accuracy': 0,
                    'r2_total': 0, 'r2_correct': 0, 'r2_accuracy': 0,
                    'overall_accuracy': 0
                }
                # Add empty response time data
                session_data['response_time'] = {'rt': 0,
                    'r1_correct_rt': 0, 'r1_incorrect_rt': 0,
                    'r1_avg_correct_rt': 0, 'r1_avg_incorrect_rt': 0, 'r1_avg_rt': 0,
                    'r2_correct_rt': 0, 'r2_incorrect_rt': 0,
                    'r2_avg_correct_rt': 0, 'r2_avg_incorrect_rt': 0, 'r2_avg_rt': 0,
                    'hit_rt': 0, 'false_alarm_rt': 0, 'trial_id': 0
                }
                # Add empty decision specificity data
                session_data['specificity_summary'] = {
                    'C_pokes': 0, 'C_trials': 0,
                    'D_pokes': 0, 'D_trials': 0,
                    'E_pokes': 0, 'E_trials': 0,
                    'F_pokes': 0, 'F_trials': 0,
                    'G_pokes': 0, 'G_trials': 0,
                    'C_false_alarm': 0, 'D_false_alarm': 0,
                    'E_false_alarm': 0, 'F_false_alarm': 0,
                    'G_false_alarm': 0, 'overall_false_alarm': 0
                }
        
        except Exception as e:
            print(f"Error during session data processing: {e}")
            # Default session data with zeros
            session_data = {
                'num_r1_rewards': 0,
                'num_r2_rewards': 0,
                'session_duration_sec': 0,
                'accuracy_summary': {
                    'r1_total': 0, 'r1_correct': 0, 'r1_accuracy': 0,
                    'r2_total': 0, 'r2_correct': 0, 'r2_accuracy': 0,
                    'overall_accuracy': 0
                },
                'response_time': {
                    'rt': 0,
                    'r1_correct_rt': 0, 'r1_incorrect_rt': 0,
                    'r1_avg_correct_rt': 0, 'r1_avg_incorrect_rt': 0, 'r1_avg_rt': 0,
                    'r2_correct_rt': 0, 'r2_incorrect_rt': 0,
                    'r2_avg_correct_rt': 0, 'r2_avg_incorrect_rt': 0, 'r2_avg_rt': 0,
                    'hit_rt': 0, 'false_alarm_rt': 0, 'trial_id': 0
                },
                'specificity_summary': {               
                    'C_pokes': 0, 'C_trials': 0,
                    'D_pokes': 0, 'D_trials': 0,
                    'E_pokes': 0, 'E_trials': 0,
                    'F_pokes': 0, 'F_trials': 0,
                    'G_pokes': 0, 'G_trials': 0,
                    'C_false_alarm': 0, 'D_false_alarm': 0,
                    'E_false_alarm': 0, 'F_false_alarm': 0,
                    'G_false_alarm': 0, 'overall_false_alarm': 0
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
            'r1_total': 0, 'r1_correct': 0, 'r1_accuracy': 0,
            'r2_total': 0, 'r2_correct': 0, 'r2_accuracy': 0,
            'overall_accuracy': 0
        })

    @staticmethod
    def get_decision_specificity(data_path):
        """
        Static method to calculate decision specificity for each trial in a single session.
        
        Parameters:
        -----------
        data_path : str or Path
            Path to session data directory
            
        Returns:
        --------
        dict
            Dictionary with decision specificity metrics or None if calculation fails
        """
        root = Path(data_path)
        
        # Process the given directory directly
        print(f"Processing decision specificity for: {root}")
        
        # Create a temporary instance to access the _get_session_data method
        temp_instance = RewardAnalyser.__new__(RewardAnalyser)
        session_data = temp_instance._get_session_data(root)

        return session_data.get('specificity', {'C_pokes': 0, 'C_trials': 0,
                                                'D_pokes': 0, 'D_trials': 0, 'E_pokes': 0, 'E_trials': 0,
                                                'F_pokes': 0, 'F_trials': 0, 'G_pokes': 0, 'G_trials': 0,
                                                'C_false_alarm': 0,
                                                'D_false_alarm': 0, 'E_false_alarm': 0,
                                                'F_false_alarm': 0, 'G_false_alarm': 0, 
                                                'overall_false_alarm': 0
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
        return session_data.get('response_time', {'rt': 0,
                        'r1_correct_rt': 0, 'r1_incorrect_rt': 0,
                        'r1_avg_correct_rt': 0, 'r1_avg_incorrect_rt': 0, 'r1_avg_rt': 0,
                        'r2_correct_rt': 0, 'r2_incorrect_rt': 0,
                        'r2_avg_correct_rt': 0, 'r2_avg_incorrect_rt': 0, 'r2_avg_rt': 0,
                        'hit_rt': 0, 'false_alarm_rt': 0, 'trial_id': 0
                    })

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
                            match = re.search(r'_Stage(\d+)', seq['name'])
                            if match:
                                if 'FreeRun' in seq['name']:
                                    stage_found = int(match.group(1)) + 7
                                else:
                                    stage_found = match.group(1)
                                return stage_found
                            else:
                                if 'Doubles' in seq['name']:
                                    stage_found = 8
                elif isinstance(seq_group, dict) and 'name' in seq_group:
                    # Handle case where outer list contains dicts directly
                    print(f"Found sequence name: {seq_group['name']}")
                    match = re.search(r'_Stage(\d+)', seq_group['name'])
                    if match:
                        if 'FreeRun' in seq_group['name']:
                            stage_found = int(match.group(1)) + 7
                        else:
                            stage_found = match.group(1)
                        return stage_found
                    else:
                        if 'Doubles' in seq_group['name']:
                            stage_found = 8
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

def detect_stage(root):
    """
    Extracts the stage from metadata if available.
    Handles nested structure of sequences in metadata.
    """
    
    path_root = Path(root)
    metadata_reader = utils.SessionData()
    session_settings = utils.load_json(metadata_reader, path_root/"SessionSettings")
    stage_found = None
    sequences = session_settings.iloc[0]['metadata'].sequences
        
    # Handle the nested list structure
    if isinstance(sequences, list):
        # Iterate through outer list
        for seq_group in sequences:
            if isinstance(seq_group, list):
                # Iterate through inner list
                for seq in seq_group:
                    if isinstance(seq, dict) and 'name' in seq:
                        print(f"Found sequence name: {seq['name']}")
                        match = re.search(r'_Stage(\d+)', seq['name'])
                        if match:
                            if 'FreeRun' in seq['name']:
                                stage_found = int(match.group(1)) + 7
                            else:
                                stage_found = match.group(1)
                            return stage_found
                        else:
                            if 'Doubles' in seq['name']:
                                stage_found = 8
            elif isinstance(seq_group, dict) and 'name' in seq_group:
                # Handle case where outer list contains dicts directly
                print(f"Found sequence name: {seq_group['name']}")
                match = re.search(r'_Stage(\d+)', seq_group['name'])
                if match:
                    if 'FreeRun' in seq_group['name']:
                        stage_found = int(match.group(1)) + 7
                    else:
                        stage_found = match.group(1)
                    return stage_found
                else:
                    if 'Doubles' in seq_group['name']:
                        stage_found = 8
                        return stage_found
        
    print(f"Final stage detected: {stage_found}")
    return stage_found if stage_found else "Unknown"

def calculate_overall_decision_accuracy(events_df):
    """
    Calculate decision accuracy for r1/r2 trials.
    
    Parameters:
    -----------
    events_df : pandas.DataFrame
        DataFrame containing trial events with columns: 'timestamp', 'EndInitiation',
        'r1_olf_valve', 'r2_olf_valve', 'r1_poke', 'r2_poke'
        
    Returns:
    --------
    dict
        Dictionary containing accuracy metrics for r1, r2, and overall trials
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

    # Process each trial
    for end_idx in end_initiation_indices:
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
            # else:
            #     closest_valve_idx = i
            #     trial_type = 'nonR'
            #     break 
                
        # Skip if no valve activation found before this trial end
        if closest_valve_idx is None:
            continue
            
        # Count trial by type
        if trial_type == 'r1':
            r1_total += 1
        elif trial_type == 'r2':
            r2_total += 1
            
        # Find the first poke after trial end
        for j in range(end_idx + 1, len(events_df)):
            if events_df.loc[j, 'r1_poke'] or events_df.loc[j, 'r2_poke']:
                # Correct if poke matches trial type
                if trial_type == 'r1' and events_df.loc[j, 'r1_poke']:
                    r1_correct += 1
                elif trial_type == 'r2' and events_df.loc[j, 'r2_poke']:
                    r2_correct += 1
                break

    # Calculate accuracy percentages with safety checks for division by zero
    r1_accuracy = (r1_correct / r1_total * 100) if r1_total > 0 else 0
    r2_accuracy = (r2_correct / r2_total * 100) if r2_total > 0 else 0
    overall_accuracy = ((r1_correct + r2_correct) / (r1_total + r2_total) * 100) if (r1_total + r2_total) > 0 else 0
    
    # Return detailed accuracy metrics
    return {
        'r1_total': r1_total,
        'r1_correct': r1_correct,
        'r1_accuracy': r1_accuracy,
        'r2_total': r2_total,
        'r2_correct': r2_correct,
        'r2_accuracy': r2_accuracy,
        'overall_accuracy': overall_accuracy
    }

def calculate_overall_decision_specificity_freerun(events_df): # TODO
    """
    Calculate decision specificity for r1/r2 trials.
    
    Parameters:
    -----------
    events_df : pandas.DataFrame
        DataFrame containing trial events with columns: 'timestamp', 'EndInitiation',
        'r1_olf_valve', 'r2_olf_valve', 'odourC_olf_valve', 'odourD_olf_valve', 
        'odourE_olf_valve', 'odourF_olf_valve', 'odourG_olf_valve', 'r1_poke', 'r2_poke'
        
    Returns:
    --------
    dict
        Dictionary containing specificity metrics for r1, r2, and overall trials
    """

    # Initialize counters
    r1_correct = 0
    r1_total = 0
    r2_correct = 0
    r2_total = 0
    C_total = C1_poke = C2_poke = 0
    D_total = D1_poke = D2_poke = 0
    E_total = E1_poke = E2_poke = 0
    F_total = F1_poke = F2_poke = 0
    G_total = G1_poke = G2_poke = 0

    
    # Find all trial end points
    end_initiation_indices = events_df.index[events_df['EndInitiation'] == True].tolist()

    # Process each trial
    for end_idx in end_initiation_indices:
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
        if trial_type == 'r1':
            r1_total += 1
        elif trial_type == 'r2':
            r2_total += 1
        elif trial_type == 'C':
            C_total += 1
        elif trial_type == 'D':
            D_total += 1
        elif trial_type == 'E':
            E_total += 1
        elif trial_type == 'F':
            F_total += 1
        elif trial_type == 'G':
            G_total += 1

        # Find the first poke after trial end
        for j in range(end_idx + 1, len(events_df)):
            if events_df.loc[j, 'r1_poke'] or events_df.loc[j, 'r2_poke']:
                # Correct if poke matches trial type
                if trial_type == 'r1' and events_df.loc[j, 'r1_poke']:
                    r1_correct += 1
                elif trial_type == 'r2' and events_df.loc[j, 'r2_poke']:
                    r2_correct += 1
                elif trial_type == 'C' and events_df.loc[j, 'r1_poke']:
                    C1_poke += 1
                elif trial_type == 'C' and events_df.loc[j, 'r2_poke']:
                    C2_poke += 1
                elif trial_type == 'D' and events_df.loc[j, 'r1_poke']:
                    D1_poke += 1
                elif trial_type == 'D' and events_df.loc[j, 'r2_poke']:
                    D2_poke += 1
                elif trial_type == 'E' and events_df.loc[j, 'r1_poke']:
                    E1_poke += 1
                elif trial_type == 'E' and events_df.loc[j, 'r2_poke']:
                    E2_poke += 1
                elif trial_type == 'F' and events_df.loc[j, 'r1_poke']:
                    F1_poke += 1
                elif trial_type == 'F' and events_df.loc[j, 'r2_poke']:
                    F2_poke += 1
                elif trial_type == 'G' and events_df.loc[j, 'r1_poke']:
                    G1_poke += 1
                elif trial_type == 'G' and events_df.loc[j, 'r2_poke']:
                    G2_poke += 1
                break
        
    # Calculate specificity percentages with safety checks for division by zero
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
    print('Calculating response times...')
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

        # Skip if no valve activation found before this trial end
        if closest_valve_idx is None:
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

def get_decision_specificity(data_path):
    """
    Calculate decision specificity for each trial in a single session.
    
    Parameters:
    -----------
    data_path : str or Path
        Path to session data directory
        
    Returns:
    --------
    dict
        Dictionary with decision specificity metrics or None if calculation fails
    """
    return RewardAnalyser.get_decision_specificity(data_path)
