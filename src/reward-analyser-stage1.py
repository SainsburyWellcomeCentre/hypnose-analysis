#!/usr/bin/env python3
# filepath: reward-analyser-stage1.py

import argparse
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import zoneinfo
import utils
import harp

def main():
    parser = argparse.ArgumentParser(
        description="Analyze rewards and session length."
    )
    parser.add_argument("data_path", help="Path to the root folder containing behavior data.")
    parser.add_argument("--reward_a", type=float, default=8.0, help="Volume (µL) per Reward A.")
    parser.add_argument("--reward_b", type=float, default=8.0, help="Volume (µL) per Reward B.")
    args = parser.parse_args()

    # Load data
    root = Path(args.data_path)
    behavior_reader = harp.reader.create_reader('device_schemas/behavior.yml', epoch=harp.io.REFERENCE_EPOCH)
    olfactometer_reader = harp.reader.create_reader('device_schemas/olfactometer.yml', epoch=harp.io.REFERENCE_EPOCH)

    # Load relevant data streams
    digital_input_data = utils.load(behavior_reader.DigitalInputState, root/"Behavior")
    pulse_supply_1 = utils.load(behavior_reader.PulseSupplyPort1, root/"Behavior")
    pulse_supply_2 = utils.load(behavior_reader.PulseSupplyPort2, root/"Behavior")
    heartbeat = utils.load(behavior_reader.TimestampSeconds, root/"Behavior")

    # Convert time index to column
    for df in [heartbeat, digital_input_data, pulse_supply_1, pulse_supply_2]:
        df.reset_index(inplace=True)

    # Derive real-time offset from the folder name (assuming UTC, then convert to local time)
    real_time_str = root.as_posix().split('/')[-1]
    real_time_ref_utc = datetime.datetime.strptime(real_time_str, '%Y-%m-%dT%H-%M-%S').replace(tzinfo=datetime.timezone.utc)
    uk_tz = zoneinfo.ZoneInfo("Europe/London")
    real_time_ref = real_time_ref_utc.astimezone(uk_tz)

    # Compute offset for hardware timestamps
    start_time_hardware = heartbeat['Time'].iloc[0]
    start_time_dt = start_time_hardware.to_pydatetime()
    if start_time_dt.tzinfo is None:
        start_time_dt = start_time_dt.replace(tzinfo=uk_tz)
    real_time_offset = real_time_ref - start_time_dt

    # Shift relevant data by the real-time offset
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
    total_vol_r1 = num_r1_rewards * args.reward_a
    total_vol_r2 = num_r2_rewards * args.reward_b
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

if __name__ == "__main__":
    main()