import re
import sys
import argparse
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import zoneinfo
import src.session_utils as utils  
import harp
import yaml
from functools import reduce


def process_olfactometer_valves(df, valve_col, on_label, off_label, event_frames):
    """
    Finds the ON and OFF events for each olfactometer-valve combination
    """
    if not df.empty and valve_col in df.columns:
        try:
            # ON events
            on_df = df[df[valve_col] == True].copy()
            if not on_df.empty:
                on_df = on_df[['Time']]
                on_df[on_label] = True
                event_frames.append(on_df)
                print(f"Added {len(on_df)} {on_label} events")

                # OFF events
                df_sorted = df.sort_values('Time').reset_index(drop=True)
                valve_prev = df_sorted[valve_col].shift(1)
                valve_now = df_sorted[valve_col]
                off_mask = (valve_prev == True) & (valve_now == False)
                off_df = df_sorted.loc[off_mask, ['Time']].copy()
                off_df[off_label] = True
                event_frames.append(off_df)
                print(f"Added {len(off_df)} {off_label} events")
        except Exception as e:
            print(f"Error processing {on_label}: {e}")

        return event_frames
    