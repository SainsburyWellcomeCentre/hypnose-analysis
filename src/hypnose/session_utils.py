import os
import json
from dotmap import DotMap
import pandas as pd
from pathlib import Path
from glob import glob
from aeon.io.reader import Reader, Csv
import aeon.io.api as api


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
    # print(pattern)
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

def create_unique_series(events_df):
    """Creates a unique-timestamp boolean series adding slight offsets to duplicate timestamps."""
    timestamps = events_df['Time']
    if len(timestamps) != len(set(timestamps)):
        unique_timestamps = []
        seen = set()
        for ts in timestamps:
            counter = 0
            ts_modified = ts
            while ts_modified in seen:
                counter += 1
                ts_modified = ts + pd.Timedelta(microseconds=counter)
            seen.add(ts_modified)
            unique_timestamps.append(ts_modified)
        timestamps = unique_timestamps
    return pd.Series(True, index=timestamps)

def find_session_roots(subject_folder):
    """
    Find all session root directories for a subject following the structure:
    subject_folder/ses-*_date-*/behav/*
    Returns a list of tuples: (session_id, session_date, session_path)
    """
    subject_path = Path(subject_folder)
    session_roots = []
    
    # Find all session directories following pattern 'ses-*_date-*'
    session_dirs = list(subject_path.glob('ses-*_date-*/behav/*'))
    
    for session_dir in session_dirs:
        if not session_dir.is_dir() or not (session_dir / "SessionSettings").exists():
            continue
            
        # Extract session ID and date from parent directory names
        parent_dir = session_dir.parent.parent.name
        try:
            # Parse the ses-X_date-YYYYMMDD format
            parts = parent_dir.split('_')
            session_id = parts[0].replace('ses-', '')
            session_date = parts[1].replace('date-', '')
            session_roots.append((session_id, session_date, session_dir))
        except (IndexError, AttributeError):
            print(f"Warning: Could not parse session information from {parent_dir}")
            session_roots.append(("unknown", "unknown", session_dir))
    
    # Sort session roots by session_id (numerically if possible)
    session_roots.sort(key=lambda x: int(x[0]) if x[0].isdigit() else float('inf'))
    
    return session_roots