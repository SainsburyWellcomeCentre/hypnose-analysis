from pathlib import Path
import pandas as pd
import datetime
import zoneinfo
import re


class TimeProcessor:
    """Utility class for processing timestamps and time alignment"""
    
    @staticmethod
    def calculate_real_time_offset(root: Path, heartbeat: pd.DataFrame) -> pd.Timedelta:
        """Calculate real-time offset for timestamp alignment"""
        if heartbeat.empty or 'Time' not in heartbeat.columns or len(heartbeat) == 0:
            return pd.Timedelta(0)
        
        try:
            # Extract timestamp from directory name
            real_time_str = root.name
            match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', real_time_str)
            if not match:
                real_time_str = root.parent.name
                match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', real_time_str)
            
            if not match:
                return pd.Timedelta(0)
            
            real_time_str = match.group(0)
            real_time_ref_utc = datetime.datetime.strptime(real_time_str, '%Y-%m-%dT%H-%M-%S')
            real_time_ref_utc = real_time_ref_utc.replace(tzinfo=datetime.timezone.utc)
            uk_tz = zoneinfo.ZoneInfo("Europe/London")
            real_time_ref = real_time_ref_utc.astimezone(uk_tz)
            
            start_time_hardware = heartbeat['Time'].iloc[0]
            start_time_dt = start_time_hardware.to_pydatetime()
            if start_time_dt.tzinfo is None:
                start_time_dt = start_time_dt.replace(tzinfo=uk_tz)
            
            return real_time_ref - start_time_dt
            
        except Exception as e:
            print(f"Error calculating real-time offset: {e}")
            return pd.Timedelta(0)
    
    @staticmethod
    def apply_time_offset(dataframes: list, offset: pd.Timedelta):
        """Apply time offset to list of DataFrames"""
        for df in dataframes:
            if not df.empty and 'Time' in df.columns:
                df['Time'] = df['Time'] + offset
    
    @staticmethod
    def create_timestamp_mapper(heartbeat: pd.DataFrame) -> pd.Series:
        """Create timestamp interpolation mapper"""
        if heartbeat.empty or 'Time' not in heartbeat.columns or 'TimestampSeconds' not in heartbeat.columns:
            return pd.Series()
        
        heartbeat['Time'] = pd.to_datetime(heartbeat['Time'], errors='coerce')
        return pd.Series(data=heartbeat['Time'].values, index=heartbeat['TimestampSeconds'])
    
    @staticmethod
    def interpolate_time(seconds: float, timestamp_mapper: pd.Series) -> pd.Timestamp:
        """Interpolate timestamps from seconds"""
        if timestamp_mapper.empty:
            return pd.NaT
        
        int_seconds = int(seconds)
        fractional_seconds = seconds % 1
        
        if int_seconds in timestamp_mapper.index:
            base_time = timestamp_mapper.loc[int_seconds]
            return base_time + pd.to_timedelta(fractional_seconds, unit='s')
        
        return pd.NaT
