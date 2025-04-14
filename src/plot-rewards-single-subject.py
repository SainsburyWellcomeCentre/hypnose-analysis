import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import argparse
import re
from pathlib import Path
import harp
import utils
from analysis import detect_stage

def process_subject_sessions(subject_folder):
    """
    Process all sessions for a subject and return a DataFrame with rewards data
    
    Args:
        subject_folder: Path to the subject's data folder
    
    Returns:
        DataFrame with session information including reward counts
    """
    base_dir = Path(subject_folder)
    
    # Find all session directories for this subject, filtering out non-directories
    session_dirs = [d for d in base_dir.glob('ses-*_date-*/behav/*') if d.is_dir()]
    
    print(f"Found {len(session_dirs)} sessions for subject {base_dir.name}")
    
    # Dictionary to store processed data for each session
    sessions_data = {}
    
    # Group directories by session_id
    session_groups = {}
    for session_dir in session_dirs:
        # Extract session ID from path
        session_match = re.search(r'ses-(\d+)_date-(\d+)', str(session_dir))
        if session_match:
            session_id = session_match.group(1)
            session_date = session_match.group(2)
        else:
            # Use directory name as fallback
            session_id = session_dir.parent.name
        
        # Add directory to the corresponding session group
        if session_id not in session_groups:
            session_groups[session_id] = []
        session_groups[session_id].append(session_dir)
    
    # Process each session (potentially with multiple files)
    for session_id, dirs in sorted(session_groups.items(), key=lambda x: int(x[0]) if x[0].isdigit() else float('inf')):
        try:
            print(f"\nProcessing session: {session_id} with {len(dirs)} directories")
            
            # Create readers for data formats
            behavior_reader = harp.reader.create_reader('device_schemas/behavior.yml', epoch=harp.io.REFERENCE_EPOCH)
            
            # Initialize counters and data containers for this session
            r1_poke_count_total = 0
            r2_poke_count_total = 0
            r1_reward_count_total = 0
            r2_reward_count_total = 0
            
            total_session_duration_sec = 0
            earliest_timestamp = None
            latest_timestamp = None
            session_duration_sec = None
            duration_str = "Unknown"
            
            # Store the first directory path for reference
            first_dir = dirs[0]

            # Extract date from directory name first, so it's available regardless of errors
            date_match = re.search(r'date-(\d{8})', str(first_dir))
            if date_match:
                date_str = date_match.group(1)
                # Format date as YYYY-MM-DD
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            else:
                formatted_date = first_dir.stem  # Use directory name as fallback

            # Detect stage for this session
            try:
                stage = detect_stage(first_dir)
            except Exception as e:
                print(f"Error detecting stage for session {session_id}: {e}")
                stage = "Unknown"
            
            # Process each directory within the session
            for session_dir in dirs:
                print(f"  Processing directory: {session_dir}")
                
                # Load the data streams
                try:
                    digital_input_data = utils.load(behavior_reader.DigitalInputState, session_dir/"Behavior")
                    if digital_input_data is not None:
                        r1_poke_count = digital_input_data['DIPort1'].sum() if 'DIPort1' in digital_input_data else 0
                        r2_poke_count = digital_input_data['DIPort2'].sum() if 'DIPort2' in digital_input_data else 0
                        r1_poke_count_total += r1_poke_count
                        r2_poke_count_total += r2_poke_count
                except Exception:
                    print(f"  No digital_input_data found in {session_dir}")
                
                try:
                    heartbeat = utils.load(behavior_reader.TimestampSeconds, session_dir/"Behavior")
                    if not heartbeat.empty:
                        start_time = heartbeat['TimestampSeconds'].iloc[0]
                        end_time = heartbeat['TimestampSeconds'].iloc[-1]
                        
                        # Calculate duration for this directory
                        dir_duration = end_time - start_time
                        total_session_duration_sec += dir_duration
                        
                        # Also keep track of global earliest/latest for reference
                        if earliest_timestamp is None or start_time < earliest_timestamp:
                            earliest_timestamp = start_time
                        
                        if latest_timestamp is None or end_time > latest_timestamp:
                            latest_timestamp = end_time
                        
                        print(f"  Directory duration: {dir_duration:.2f} seconds")
                except Exception as e:
                    print(f"  Error loading heartbeat from {session_dir}: {e}")
                
                # Process reward events
                try:
                    pulse_supply_1 = utils.load(behavior_reader.PulseSupplyPort1, session_dir/"Behavior")
                    r1_reward_count_total += len(pulse_supply_1)
                except Exception:
                    pass
                
                try:
                    pulse_supply_2 = utils.load(behavior_reader.PulseSupplyPort2, session_dir/"Behavior")
                    r2_reward_count_total += len(pulse_supply_2)
                except Exception:
                    pass
            
            # Calculate session duration across all files
            if total_session_duration_sec > 0:
                session_duration_sec = total_session_duration_sec
                
                # Convert to hours, minutes, seconds
                hours = int(session_duration_sec // 3600)
                minutes = int((session_duration_sec % 3600) // 60)
                seconds = int(session_duration_sec % 60)
                
                duration_str = f"{hours}h {minutes}m {seconds}s"
                
                # Also calculate total timespan (for reference)
                if earliest_timestamp is not None and latest_timestamp is not None:
                    total_span_sec = latest_timestamp - earliest_timestamp
                    print(f"  Total session duration (sum): {duration_str} ({session_duration_sec:.2f} sec)")
                    print(f"  Total session timespan (start to end): {total_span_sec:.2f} sec")
            
            # Now create the dictionary after all the variables have been properly set
            sessions_data[session_id] = {
                'dir': first_dir,
                'date': formatted_date,
                'num_files': len(dirs),
                'duration_sec': session_duration_sec,
                'duration': duration_str,
                'r1_pokes': r1_poke_count_total,
                'r2_pokes': r2_poke_count_total, 
                'r1_rewards': r1_reward_count_total,
                'r2_rewards': r2_reward_count_total,
                'stage': stage
            }
            
            print(f"Successfully processed session {session_id}")
            
        except Exception as e:
            print(f"Error processing session {session_id}: {e}")
    
    # Create DataFrame from sessions data
    sessions_df = pd.DataFrame.from_dict(sessions_data, orient='index')
    sessions_df.index.name = 'session_id'
    return sessions_df

def plot_rewards_by_date(sessions_df, output_file=None, subject_name=None):
    """
    Plot total rewards by session date, showing all days in the date range.
    Colors points based on training stage.
    
    Args:
        sessions_df: DataFrame with session data
        output_file: Optional path to save the plot
        subject_name: Name of the subject for plot title
    """
    # Check if DataFrame is empty or missing required columns
    if sessions_df.empty:
        print("No data available to plot.")
        return
    
    # Check if required columns exist
    required_columns = ['r1_rewards', 'r2_rewards', 'date']
    missing_columns = [col for col in required_columns if col not in sessions_df.columns]
    
    if missing_columns:
        print(f"Missing required columns for plotting: {missing_columns}")
        return
        
    # Add a column for total rewards and convert the date to a datetime object
    sessions_df['total_rewards'] = sessions_df['r1_rewards'] + sessions_df['r2_rewards']
    sessions_df['date'] = pd.to_datetime(sessions_df['date'])
    
    # Extract only the calendar date (ignoring time) for each session
    sessions_df['calendar_date'] = sessions_df['date'].dt.date
    
    # Add a column for the day of the week
    sessions_df['day_of_week'] = pd.to_datetime(sessions_df['calendar_date']).dt.day_name()
    
    # Create a complete date range from the first to the last session date
    full_date_range = pd.date_range(start=sessions_df['calendar_date'].min(), 
                                   end=sessions_df['calendar_date'].max(), freq='D')
    
    # Create a new DataFrame with the full date range
    full_sessions_df = pd.DataFrame({'calendar_date': full_date_range.date})
    
    # Merge the original sessions_df with the full date range DataFrame
    full_sessions_df = full_sessions_df.merge(sessions_df, on='calendar_date', how='left')
    
    # Fill missing day_of_week values for the full date range
    full_sessions_df['day_of_week'] = pd.to_datetime(full_sessions_df['calendar_date']).dt.day_name()
    
    # Define a color map for different stages
    stage_colors = {
        '1': 'lightblue',
        '2': 'lightgreen',
        '3': 'salmon',
        '4': 'purple',
        '5': 'orange',
        '6': 'brown',
        '7': 'pink',
        '8': 'gray',
        'Unknown': 'black'
    }
    
    # Plot total rewards vs. session date
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Group by stage and plot each group with its color
    stages_present = set()
    has_data = ~full_sessions_df['total_rewards'].isna()
    
    for stage in stage_colors:
        mask = (full_sessions_df['stage'] == stage) & has_data
        if mask.any():
            stages_present.add(stage)
            ax.scatter(full_sessions_df.loc[mask, 'calendar_date'], 
                       full_sessions_df.loc[mask, 'total_rewards'], 
                       color=stage_colors[stage], alpha=0.7, edgecolor='black',
                       s=80, label=f'Stage {stage}')
    
    # Format the x-axis with custom tick labels
    dates = full_sessions_df['calendar_date']
    
    # Create custom tick labels with month/day and weekday
    date_labels = []
    for date in dates:
        dt = pd.to_datetime(date)
        month_day = dt.strftime('%m/%d')
        weekday = dt.strftime('%a')
        date_labels.append(f"{month_day} - {weekday}")
    
    # Set the x-ticks and labels
    ax.set_xticks([pd.Timestamp(d) for d in dates])
    ax.set_xticklabels(date_labels, rotation=45, ha='right')
    
    # Format the plot
    ax.set_ylabel('Total Rewards')
    
    # Add title with subject name if provided
    if subject_name:
        ax.set_title(f'{subject_name}')
    else:
        ax.set_title('')
        
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Remove top and bottom spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Format the plot
    fig.tight_layout()
    
    # Save the plot if an output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    # Show the plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot total rewards across days for a subject")
    parser.add_argument("subject_folder", help="Path to the subject's data folder")
    parser.add_argument("--output", "-o", help="Path to save the output plot (optional)")
    args = parser.parse_args()
    
    # Extract subject name from folder path and keep only the "sub-XXX" part
    subject_folder = Path(args.subject_folder)
    full_subject_name = subject_folder.name
    
    # Use regex to extract just the "sub-XXX" part
    subject_match = re.match(r'(sub-\d+).*', full_subject_name)
    if subject_match:
        subject_name = subject_match.group(1)
    else:
        subject_name = full_subject_name  # Fallback to full name if pattern doesn't match
    
    # Process subject sessions
    sessions_df = process_subject_sessions(args.subject_folder)
    
    # Check if DataFrame is empty
    if sessions_df.empty:
        print("\nNo session data was successfully processed.")
        return
    
    # Display summary table - update to use only the columns that actually exist
    print("\nSummary of sessions:")
    
    # Check if columns exist before trying to access them
    display_columns = []
    for col in ['date', 'duration', 'r1_rewards', 'r2_rewards', 'stage']:
        if col in sessions_df.columns:
            display_columns.append(col)
    
    if display_columns:
        print(sessions_df[display_columns].to_string())
    else:
        print("No columns available to display.")
    
    # Plot rewards by date, passing subject name for title
    plot_rewards_by_date(sessions_df, args.output, subject_name)
    
    # Save CSV of sessions data
    if args.output:
        csv_path = str(Path(args.output).with_suffix('.csv'))
        sessions_df.to_csv(csv_path)
        print(f"Session data saved to {csv_path}")

if __name__ == "__main__":
    main()