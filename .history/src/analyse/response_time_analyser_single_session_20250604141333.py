import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import harp

from src import utils
from src.analysis import RewardAnalyser, get_decision_accuracy, detect_stage, get_response_time

# Filter out specific warnings
warnings.filterwarnings(
    "ignore", 
    message="Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated",
    category=FutureWarning
)

def analyze_session_folder(session_folder, reward_a=8.0, reward_b=8.0, verbose=False):
    """
    Analyze all behavioral sessions within a given session folder
    
    Parameters:
    -----------
    session_folder : str or Path
        Path to the session folder (e.g., ses-YYY_date-YYYYMMDD/)
    reward_a : float
        Volume (µL) per Reward A
    reward_b : float
        Volume (µL) per Reward B
    verbose : bool
        Whether to print detailed debug information
    
    Returns:
    --------
    dict
        Session summary with combined data
    """
    session_folder = Path(session_folder)
    behav_folder = session_folder / "behav"
    
    if not behav_folder.exists():
        print(f"No behavior folder found at {behav_folder}")
        return None
    
    # Collect all session directories
    session_dirs = [d for d in behav_folder.iterdir() if d.is_dir()]
    print(f"Found {len(session_dirs)} behavioral sessions in {behav_folder}")
    
    # Variables to track overall session
    all_rewards_r1 = 0
    all_rewards_r2 = 0
    total_duration_sec = 0
    all_r1_correct = 0
    all_r1_total = 0
    all_r2_correct = 0
    all_r2_total = 0
    stage = None
    all_r1_correct_rt = []
    all_r1_incorrect_rt = []
    all_r2_correct_rt = []
    all_r2_incorrect_rt = []
    
    # Per-session results for detailed output
    session_results = []
    
    # Process each session
    for session_dir in sorted(session_dirs):
        print(f"\nProcessing session: {session_dir.name}")
        
        # Only process if session settings exist
        if not (session_dir / "SessionSettings").exists():
            print(f"No SessionSettings found in {session_dir.name}, skipping")
            continue

        # Load session settings
        metadata_reader = utils.SessionData()
        try:
            session_settings = utils.load_json(metadata_reader, session_dir / "SessionSettings")
            
            # Detect stage if not already set
            if stage is None:
                stage = detect_stage(session_dir)
                if verbose:
                    print(f"Detected stage: {stage}")
        except Exception as e:
            print(f"Error loading session settings: {e}")
            continue
        
        # Create analyzer instance
        try:
            analyzer = RewardAnalyser(session_settings)
        except Exception as e:
            print(f"Error creating analyzer: {e}")
            continue
        
        # Calculate session length and rewards
        try:
            # Get basic session info by running the appropriate analyzer
            print(f"Running reward analysis for stage {stage}")
            if stage == "1":
                analyzer._reward_analyser_stage1(session_dir, reward_a, reward_b)
            else:
                analyzer._reward_analyser_stage2to8(session_dir, reward_a, reward_b)
        except Exception as e:
            print(f"Error running reward analysis: {e}")
        
        # Calculate decision accuracy
        accuracy_data = get_decision_accuracy(session_dir)

        # Calculate response time
        response_time = get_response_time(session_dir)
        
        # Extract reward and duration data
        session_info = {}
        
        # Reload data to extract counts (this is more reliable than capturing print output)
        try:
            # Use absolute path to device_schemas directory in project root
            project_root = Path(__file__).resolve().parents[2]  # Go up from src/analyse to the project root
            schema_path = project_root / 'device_schemas/behavior.yml'
            behavior_reader = harp.reader.create_reader(str(schema_path), epoch=harp.io.REFERENCE_EPOCH)
            
            # Fix: Changed 'root' to 'session_dir' to correctly reference the current session directory
            pulse_supply_1 = utils.load(behavior_reader.PulseSupplyPort1, session_dir/"Behavior")
            pulse_supply_2 = utils.load(behavior_reader.PulseSupplyPort2, session_dir/"Behavior")
            heartbeat = utils.load(behavior_reader.TimestampSeconds, session_dir/"Behavior")
            
            # Ensure variables are properly initialized with default values
            num_r1_rewards = 0
            num_r2_rewards = 0
            session_duration_sec = 0
            
            # Calculate rewards
            if pulse_supply_1 is not None and not pulse_supply_1.empty:
                num_r1_rewards = len(pulse_supply_1)
            if pulse_supply_2 is not None and not pulse_supply_2.empty:
                num_r2_rewards = len(pulse_supply_2)
                
            r1_volume = num_r1_rewards * reward_a
            r2_volume = num_r2_rewards * reward_b
            
            # Calculate session duration
            if heartbeat is not None and not heartbeat.empty and len(heartbeat) > 1:
                start_time_sec = heartbeat['TimestampSeconds'].iloc[0]
                end_time_sec = heartbeat['TimestampSeconds'].iloc[-1]
                session_duration_sec = end_time_sec - start_time_sec
            
            # Update session info
            session_info = {
                'session_name': session_dir.name,
                'stage': stage,
                'r1_rewards': num_r1_rewards,
                'r2_rewards': num_r2_rewards,
                'r1_volume': r1_volume,
                'r2_volume': r2_volume,
                'duration_sec': session_duration_sec
            }
            
            # Update totals
            all_rewards_r1 += num_r1_rewards
            all_rewards_r2 += num_r2_rewards
            total_duration_sec += session_duration_sec
            
        except Exception as e:
            print(f"Error extracting session metrics: {e}")
            # Make sure we set default values if there's an error
            num_r1_rewards = 0
            num_r2_rewards = 0
            r1_volume = 0
            r2_volume = 0
            session_duration_sec = 0
            
            session_info = {
                'session_name': session_dir.name,
                'stage': stage,
                'r1_rewards': num_r1_rewards,
                'r2_rewards': num_r2_rewards,
                'r1_volume': r1_volume,
                'r2_volume': r2_volume,
                'duration_sec': session_duration_sec
            }
        
        # Add accuracy data 
        if accuracy_data:
            all_r1_correct += accuracy_data['r1_correct']
            all_r1_total += accuracy_data['r1_total']
            all_r2_correct += accuracy_data['r2_correct']
            all_r2_total += accuracy_data['r2_total']
            
            session_info.update({
                'r1_correct': accuracy_data['r1_correct'],
                'r1_total': accuracy_data['r1_total'],
                'r1_accuracy': accuracy_data['r1_accuracy'],
                'r2_correct': accuracy_data['r2_correct'],
                'r2_total': accuracy_data['r2_total'],
                'r2_accuracy': accuracy_data['r2_accuracy'],
                'overall_accuracy': accuracy_data['overall_accuracy']
            })
        else:
            session_info.update({
                'r1_correct': 0,
                'r1_total': 0,
                'r1_accuracy': 0,
                'r2_correct': 0,
                'r2_total': 0,
                'r2_accuracy': 0,
                'overall_accuracy': 0
            })

        # Add response time data 
        if response_time:  
            all_r1_correct_rt += response_time['r1_correct_rt']
            all_r1_incorrect_rt += response_time['r1_incorrect_rt']
            all_r2_correct_rt += response_time['r2_correct_rt']
            all_r2_incorrect_rt += response_time['r2_incorrect_rt']
            
            session_info.update({
                'r1_correct_rt': response_time['r1_correct_rt'],
                'r1_incorrect_rt': response_time['r1_incorrect_rt'],
                'r1_avg_correct_rt': response_time['r1_avg_correct_rt'],
                'r1_avg_incorrect_rt': response_time['r1_avg_incorrect_rt'],
                'r1_avg_rt': response_time['r1_avg_rt'],
                'r2_correct_rt': response_time['r2_correct_rt'],
                'r2_incorrect_rt': response_time['r2_incorrect_rt'],
                'r2_avg_correct_rt': response_time['r2_avg_correct_rt'],
                'r2_avg_incorrect_rt': response_time['r2_avg_incorrect_rt'],
                'r2_avg_rt': response_time['r2_avg_rt'],
                'hit_rt': response_time['hit_rt'],
                'false_alarm_rt': response_time['false_alarm_rt']
            })
        
        else:
            session_info.update({
                'r1_correct_rt': np.nan,
                'r1_incorrect_rt': np.nan,
                'r1_avg_correct_rt': np.nan,
                'r1_avg_incorrect_rt': np.nan,
                'r1_avg_rt': np.nan,
                'r2_correct_rt': np.nan,
                'r2_incorrect_rt': np.nan,
                'r2_avg_correct_rt': np.nan,
                'r2_avg_incorrect_rt': np.nan,
                'r2_avg_rt': np.nan,
                'hit_rt': np.nan,
                'false_alarm_rt': np.nan 
            })

        session_results.append(session_info)
        
        # Print session summary
        print(f"\nSession Summary for {session_dir.name}:")
        print(f"  Stage: {stage}")
        print(f"  Duration: {session_duration_sec:.1f} seconds")
        print(f"  Rewards: R1={num_r1_rewards} ({r1_volume:.1f}µL), R2={num_r2_rewards} ({r2_volume:.1f}µL)")
        if accuracy_data:
            print(f"  Accuracy: R1={accuracy_data['r1_accuracy']:.1f}% ({accuracy_data['r1_correct']}/{accuracy_data['r1_total']}), "
                  f"R2={accuracy_data['r2_accuracy']:.1f}% ({accuracy_data['r2_correct']}/{accuracy_data['r2_total']})")
            print(f"  Overall: {accuracy_data['overall_accuracy']:.1f}%")
    
    # Calculate overall accuracy
    all_r1_accuracy = (all_r1_correct / all_r1_total * 100) if all_r1_total > 0 else 0
    all_r2_accuracy = (all_r2_correct / all_r2_total * 100) if all_r2_total > 0 else 0
    all_overall_accuracy = ((all_r1_correct + all_r2_correct) / (all_r1_total + all_r2_total) * 100) if (all_r1_total + all_r2_total) > 0 else 0
    
    # Calculate overall response time 
    all_r1_rt = np.mean(np.concatenate([all_r1_correct_rt, all_r1_incorrect_rt]))
    all_r2_rt = np.mean(np.concatenate([all_r2_correct_rt, all_r2_incorrect_rt]))
    all_hit_rt = np.mean(np.concatenate([all_r1_correct_rt, all_r2_correct_rt]))
    all_false_alarm_rt = np.mean(np.concatenate([all_r1_incorrect_rt, all_r2_incorrect_rt]))

    # Format time in a readable way
    h = int(total_duration_sec // 3600)
    m = int((total_duration_sec % 3600) // 60)
    s = int(total_duration_sec % 60)
    
    # Print overall summary
    print("\n========== Overall Session Summary ==========")
    print(f"Session folder: {session_folder}")
    print(f"Stage: {stage}")
    print(f"Total duration: {h}h {m}m {s}s")
    print(f"Total rewards: R1={all_rewards_r1} ({all_rewards_r1 * reward_a:.1f}µL), R2={all_rewards_r2} ({all_rewards_r2 * reward_b:.1f}µL)")
    print(f"Combined total rewards: {all_rewards_r1 + all_rewards_r2} ({all_rewards_r1 * reward_a + all_rewards_r2 * reward_b:.1f}µL)")
    print(f"Response time: R1={all_r1_rt:.1f}%, R2={all_r2_rt:.1f}%")
    print(f"Response time: Hit={all_hit_rt:.1f}%, FalseAlarm={all_false_alarm_rt:.1f}%")
    print(f"Decision accuracy: R1={all_r1_accuracy:.1f}% ({all_r1_correct}/{all_r1_total}), R2={all_r2_accuracy:.1f}% ({all_r2_correct}/{all_r2_total})")
    print(f"Overall accuracy: {all_overall_accuracy:.1f}%")
    
    # Create results dictionary
    results = {
        'session_folder': str(session_folder),
        'stage': stage,
        'total_duration_sec': total_duration_sec,
        'r1_rewards': all_rewards_r1,
        'r2_rewards': all_rewards_r2,
        'total_rewards': all_rewards_r1 + all_rewards_r2,
        'r1_volume': all_rewards_r1 * reward_a,
        'r2_volume': all_rewards_r2 * reward_b,
        'total_volume': all_rewards_r1 * reward_a + all_rewards_r2 * reward_b,
        'r1_correct': all_r1_correct,
        'r1_total': all_r1_total,
        'r1_accuracy': all_r1_accuracy,
        'r2_correct': all_r2_correct,
        'r2_total': all_r2_total, 
        'r2_accuracy': all_r2_accuracy,
        'overall_accuracy': all_overall_accuracy,
        'sessions': session_results,
        'all_r1_correct_rt': all_r1_correct_rt,
        'all_r2_correct_rt': all_r2_correct_rt,
        'all_r1_incorrect_rt': all_r1_incorrect_rt,
        'all_r2_incorrect_rt': all_r2_incorrect_rt,
        'all_r1_rt': all_r1_rt,
        'all_r2_rt': all_r2_rt,
        'all_hit_rt': all_hit_rt,
        'all_false_alarm_rt': all_false_alarm_rt
    }
    
    return results

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("/Volumes/harris/hypnose/rawdata/sub-026_id-077/ses-50_date-20250603")

    parser = argparse.ArgumentParser(description="Analyze all behavioral sessions in a folder")
    parser.add_argument("session_folder", help="Path to the session folder (e.g., sub-XXX/ses-YYY_date-YYYYMMDD)")
    parser.add_argument("--reward_a", type=float, default=4.0, help="Volume (µL) per Reward A")
    parser.add_argument("--reward_b", type=float, default=4.0, help="Volume (µL) per Reward B")
    parser.add_argument("--verbose", action="store_true", help="Print detailed debug information")
    
    args = parser.parse_args()
    
    result = analyze_session_folder(
        args.session_folder, 
        reward_a=args.reward_a, 
        reward_b=args.reward_b,
        verbose=args.verbose
    )
