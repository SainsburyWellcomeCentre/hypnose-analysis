import argparse
import sys, os, re
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import harp
import yaml
from collections import defaultdict

from src import utils
from src.analysis import RewardAnalyser, get_decision_accuracy, get_response_time, \
    get_decision_sensitivity, get_false_alarm, get_sequence_completion, get_false_alarm_bias
from src.processing.detect_stage import detect_stage

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
    stage = None

    # Accuracy variables 
    all_r1_correct = 0
    all_r1_total = 0
    all_r2_correct = 0
    all_r2_total = 0
    
    # Response time variables
    all_rt = []
    all_r1_correct_rt = []
    all_r1_incorrect_rt = []
    all_r2_correct_rt = []
    all_r2_incorrect_rt = []
    all_trial_id = []

    # false alarm variables 
    all_C_pokes = []
    all_C_trials = []
    all_D_pokes = []
    all_D_trials = []
    all_E_pokes = []
    all_E_trials = []
    all_F_pokes = []
    all_F_trials = []
    all_G_pokes = []
    all_G_trials = []

    # false alarm bias variables 
    all_odour_interval_pokes = defaultdict(lambda: defaultdict(int))
    all_odour_interval_trials = defaultdict(lambda: defaultdict(int))
    all_odour_interval_false_alarm = defaultdict(lambda: defaultdict(lambda: None))
    all_interval_pokes = {}
    all_interval_trials = {}
    all_interval_false_alarm = {}

    all_odour_same_olf_pokes = defaultdict(int)
    all_odour_same_olf_trials = defaultdict(int)
    all_odour_same_olf_false_alarm = {}
    all_odour_diff_olf_pokes = defaultdict(int)
    all_odour_diff_olf_trials = defaultdict(int)
    all_odour_diff_olf_false_alarm = {}
    all_same_olf_pokes = 0
    all_same_olf_trials = 0
    all_same_olf_false_alarm = {}
    all_diff_olf_pokes = 0
    all_diff_olf_trials = 0
    all_diff_olf_false_alarm = {}

    # sequence completion variables 
    all_rew_trials = 0
    all_non_rew_trials = 0
    
    # sensitivity varibles 
    all_r1_respond = 0
    all_r1_total = 0
    all_r2_respond = 0
    all_r2_total = 0
    
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
            if stage == 1:
                analyzer._reward_analyser_stage1(session_dir, reward_a, reward_b)
            else:
                analyzer._reward_analyser_stage2to8(session_dir, reward_a, reward_b)
        except Exception as e:
            print(f"Error running reward analysis: {e}")
        
        # Calculate decision accuracy
        accuracy_data = get_decision_accuracy(session_dir)

        # Calculate response time
        response_time = get_response_time(session_dir)

        # Calculate false alarms
        false_alarm = get_false_alarm(session_dir)

        # Calculate false alarm bias
        false_alarm_bias = get_false_alarm_bias(session_dir)

        # Calculate sequence completion 
        sequence_completion = get_sequence_completion(session_dir)

        # Calculate decision sensitivity
        sensitivity = get_decision_sensitivity(session_dir)
        
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
            all_rt.append(response_time['rt'])
            all_r1_correct_rt.append(response_time['r1_correct_rt'])
            all_r1_correct_rt.append(response_time['r1_correct_rt'])
            all_r1_incorrect_rt.append(response_time['r1_incorrect_rt'])
            all_r2_correct_rt.append(response_time['r2_correct_rt'])
            all_r2_incorrect_rt.append(response_time['r2_incorrect_rt'])
            all_trial_id.append(response_time['trial_id'])

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
                'false_alarm_rt': response_time['false_alarm_rt'], 
                'trial_id': response_time['trial_id']
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
                'false_alarm_rt': np.nan, 
                'trial_id': np.nan
            })

        # Add false alarm data
        if false_alarm:
            all_C_pokes.append(false_alarm['C_pokes']) 
            all_C_trials.append(false_alarm['C_trials'])
            all_D_pokes.append(false_alarm['D_pokes']) 
            all_D_trials.append(false_alarm['D_trials'])
            all_E_pokes.append(false_alarm['E_pokes']) 
            all_E_trials.append(false_alarm['E_trials'])
            all_F_pokes.append(false_alarm['F_pokes']) 
            all_F_trials.append(false_alarm['F_trials'])
            all_G_pokes.append(false_alarm['G_pokes']) 
            all_G_trials.append(false_alarm['G_trials'])

            session_info.update({
                'C_pokes': false_alarm['C_pokes'],
                'C_trials': false_alarm['C_trials'],
                'D_pokes': false_alarm['D_pokes'],
                'D_trials': false_alarm['D_trials'],
                'E_pokes': false_alarm['E_pokes'],
                'E_trials': false_alarm['E_trials'],
                'F_pokes': false_alarm['F_pokes'],
                'F_trials': false_alarm['F_trials'],
                'G_pokes': false_alarm['G_pokes'],
                'G_trials': false_alarm['G_trials'],
                'C_false_alarm': false_alarm['C_false_alarm'],
                'D_false_alarm': false_alarm['D_false_alarm'],
                'E_false_alarm': false_alarm['E_false_alarm'],
                'F_false_alarm': false_alarm['F_false_alarm'],
                'G_false_alarm': false_alarm['G_false_alarm'],
                'overall_false_alarm': false_alarm['overall_false_alarm'],
            })
        else:
            session_info.update({
                'C_pokes': np.nan,
                'C_trials': np.nan,
                'D_pokes': np.nan,
                'D_trials': np.nan,
                'E_pokes': np.nan,
                'E_trials': np.nan,
                'F_pokes': np.nan,
                'F_trials': np.nan,
                'G_pokes': np.nan,
                'G_trials': np.nan,
                'C_false_alarm': np.nan,
                'D_false_alarm': np.nan,
                'E_false_alarm': np.nan,
                'F_false_alarm': np.nan,
                'G_false_alarm': np.nan,
                'overall_false_alarm': np.nan,
            })

        # Add false alarm bias data 
        if false_alarm_bias and all(value != 0 for value in false_alarm_bias.values()):
            nonR_odours = false_alarm_bias['odour_interval_pokes'].keys()
            first_odour = next(iter(false_alarm_bias['odour_interval_pokes']))
            intervals = false_alarm_bias['odour_interval_pokes'][first_odour].keys()
    
            for odour in nonR_odours:
                # time bias
                for interval in intervals:
                    all_odour_interval_pokes[odour][interval] += false_alarm_bias['odour_interval_pokes'][odour][interval]
                    all_odour_interval_trials[odour][interval] += false_alarm_bias['odour_interval_trials'][odour][interval]
                
                # olfactometer bias
                all_odour_same_olf_pokes[odour] += false_alarm_bias['odour_same_olf_pokes'][odour]
                all_odour_same_olf_trials[odour] += false_alarm_bias['odour_same_olf_trials'][odour]
                all_odour_diff_olf_pokes[odour] += false_alarm_bias['odour_diff_olf_pokes'][odour]
                all_odour_diff_olf_trials[odour] += false_alarm_bias['odour_diff_olf_trials'][odour]

            all_same_olf_pokes += false_alarm_bias['same_olf_pokes']
            all_same_olf_trials += false_alarm_bias['same_olf_trials']
            all_diff_olf_pokes += false_alarm_bias['diff_olf_pokes']
            all_diff_olf_trials += false_alarm_bias['diff_olf_trials']

            session_info.update({'odour_interval_pokes': false_alarm_bias['odour_interval_pokes'],
                    'odour_interval_trials': false_alarm_bias['odour_interval_trials'],
                    'odour_interval_false_alarm': false_alarm_bias['odour_interval_false_alarm'],
                    'interval_pokes': false_alarm_bias['interval_pokes'],
                    'interval_trials': false_alarm_bias['interval_trials'],
                    'interval_false_alarm': false_alarm_bias['interval_false_alarm'],
                    'odour_same_olf_pokes': false_alarm_bias['odour_same_olf_pokes'], 
                    'odour_same_olf_trials': false_alarm_bias['odour_same_olf_trials'], 
                    'odour_same_olf_false_alarm': false_alarm_bias['odour_same_olf_false_alarm'], 
                    'odour_diff_olf_pokes': false_alarm_bias['odour_diff_olf_pokes'], 
                    'odour_diff_olf_trials': false_alarm_bias['odour_diff_olf_trials'], 
                    'odour_diff_olf_false_alarm': false_alarm_bias['odour_diff_olf_false_alarm'], 
                    'same_olf_pokes': false_alarm_bias['same_olf_pokes'], 
                    'same_olf_trials': false_alarm_bias['same_olf_trials'], 
                    'same_olf_false_alarm': false_alarm_bias['same_olf_false_alarm'], 
                    'diff_olf_pokes': false_alarm_bias['diff_olf_pokes'], 
                    'diff_olf_trials': false_alarm_bias['diff_olf_trials'], 
                    'diff_olf_false_alarm': false_alarm_bias['diff_olf_false_alarm']
                    })
        else:
            session_info.update({'odour_interval_pokes': 0,
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
                    'odour_diff_olf_false_alarm': 0, 
                    'same_olf_pokes': 0, 
                    'same_olf_trials': 0, 
                    'same_olf_false_alarm': 0, 
                    'diff_olf_pokes': 0, 
                    'diff_olf_trials': 0, 
                    'diff_olf_false_alarm': 0
                    })
            
        # Add sequence completion data
        if sequence_completion:
            all_rew_trials += sequence_completion['rew_trials']
            all_non_rew_trials += sequence_completion['non_rew_trials']
            
            session_info.update({
                'rew_trials': sequence_completion['rew_trials'],
                'non_rew_trials': sequence_completion['non_rew_trials'],
                'completion_ratio': sequence_completion['completion_ratio']
            })
        else:
            session_info.update({
                'rew_trials': 0,
                'non_rew_trials': 0,
                'completion_ratio': 0
            })    
        
        # Add sensitivity data
        if sensitivity:
            all_r1_respond += sensitivity['r1_respond']
            # all_r1_total += sensitivity['r1_total']
            all_r2_respond += sensitivity['r2_respond']
            # all_r2_total += sensitivity['r2_total']

            session_info.update({
                'r1_respond': sensitivity['r1_respond'],
                'r1_total': sensitivity['r1_total'],
                'r1_sensitivity': sensitivity['r1_sensitivity'],
                'r2_respond': sensitivity['r2_respond'],
                'r2_total': sensitivity['r2_total'],
                'r2_sensitivity': sensitivity['r2_sensitivity'],
                'overall_sensitivity': sensitivity['overall_sensitivity']
            })
        else:
            session_info.update({
                'r1_respond': 0,
                'r1_total': 0,
                'r1_sensitivity': 0,
                'r2_respond': 0,
                'r2_total': 0,
                'r2_sensitivity': 0,
                'overall_sensitivity': 0
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
        
        if response_time and response_time['rt']:
            print(f"  Response Time: R1={response_time['r1_avg_rt']:.1f}, \
                  R2={response_time['r2_avg_rt']:.1f}")
        
        if false_alarm:
            print(f"  False alarm: C={false_alarm['C_false_alarm']:.1f}% D={false_alarm['D_false_alarm']:.1f}%, "
                  f"E={false_alarm['E_false_alarm']:.1f}% F={false_alarm['F_false_alarm']:.1f}%, "
                  f"G={false_alarm['G_false_alarm']:.1f}%")
            print(f"  Overall false alarm rate: {false_alarm['overall_false_alarm']:.1f}%")
        
        if false_alarm_bias and all(value != 0 for value in false_alarm_bias.values()):
            for odour, rate in false_alarm_bias['odour_same_olf_false_alarm'].items():
                print(f"\nFalse alarm same-olfactometer bias rates for odour {odour}: {rate:.1f}%")
            for odour, rate in false_alarm_bias['odour_diff_olf_false_alarm'].items():
                print(f"\nFalse alarm diff-olfactometer bias rates for odour {odour}: {rate:.1f}%")                
            print(f"  Overall false alarm same-olfactometer bias rate: {false_alarm_bias['same_olf_false_alarm']:.1f}%")
            print(f"  Overall false alarm diff-olfactometer bias rate: {false_alarm_bias['diff_olf_false_alarm']:.1f}%")

        if sequence_completion and all(value != 0 for value in sequence_completion.values()):
            print(f"  Sequence completion ratio: {sequence_completion['completion_ratio']:.1f}%")
    
        if sensitivity and all(value != 0 for value in sensitivity.values()):
            print(f"  Sensitivity: A={sensitivity['r1_sensitivity']:.1f}% ({sensitivity['r1_respond']}/{sensitivity['r1_total']}), "
                  f"B={sensitivity['r2_sensitivity']:.1f}% ({sensitivity['r2_respond']}/{sensitivity['r2_total']})")
            print(f"  Overall: {sensitivity['overall_sensitivity']:.1f}%")
        
    # Calculate overall accuracy
    all_r1_accuracy = (all_r1_correct / all_r1_total * 100) if all_r1_total > 0 else 0
    all_r2_accuracy = (all_r2_correct / all_r2_total * 100) if all_r2_total > 0 else 0
    all_overall_accuracy = ((all_r1_correct + all_r2_correct) / (all_r1_total + all_r2_total) * 100) if (all_r1_total + all_r2_total) > 0 else 0
    
    # Calculate overall response time 
    all_r1_correct_rt = np.array(sum(all_r1_correct_rt, []))
    all_r1_incorrect_rt = np.array(sum(all_r1_incorrect_rt, []))
    all_r2_correct_rt = np.array(sum(all_r2_correct_rt, []))
    all_r2_incorrect_rt = np.array(sum(all_r2_incorrect_rt, []))
    all_trial_id = np.array(np.concatenate(all_trial_id))
    all_rt = np.array(sum(all_rt, []))

    window_size = 10
    avg_response_time = np.convolve(all_rt, np.ones(window_size)/window_size, mode='valid')

    all_r1_rt = np.mean(np.concatenate([all_r1_correct_rt, all_r1_incorrect_rt]))
    all_r2_rt = np.mean(np.concatenate([all_r2_correct_rt, all_r2_incorrect_rt]))
    all_hit_rt = np.mean(np.concatenate([all_r1_correct_rt, all_r2_correct_rt]))
    all_false_alarm_rt = np.mean(np.concatenate([all_r1_incorrect_rt, all_r2_incorrect_rt]))

    # Calculate overall false alarm 
    all_C_pokes = np.sum(all_C_pokes)
    all_D_pokes = np.sum(all_D_pokes)
    all_E_pokes = np.sum(all_E_pokes)
    all_F_pokes = np.sum(all_F_pokes)
    all_G_pokes = np.sum(all_G_pokes)
    
    all_C_trials = np.sum(all_C_trials)
    all_D_trials = np.sum(all_D_trials)
    all_E_trials = np.sum(all_E_trials)
    all_F_trials = np.sum(all_F_trials)
    all_G_trials = np.sum(all_G_trials)

    all_C_false_alarm = (all_C_pokes / all_C_trials * 100) if all_C_trials > 0 else 0
    all_D_false_alarm = (all_D_pokes / all_D_trials * 100) if all_D_trials > 0 else 0
    all_E_false_alarm = (all_E_pokes / all_E_trials * 100) if all_E_trials > 0 else 0
    all_F_false_alarm = (all_F_pokes / all_F_trials * 100) if all_F_trials > 0 else 0
    all_G_false_alarm = (all_G_pokes / all_G_trials * 100) if all_G_trials > 0 else 0

    all_nonR_pokes = all_C_pokes + all_D_pokes + all_E_pokes + all_F_pokes + all_G_pokes
    all_nonR_trials = all_C_trials + all_D_trials + all_E_trials + all_F_trials + all_G_trials
    all_overall_false_alarm = (all_nonR_pokes / all_nonR_trials * 100) if all_nonR_trials > 0 else 0

    # Calculate overall false alarm time and olfactometer bias 
    if stage > 8 and stage < 9:
        for odour in nonR_odours:
            for interval in intervals:
                all_odour_interval_false_alarm[odour][interval] = (all_odour_interval_pokes[odour][interval] / all_odour_interval_trials[odour][interval] * 100) if all_odour_interval_trials[odour][interval] else 0

            all_odour_same_olf_false_alarm[odour] = (all_odour_same_olf_pokes[odour] / all_odour_same_olf_trials[odour] * 100) if all_odour_same_olf_trials[odour] else 0
            all_odour_diff_olf_false_alarm[odour] = (all_odour_diff_olf_pokes[odour] / all_odour_diff_olf_trials[odour] * 100) if all_odour_diff_olf_trials[odour] else 0

        all_same_olf_false_alarm = (all_same_olf_pokes / all_same_olf_trials * 100) if all_same_olf_trials else 0
        all_diff_olf_false_alarm = (all_diff_olf_pokes / all_diff_olf_trials * 100) if all_diff_olf_trials else 0

        for interval in intervals:
            all_interval_pokes[interval] = np.sum([all_odour_interval_pokes[odour][interval] for odour in nonR_odours])
            all_interval_trials[interval] = np.sum([all_odour_interval_trials[odour][interval] for odour in nonR_odours])
            all_interval_false_alarm[interval] = (all_interval_pokes[interval] / all_interval_trials[interval] * 100) if all_interval_trials[interval] else 0
    else:
        all_odour_same_olf_false_alarm = 0
        all_odour_diff_olf_false_alarm = 0 
        all_same_olf_false_alarm = 0 
        all_diff_olf_false_alarm = 0 
        all_odour_interval_false_alarm = 0 
        all_interval_pokes = 0 
        all_interval_trials = 0 
        all_interval_false_alarm = 0 

    # Calculate overall sequence completion ratio
    if stage >= 9:
        overall_completion_ratio = all_rew_trials / (all_rew_trials + all_non_rew_trials) * 100 if (all_rew_trials + all_non_rew_trials) > 0 else 0
    else:
        overall_completion_ratio = 0

    # Calculate overall sensitivity
    if stage >= 8.2:
        all_r1_sensitivity = (all_r1_respond / all_r1_total * 100) if all_r1_total > 0 else 0
        all_r2_sensitivity = (all_r2_respond / all_r2_total * 100) if all_r2_total > 0 else 0
        all_overall_sensitivity = ((all_r1_respond + all_r2_respond) / (all_r1_total + all_r2_total) * 100) if (all_r1_total + all_r2_total) > 0 else 0
    else:
        all_r1_sensitivity = 0
        all_r2_sensitivity = 0
        all_overall_sensitivity = 0
    
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
    print(f"Response time: R1={all_r1_rt:.1f} s, R2={all_r2_rt:.1f} s")
    print(f"Response time: Hit={all_hit_rt:.1f} s, FalseAlarm={all_false_alarm_rt:.1f} s")
    if stage > 7:
        print(f"False alarm rate: C={all_C_false_alarm:.1f}%, D={all_D_false_alarm:.1f}%, E={all_E_false_alarm:.1f}%, F={all_F_false_alarm:.1f}%, G={all_G_false_alarm:.1f}%")
        print(f"Overall false alarm rate: {all_overall_false_alarm:.1f}%")
    if stage >= 9:
        print(f"Overall sequence completion: {overall_completion_ratio:.1f}%")
    if stage > 8 and stage < 9:
        print(f"False alarm same-olfactometer bias: {all_same_olf_false_alarm:.1f}%")
        print(f"False alarm diff-olfactometer bias: {all_diff_olf_false_alarm:.1f}%")
    if stage >= 8.2:
        print(f"Sensitivity: A={all_r1_sensitivity:.1f}% ({all_r1_respond}/{all_r1_total}), B={all_r2_sensitivity:.1f}% ({all_r2_respond}/{all_r2_total})")
        print(f"Overall sensitivity: {all_overall_sensitivity:.1f}%")
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
        'all_false_alarm_rt': all_false_alarm_rt,
        'avg_response_time': avg_response_time, 
        'all_trial_id': all_trial_id,
        'all_C_false_alarm': all_C_false_alarm,
        'all_D_false_alarm': all_D_false_alarm,
        'all_E_false_alarm': all_E_false_alarm,
        'all_F_false_alarm': all_F_false_alarm,
        'all_G_false_alarm': all_G_false_alarm,
        'all_overall_false_alarm': all_overall_false_alarm,
        'overall_completion_ratio': overall_completion_ratio, 
        'all_odour_interval_pokes': all_odour_interval_pokes,
        'all_odour_interval_trials': all_odour_interval_trials,
        'all_odour_interval_false_alarm': all_odour_interval_false_alarm,
        'all_interval_pokes': all_interval_pokes,
        'all_interval_trials': all_interval_trials,
        'all_interval_false_alarm': all_interval_false_alarm,
        'all_odour_same_olf_pokes': all_odour_same_olf_pokes, 
        'all_odour_same_olf_trials': all_odour_same_olf_trials, 
        'all_odour_same_olf_false_alarm': all_odour_same_olf_false_alarm, 
        'all_odour_diff_olf_pokes': all_odour_diff_olf_pokes, 
        'all_odour_diff_olf_trials': all_odour_diff_olf_trials, 
        'all_odour_same_olf_false_alarm': all_odour_same_olf_false_alarm, 
        'all_same_olf_pokes': all_same_olf_pokes, 
        'all_same_olf_trials': all_same_olf_trials, 
        'all_same_olf_false_alarm': all_same_olf_false_alarm, 
        'all_diff_olf_pokes': all_diff_olf_pokes, 
        'all_diff_olf_trials': all_diff_olf_trials, 
        'all_diff_olf_false_alarm': all_diff_olf_false_alarm,
        'r1_respond': all_r1_respond,
        'r1_sensitivity': all_r1_sensitivity,
        'r2_respond': all_r2_respond,
        'r2_sensitivity': all_r2_sensitivity,
        'overall_sensitivity': all_overall_sensitivity,
    }
    
    return results

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # sys.argv.append("/Volumes/harris/hypnose/rawdata/sub-026_id-077/ses-59_date-20250616")
        sys.argv.append("/Volumes/harris/hypnose/rawdata/sub-025_id-076/ses-81_date-20250715")

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
