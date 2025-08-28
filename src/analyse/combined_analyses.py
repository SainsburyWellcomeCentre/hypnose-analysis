import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import matplotlib.dates as mdates
from collections import defaultdict
from src.analysis import get_decision_accuracy, get_decision_sensitivity, get_false_alarm, \
    get_sequence_completion, detect_stage, get_false_alarm_bias
import src.utils as utils

def calculate_combined_session_accuracy(subject_path, stage=None, sessions=None):
    """
    Calculate combined decision accuracy across sessions (from a specific stage or from specific sessions)
    """
    # Use utils.find_session_roots instead of the local function
    session_roots = utils.find_session_roots(subject_path)
    
    if not session_roots:
        print(f"No valid session directories found in {subject_path}")
        return
        
    # Group sessions by session_id and session_date
    grouped_sessions = {}
    for session_id, session_date, session_path in session_roots:
        try:
            detected_stage = detect_stage(session_path)
        except Exception as e:
            print(f"  Could not detect stage for {session_path.name}: {e}")
            continue

        # Apply filtering rules
        session_id_int = int(session_id)
        session_allowed = (sessions is None or session_id_int in sessions)
        if stage is None:
            stage_allowed = True
        elif isinstance(stage, int):
            stage_allowed = int(detected_stage) == stage
        else:
            stage_allowed = detected_stage == stage

        if session_allowed and stage_allowed:
            key = (session_id, session_date)
            grouped_sessions.setdefault(key, []).append(session_path)
    
    # Create a list to store combined results
    results = []
    
    # Process each grouped session
    for (session_id, session_date), session_paths in sorted(grouped_sessions.items()):
        print(f"\nProcessing Session ID: {session_id}, Date: {session_date}")
        print(f"Found {len(session_paths)} directories within this session")
        
        # Initialize combined metrics
        total_r1_correct = 0
        total_r1_repond = 0
        total_r2_correct = 0
        total_r2_repond = 0
        
        # Directory-specific results for detailed information
        dir_results = []
        
        # Process each directory within the session
        for session_path in session_paths:
            print(f"  Processing directory: {session_path.name}")
            try:
                # Get accuracy data for this directory
                accuracy = get_decision_accuracy(session_path)

                if accuracy and accuracy != {
                    'r1_respond': 0, 'r1_correct': 0, 'r1_accuracy': 0,
                    'r2_respond': 0, 'r2_correct': 0, 'r2_accuracy': 0,
                    'overall_accuracy': 0
                }:
                    # Add to totals
                    total_r1_correct += accuracy['r1_correct']
                    total_r1_repond += accuracy['r1_respond']
                    total_r2_correct += accuracy['r2_correct']
                    total_r2_repond += accuracy['r2_respond']
                    
                    dir_info = {
                        'directory': session_path.name,
                        'r1_correct': accuracy['r1_correct'],
                        'r1_respond': accuracy['r1_respond'],
                        'r1_accuracy': accuracy['r1_accuracy'],
                        'r2_correct': accuracy['r2_correct'],
                        'r2_respond': accuracy['r2_respond'],
                        'r2_accuracy': accuracy['r2_accuracy'],
                        'overall_accuracy': accuracy['overall_accuracy']
                    }
                    
                    print(f"    R1: {accuracy['r1_correct']}/{accuracy['r1_respond']} ({accuracy['r1_accuracy']:.1f}%), "
                          f"R2: {accuracy['r2_correct']}/{accuracy['r2_respond']} ({accuracy['r2_accuracy']:.1f}%)")
                    
                    dir_results.append(dir_info)
                else:
                    print(f"    No valid accuracy data found")
                    
            except Exception as e:
                print(f"    Error processing directory {session_path.name}: {str(e)}")
        
        # Calculate combined accuracy values
        if total_r1_repond + total_r2_repond > 0:
            # Calculate overall accuracy across all directories in this session
            r1_accuracy = (total_r1_correct / total_r1_repond) if total_r1_repond > 0 else 0
            r2_accuracy = (total_r2_correct / total_r2_repond) if total_r2_repond > 0 else 0
            overall_accuracy = ((total_r1_correct + total_r2_correct) / 
                               (total_r1_repond + total_r2_repond)) if (total_r1_repond + total_r2_repond) > 0 else 0
            
            # Store combined session results
            session_result = {
                'session_id': session_id,
                'session_date': session_date,
                'overall_accuracy': overall_accuracy,
                'r1_accuracy': r1_accuracy,
                'r2_accuracy': r2_accuracy,
                'r1_respond': total_r1_repond,
                'r2_respond': total_r2_repond,
                'total_trials': total_r1_repond + total_r2_repond,
                'r1_correct': total_r1_correct,
                'r2_correct': total_r2_correct,
                'directory_count': len(session_paths),
                'directories': dir_results
            }
            
            print(f"  Combined results for Session {session_id}:")
            print(f"    R1: {total_r1_correct}/{total_r1_repond} ({r1_accuracy:.1%})")
            print(f"    R2: {total_r2_correct}/{total_r2_repond} ({r2_accuracy:.1%})")
            print(f"    Overall: {total_r1_correct + total_r2_correct}/{total_r1_repond + total_r2_repond} ({overall_accuracy:.1%})")
            
            results.append(session_result)
        else:
            print(f"  No valid trials found for Session {session_id}")
            results.append({
                'session_id': session_id,
                'session_date': session_date,
                'overall_accuracy': np.nan,
                'r1_accuracy': np.nan,
                'r2_accuracy': np.nan,
                'r1_respond': 0,
                'r2_respond': 0,
                'total_trials': 0,
                'r1_correct': 0,
                'r2_correct': 0,
                'directory_count': len(session_paths),
                'directories': dir_results
            })
    
    # Create DataFrame from combined results
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'directories'} for r in results])
    
    # Print summary
    print("\nSummary of Combined Session Accuracies:")
    print("=======================================")
    for _, row in results_df.iterrows():
        if pd.notna(row['overall_accuracy']):
            print(f"Session {row['session_id']} ({row['session_date']}): " 
                  f"Overall {row['overall_accuracy']:.2%}, "
                  f"R1 {row['r1_accuracy']:.2%} ({row['r1_respond']} trials), "
                  f"R2 {row['r2_accuracy']:.2%} ({row['r2_respond']} trials)")
        else:
            print(f"Session {row['session_id']} ({row['session_date']}): No valid trials found")

    return results_df


def calculate_combined_session_sensitivity(subject_path, stage=None, sessions=None):
    """
    Calculate combined decision sensitivity across sessions (from a specific stage or from specific sessions)
    """
    # Use utils.find_session_roots instead of the local function
    session_roots = utils.find_session_roots(subject_path)
    
    if not session_roots:
        print(f"No valid session directories found in {subject_path}")
        return
        
    # Group sessions by session_id and session_date
    grouped_sessions = {}
    for session_id, session_date, session_path in session_roots:
        try:
            detected_stage = detect_stage(session_path)
        except Exception as e:
            print(f"  Could not detect stage for {session_path.name}: {e}")
            continue

        # Apply filtering rules
        session_id_int = int(session_id)
        session_allowed = (sessions is None or session_id_int in sessions)
        if stage is None:
            stage_allowed = True
        elif isinstance(stage, int):
            stage_allowed = int(detected_stage) == stage
        else:
            stage_allowed = detected_stage == stage

        if session_allowed and stage_allowed:
            key = (session_id, session_date)
            grouped_sessions.setdefault(key, []).append(session_path)
    
    # Create a list to store combined results
    results = []
    
    # Process each grouped session
    for (session_id, session_date), session_paths in sorted(grouped_sessions.items()):
        print(f"\nProcessing Session ID: {session_id}, Date: {session_date}")
        print(f"Found {len(session_paths)} directories within this session")
        
        # Initialize combined metrics
        total_r1_respond = 0
        total_r1_trials = 0
        total_r2_respond = 0
        total_r2_trials = 0
        
        # Directory-specific results for detailed information
        dir_results = []
        
        # Process each directory within the session
        for session_path in session_paths:
            print(f"  Processing directory: {session_path.name}")
            try:
                # Get sensitivity data for this directory
                sensitivity = get_decision_sensitivity(session_path)

                if sensitivity and sensitivity != {
                    'r1_total': 0, 'r1_respond': 0, 'r1_sensitivity': 0,
                    'r2_total': 0, 'r2_respond': 0, 'r2_sensitivity': 0,
                    'overall_sensitivity': 0
                }:
                    # Add to totals
                    total_r1_respond += sensitivity['r1_respond']
                    total_r1_trials += sensitivity['r1_total']
                    total_r2_respond += sensitivity['r2_respond']
                    total_r2_trials += sensitivity['r2_total']
                    
                    dir_info = {
                        'directory': session_path.name,
                        'r1_respond': sensitivity['r1_respond'],
                        'r1_trials': sensitivity['r1_total'],
                        'r1_sensitivity': sensitivity['r1_sensitivity'],
                        'r2_respond': sensitivity['r2_respond'],
                        'r2_trials': sensitivity['r2_total'],
                        'r2_sensitivity': sensitivity['r2_sensitivity'],
                        'overall_sensitivity': sensitivity['overall_sensitivity']
                    }
                    
                    print(f"    R1: {sensitivity['r1_respond']}/{sensitivity['r1_total']} ({sensitivity['r1_sensitivity']:.1f}%), "
                          f"R2: {sensitivity['r2_respond']}/{sensitivity['r2_total']} ({sensitivity['r2_sensitivity']:.1f}%)")
                    
                    dir_results.append(dir_info)
                else:
                    print(f"    No valid sensitivity data found")
                    
            except Exception as e:
                print(f"    Error processing directory {session_path.name}: {str(e)}")
        
        # Calculate combined sensitivity values
        if total_r1_trials + total_r2_trials > 0:
            # Calculate overall sensitivity across all directories in this session
            r1_sensitivity = (total_r1_respond / total_r1_trials) if total_r1_trials > 0 else 0
            r2_sensitivity = (total_r2_respond / total_r2_trials) if total_r2_trials > 0 else 0
            overall_sensitivity = ((total_r1_respond + total_r2_respond) / 
                               (total_r1_trials + total_r2_trials)) if (total_r1_trials + total_r2_trials) > 0 else 0
            
            # Store combined session results
            session_result = {
                'session_id': session_id,
                'session_date': session_date,
                'overall_sensitivity': overall_sensitivity,
                'r1_sensitivity': r1_sensitivity,
                'r2_sensitivity': r2_sensitivity,
                'r1_trials': total_r1_trials,
                'r2_trials': total_r2_trials,
                'total_trials': total_r1_trials + total_r2_trials,
                'r1_respond': total_r1_respond,
                'r2_respond': total_r2_respond,
                'directory_count': len(session_paths),
                'directories': dir_results
            }
            
            print(f"  Combined results for Session {session_id}:")
            print(f"    R1: {total_r1_respond}/{total_r1_trials} ({r1_sensitivity:.1%})")
            print(f"    R2: {total_r2_respond}/{total_r2_trials} ({r2_sensitivity:.1%})")
            print(f"    Overall: {total_r1_respond + total_r2_respond}/{total_r1_trials + total_r2_trials} ({overall_sensitivity:.1%})")
            
            results.append(session_result)
        else:
            print(f"  No valid trials found for Session {session_id}")
            results.append({
                'session_id': session_id,
                'session_date': session_date,
                'overall_sensitivity': np.nan,
                'r1_sensitivity': np.nan,
                'r2_sensitivity': np.nan,
                'r1_trials': 0,
                'r2_trials': 0,
                'total_trials': 0,
                'r1_respond': 0,
                'r2_respond': 0,
                'directory_count': len(session_paths),
                'directories': dir_results
            })
    
    # Create DataFrame from combined results
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'directories'} for r in results])
    
    # Print summary
    print("\nSummary of Combined Session Sensitivities:")
    print("=======================================")
    for _, row in results_df.iterrows():
        if pd.notna(row['overall_sensitivity']):
            print(f"Session {row['session_id']} ({row['session_date']}): " 
                  f"Overall {row['overall_sensitivity']:.1%}, "
                  f"R1 {row['r1_sensitivity']:.1%} ({row['r1_trials']} trials), "
                  f"R2 {row['r2_sensitivity']:.1%} ({row['r2_trials']} trials)")
        else:
            print(f"Session {row['session_id']} ({row['session_date']}): No valid trials found")

    return results_df


def calculate_combined_session_false_alarms(subject_path, stage=None, sessions=None):
    """
    Calculate combined false alarm rate across sessions (from a specific stage or from specific sessions)
    """
    # Use utils.find_session_roots instead of the local function
    session_roots = utils.find_session_roots(subject_path)
    
    if not session_roots:
        print(f"No valid session directories found in {subject_path}")
        return
     
    # Group sessions by session_id and session_date
    grouped_sessions = {}
    for session_id, session_date, session_path in session_roots:
        try:
            detected_stage = detect_stage(session_path)
        except Exception as e:
            print(f"  Could not detect stage for {session_path.name}: {e}")
            continue

        # Apply filtering rules
        session_id_int = int(session_id)
        session_allowed = (sessions is None or session_id_int in sessions)
        if stage is None:
            stage_allowed = True
        elif isinstance(stage, int):
            stage_allowed = int(detected_stage) == stage
        else:
            stage_allowed = detected_stage == stage

        if session_allowed and stage_allowed:
            key = (session_id, session_date)
            grouped_sessions.setdefault(key, []).append(session_path)
    
    # Create a list to store combined results
    results = []
    
    # Process each grouped session
    for (session_id, session_date), session_paths in sorted(grouped_sessions.items()):
        print(f"\nProcessing Session ID: {session_id}, Date: {session_date}")
        print(f"Found {len(session_paths)} directories within this session")
        
        # Initialize combined metrics
        total_C_pokes = 0
        total_D_pokes = 0
        total_E_pokes = 0
        total_F_pokes = 0
        total_G_pokes = 0
        total_C_trials = 0
        total_D_trials = 0
        total_E_trials = 0
        total_F_trials = 0
        total_G_trials = 0
        
        # Directory-specific results for detailed information
        dir_results = []
        
        # Process each directory within the session
        for session_path in session_paths:
            print(f"  Processing directory: {session_path.name}")
            try:
                # Get false alarm data for this directory
                false_alarm = get_false_alarm(session_path)

                if false_alarm and false_alarm != {'C_pokes': 0, 'C_trials': 0,
                        'D_pokes': 0, 'D_trials': 0, 'E_pokes': 0, 'E_trials': 0,
                        'F_pokes': 0, 'F_trials': 0, 'G_pokes': 0, 'G_trials': 0,
                        'C_false_alarm': 0,
                        'D_false_alarm': 0, 'E_false_alarm': 0,
                        'F_false_alarm': 0, 'G_false_alarm': 0, 
                        'overall_false_alarm': 0
                    }:
                    # Add to totals 
                    total_C_pokes += false_alarm['C_pokes']
                    total_D_pokes += false_alarm['D_pokes']
                    total_E_pokes += false_alarm['E_pokes']
                    total_F_pokes += false_alarm['F_pokes']
                    total_G_pokes += false_alarm['G_pokes']
                    total_C_trials += false_alarm['C_trials']
                    total_D_trials += false_alarm['D_trials']
                    total_E_trials += false_alarm['E_trials']
                    total_F_trials += false_alarm['F_trials']
                    total_G_trials += false_alarm['G_trials']

                    dir_info = {
                        'directory': session_path.name,
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
                        'overall_false_alarm': false_alarm['overall_false_alarm']
                    }
                    
                    print(f" C: {false_alarm['C_false_alarm']:.1f}%, D: {false_alarm['D_false_alarm']:.1f}%, "
                          f" E: {false_alarm['E_false_alarm']:.1f}%, F: {false_alarm['F_false_alarm']:.1f}%, "
                          f" G: {false_alarm['G_false_alarm']:.1f}%, overall: {false_alarm['overall_false_alarm']:.1f}%, ")
                    
                    dir_results.append(dir_info)
                else:
                    print(f"    No valid false alarm data found")
                    
            except Exception as e:
                print(f"    Error processing directory {session_path.name}: {str(e)}")
        
        # Calculate combined false alarm values
        total_C_pokes = np.sum(total_C_pokes)
        total_D_pokes = np.sum(total_D_pokes)
        total_E_pokes = np.sum(total_E_pokes)
        total_F_pokes = np.sum(total_F_pokes)
        total_G_pokes = np.sum(total_G_pokes)
        
        total_C_trials = np.sum(total_C_trials)
        total_D_trials = np.sum(total_D_trials)
        total_E_trials = np.sum(total_E_trials)
        total_F_trials = np.sum(total_F_trials)
        total_G_trials = np.sum(total_G_trials)

        total_C_false_alarm = (total_C_pokes / total_C_trials * 100) if total_C_trials > 0 else 0
        total_D_false_alarm = (total_D_pokes / total_D_trials * 100) if total_D_trials > 0 else 0
        total_E_false_alarm = (total_E_pokes / total_E_trials * 100) if total_E_trials > 0 else 0
        total_F_false_alarm = (total_F_pokes / total_F_trials * 100) if total_F_trials > 0 else 0
        total_G_false_alarm = (total_G_pokes / total_G_trials * 100) if total_G_trials > 0 else 0

        total_nonR_pokes = total_C_pokes + total_D_pokes + total_E_pokes + total_F_pokes + total_G_pokes
        total_nonR_trials = total_C_trials + total_D_trials + total_E_trials + total_F_trials + total_G_trials
        total_overall_false_alarm = (total_nonR_pokes / total_nonR_trials * 100) if total_nonR_trials > 0 else 0

        if total_nonR_trials > 0:
            # Store combined session results
            session_result = {
                'session_id': session_id,
                'session_date': session_date,
                'total_C_pokes': total_C_pokes,
                'total_D_pokes': total_D_pokes,
                'total_E_pokes': total_E_pokes,
                'total_F_pokes': total_F_pokes,
                'total_G_pokes': total_G_pokes,
                'total_C_trials': total_C_trials,
                'total_D_trials': total_D_trials,
                'total_E_trials': total_E_trials,
                'total_F_trials': total_F_trials,
                'total_G_trials': total_G_trials,
                'total_C_false_alarm': total_C_false_alarm,
                'total_D_false_alarm': total_D_false_alarm,
                'total_E_false_alarm': total_E_false_alarm,
                'total_F_false_alarm': total_F_false_alarm,
                'total_G_false_alarm': total_G_false_alarm,
                'total_overall_false_alarm': total_overall_false_alarm,
                'directory_count': len(session_paths),
                'directories': dir_results
            }
            
            print(f"  Combined results for Session {session_id}:")
            print(f"  False alarm rate: C={total_C_false_alarm:.1f}%, D={total_D_false_alarm:.1f}%, E={total_E_false_alarm:.1f}%, F={total_F_false_alarm:.1f}%, G={total_G_false_alarm:.1f}%")
            print(f"  Overall false alarm rate: {total_overall_false_alarm:.1f}%")
        
            results.append(session_result)
        else:
            print(f"  No valid non-rewarded odour trials found for Session {session_id}")
            results.append({
                'session_id': session_id,
                'session_date': session_date,
                'total_C_pokes': 0,
                'total_D_pokes': 0,
                'total_E_pokes': 0,
                'total_F_pokes': 0,
                'total_G_pokes': 0,
                'total_C_trials': 0,
                'total_D_trials': 0,
                'total_E_trials': 0,
                'total_F_trials': 0,
                'total_G_trials': 0,
                'total_C_false_alarm': 0,
                'total_D_false_alarm': 0,
                'total_E_false_alarm': 0,
                'total_F_false_alarm': 0,
                'total_G_false_alarm': 0,
                'total_overall_false_alarm': 0,
                'directory_count': len(session_paths),
                'directories': dir_results
            })
    
    # Create DataFrame from combined results
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'directories'} for r in results])
    
    # Print summary
    print("\nSummary of Combined Session False Alarm Rate:")
    print("=======================================")
    for _, row in results_df.iterrows():
        if pd.notna(row['total_overall_false_alarm']):
            print(f"Session {row['session_id']} ({row['session_date']}): " 
                  f"Overall {row['total_overall_false_alarm']:.1f}%, "
                  f"C {row['total_C_false_alarm']:.1f}%, "
                  f"D {row['total_D_false_alarm']:.1f}%, "
                  f"E {row['total_E_false_alarm']:.1f}%, "
                  f"F {row['total_F_false_alarm']:.1f}%, "
                  f"G {row['total_G_false_alarm']:.1f}%")
            
        else:
            print(f"Session {row['session_id']} ({row['session_date']}): No valid trials found")
    
    return results_df


def calculate_combined_session_false_alarm_bias(subject_path, stage=None, sessions=None):
    """
    Calculate combined false alarm bias across sessions (from a specific stage or from specific sessions)
    """
    # Use utils.find_session_roots instead of the local function
    session_roots = utils.find_session_roots(subject_path)
    
    if not session_roots:
        print(f"No valid session directories found in {subject_path}")
        return
        
    # Group sessions by session_id and session_date
    grouped_sessions = {}
    for session_id, session_date, session_path in session_roots:
        try:
            detected_stage = detect_stage(session_path)
        except Exception as e:
            print(f"  Could not detect stage for {session_path.name}: {e}")
            continue

        # Apply filtering rules
        session_id_int = int(session_id)
        session_allowed = (sessions is None or session_id_int in sessions)
        if stage is None:
            stage_allowed = True
        elif isinstance(stage, int):
            stage_allowed = int(detected_stage) == stage
        else:
            stage_allowed = detected_stage == stage

        if session_allowed and stage_allowed:
            key = (session_id, session_date)
            grouped_sessions.setdefault(key, []).append(session_path)
    
    # Create a list to store combined results
    results = []
    
    # Process each grouped session
    for (session_id, session_date), session_paths in sorted(grouped_sessions.items()):
        print(f"\nProcessing Session ID: {session_id}, Date: {session_date}")
        print(f"Found {len(session_paths)} directories within this session")
        
        # Initialize combined metrics
        total_odour_interval_pokes = defaultdict(lambda: defaultdict(int))
        total_odour_interval_trials = defaultdict(lambda: defaultdict(int))
        total_odour_interval_false_alarm = defaultdict(lambda: defaultdict(lambda: None))
        total_interval_pokes = {}
        total_interval_trials = {}
        total_interval_false_alarm = {}

        total_odour_same_olf_pokes = defaultdict(int)
        total_odour_same_olf_trials = defaultdict(int)
        total_odour_same_olf_false_alarm = {}
        total_odour_diff_olf_pokes = defaultdict(int)
        total_odour_diff_olf_trials = defaultdict(int)
        total_odour_diff_olf_false_alarm = {}
        total_odour_false_alarm_trials = defaultdict(int)
        total_same_olf_pokes = 0
        total_same_olf_trials = 0
        total_same_olf_false_alarm = {}
        total_diff_olf_pokes = 0
        total_diff_olf_trials = 0
        total_diff_olf_false_alarm = {}

        total_odour_same_olf_rew_pairing = defaultdict(int)
        total_odour_diff_olf_rew_pairing = defaultdict(int)
        total_odour_same_olf_rew_false_alarm = {}
        total_odour_diff_olf_rew_false_alarm = {}
        total_same_olf_rew_pairing = 0
        total_diff_olf_rew_pairing = 0
        
        # Directory-specific results for detailed information
        dir_results = []
        
        # Process each directory within the session
        for session_path in session_paths:
            print(f"  Processing directory: {session_path.name}")
            try:
                # Get false alarm bias data for this directory
                false_alarm_bias = get_false_alarm_bias(session_path)

                if false_alarm_bias and false_alarm_bias != {'odour_interval_pokes': 0, 
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
                                                            'diff_olf_false_alarm': 0,
                                                            'odour_false_alarm_trials': 0,
                                                            'odour_same_olf_rew_pairing': 0,
                                                            'odour_diff_olf_rew_pairing': 0,
                                                            'same_olf_rew_pairing': 0,
                                                            'diff_olf_rew_pairing': 0,
                                                            'same_olf_rew_false_alarm': 0,
                                                            'diff_olf_rew_false_alarm': 0
                    }:
                    # Add to totals 
                    nonR_odours = false_alarm_bias['odour_interval_pokes'].keys()
                    first_odour = next(iter(false_alarm_bias['odour_interval_pokes']))
                    intervals = false_alarm_bias['odour_interval_pokes'][first_odour].keys()
            
                    for odour in nonR_odours:
                        # time bias
                        for interval in intervals:
                            total_odour_interval_pokes[odour][interval] += false_alarm_bias['odour_interval_pokes'][odour][interval]
                            total_odour_interval_trials[odour][interval] += false_alarm_bias['odour_interval_trials'][odour][interval]
                            
                        # olfactometer bias 
                        total_odour_same_olf_pokes[odour] += false_alarm_bias['odour_same_olf_pokes'][odour]
                        total_odour_same_olf_trials[odour] += false_alarm_bias['odour_same_olf_trials'][odour]
                        total_odour_diff_olf_pokes[odour] += false_alarm_bias['odour_diff_olf_pokes'][odour]
                        total_odour_diff_olf_trials[odour] += false_alarm_bias['odour_diff_olf_trials'][odour]
                        total_odour_false_alarm_trials[odour] += false_alarm_bias['odour_false_alarm_trials'][odour]

                        # olfactometer-reward side bias
                        total_odour_same_olf_rew_pairing[odour] += false_alarm_bias['odour_same_olf_rew_pairing'][odour]
                        total_odour_diff_olf_rew_pairing[odour] += false_alarm_bias['odour_diff_olf_rew_pairing'][odour]

                    total_same_olf_pokes += false_alarm_bias['same_olf_pokes']
                    total_same_olf_trials += false_alarm_bias['same_olf_trials']
                    total_diff_olf_pokes += false_alarm_bias['diff_olf_pokes']
                    total_diff_olf_trials += false_alarm_bias['diff_olf_trials']

                    total_same_olf_rew_pairing += false_alarm_bias['same_olf_rew_pairing']
                    total_diff_olf_rew_pairing += false_alarm_bias['diff_olf_rew_pairing']

                    dir_info = {
                        'directory': session_path.name,
                        'odour_interval_pokes': false_alarm_bias['odour_interval_pokes'],
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
                        'diff_olf_false_alarm': false_alarm_bias['diff_olf_false_alarm'],
                        'odour_false_alarm_trials': false_alarm_bias['odour_false_alarm_trials'],
                        'odour_same_olf_rew_pairing': false_alarm_bias['odour_same_olf_rew_pairing'],
                        'odour_diff_olf_rew_pairing': false_alarm_bias['odour_diff_olf_rew_pairing'],
                        'odour_same_olf_rew_false_alarm': false_alarm_bias['odour_same_olf_rew_false_alarm'],
                        'odour_diff_olf_rew_false_alarm': false_alarm_bias['odour_diff_olf_rew_false_alarm'],
                        'same_olf_rew_pairing': false_alarm_bias['same_olf_rew_pairing'],
                        'diff_olf_rew_pairing': false_alarm_bias['diff_olf_rew_pairing'],
                        'same_olf_rew_false_alarm': false_alarm_bias['same_olf_rew_false_alarm'],
                        'diff_olf_rew_false_alarm': false_alarm_bias['diff_olf_rew_false_alarm']
                    }
                    # for odour in nonR_odours:
                    #     print(f"\nFalse alarm bias rates for odour {odour}:")
                    #     for interval, rate in false_alarm_bias['odour_interval_false_alarm'][odour].items():
                    #         print(f"  Interval {interval}: {rate:.1f}%")

                    for interval, rate in false_alarm_bias['interval_false_alarm'].items():
                        print(f"  Overall false alarm bias rate for interval {interval}: {rate:.1f}%")                

                    dir_results.append(dir_info)
                else:
                    print(f"    No valid false alarm time bias data found")
                    
            except Exception as e:
                print(f"    Error processing directory {session_path.name}: {str(e)}")

        total_olf_pokes = np.sum([total_odour_false_alarm_trials[odour] for odour in nonR_odours])
        for odour in nonR_odours:
            # time bias
            for interval in intervals:
                total_odour_interval_false_alarm[odour][interval] = (total_odour_interval_pokes[odour][interval] / total_odour_false_alarm_trials[odour] * 100) if total_odour_false_alarm_trials[odour] else 0
        
            # olfactometer bias
            total_odour_same_olf_false_alarm[odour] = (total_odour_same_olf_pokes[odour] / total_odour_false_alarm_trials[odour] * 100) if total_odour_false_alarm_trials[odour] else 0
            total_odour_diff_olf_false_alarm[odour] = (total_odour_diff_olf_pokes[odour] / total_odour_false_alarm_trials[odour] * 100) if total_odour_false_alarm_trials[odour] else 0

            # reward side bias
            total_odour_same_olf_rew_false_alarm[odour] = (total_odour_same_olf_rew_pairing[odour] / total_odour_false_alarm_trials[odour] * 100) if total_odour_false_alarm_trials[odour] else 0
            total_odour_diff_olf_rew_false_alarm[odour] = (total_odour_diff_olf_rew_pairing[odour] / total_odour_false_alarm_trials[odour] * 100) if total_odour_false_alarm_trials[odour] else 0

        # olfactometer bias
        total_same_olf_false_alarm = (total_same_olf_pokes / total_olf_pokes * 100) if total_olf_pokes else 0
        total_diff_olf_false_alarm = (total_diff_olf_pokes / total_olf_pokes * 100) if total_olf_pokes else 0

        # reward side bias
        total_same_olf_rew_false_alarm = (total_same_olf_rew_pairing / total_olf_pokes * 100) if total_olf_pokes else 0
        total_diff_olf_rew_false_alarm = (total_diff_olf_rew_pairing / total_olf_pokes * 100) if total_olf_pokes else 0
        
        # time bias
        for interval in intervals:
            total_interval_pokes[interval] = np.sum([total_odour_interval_pokes[odour][interval] for odour in nonR_odours])
            total_interval_trials[interval] = np.sum([total_odour_interval_trials[odour][interval] for odour in nonR_odours])
            total_interval_false_alarm[interval] = (total_interval_pokes[interval] / total_olf_pokes * 100) if total_olf_pokes > 0 else 0
    
        if any(interval_trials > 0 for interval_trials in total_interval_trials.values()):
            # Store combined session results
            session_result = {
                'session_id': session_id,
                'session_date': session_date,
                'total_odour_interval_pokes': total_odour_interval_pokes,
                'total_odour_interval_trials': total_odour_interval_trials,
                'total_odour_interval_false_alarm': total_odour_interval_false_alarm,
                'total_interval_pokes': total_interval_pokes,
                'total_interval_trials': total_interval_trials,
                'total_interval_false_alarm': total_interval_false_alarm,
                'total_odour_same_olf_pokes': total_odour_same_olf_pokes, 
                'total_odour_same_olf_trials': total_odour_same_olf_trials, 
                'total_odour_same_olf_false_alarm': total_odour_same_olf_false_alarm, 
                'total_odour_diff_olf_pokes': total_odour_diff_olf_pokes, 
                'total_odour_diff_olf_trials': total_odour_diff_olf_trials, 
                'total_odour_diff_olf_false_alarm': total_odour_diff_olf_false_alarm, 
                'total_same_olf_pokes': total_same_olf_pokes, 
                'total_same_olf_trials': total_same_olf_trials, 
                'total_same_olf_false_alarm': total_same_olf_false_alarm, 
                'total_diff_olf_pokes': total_diff_olf_pokes, 
                'total_diff_olf_trials': total_diff_olf_trials, 
                'total_diff_olf_false_alarm': total_diff_olf_false_alarm,
                'total_odour_same_olf_rew_pairing': total_odour_same_olf_rew_pairing,
                'total_odour_diff_olf_rew_pairing': total_odour_diff_olf_rew_pairing,
                'total_odour_same_olf_rew_false_alarm': total_odour_same_olf_rew_false_alarm,
                'total_odour_diff_olf_rew_false_alarm': total_odour_diff_olf_rew_false_alarm,
                'total_same_olf_rew_pairing': total_same_olf_rew_pairing,
                'total_diff_olf_rew_pairing': total_diff_olf_rew_pairing,
                'total_same_olf_rew_false_alarm': total_same_olf_rew_false_alarm,
                'total_diff_olf_rew_false_alarm': total_diff_olf_rew_false_alarm,
                'total_odour_false_alarm_trials': total_odour_false_alarm_trials,
                'directory_count': len(session_paths),
                'directories': dir_results
            }
            
            print(f"  Combined results for Session {session_id}:")
            # for odour in nonR_odours:
            #     print(f"\nFalse alarm bias rates for odour {odour}:")
            #     for interval, rate in total_odour_interval_false_alarm[odour].items():
            #         print(f"  Interval {interval}: {rate:.1f}%")

            for interval, rate in total_interval_false_alarm.items():
                print(f"  False alarm bias rate for interval {interval}: {rate:.1f}%")
        
            print(f"  False alarm same-olfactometer bias: {total_same_olf_false_alarm:.1f}%")
            print(f"  False alarm diff-olfactometer bias: {total_diff_olf_false_alarm:.1f}%")
            print(f"  False alarm same-olfactometer-reward bias rate: {total_same_olf_rew_false_alarm:.1f}%")
            print(f"  False alarm diff-olfactometer-reward bias rate: {total_diff_olf_rew_false_alarm:.1f}%")
        
            results.append(session_result)
        else:
            print(f"  No valid non-rewarded trials found for Session {session_id}")
            results.append({
                'session_id': session_id,
                'session_date': session_date,
                'total_odour_interval_pokes': total_odour_interval_pokes,
                'total_odour_interval_trials': total_odour_interval_trials,
                'total_odour_interval_false_alarm': total_odour_interval_false_alarm,
                'total_interval_pokes': total_interval_pokes,
                'total_interval_trials': total_interval_trials,
                'total_interval_false_alarm': total_interval_false_alarm,
                'total_odour_same_olf_pokes': total_odour_same_olf_pokes, 
                'total_odour_same_olf_trials': total_odour_same_olf_trials, 
                'total_odour_same_olf_false_alarm': total_odour_same_olf_false_alarm, 
                'total_odour_diff_olf_pokes': total_odour_diff_olf_pokes, 
                'total_odour_diff_olf_trials': total_odour_diff_olf_trials, 
                'total_odour_diff_olf_false_alarm': total_odour_diff_olf_false_alarm, 
                'total_same_olf_pokes': total_same_olf_pokes, 
                'total_same_olf_trials': total_same_olf_trials, 
                'total_same_olf_false_alarm': total_same_olf_false_alarm, 
                'total_diff_olf_pokes': total_diff_olf_pokes, 
                'total_diff_olf_trials': total_diff_olf_trials, 
                'total_diff_olf_false_alarm': total_diff_olf_false_alarm,
                'directory_count': len(session_paths),
                'directories': dir_results
            })
    
    # Create DataFrame from combined results
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'directories'} for r in results])
    
    # Print summary
    print("\nSummary of Combined Session False Alarms:")
    print("=======================================")
    for _, row in results_df.iterrows():
        if pd.notna(row['total_interval_false_alarm']):
            print(f"Session {row['session_id']} ({row['session_date']}): ") 
            
            for interval, rate in row['total_interval_false_alarm'].items():
                print(f"  Overall false alarm bias rate for interval {interval}: {rate:.1f}%")
            
            print(f"False alarm same-olfactometer bias rate: {row['total_same_olf_false_alarm']:.1f}%")
            print(f"False alarm diff-olfactometer bias rate: {row['total_diff_olf_false_alarm']:.1f}%")

            print(f"False alarm same-olfactometer-reward bias: {row['total_same_olf_rew_false_alarm']:.1f}%")
            print(f"False alarm diff-olfactometer-reward bias: {row['total_diff_olf_rew_false_alarm']:.1f}%")

        else:
            print(f"Session {row['session_id']} ({row['session_date']}): No valid trials found")
    
    return results_df
    

def calculate_combined_session_completion_commitment(subject_path, stage=None, sessions=None):
    """
    Calculate combined sequence completion and commitment ratios across sessions (from a specific stage or from specific sessions)
    """
    # Use utils.find_session_roots instead of the local function
    session_roots = utils.find_session_roots(subject_path)
    
    if not session_roots:
        print(f"No valid session directories found in {subject_path}")
        return
    
    # Group sessions by session_id and session_date
    grouped_sessions = {}
    for session_id, session_date, session_path in session_roots:
        try:
            detected_stage = detect_stage(session_path)
        except Exception as e:
            print(f"  Could not detect stage for {session_path.name}: {e}")
            continue

        # Apply filtering rules
        session_id_int = int(session_id)
        session_allowed = (sessions is None or session_id_int in sessions)
        if stage is None:
            stage_allowed = True
        elif isinstance(stage, int):
            stage_allowed = int(detected_stage) == stage
        else:
            stage_allowed = detected_stage == stage

        if session_allowed and stage_allowed:
            key = (session_id, session_date)
            grouped_sessions.setdefault(key, []).append(session_path)
    
    # Create a list to store combined results
    results = []
    
    # Process each grouped session
    for (session_id, session_date), session_paths in sorted(grouped_sessions.items()):
        print(f"\nProcessing Session ID: {session_id}, Date: {session_date}")
        print(f"Found {len(session_paths)} directories within this session")
        
        # Initialize combined metrics
        all_initiated_sequences = 0
        all_terminated_sequences = 0
        all_complete_sequences = 0
        all_early_rew_sampling = 0 
        
        # Directory-specific results for detailed information
        dir_results = []
        
        # Process each directory within the session
        for session_path in session_paths:
            print(f"  Processing directory: {session_path.name}")
            try:
                # Get sequence completion data for this directory
                sequence_completion = get_sequence_completion(session_path)

                if sequence_completion and sequence_completion != {
                    'initiated_sequences': 0, 'terminated_sequences': 0, 'complete_sequences': 0,
                    'early_rew_sampling': 0, 'completion_ratio': 0, 'commitment_ratio': 0,
                    'early_rew_sampling_ratio': 0
                }:
                    # Add to totals
                    all_initiated_sequences += sequence_completion['initiated_sequences']
                    all_terminated_sequences += sequence_completion['terminated_sequences']
                    all_complete_sequences += sequence_completion['complete_sequences']
                    all_early_rew_sampling += sequence_completion['early_rew_sampling']

                    dir_info = {
                        'directory': session_path.name,
                        'initiated_sequences': sequence_completion['initiated_sequences'],
                        'terminated_sequences': sequence_completion['terminated_sequences'],
                        'complete_sequences': sequence_completion['complete_sequences'],
                        'early_rew_sampling': sequence_completion['early_rew_sampling'],
                        'completion_ratio': sequence_completion['completion_ratio'],
                        'commitment_ratio': sequence_completion['commitment_ratio'],
                        'early_rew_sampling_ratio': sequence_completion['early_rew_sampling_ratio']
                        }
                    
                    print(f"  Sequence completion: {sequence_completion['completion_ratio']:.1f}%")
                    print(f"  Sequence commitment: {sequence_completion['commitment_ratio']:.1f}%")
                    
                    dir_results.append(dir_info)
                else:
                    print(f"    No valid sequence completion data found")
                    
            except Exception as e:
                print(f"    Error processing directory {session_path.name}: {str(e)}")
        
        # Calculate combined sequence completion and commitment values
        if all_terminated_sequences > 0:
            # Calculate overall sequence completion and commitment across all directories in this session
            overall_completion_ratio = all_complete_sequences / all_terminated_sequences * 100 if all_terminated_sequences > 0 else 0
            overall_commitment_ratio = all_complete_sequences / all_initiated_sequences * 100 if all_initiated_sequences > 0 else 0
            overall_early_rew_sampling_ratio = all_early_rew_sampling / all_complete_sequences * 100 if all_complete_sequences > 0 else 0
    
            # Store combined session results
            session_result = {
                'session_id': session_id,
                'session_date': session_date,
                'overall_completion_ratio': overall_completion_ratio,
                'overall_commitment_ratio': overall_commitment_ratio,
                'overall_early_rew_sampling_ratio': overall_early_rew_sampling_ratio,
                'directory_count': len(session_paths),
                'directories': dir_results
            }
            
            print(f"  Combined results for Session {session_id}:")
            print(f"    Overall sequence completion ratio: {overall_completion_ratio:.1f}%")
            print(f"    Overall sequence commitment ratio: {overall_commitment_ratio:.1f}%")
            print(f"    Overall early reward sampling ratio: {overall_early_rew_sampling_ratio:.1f}%")
            
            results.append(session_result)
        else:
            print(f"  No valid trials found for Session {session_id}")
            results.append({
                'session_id': session_id,
                'session_date': session_date,
                'overall_completion_ratio': np.nan,
                'overall_commitment_ratio': np.nan,
                'overall_early_rew_sampling_ratio': np.nan,
                'directory_count': len(session_paths),
                'directories': dir_results
            })
    
    # Create DataFrame from combined results
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'directories'} for r in results])
    
    # Print summary
    print("\nSummary of Combined Session Sequence Completion:")
    print("=======================================")
    for _, row in results_df.iterrows():
        if pd.notna(row['overall_completion_ratio']):
            print(f"Session {row['session_id']} ({row['session_date']}): " 
                  f"Overall {row['overall_completion_ratio']:.1f}%, "
                  f"Overall {row['overall_commitment_ratio']:.1f}%, "
                  f"Overall {row['overall_early_rew_sampling_ratio']:.1f}%, ")
        else:
            print(f"Session {row['session_id']} ({row['session_date']}): No valid trials found")
    
    return results_df


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("/Volumes/harris/hypnose/rawdata/sub-020_id-072")
        
    parser = argparse.ArgumentParser(description="Calculate and plot decision accuracy across sessions")
    parser.add_argument("subject_folder", help="Path to the subject's folder containing session data")
    parser.add_argument("--sessions", default=np.arange(31,91), help="List of session IDs (optional)") 
    parser.add_argument("--stage", "--s", default=7, help="Stage to be analysed (optional)")
    args = parser.parse_args()
    
    calculate_combined_session_accuracy(args.subject_folder, args.stage, args.sessions)
    