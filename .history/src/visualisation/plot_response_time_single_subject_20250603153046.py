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
from src.analysis import get_decision_accuracy, get_response_time
import src.utils as utils


def calculate_session_response_time(session_path):
    """
    Calculate decision response time for the session.
    
    Args:
        session_path (Path): Path to the session directory
        
    Returns:
        dict: Contains 'r1_correct_rt', 'r1_incorrect_rt', 'r1_avg_correct_rt', 'r1_avg_incorrect_rt', 'r1_avg_rt',
        'r2_correct_rt', 'r2_incorrect_rt', 'r2_avg_correct_rt', 'r2_avg_incorrect_rt', 'r2_avg_rt', 
        'hit_rt', 'false_alarm_rt' response time values
    """
    # Don't look for JSON files - send the session path directly to get_response_time
    try:
        # Get response times for this session
        response_time = get_response_time(session_path)
        
        if response_time is None:
            print(f"No response time data available for {session_path}")
            return {
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
            }
        
        return {
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
        }
        
    except Exception as e:
        print(f"Error processing session {session_path}: {str(e)}")
        return {
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
        }















































































































































































































