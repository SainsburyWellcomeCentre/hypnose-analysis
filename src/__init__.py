"""
Hypnose Analysis - Tools for processing and analysing hypnose behavioral data
"""

# Import the most commonly used components for easy access
from .processing import detect_subject_stages
from .visualisation import plot_accuracy_single_subject, plot_rewards_single_subject, plot_rewards_multiple_subjects

__version__ = '0.1.0'