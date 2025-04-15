import pandas as pd
from pathlib import Path
import utils
from analysis import RewardAnalyser
import warnings

# Filter out the specific FutureWarning about downcasting
warnings.filterwarnings(
    "ignore", 
    message="Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated",
    category=FutureWarning
)

# Step 1: Load session settings
metadata_reader = utils.SessionData()
root = Path('/Volumes/harris/hypnose/rawdata/sub-020_id-072/ses-16_date-20250407/behav/2025-04-07T14-52-48')
session_settings = utils.load_json(metadata_reader, root/"SessionSettings")

# Step 2: Create analyzer instance
analyzer = RewardAnalyser(session_settings)

# Step 3: Run the analysis
reward_value = 4.0  # Single reward value for both A and B
analyzer.run(
    data_path=root, 
    reward_a=reward_value,
    reward_b=reward_value
)