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
root = Path('/Volumes/harris/hypnose/rawdata/sub-028_id-069/ses-03_date-20250411/behav/2025-04-11T07-35-41')
session_settings = utils.load_json(metadata_reader, root/"SessionSettings")

# Step 2: Create analyzer instance
analyzer = RewardAnalyser(session_settings)

# Step 3: Run the analysis
analyzer.run(
    data_path=root, 
    reward_a=8.0,  # Default value, can be adjusted as needed
    reward_b=8.0   # Default value, can be adjusted as needed
)