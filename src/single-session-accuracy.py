import pandas as pd
from pathlib import Path
import utils
from analysis import RewardAnalyser

# Step 1: Load session settings
metadata_reader = utils.SessionData()
root = Path('/Volumes/harris/hypnose/rawdata/sub-027_id-068/ses-01_date-20250409/behav/2025-04-09T07-45-42')
session_settings = utils.load_json(metadata_reader, root/"SessionSettings")

# Step 2: Create analyzer instance
analyzer = RewardAnalyser(session_settings)

# Step 3: Run the analysis
analyzer.run(
    data_path=root, 
    reward_a=8.0,  # Default value, can be adjusted as needed
    reward_b=8.0   # Default value, can be adjusted as needed
)