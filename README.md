# Hypnose Processing

This repository is utilised for processing and visualising data acquired from Hypnose Harris lab project.

## Features

- Extracting mouse interactions with ports, olfactometer commands and synchronisation with cameras/EEG recordings

- Analysing and visualising behavioral metrics

## How to Use

1. Clone the repository

Within your working directory use a terminal to clone the repo to your local folder:

```git clone https://github.com/vlkuzun/hypnose-processing.git```

2. Create a conda virtual environment with preferred replacement for env-name and activate

```conda create -n env-name python=3.12``` 

3. Add virtual environment as kernel to run notebooks 

```conda install -c conda-forge notebook ipykernel```

```python -m ipykernel install --user --name=env-name --display-name="Display Name"```

4. Change path variables to use on your PC

In utils.py: 
- Change project_root to match your folder (line 3)
- For windows users: match storage folder path within load_experiment base_path = ... (line 150), batch_analyze_sessions base_path (line 4415), and cut_video base_dir (line 4461)

In behavioral_metrics.py:
- Change project_root to match your folder (line 3)
- For windows users: match base directory path within load_session_results base_dir = ... (line 31), merged_results_output_dir derivatives directory path (line 308), batch_run_all_metrics_with_merge directory path (line 359)

In hypnose_testing.ipynb and metrics_analysis.ipynb:
- Change project root to match your folder (initial import cell)


## Running Analysis: 

The analysis consists of two parts: **trial classification** and **behavioral metric calculation**

1. Trial Classification

The hypnose_testing notebook runs the trial classification. All functions used in this notebook are in the utils.py file. 
batch_analyze_sessions can run on any combination of dates and subjids to run analysis on several subjects or dates at ones. If one parameter is None, it will run on all subjects for date(s) provided or all dates for subject(s) provided. Results are saved as json and csv combination. A summary txt file is saved per session analyzed. 
plot_valve_and_poke_events can be used to visualize all valve states, with option to specify a time window. 
cut_video can be used to cut a short video of the experiment with a defined time window. 

2. Behavioral Metric Calculation

The metrics_analysis notebook runs the behavioral metric calculation. All functions used in this notebook are in the behavioral_metrics.py file. 
To add another metric calculation, add the definition as an independent function, and call it within run_all_metrics. 
batch_run_all_metrics_with_merge can run on any combination of dates and subjids. Further, a protocol filter can be applied to only run on sessions under same protocol (within or across subjects). 
Results are saved per session and merged for all sessions analyzed, either within the subject directory, or in the merged directory at the subject directory level for multi-subject runs.
Results are saved as a json and csv file combination with a summary txt file. 