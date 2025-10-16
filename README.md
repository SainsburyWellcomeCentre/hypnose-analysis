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

In classification_utils.py: 
- Change project_root to match your folder (line 3)
- For windows users: match storage folder path within load_experiment base_path = ... (line 150), batch_analyze_sessions base_path (line 4415), and cut_video base_dir (line 4461)

In metrics_utils.py:
- Change project_root to match your folder (line 3)
- For windows users: match base directory path within load_session_results base_dir = ... (line 31), merged_results_output_dir derivatives directory path (line 308), batch_run_all_metrics_with_merge directory path (line 359)

In trial_classification.ipynb and metrics_analysis.ipynb:
- Change project root to match your folder (initial import cell)


## Running Analysis: 

The analysis consists of two parts: **trial classification** and **behavioral metric calculation**

1. Trial Classification

The trial_classification notebook runs the trial classification. All functions used in this notebook are in the classification_utils.py file. 

batch_analyze_sessions can run on any combination of dates and subjids to run analysis on several subjects or dates at ones. If one parameter is None, it will run on all subjects for date(s) provided or all dates for subject(s) provided. Results are saved as json and csv combination. A summary txt file is saved per session analyzed. 

plot_valve_and_poke_events can be used to visualize all valve states, with option to specify a time window. 

cut_video can be used to cut a short video of the experiment with a defined time window. 

2. Behavioral Metric Calculation

The metrics_analysis notebook runs the behavioral metric calculation. All functions used in this notebook are in the metrics_utils.py file. 

To add another metric calculation, add the definition as an independent function, and call it within run_all_metrics. 

batch_run_all_metrics_with_merge can run on any combination of dates and subjids. Further, a protocol filter can be applied to only run on sessions under same protocol (within or across subjects). 

Results are saved per session and merged for all sessions analyzed, either within the subject directory, or in the merged directory at the subject directory level for multi-subject runs.

Results are saved as a json and csv file combination with a summary txt file. 


## Running analysis on the HPC: 
1. Connect to the HPC 

2. Navigate to home (cd) and install miniconda for linux (x86_64)

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

    bash miniconda.sh
    source ~/.bashrc (if missing, run touch ~/.bashrc followed by ~/miniconda3/bin/conda init bash)
    conda activate

3. Create hypnose_hpc env

    conda create -n hypnose_hpc python=3.12.11
    conda activate hypnose_hpc
    conda install numpy pandas pyyaml swc-aeon moviepy opencv matplotlib ipython harp-python dotmap

4. Update SLURM 

    Navigate to cd /ceph/harris/hypnose/hypnose-analysis/

    Update the .sh file to specify parameters for analysis
        nano run_batch_analysis.sh --> edit the changes and exit via ^X

5. Run Analysis

    Activate conda environment 

    Submit SLURM job via sbatch run_batch_analysis.sh

6. Output
    
    Slurm error and output files are saved in ceph/harris/hypnose/hpc_output
    Data is saved as usual in the derivatives directory