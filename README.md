# Hypnose Processing

This repository is utilised for processing and visualising data acquired from Hypnose Harris lab project.

## Features

- Extracting mouse interactions with ports, olfactometer commands and synchronisation with cameras/EEG recordings

- Analysing and visualising behavioral metrics

## How to Use

1. Clone the repository

Within your working directory use a terminal to clone the repo to your local folder:

```git clone github.com/SainsburyWellcomeCentre/hypnose-analysis```

2. Create a conda virtual environment with preferred replacement for env-name and activate

```conda create -n env-name python=3.12``` 

3. Add virtual environment as kernel to run notebooks 

```conda install -c conda-forge notebook ipykernel```

```python -m ipykernel install --user --name=env-name --display-name="Display Name"```

4. Install dependencies and editable install of hypnose-analysis package

```pip install -e .```

5. Symlink: 

Directories in this repo are resolved with a symlink inside /data pointing to the mounted server containing data. Depending on local structure of mounting the server, the symlink may need adjusting. 

Create SymLink in Windows (*requirement*: ceph server mounted at Z:): 

- Open a PowerShell Terminal as Administrator

- cd into hypnose-analysis (repo main folder)

- Remove any possible existing items in the symlink folder by running  ```Remove-Item -LiteralPath .\data\rawdata -Recurse -Force```

- Non-persistently map the mounted server for this session by running ```net use Z: "\\ceph-gw02.hpc.swc.ucl.ac.uk\harris" /persistent:no```
 
- Confirm path exists by running ```Test-Path "Z:\hypnose\rawdata" ``` should return True

- Create SymLink to ceph data storage by running ```New-Item -ItemType SymbolicLink -Path ".\data\rawdata" -Target "Z:\hypnose\rawdata" ```

- SymLink should appear in the repo folder

## Running Analysis: 

The analysis consists of two parts: **trial classification** and **behavioral metric calculation**

1. Trial Classification

The trial_classification notebook runs the trial classification. All functions used in this notebook are in the classification_utils.py file. 

batch_analyze_sessions can run on any combination of dates and subjids to run analysis on several subjects or dates at ones. If one parameter is None, it will run on all subjects for date(s) provided or all dates for subject(s) provided. Results are saved as json and csv combination. A summary txt file is saved per session analyzed. 

plot_valve_and_poke_events can be used to visualize all valve states, with option to specify a time window. 

cut_video can be used to cut a short video of the experiment with a defined time window. 

1.1. Output

The trial classification returns a trial_data.parquet / .csv file containing the following data columns: 

Trial Identification

    - global_trial_id (int): Count of all trials across runs, starting at 0
    - trial_id (int): Non-global trial ID per run, starts at 0 per run
    - run_id (int): Run number the data is from, starting at 1
    - attempt_number (int): Which attempt of poking the first odor started the sequence

Sequence Timing

    - initiation_sequence_time (timestamp ISO 8601): Time the initiation sequence starts for this trial. From here, the mouse can poke
    - sequence_start (timestamp ISO 8601): Time the mouse successfully initiated a sequence
    - sequence_start_corrected (timestamp ISO 8601): Corrected sequence start if sequence contained position 1 attempts. For most trials, similar to normal sequence start
    - sequence_end (timestamp ISO 8601): Time the sequence ends (either reward port poke or begin of next sequence)
    - timestamp (timestamp ISO 8601): Same as sequence_start; the time the successful poke for starting the sequence occurred

Odor Sequence Information

    - odor_name (string): Name of the first odor (e.g., OdorA)
    - odor_sequence (list): List of all odors in the sequence from first to last
    - num_odors (int): Number of odors in this sequence
    - last_odor (string): Last odor in odor_sequence, which is when the animal left the odor cue port
    - sequence_name (string): Name of the protocol used in this run
    - continuous_poke_time_ms (float): Time the first odor was poked to initiate the sequence
    - required_min_sampling_time_ms (float): Minimum poke time required for the first odor to initiate the sequence
    - minimum_sampling_time_ms_by_odor (dict): Dictionary containing all odors and their respective minimum sampling times

Detailed Event Data

    - position_valve_times (dict): Dictionary of positions containing position name, odor presented (odor_name), valve start time (valve_start), valve end time (valve_end), valve duration (valve_duration_ms), and required minimum sampling time (required_min_sampling_time_ms)
    - position_poke_times (dict): Dictionary of positions containing position name, odor presented (odor_name), poke time for that odor (poke_time_ms), poke odor start (poke_odor_start), poke odor end (poke_odor_end), time of first poke in (poke_first_in), and required minimum sampling time (required_min_sampling_time_ms)
    - presentations (list): Contains index_in_trial, odor_name, valve_start, valve_end, valve_duration_ms, poke_time_ms, poke_first_in, required_min_sampling_time_ms, and is_last_event for all indices in the sequence
    - last_event_index (int): Index where the last event (odor) appeared

Hidden Rule Information

    - hidden_rule_location (int): Currently always 0. Do not use this information
    - hidden_rule_locations (list): All locations the hidden rule odor can appear
    - hidden_rule_positions (list): Index-corrected list of hidden rule locations (+1 for each index)
    - hit_hidden_rule (boolean): Whether the hidden rule appeared in the sequence
    - hidden_rule_hit_indices (list): The index where the hidden rule appeared
    - hidden_rule_hit_positions (list): The position where the hidden rule was hit (similar to index + 1)
    - hidden_rule_success (boolean): Whether the animal successfully completed the hidden rule trial (got to await_reward state by leaving at the hidden rule odor; can still be unrewarded or timeout)
    - hidden_rule_success_position (string): Same as hidden_rule_hit_position, but only in trials where hidden_rule_success is True
    - enough_odors_for_hr (boolean): Whether enough odors were presented for hidden rule to be possible

Reward Information

    - await_reward_time (timestamp ISO 8601): Timestamp of when await reward state was triggered
    - first_supply_time (timestamp ISO 8601): Time point the supply port was activated the first time (reward delivered)
    - first_supply_port (int): In rewarded trials, which supply port the animal poked first (1 for A, 2 for B)
    - first_supply_odor_identity (string): In rewarded trials, the identity of the first supply port poked (A or B)
    - supply1_count (int): In rewarded trials, 1 or 0 indicating whether first poke happened at supply port 1
    - supply2_count (int): In rewarded trials, 1 or 0 indicating whether first poke happened at supply port 2
    - total_supply_count (int): In rewarded trials, 1 or NaN indicating whether a supply port poke happened
    - poke_window_end (timestamp ISO 8601): Time when the reward window ended, indicating timeout or unrewarded trial
    - response_time_ms (float): In rewarded, unrewarded, or timeout trials, time between odor cue port poke out and first supply port poke
    - response_time_category (string): Categorical label for completed sequences: "rewarded", "unrewarded", or "timeout_delayed"

Non-Rewarded Trial Information

    - port1_pokes_count (int): In non-rewarded completed trials (unrewarded or timeout), number of pokes in port 1
    - port2_pokes_count (int): In non-rewarded completed trials, number of pokes in port 2
    - total_reward_pokes (int): In unrewarded trials, count of pokes in the wrong supply port
    - first_reward_poke_time (timestamp ISO 8601): In unrewarded trials, time of supply port poke
    - first_reward_poke_port (int): In unrewarded trials, port number of supply port poke
    - first_reward_poke_odor_identity (string): In unrewarded trials, identity of incorrect reward port poke (A or B)

Aborted Trial Information

    - is_aborted (boolean): Identifier if trial was aborted or not
    - abortion_type (string): Type of abortion: "initiation_abortion" or "reinitiation_abortion"
    - abortion_time (timestamp ISO 8601): Time point abortion happened (poke out of cue port)
    - last_odor_position (int): In aborted trials, position where the last odor in sequence appeared
    - last_odor_name (string): In aborted trials, name of last odor
    - last_odor_valve_duration_ms (float): In aborted trials, duration of last odor valve
    - last_odor_poke_time_ms (float): In aborted trials, poke duration of last odor
    - last_required_min_sampling_time_ms (float): In aborted trials, required minimum sampling time for that odor, defining abortion classification (initiation or re-initiation)

False Alarm Information

    - fa_label (string): In aborted trials, false alarm classification: "fa_time_in" (within response time window), "fa_time_out" (up to 3x the response time window), "fa_late" (later than that), or "nFA" (no false alarm)
    - fa_time (timestamp ISO 8601): In false alarm trials, when the false alarm happened (supply port poke)
    - fa_latency_ms (float): Time between leaving odor cue port and poking either supply port in aborted trials
    - fa_port (int): In false alarm trials, port ID (1 for A, 2 for B) of first supply port poke


2. Behavioral Metric Calculation

The metrics_analysis notebook runs the behavioral metric calculation. All functions used in this notebook are in the metrics_utils.py file. 

To add another metric calculation, add the definition as an independent function, and call it within run_all_metrics. 

batch_run_all_metrics_with_merge can run on any combination of dates and subjids. Further, a protocol filter can be applied to only run on sessions under same protocol (within or across subjects). 

Results are saved per session and merged for all sessions analyzed, either within the subject directory, or in the merged directory at the subject directory level for multi-subject runs.

Results are saved as a json and csv file combination with a summary txt file. 