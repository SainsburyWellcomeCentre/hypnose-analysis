#!/bin/bash
#
#SBATCH -p cpu                      # Partition (queue)
#SBATCH -N 1                        # Number of nodes
#SBATCH -n 4                        # Number of cores (adjust as needed)
#SBATCH --mem 16G                   # Total memory required (adjust as needed)
#SBATCH -t 0-2:00:00                # Time (D-HH:MM:SS)
#SBATCH -o /ceph/harris/hypnose/hpc_output/slurm.%j.out  # STDOUT
#SBATCH -e /ceph/harris/hypnose/hpc_output/slurm.%j.err  # STDERR
#
# Initialize Conda in the script
eval "$(conda shell.bash hook)"

# Activate the Conda environment
conda activate hypnose_hpc

# Change directory 
cd /ceph/harris/hypnose/hypnose-analysis/notebooks/

# Run the script
# for dates, either select specific dates with --dates YYYYMMDD YYYYMMDD ...
# or a date range with --start_date YYYYMMDD --end_date YYYYMMDD
# or a start date with --start_date YYYYMMDD (will run for 365 days unless changed in hpc_trial_analysis.py)
python hpc_trial_analysis.py --subjids 20 25 26 --start_date 20251001 --end_date 20251010 --save --print_summary
