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

python hpc_trial_analysis.py --dates 20251001 20251002 --save --print_summary
