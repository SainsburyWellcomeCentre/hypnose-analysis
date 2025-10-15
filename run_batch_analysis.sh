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

python hpc_trial_analysis.py --subjids 20 25 26 32 34 36 --dates 20250801 20250804 20250805 20250806 20250807 20250808 20250811 20250812 20250813 20250814 20250815 20250818 20250819 20250820 20250821 20250825 20250826 20250827 20250828 20250829 20250901 20250902 20250903 20250904 20250905 20250907 20250908 20250909 20250910 20250911 20250912 20250915 20250916 20250917 20250918 20250922 20250923 20250924 20250925 20250929 20250930 20251001 20251002 20251003 20251006 20251007 20251008 20251009 20251010 --save --print_summary
