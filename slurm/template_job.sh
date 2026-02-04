#!/bin/bash
#SBATCH --job-name=my_job_name       # Job name
#SBATCH --partition=<PARTITION>      # Partition (queue) name
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=<THREADS>    # Number of CPU cores per task
#SBATCH --mem=<MEMORY>gb             # Job memory request
#SBATCH --time=<TIME>                # Time limit hrs:min:sec
#SBATCH --output=logs/job_%j.out     # Standard output log
#SBATCH --error=logs/job_%j.err      # Standard error log

# ==============================================================================
# TEMPLATE SLURM SCRIPT
# Replace <PARTITION>, <THREADS>, <MEMORY>, <TIME> and the command below.
# ==============================================================================

set -euo pipefail

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""

# ------------------------------------------------------------------------------
# 1. Load Modules
# ------------------------------------------------------------------------------
# Example for C++:
# module load intel
# module load hdf5
#
# Example for Python:
# module load python/3.10

# ------------------------------------------------------------------------------
# 2. Environment Setup
# ------------------------------------------------------------------------------
# Example for Python Virtual Env:
# source ./venv/bin/activate

# ------------------------------------------------------------------------------
# 3. Run Command
# ------------------------------------------------------------------------------

echo "Starting job..."

# Replace this with your command
# ./build/qsolver <args>
# or
# python3 my_script.py <args>

echo "Job finished."
