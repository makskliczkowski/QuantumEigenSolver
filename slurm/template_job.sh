#!/bin/bash
#SBATCH --job-name=my_job_name       # Job name
#SBATCH --partition=<PARTITION>      # Partition (queue) name
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=<THREADS>    # Number of CPU cores per task
#SBATCH --mem=<MEMORY>gb             # Job memory request (in GB)
#SBATCH --time=<TIME>                # Time limit hrs:min:sec
#SBATCH --output=logs/job_%j.out     # Standard output log
#SBATCH --error=logs/job_%j.err      # Standard error log

# ==============================================================================
# TEMPLATE SLURM SCRIPT
# 1. Copy this file: cp slurm/template_job.sh my_job.sh
# 2. Replace <PARTITION>, <THREADS>, <MEMORY>, <TIME>.
# 3. Replace <COMMAND> below.
# 4. Submit: sbatch my_job.sh
# ==============================================================================

# Fail on errors
set -euo pipefail

# Ensure logs directory exists
mkdir -p logs

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""

# ------------------------------------------------------------------------------
# 1. Load Modules
# ------------------------------------------------------------------------------
# Example:
# module load intel hdf5
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

# <COMMAND> - Replace the line below with your command
echo "Replace this line with your command"
# ./build/qsolver <args>
# python3 my_script.py <args>

echo "Job finished."
