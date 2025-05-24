#!/bin/bash
#SBATCH -N1
#SBATCH -c1
#SBATCH --mem=1gb
#SBATCH --time=49:59:59
#SBATCH -o /home/klimak/Codes/QuantumEigenSolver/Python/projects/2025_um_evolotion/SLURM/out-%j-fun=2025_um_evo,Ns=8-10,a=0.62-0.06-6.out
#SBATCH -e /home/klimak/Codes/QuantumEigenSolver/Python/projects/2025_um_evolotion/SLURM/err-%j-fun=2025_um_evo,Ns=8-10,a=0.62-0.06-6.err
#SBATCH --job-name=fun=2025_um_evo,Ns=8-10,a=0.62-0.06-6

# Export job ID for use in script
export SLURM_JOB_ID=${SLURM_JOB_ID}

# Set up scratch directory
# SAVDIR="/lustre/tmp/slurm/${SLURM_JOB_ID}"
# mkdir -p "${SAVDIR}"

# Log the job directory
echo -e "${SAVDIR}\t$(date)" >> /home/klimak/Codes/QuantumEigenSolver/Python/projects/2025_um_evolotion/slurm_ids_run.log

# Load required modules
source /usr/local/sbin/modules.sh

# Load necessary modules and install for 


module load python
module load HDF5

# Change to working directory
cd /home/klimak/Codes/QuantumEigenSolver/Python/projects/2025_um_evolotion

# Run the main computation
python3 /home/klimak/Codes/QuantumEigenSolver/Python/projects/2025_um_evolotion/evolution.py     \
    /home/klimak/Codes/QuantumEigenSolver/Python/projects/2025_um_evolotion/DATA/RANDOM_MODELS_EVO_2025_UM                   \
    0.62 0.06 6   \
    200 8 10  \
    1 100000 1      \
    > /home/klimak/Codes/QuantumEigenSolver/Python/projects/2025_um_evolotion/LOG/RANDOM_MODELS_EVO_2025_UM/log_fun=2025_um_evo,Ns=8-10,a=0.62-0.06-6.log 2>&1

# Copy results back (if uncommented)
# rsync -a --ignore-existing --remove-source-files "${SAVDIR}/"* /home/klimak/Codes/QuantumEigenSolver/Python/projects/2025_um_evolotion/DATA/RANDOM_MODELS_EVO_2025_UM/
# rm -rf "${SAVDIR}/"*

# Clean up scratch directory (if using TMPDIR)
# rmdir "${SAVDIR}" 2>/dev/null || true
