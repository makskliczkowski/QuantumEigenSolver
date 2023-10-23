#!/bin/bash
#SBATCH -N1
#SBATCH -c48
#SBATCH --mem=192gb
#SBATCH -t1
#SBATCH --time=119:59:59
#SBATCH --array=1-30 # lista ID podzadań

Ns=$1
Asize=$2

source /usr/local/sbin/modules.sh
module load Python
module load HDF5
module load intel 

echo $Ns
echo $Asize
echo ${SLURM_ARRAY_TASK_ID}

# echo "python runSweepArray.py $Ns $Asize ${SLURM_ARRAY_TASK_ID}"
python runSweepArray.py ${Ns} ${Asize} ${SLURM_ARRAY_TASK_ID}