#PBS -S /bin/bash
#PBS -q main
#PBS -l walltime=167:59:59
#PBS -l select=1:ncpus=24:mem=16GB
set OMP_NUM_THREADS=24
export MKL_SERIAL=yes

cd $PBS_O_WORKDIR/
module load Armadillo
module load mkl
module load openmpi



echo "Start exec"
nohup ./rbm.o -f input.txt ./results/ >& results.txt