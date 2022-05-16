#PBS -S /bin/bash
#PBS -q main
#PBS -l walltime=167:59:59
#PBS -l select=1:ncpus=24:mem=16GB


source /opt/intel/oneapi/setvars.sh
echo "Start exec"
nohup ./rbm.o -f input.txt ./results/ >& results.txt
echo "Finished exec"