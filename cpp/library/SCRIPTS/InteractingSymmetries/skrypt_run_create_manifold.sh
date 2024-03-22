#!/bin/bash
#SBATCH -N1
#SBATCH -c1
#SBATCH --mem=1gb
#SBATCH --time=1:00:00
#SBATCH --job-name=vqmc-loop

L=$1
PARS=$2
TIM=$3
MEM=$4
CPU=$5
FUN=$6
SSYYMS=$(tr -d ' ' <<< "$PARS")

cd /home/klimak97/CODES/QuantumEigenSolver/cpp/library/

# start writing
a="L=${L}_PARS=${SSYYMS}"
echo "#!/bin/bash" >> ${a}
echo "#SBATCH -N1" >> ${a}
echo "#SBATCH -c${CPU}" >> ${a}
echo "#SBATCH --mem=${MEM}" >> ${a}
echo "#SBATCH --time=${TIM}" >> ${a}
echo >> ${a}
echo "#SBATCH --job-name=${a}" >> ${a}
echo >> ${a}
echo "source /usr/local/sbin/modules.sh" >> ${a}
echo >> ${a}
echo "module load intel/2022b" >> ${a}
echo >> ${a}
echo "module load HDF5" >> ${a}
echo >> ${a}
echo "cd /home/klimak97/CODES/QuantumEigenSolver/cpp/library/" >> ${a}

echo "./qsolver.o -fun ${FUN} ${PARS} -bc 0 -l 0 -d 1 -Lx ${L} -Ly 1 -Lz 1 -th ${CPU} -dir Patrycja/ >& ./LOG/log_man_${a}.txt" >> ${a}
sbatch ${a} 
# echo ${a}
echo $?
rm ${a}