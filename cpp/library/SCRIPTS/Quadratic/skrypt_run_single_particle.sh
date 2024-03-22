#!/bin/bash
#SBATCH -N1
#SBATCH -c1
#SBATCH --mem=1gb
#SBATCH --time=1:00:00
#SBATCH --job-name=vqmc-loop
VALNONE=-1000
L=$1
PARS=$2
TIM=$3
MEM=$4
CPU=$5
FUN=$6

lustredir=/home/klimak97/mylustre-hpc-maciek/
# create log directory
# mkdir ${lustredir}/QSolver
LOGDIR=${lustredir}/QSolver/LOG
# mkdir -p ${LOGDIR}
SAVDIR=${lustredir}/QSolver/DATA
# mkdir -p ${SAVDIR}

cd /home/klimak97/CODES/QuantumEigenSolver/cpp/library/

SSYYMS=$(tr -d ' ' <<< "$PARS")

b="Quadratic_L=${L},${SSYYMS}"
a="L=${L}"
echo "#!/bin/bash" >> ${a}
echo "#SBATCH -N1" >> ${a}
echo "#SBATCH -c${CPU}" >> ${a}
echo "#SBATCH --mem=${MEM}" >> ${a}
echo "#SBATCH --time=${TIM}" >> ${a}
echo "#SBATCH -o ./SLURM/out-%j.out" >> ${a}
echo >> ${a}
echo "#SBATCH --job-name=${b}" >> ${a}
echo >> ${a}
echo "source /usr/local/sbin/modules.sh" >> ${a}
echo >> ${a}
echo "module load intel/2022b" >> ${a}
echo >> ${a}
echo "module load HDF5" >> ${a}
echo >> ${a}
echo "cd /home/klimak97/CODES/QuantumEigenSolver/cpp/library/" >> ${a}
echo >> ${a}
echo "./qsolver.o -fun ${FUN} ${PARS} -bc 0 -l 0 -d 1 -Lx ${L} -Ly 1 -Lz 1 -th ${CPU} -dir ${SAVDIR}/ >& ${LOGDIR}/log_SP_${b}.log" >> ${a}
sbatch ${a} 
# echo ${a}
echo $?
rm ${a}
# -x ${r} -px ${px} -py ${py} -pz ${pz} -k ${k} -U1 ${u1}
# echo "finished"