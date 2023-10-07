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

cd /home/klimak97/CODES/QuantumEigenSolver/cpp/library/

# check the boundary conditions
# if [ $BC -eq 0 ]
# then
#     echo "doing PBC"
# else
#     echo "doing OBC"
# fi
SSYYMS=$(tr -d ' ' <<< "$PARS")

a="L=${L},${SSYYMS}"
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
echo "module load intel/2022a" >> ${a}
echo >> ${a}
echo "module load HDF5" >> ${a}
echo >> ${a}
echo "cd /home/klimak97/CODES/QuantumEigenSolver/cpp/library/" >> ${a}
echo >> ${a}
echo "./qsolver.o -fun ${FUN} ${PARS} -bc 0 -l 0 -d 1 -Lx ${L} -Ly 1 -Lz 1 -th ${CPU} -dir SP_BIG/ >& ./LOG/log_SP_${a}.txt" >> ${a}
sbatch ${a} 
# echo ${a}
echo $?
rm ${a}
# -x ${r} -px ${px} -py ${py} -pz ${pz} -k ${k} -U1 ${u1}
# echo "finished"