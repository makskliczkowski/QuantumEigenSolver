#!/bin/bash
#SBATCH -N1
#SBATCH -c1
#SBATCH --mem=1gb
#SBATCH --time=1:00:00
#SBATCH --job-name=vqmc-loop
VALNONE=-1000
L=$1
dlt1=$2
eta1=$3
SYMS=$4
TIM=$5
MEM=$6
CPU=$7
FUN=$8
# u1=$4
# px=$5
# py=$6
# pz=$7
# r=$8
# k=$9

cd /home/klimak97/CODES/QuantumEigenSolver/cpp/library/

# check the boundary conditions
# if [ $BC -eq 0 ]
# then
#     echo "doing PBC"
# else
#     echo "doing OBC"
# fi
SSYYMS=$(tr -d ' ' <<< "$SYMS")

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
echo "./qsolver.o -fun ${FUN} -mod 1 -bc 0 -l 0 -d 1 -Lx ${L} -Ly 1 -Lz 1 -J1 1.0 -J2 0 -hx 0 -hz 0 -eta1 ${eta1} -eta2 0 -dlt1 ${dlt1} -dlt2 0 -S 1 ${SYMS} -th ${CPU} -dir SUSY_MISSING/ >& ./LOG/log_${a}.txt" >> ${a}
sbatch ${a} 
# echo ${a}
echo $?
rm ${a}
# -x ${r} -px ${px} -py ${py} -pz ${pz} -k ${k} -U1 ${u1}
# echo "finished"