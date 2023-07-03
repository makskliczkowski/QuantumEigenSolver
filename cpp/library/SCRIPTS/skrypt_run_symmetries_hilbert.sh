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
TIM=$4
MEM=$5
CPU=$6

cd /home/klimak97/CODES/QuantumEigenSolver/cpp/library/

a="L=${L},${eta1},${dlt1}"
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
echo "module load intel" >> ${a}
echo >> ${a}
echo "module load HDF5" >> ${a}
echo >> ${a}
echo "cd /home/klimak97/CODES/QuantumEigenSolver/cpp/library/" >> ${a}
echo >> ${a}
echo "./qsolver.o -fun 23 -mod 1 -bc 0 -l 0 -d 1 -Lx ${L} -Ly 1 -Lz 1 -J1 1.0 -J2 0 -hx 0 -hz 0 -eta1 ${eta1} -eta2 0 -dlt1 ${dlt1} -dlt2 0 -S 1 -th ${CPU} -dir HILBERT/ >& ./LOG/log_${a}.txt" >> ${a}
sbatch ${a} 
echo $?
rm ${a}
