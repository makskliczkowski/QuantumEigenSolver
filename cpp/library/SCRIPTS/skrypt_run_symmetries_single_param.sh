#!/bin/bash
#SBATCH -N1
#SBATCH -c1
#SBATCH --mem=1gb
#SBATCH --time=1:00:00
#SBATCH --job-name=vqmc-loop

L=$1
h=$2
g=$3
J2=$4
eta=$5
BC=$6
ks=$7
ps=$8
xs=$9

delta=0.3

lhalf=$(echo "$L / 2"|bc)

dir="/home/klimak97/VQMC_sub/HamiltonianSolver_ed_vqmc/cpp/VQMC_S/"
cd /home/klimak97/VQMC_sub/HamiltonianSolver_ed_vqmc/cpp/VQMC_S/

# start writing
a="L=${L}_g=${g},h=${h},J2=${J2},eta=${eta},d=${delta},x=${xs},p=${ps},k=${ks},bc=${BC}"
echo "#!/bin/bash" >> ${a}
echo "#SBATCH -N1" >> ${a}
echo "#SBATCH -c24" >> ${a}
echo "#SBATCH --mem=192gb" >> ${a}
echo "#SBATCH --time=399:59:59" >> ${a}
echo >> ${a}
echo "#SBATCH --job-name=${a}" >> ${a}

echo "source /usr/local/sbin/modules.sh" >> ${a}
echo >> ${a}
echo "module load intel" >> ${a}
echo >> ${a}
echo "module load OpenMPI" >> ${a}
echo >> ${a}
echo "module load HDF5" >> ${a}
echo >> ${a}
echo "cd /home/klimak97/VQMC_sub/HamiltonianSolver_ed_vqmc/cpp/VQMC_S/" >> ${a}
echo >> ${a}
echo "./vqmc.o -fun 21 -mod 4 -bc ${BC} -l 0 -d 1 -lx ${L} -ly 1 -J 1 -J2 ${J2} -h ${h} -g ${g} -eta ${eta} -eta2 ${eta} -dlt ${delta} -dlt2 ${delta} -w 0 -S 0 -su2 ${lhalf} -th 24 -ks ${ks} -ps ${ps} -xs ${xs} -dir Data/resultsXYZ${L}/ >& ./all_xyz/log_${a}.txt" >> ${a}
echo "      ${a}"
sbatch ${a}
rm ${a}
