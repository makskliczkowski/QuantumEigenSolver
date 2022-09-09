#!/bin/bash
L=$1
h=$2
g=$3
J=$4
k=$5
p=$6
x=$7
S=$8
SS=$9
dir="/home/klimak97/VQMC/HamiltonianSolver_ed_vqmc/cpp/VQMC_S/"
echo $dir
cd /home/klimak97/VQMC/HamiltonianSolver_ed_vqmc/cpp/VQMC_S/


#source /etc/profile.d/modules.sh
#source /etc/profile.d/modules-local.sh





a="L=${L}_k=${k},p=${p},x=${x},g=${g},h=${h},J=${J}"
b=${a}

echo "#!/bin/bash" >> ${a}
echo "#SBATCH -N1" >> ${a}
echo "#SBATCH -c24" >> ${a}
echo "#SBATCH --mem=24gb" >> ${a}
echo "#SBATCH --time=199:59:59" >> ${a}
echo >> ${a}
echo "#SBATCH --job-name=${a}" >> ${a}

echo "source /usr/local/sbin/modules.sh" >> ${a}
echo >> ${a}
echo "module load intel/2021a" >> ${a}
echo >> ${a}
echo "module load OpenMPI/4.1.1-GCC-10.3.0" >> ${a}
echo >> ${a}
echo "cd /home/klimak97/VQMC/HamiltonianSolver_ed_vqmc/cpp/VQMC_S/" >> ${a}
echo >> ${a}
echo "vqmc.o -bc 0 -l 0 -d 1 -lx ${L} -ly 1 -J ${J} -h ${h} -g ${g} -th 24 -ks ${k} -ps ${p} -xs ${x} -S ${S} -SS ${SS} -dir resultsSym/ >& log_${b}.txt" >> ${a}
sbatch ${a}
rm ${a}