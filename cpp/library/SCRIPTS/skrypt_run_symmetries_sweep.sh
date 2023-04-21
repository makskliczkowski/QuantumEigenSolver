#!/bin/bash
#SBATCH -N1
#SBATCH -c1
#SBATCH --mem=1gb
#SBATCH --time=1:00:00
#SBATCH --job-name=vqmc-loop
L=$1
h=$2
g=$3
dlt1=$4
eta=$5
S=$6
mod=$7
BC=$8
folder=$9

dir="/home/klimak97/VQMC_sub/HamiltonianSolver_ed_vqmc/cpp/VQMC_S/"
echo $dir
cd /home/klimak97/VQMC_sub/HamiltonianSolver_ed_vqmc/cpp/VQMC_S/



# check the boundary conditions
if [ $BC -eq 0 ]
then
    echo "doing PBC"
else
    echo "doing OBC"
fi

a="L=${L}_g=${g},h=${h},dlta=${dlt1},eta=${eta},S=${S},bc=${BC}"
echo "#!/bin/bash" >> ${a}
echo "#SBATCH -N1" >> ${a}
echo "#SBATCH -c24" >> ${a}
echo "#SBATCH --mem=192gb" >> ${a}
echo "#SBATCH --time=299:59:59" >> ${a}
echo >> ${a}
echo "#SBATCH --job-name=${a}" >> ${a}
echo >> ${a}
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
echo "./vqmc.o -fun 22 -mod ${mod} -bc ${BC} -l 0 -d 1 -lx ${L} -ly 1 -J 1.0 -h ${h} -g ${g} -eta ${eta} -eta2 ${eta} -dlt ${dlt1} -S ${S} -th 24 -dir ${folder}/resultsXYZ${L}/ >& ./all_xyz/log_${a}.txt" >> ${a}
sbatch ${a}
rm ${a}

echo "finished"