#!/bin/bash
#SBATCH -N1
#SBATCH -c1
#SBATCH --mem=1gb
#SBATCH --time=1:00:00
#SBATCH --job-name=vqmc-loop
L=$1
h=$2
g=$3
J=$4
S=$5
SS=$6

dir="/home/klimak97/VQMC/HamiltonianSolver_ed_vqmc/cpp/VQMC_S/"
echo $dir
cd /home/klimak97/VQMC/HamiltonianSolver_ed_vqmc/cpp/VQMC_S/

xs=(0 1)
if [ $h -eq 0.0 ]
then
    echo "doing spin flip"
else
    echo "not doing spin flip"
    xs=(1)
fi
one=1
lhalf=$(echo "$L / 2"|bc)
all_k=$(echo "$L - $one"|bc)
pss=(0 1)


kss=($(seq 0 1 $all_k))
for ks in ${kss[@]};do
    echo "ks=${ks}"
    for xss in ${xs[@]};do
        echo "xs=${xss}"
        if [ $ks -eq 0 ] || [ $ks -eq $lhalf ];
        then
            for ps in ${pss[@]};do
                echo "ps=$ps"
                a="L=${L}_g=${g},h=${h},J=${J},S=${S},SS=${SS},x=${xss},p=${ps},k=${ks}"
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
                echo "./vqmc.o -bc 0 -l 0 -d 1 -lx ${L} -ly 1 -J ${J} -h ${h} -g ${g} -th 24 -ks ${ks} -ps ${ps} -xs ${xss} -S ${S} -SS ${SS} -dir resultsSym${L}/ >& ./all/log_${a}.txt" >> ${a}
                sbatch ${a}
                rm ${a}

            done
        else
            a="L=${L}_g=${g},h=${h},J=${J},S=${S},SS=${SS},x=${xss},p=${1},k=${ks}"
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
            echo "./vqmc.o -bc 0 -l 0 -d 1 -lx ${L} -ly 1 -J ${J} -h ${h} -g ${g} -th 24 -ks ${ks} -ps ${1} -xs ${xss} -S ${S} -SS ${SS} -dir resultsSym${L}/ >& ./all/log_${a}.txt" >> ${a}
            sbatch ${a}
            rm ${a}
        fi
    done
done
echo "finished"