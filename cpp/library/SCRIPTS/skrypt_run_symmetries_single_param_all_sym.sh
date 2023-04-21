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
S=$7


dir="/home/klimak97/VQMC_sub/HamiltonianSolver_ed_vqmc/cpp/VQMC_S/"
#echo $dir
cd /home/klimak97/VQMC_sub/HamiltonianSolver_ed_vqmc/cpp/VQMC_S/

xs=(0 1)
# check the spin flip
if [ $h -eq 0 ]
then
    echo "doing spin flip"
else
    echo "not doing spin flip"
    xs=(1)
fi

# check the boundary conditions
if [ $BC -eq 0 ]
then
    echo "doing PBC"
else
    echo "doing OBC"
fi

# define some stuff
one=1
lhalf=$(echo "$L / 2"|bc)
all_k=$(echo "$L - $one"|bc)
pss=(0 1)

if [ $S -eq 0 ]
then
    echo "NOSYM"
    bash ./skrypt_run_single_param.sh ${L} ${h} ${g} ${J2} ${eta} ${BC} ${S}
    exit 0
fi


# doing symmetries
if [ $BC -eq 0 ]
then
    # do the translation as well
    kss=($(seq 0 1 $all_k))
    for ks in ${kss[@]};do
        echo "ks=${ks}"
        for xss in ${xs[@]};do
            echo "xs=${xss}"
            if [ $ks -eq 0 ] || [ $ks -eq $lhalf ];
            then
                for ps in ${pss[@]};do
                    echo "ps=$ps"
                    bash ./skrypt_run_symmetries_single_param.sh ${L} ${h} ${g} ${J2} ${eta} ${BC} ${ks} ${ps} ${xss}
                done
            else
                bash ./skrypt_run_symmetries_single_param.sh ${L} ${h} ${g} ${J2} ${eta} ${BC} ${ks} ${ps} ${xss}
            fi
        done
    done
else
    for xss in ${xs[@]};do
    echo "xs=${xss}"
        # do only the parity
        for ps in ${pss[@]};do
            bash ./skrypt_run_symmetries_single_param.sh ${L} ${h} ${g} ${J2} ${eta} ${BC} 0 ${ps} ${xss}
        done
    done
fi
echo "finished"