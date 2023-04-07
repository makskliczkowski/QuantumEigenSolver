#!/bin/bash

Lsu2=$1
Lnsu2=$2
BC=$3

mod=3

# do SU(2), hz = 0, sweep delta_b
eta_a=0
eta_b=0
hz=0
g=0

delta_a_start=0.1
delta_a_end=0.5
delta_a_step=.05
folder="xyz_su2_bc=${BC}"
for delta_a in $(seq $delta_a_start $delta_a_step $delta_a_end)
do
    echo $delta_a
    sbatch ./skrypt_run_symmetries_sweep.sh ${Lsu2} ${hz} ${g} ${delta_a} ${eta_a} 1 ${mod} ${BC} ${folder}
done


# do no SU(2), delta_b = 0.9, sweep hz
eta_a=0.5
eta_b=0.5
g=0

delta_b=0.9
h_start=0.05
h_end=0.7
h_step=.05
folder="xyz_nsu2_bc=${BC}"
for h in $(seq $h_start $h_step $h_end)
do
    echo $h
    sbatch ./skrypt_run_symmetries_sweep.sh ${Lnsu2} ${h} ${g} ${delta_b} ${eta_a} 1 ${mod} ${BC} ${folder}
done
 
