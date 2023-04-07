#!/bin/bash

Lx=$1
Ly=$2
J_step=$3

J_start=-2.0
J_end=2.0
folder="classical_results"
for J in $(seq $J_start $J_step $J_end)
do
    echo ${J}
    sbatch ./skrypt_run_classical.sh ${Lx} ${Ly} ${J} ${folder}
done