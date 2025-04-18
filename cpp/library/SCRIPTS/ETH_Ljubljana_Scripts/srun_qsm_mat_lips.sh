#!/bin/bash
#SBATCH -N1
#SBATCH -c1
#SBATCH --mem=1gb
#SBATCH --time=1:00:00
#SBATCH --job-name=vqmc-loop
Ns=$1
PARS=$2
TIM=$3
MEM=$4
CPU=$5
FUN=$6

RUN_DIR=/home/kliczkowski/CODES/QuantumEigenSolver/cpp/library/
cd ${RUN_DIR}

savedir=/scratch/kliczkowski/FADING_RAN_MODELS/FINAL/
mkdir -p ${savedir}

# create log directory
SAVDIR=${savedir}/DATA
mkdir -p $SAVDIR
LOGDIR=${RUN_DIR}/LOG/ETH
mkdir -p $LOGDIR
SLURMDIR=${RUN_DIR}/SLURM
mkdir -p $SLURMDIR

# parameters
SSYYMS=$(tr -d ' ' <<< "$PARS")

b="mat_ns=${Ns},fun=${FUN},${SSYYMS}"
arhname="Ns=${Ns},${b},${SLURM_JOBID}.tar.gz"
a="ns=${Ns}"
echo "#!/bin/bash" >> ${a}
echo "#SBATCH -N1" >> ${a}
echo "#SBATCH --cpus-per-task=${CPU}" >> ${a}
echo "#SBATCH --mem-per-cpu=${MEM}" >> ${a}
echo "#SBATCH --time=${TIM}" >> ${a}
echo "#SBATCH --partition=long" >> ${a}
echo "#SBATCH --output=${SLURMDIR}/out-%j-${b}.out" >> ${a}
echo "#SBATCH --job-name=${b}" >> ${a}
echo "#SBATCH --error=${SLURMDIR}/err_${b}_log-%A-%a.out" >> ${a}
echo >> ${a}
echo "module load intel" >> ${a}
echo >> ${a}
echo "module load HDF5" >> ${a}
echo >> ${a}
echo "cd ${RUN_DIR}" >> ${a}
echo >> ${a}
echo "${RUN_DIR}/qsolver.o -Ntot ${Ns} -plrb_mb 1 -Ntots 7 -fun ${FUN} ${PARS} -op 'Sz/L;Sz/L_2;Sz/0;Sz/Lm1;Sz/Lm1-L;Sz/0-1;Sx/0-L' -th ${CPU} -dir ${SAVDIR}/ >& ${LOGDIR}/log_${b}.log" >> ${a}
echo >> ${a}
# echo "tar -cvzf ${SAVDIR}/${arhname} ${SAVDIR}" >> ${a}
# echo "mv ${SAVDIR}/${arhname} ${lustredir} && rm -rf ${SAVDIR}" >> ${a} 

sbatch ${a} 
# echo ${a}
# echo ${b}
echo $?
rm ${a}