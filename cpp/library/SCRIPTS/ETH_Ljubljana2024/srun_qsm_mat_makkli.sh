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

RUN_DIR=/home/makkli4548/CODES/QuantumEigenSolver/cpp/library/
cd ${RUN_DIR}

lustredir=/home/makkli4548/DATA
mkdir -p ${lustredir}

# create log directory
SAVDIR=${TMPDIR}/DATA/${SLURM_JOBID}
mkdir -p $SAVDIR
LOGDIR=${RUN_DIR}/LOG/RANDOM_MODELS
mkdir -p $LOGDIR

# parameters
SSYYMS=$(tr -d ' ' <<< "$PARS")

b="qsm_mat_ns=${Ns}_${SLURM_JOBID}_${EPOCHREALTIME}"
arhname="Ns=${Ns},${b},${SLURM_JOBID}.tar.gz"
a="ns=${Ns}"
echo "#!/bin/bash" >> ${a}
echo "#SBATCH -N1" >> ${a}
echo "#SBATCH -c${CPU}" >> ${a}
echo "#SBATCH --mem=${MEM}" >> ${a}
echo "#SBATCH --time=${TIM}" >> ${a}
echo "#SBATCH -o ./SLURM/out-%j-${b}.out" >> ${a}
echo "#SBATCH --job-name=${b}" >> ${a}
echo "source /usr/local/sbin/modules.sh" >> ${a}
echo >> ${a}
echo "module load intel/2022b" >> ${a}
echo >> ${a}
echo "module load HDF5" >> ${a}
echo >> ${a}
echo "cd ${RUN_DIR}" >> ${a}
echo >> ${a}
echo "${RUN_DIR}/qsolver.o -Ntot ${Ns} -plrb_mb 1 -Ntots 7 -op 'Sz/L;Sz/0;Sx/0;Sz/0-1;Sz/0-1-2;Sz/0-1-2-3;Sz/1:Lm3:1;Sz/0-1:Lm3:1' -qsm_h r\;1.0\;0.5 -fun ${FUN} ${PARS} -th ${CPU} -dir ${SAVDIR}/ >& ${LOGDIR}/log_${b}.log" >> ${a}
echo >> ${a}
echo "tar -cvzf ${SAVDIR}/${arhname} ${SAVDIR}" >> ${a}
echo "mv ${SAVDIR}/${arhname} ${lustredir} && rm -rf ${SAVDIR}" >> ${a} 

sbatch ${a} 
# echo ${a}
# echo ${b}
echo $?
rm ${a}
