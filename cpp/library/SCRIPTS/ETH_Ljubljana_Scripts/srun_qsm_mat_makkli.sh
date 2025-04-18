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
# comment

RUN_DIR=/home/makkli4548/CODES/QuantumEigenSolver/cpp/library/
cd ${RUN_DIR}

lustredir=/home/makkli4548/mylustre/DATA_FIDELITY/
mkdir -p ${lustredir}

# create log directory
SAVDIR=${TMPDIR}/DATA/${SLURM_JOBID}
mkdir -p $SAVDIR
LOGDIR=${RUN_DIR}/LOG/RANDOM_MODELS_EVO
mkdir -p $LOGDIR

# parameters
SSYYMS=$(tr -d ' ' <<< "$PARS")

b="qsm_mat_ns=${Ns},${SSYYMS}"
arhname="Ns=${Ns},${b},\${SLURM_JOBID}.tar.gz"
a="ns=${Ns}_${SSYYMS}"
echo "#!/bin/bash" >> ${a}
echo "#SBATCH -N1" >> ${a}
echo "#SBATCH -c${CPU}" >> ${a}
echo "#SBATCH --mem=${MEM}" >> ${a}
echo "#SBATCH --time=${TIM}" >> ${a}
echo "#SBATCH -o ${RUN_DIR}/SLURM/out-%j-${b}.out" >> ${a}
echo "#SBATCH --job-name=${b}" >> ${a}
echo "export SLURM_JOB_ID=\$SLURM_JOB_ID" >> ${a}
echo >> ${a}
echo "SAVDIR=/lustre/tmp/slurm/\$SLURM_JOB_ID" >> ${a}
echo >> ${a}
echo "echo -e \"\${SAVDIR}\t${PARS}\" >> ${RUN_DIR}/slurm_ids_run.log" >> ${a}  
echo >> ${a}
echo "mkdir -p \${SAVDIR}" >> ${a}
# echo "cd \${SAVDIR}" >> ${a}
echo >> ${a}
echo "source /usr/local/sbin/modules.sh" >> ${a}
echo >> ${a}
echo "module load intel/2022b" >> ${a}
echo >> ${a}
echo "module load HDF5" >> ${a}
echo >> ${a}
echo "cd ${RUN_DIR}" >> ${a}
echo >> ${a}
echo "${RUN_DIR}/qsolver -Ntot ${Ns} -plrb_mb 1 -Ntots ${Ns} -op 'Sz/L;Sx/0;Sz/0' -eth_entro 1 -eth_ipr 1 -eth_susc 1 -eth_end '0.1;0.2;0.3;0.4' -eth_offd 1 -checkpoint 0 -fun ${FUN} ${PARS} -th ${CPU} -dir ${lustredir}/ >& ${LOGDIR}/log_${b}.log" >> ${a}
# echo "${RUN_DIR}/qsolver.o -Ntot ${Ns} -plrb_mb 1 -Ntots 7 -op 'Sz/L;Sz/0;Sx/0;Sz/0-1;Sz/0-1-2;Sz/0-1-2-3;Sz/1:Lm3:1;Sz/0-1:Lm3:1' -eth_entro 1 -eth_ipr 1 -eth_susc 1 -eth_end '0.1;0.2;0.3;0.4' -eth_offd 1 -fun ${FUN} ${PARS} -th ${CPU} -dir \${SAVDIR}/ >& ${LOGDIR}/log_${b}.log" >> ${a}
echo >> ${a}
# echo "rsync -a --ignore-existing --remove-source-files \${SAVDIR}/* ${lustredir}" >> ${a}
# echo "rm -rf \${SAVDIR}/*" >> ${a} 
echo ${TMPDIR}/DATA/${SLURM_JOBID} >> last_tmp_jobs.log
sbatch ${a} 
# echo ${a}
# echo ${b}
echo $?
rm ${a}
