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

RUN_DIR=/home/klimak97/CODES/QuantumEigenSolver/cpp/library
cd ${RUN_DIR}

lustredir=/home/klimak97/mylustre-hpc-maciek/UM2
mkdir -p ${lustredir}

# create log directory
SAVDIR=${TMPDIR}/DATA/${SLURM_JOBID}
# SAVDIR=${lustredir}
mkdir -p $SAVDIR
LOGDIR=${RUN_DIR}/LOG/RANDOM_MODELS
mkdir -p $LOGDIR

# parameters
SSYYMS=$(tr -d ' ' <<< "$PARS")

# save the SLURM ID AND CONNECTED input to it 

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
echo >> ${a}
# echo "SAVDIR=/lustre/tmp/slurm/\$SLURM_JOB_ID" >> ${a}
echo "SAVDIR=${lustredir}" >> ${a}
echo >> ${a}
echo "echo -e \"\${SAVDIR}\t${PARS}\" >> ${RUN_DIR}/slurm_ids_run.log" >> ${a}  
echo >> ${a}
echo "mkdir -p \${SAVDIR}" >> ${a}
# echo "cd \${SAVDIR}" >> ${a}§
echo >> ${a}
echo "source /usr/local/sbin/modules.sh" >> ${a}
echo >> ${a}
echo "module load intel/2022b" >> ${a}
echo >> ${a}
echo "module load HDF5" >> ${a}
echo >> ${a}
echo "cd ${RUN_DIR}" >> ${a}
echo >> ${a}
echo "${RUN_DIR}/qsolver -Ntots ${Ns} -plrb_mb 1 -op 'Sz/L;Sz/0' -eth_entro 1 -checkpoint 0 -eth_ipr 1 -eth_susc 1 -eth_end '0.1;0.2;0.3;0.4' -eth_offd 1 -fun ${FUN} ${PARS} -th ${CPU} -dir ${SAVDIR}/ >& ${LOGDIR}/log_${b}.log" >> ${a}
# echo "${RUN_DIR}/qsolver -Ntots ${Ns} -plrb_mb 1 -op 'Sz/L;Sz/0' -eth_entro 1 -checkpoint 0 -eth_ipr 1 -eth_susc 1 -eth_end '0.1;0.2;0.3;0.4' -eth_offd 1 -fun ${FUN} ${PARS} -th ${CPU} -dir \${SAVDIR}/ >& ${LOGDIR}/log_${b}.log" >> ${a}
echo >> ${a}
# echo "rsync -a --ignore-existing --remove-source-files \${SAVDIR}/* ${lustredir}" >> ${a}
# echo "rm -rf \${SAVDIR}/*" >> ${a} 
# echo ${TMPDIR}/DATA/${SLURM_JOBID} >> last_tmp_jobs.log
sbatch ${a} 
# echo ${a}
# echo ${b}
echo $?
rm ${a}
