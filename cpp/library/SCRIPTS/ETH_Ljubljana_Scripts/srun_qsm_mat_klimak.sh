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

RUN_DIR=/home/klimak97/CODES/QuantumEigenSolver/cpp/library/
cd ${RUN_DIR}

if [ "$FUN" -le 45 ]; then
    lustredir=/home/klimak97/mylustre-hpc-maciek/DATA_FIDELITY_ADDITIONAL/
else
    lustredir=/home/klimak97/mylustre-hpc-maciek/DATA_TIME_ADDITIONAL/
fi
mkdir -p ${lustredir}

# create log directory
SAVDIR=${TMPDIR}/DATA/${SLURM_JOBID}
# SAVDIR=${lustredir}
mkdir -p $SAVDIR
LOGDIR=${RUN_DIR}/LOG/RANDOM_MODELS_EVO_CHECK
mkdir -p $LOGDIR

# parameters
SSYYMS=$(tr -d ' ' <<< "$PARS")

# Extract meaningful parts from SSYYMS for a shorter job name
parsed_ssyms=$(echo "$SSYYMS" | sed 's/-/,/g')
b=",fun=${FUN},Ns=${Ns}${parsed_ssyms}"
# arhname="Ns=${Ns},${b},\${SLURM_JOBID}.tar.gz"
a="fun=${FUN},Ns=${Ns}${parsed_ssyms}"
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
# echo "SAVDIR=${lustredir}" >> ${a}
echo >> ${a}
echo "echo -e \"\${SAVDIR}\t${PARS}\" >> ${RUN_DIR}/slurm_ids_run.log" >> ${a}  
echo >> ${a}
echo "mkdir -p \${SAVDIR}" >> ${a}
# echo "cd \${SAVDIR}" >> ${a}§
echo >> ${a}
echo "source /usr/local/sbin/modules.sh" >> ${a}
echo >> ${a}
echo "module load intel" >> ${a}
echo >> ${a}
echo "module load HDF5" >> ${a}
echo >> ${a}
echo "cd ${RUN_DIR}" >> ${a}
echo >> ${a}
echo "${RUN_DIR}/qsolver -Ntots ${Ns} -Ntot ${Ns} -plrb_mb 1 -op 'Sz/0;Sx/0-1' -eth_entro 1 -checkpoint 0 -eth_ipr 1 -eth_susc 1 -eth_end '0.1;0.2;0.3;0.4' -eth_offd 1 -fun ${FUN} ${PARS} -th ${CPU} -dir ${lustredir}/ >& ${LOGDIR}/log${b}.log" >> ${a}
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
