#!/bin/bash
#SBATCH -N1
#SBATCH -c1
#SBATCH --mem=1gb
#SBATCH --time=1:00:00
#SBATCH --job-name=vqmc-loop

# Function to display usage
show_usage() {
    cat << EOF
Usage   : $0 "param_string" [OPTIONS]
    or  : $0 Ns_start Ns_end a_start a_step a_num n_rel TIM MEM CPU [OPTIONS]

Parameter string format: "Ns_start Ns_end a_start a_step a_num n_rel TIM MEM CPU time mem cpu"

Options:
    --time=TIME             Override time allocation (format: HH:MM:SS)
    --mem=MEM               Override memory allocation (in GB, format: Xgb or X)
    -c CPU, --cpu=CPU       Override CPU count
    -h, --help              Show this help message

Examples:
    $0 "0.68 0.06 1 5 16 16 1 100000 32 133:19:59 8gb 4" --time=133:19:59 --mem=16 -c16
    $0 0.68 0.06 1 5 16 16 1 100000 32 --time=133:19:59 --mem=16gb -c16
EOF
}

# Function to parse memory value and convert to GB integer
parse_memory() {
    local mem_input="$1"
    local mem_value
    
    # Remove 'gb' suffix if present and extract number
    mem_value=$(echo "$mem_input" | sed -E 's/[gG][bB]?$//' | sed 's/[^0-9]//g')
    
    if [[ ! "$mem_value" =~ ^[0-9]+$ ]] || [ "$mem_value" -le 0 ]; then
        echo "Error: Invalid memory format: $mem_input" >&2
        return 1
    fi
    
    echo "$mem_value"
}

# Function to validate time format
validate_time() {
    local time_input="$1"
    if [[ ! "$time_input" =~ ^[0-9]{1,3}:[0-5]?[0-9]:[0-5]?[0-9]$ ]]; then
        echo "Error: Invalid time format: $time_input (expected HH:MM:SS)" >&2
        return 1
    fi
    echo "$time_input"
}

# Initialize variables
Ns_start=""
Ns_end=""
a_start=""
a_step=""
a_num=""
n_rel=""
n=""
t_num=""
override_time=""
override_mem=""
override_cpu=""

# Parse arguments
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

# Check if first argument is help
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_usage
    exit 0
fi

# Determine if first argument is a parameter string or individual parameters
if [[ "$1" == *" "* ]]; then
    # First argument contains spaces, treat as parameter string
    param_string="$1"
    shift
    
    # Parse the parameter string
    read -r a_start a_step a_num n_rel Ns_start Ns_end n t_num param_time param_mem param_cpu <<< "$param_string"
    
    # Use parameters from string as defaults
    if [ -n "$param_time" ]; then
        TIM="$param_time"
    fi
    if [ -n "$param_mem" ]; then
        MEM=$(parse_memory "$param_mem") || exit 1
    fi
    if [ -n "$param_cpu" ]; then
        CPU="$param_cpu"
    fi
else
    # Individual parameters
    if [ $# -lt 9 ]; then
        echo "Error: Insufficient parameters when using individual argument format"
        show_usage
        exit 1
    fi
    
    Ns_start="$1"
    Ns_end="$2"
    a_start="$3"
    a_step="$4"
    a_num="$5"
    n_rel="$6"
    TIM="$7"
    MEM="$8"
    CPU="$9"
    shift 9
fi

# Parse remaining options
while [[ $# -gt 0 ]]; do
    case $1 in
        --time=*)
            override_time="${1#*=}"
            shift
            ;;
        --mem=*)
            override_mem="${1#*=}"
            shift
            ;;
        -c*)
            if [ "${1#-c}" != "$1" ]; then
                # Format: -c16
                override_cpu="${1#-c}"
            else
                # Format: -c 16
                shift
                override_cpu="$1"
            fi
            shift
            ;;
        --cpu=*)
            override_cpu="${1#*=}"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Apply overrides
if [ -n "$override_time" ]; then
    TIM=$(validate_time "$override_time") || exit 1
fi
if [ -n "$override_mem" ]; then
    MEM=$(parse_memory "$override_mem") || exit 1
fi
if [ -n "$override_cpu" ]; then
    CPU="$override_cpu"
fi

# Validate all required parameters are set
if [ -z "$Ns_start" ] || [ -z "$Ns_end" ] || [ -z "$a_start" ] || [ -z "$a_step" ] || \
   [ -z "$a_num" ] || [ -z "$n_rel" ] || [ -z "$TIM" ] || [ -z "$MEM" ] || [ -z "$CPU" ]; then
    echo "Error: Missing required parameters"
    show_usage
    exit 1
fi

# Validate numeric inputs
if ! [[ "$CPU" =~ ^[0-9]+$ ]] || [ "$CPU" -le 0 ]; then
    echo "Error: CPU must be a positive integer"
    exit 1
fi

if ! [[ "$MEM" =~ ^[0-9]+$ ]] || [ "$MEM" -le 0 ]; then
    echo "Error: MEM must be a positive integer"
    exit 1
fi

# Validate time format if not already validated
TIM=$(validate_time "$TIM") || exit 1

# Calculate memory per worker with proper integer division
mem_per_worker=$(( MEM / CPU ))

# Set directory paths
RUN_DIR="$(dirname "$(realpath "$0")")"
LUSTRE_DIR="${RUN_DIR}/DATA/RANDOM_MODELS_EVO_2025_UM"
LOG_DIR="${RUN_DIR}/LOG/RANDOM_MODELS_EVO_2025_UM"
SLURM_DIR="${RUN_DIR}/SLURM"

# Change to run directory
cd "${RUN_DIR}" || { 
    echo "Error: Cannot access RUN_DIR: ${RUN_DIR}" 
    exit 1 
}

echo "Configuration:"
echo "  RUN_DIR=${RUN_DIR}"
echo "  Ns_start=${Ns_start}, Ns_end=${Ns_end}"
echo "  a_start=${a_start}, a_step=${a_step}, a_num=${a_num}"
echo "  n_rel=${n_rel}"
echo "  Resources: ${TIM}, ${MEM}gb, ${CPU} CPUs"
echo "  mem_per_worker=${mem_per_worker}gb"

# Create necessary directories
mkdir -p "${LUSTRE_DIR}" "${LOG_DIR}" "${SLURM_DIR}"

# Create temporary directory if using TMPDIR
if [ -n "${TMPDIR}" ]; then
    SAVDIR="${TMPDIR}/DATA/${SLURM_JOBID}"
    mkdir -p "${SAVDIR}"
fi

# Generate job identifiers
job_params="fun=2025_um_evo,Ns=${Ns_start}-${Ns_end},a=${a_start}-${a_step}-${a_num}"
script_file="${SLURM_DIR}/${job_params}.sh"

# Create SLURM job script
cat > "${script_file}" << EOF
#!/bin/bash
#SBATCH -N1
#SBATCH -c${CPU}
#SBATCH --mem=${MEM}gb
#SBATCH --time=${TIM}
#SBATCH -o ${SLURM_DIR}/out-%j-${job_params}.out
#SBATCH -e ${SLURM_DIR}/err-%j-${job_params}.err
#SBATCH --job-name=${job_params}

# Export job ID for use in script
export SLURM_JOB_ID=\${SLURM_JOB_ID}

# Set up scratch directory
# SAVDIR="/lustre/tmp/slurm/\${SLURM_JOB_ID}"
# mkdir -p "\${SAVDIR}"

# Log the job directory
echo -e "\${SAVDIR}\t\$(date)" >> ${RUN_DIR}/slurm_ids_run.log

# Load required modules
source /usr/local/sbin/modules.sh

# Load necessary modules and install for 


module load python
module load HDF5

# Change to working directory
cd ${RUN_DIR}

# Run the main computation
python3 ${RUN_DIR}/evolution.py     \\
    ${LUSTRE_DIR}                   \\
    ${a_start} ${a_step} ${a_num}   \\
    ${n_rel} ${Ns_start} ${Ns_end}  \\
    ${n} ${t_num} ${mem_per_worker} \\
    > ${LOG_DIR}/log_${job_params}.log 2>&1

# Copy results back (if uncommented)
# rsync -a --ignore-existing --remove-source-files "\${SAVDIR}/"* ${LUSTRE_DIR}/
# rm -rf "\${SAVDIR}/"*

# Clean up scratch directory (if using TMPDIR)
# rmdir "\${SAVDIR}" 2>/dev/null || true
EOF

#! for the tests, echo the script content
echo "Generated SLURM script:"
cat "${script_file}"

# Make the script executable
# chmod +x "${script_file}"

# # Log temporary directory if using TMPDIR
# if [ -n "${TMPDIR}" ] && [ -n "${SLURM_JOBID}" ]; then
#     echo "${TMPDIR}/DATA/${SLURM_JOBID}" >> last_tmp_jobs.log
# fi

# # Submit the job
# echo "Submitting job: ${job_params}"
# if sbatch_output=$(sbatch "${script_file}" 2>&1); then
#     echo "Job submitted successfully:"
#     echo "${sbatch_output}"
#     job_id=$(echo "${sbatch_output}" | grep -o '[0-9]\+$')
#     echo "Job ID: ${job_id}"
#     exit_code=0
# else
#     echo "Job submission failed:"
#     echo "${sbatch_output}"
#     exit_code=1
# fi

# # Clean up the temporary script file
# rm -f "${script_file}"

# # Return appropriate exit code
# exit ${exit_code}