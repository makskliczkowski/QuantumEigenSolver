#!/bin/bash
#SBATCH -N1
#SBATCH -c1
#SBATCH --mem=1gb
#SBATCH --time=1:00:00
#SBATCH --job-name=vqmc-array

# Source the reusable library
PACKAGE_DIR="/home/makkli4548/CODES/QuantumEigenSolver/"
source "${PACKAGE_DIR}/slurm/scripts/slurm_lib.sh"

# Function to display usage specific to this array script
show_usage() {
    cat << EOF
Usage: $0 PARAMETER_FILE [OPTIONS]

PARAMETER_FILE format (one job per line):
    a_start a_step a_num n_rel Ns_start Ns_end n t_num

Options:
    --array=ARRAY_SPEC      SLURM array specification (e.g., 1-10, 1-20:2)
    --time=TIME             Override time allocation (format: HH:MM:SS)
    --mem=MEM               Override memory allocation (in GB, format: Xgb or X)
    -c CPU, --cpu=CPU       Override CPU count
    --name=JOB_NAME         Override job name prefix
    -h, --help              Show this help message

Examples:
    $0 parameters.txt --array=1-100 --time=24:00:00 --mem=8gb -c4
    $0 params.txt --array=1-50:5 --name=vqmc_batch_1
Using:
    ./evolution_makkli_array.sh parameters.txt --array=1-100 --time=24:00:00 --mem=8gb -c4

Parameter file example:
    0.68 0.06 1 5 16 16 1 100000
    0.70 0.06 1 5 16 16 1 100000
    0.72 0.06 1 5 16 16 1 100000
EOF
}

# ------------------------------------------------------------------------------------------------------------

# Function to validate parameter file
validate_parameter_file() {
    local param_file="$1"
    local line_num=0
    
    # Check file exists and is readable
    validate_file_exists "$param_file" "Parameter file" || return 1
    
    # Validate each line has correct number of parameters
    while IFS= read -r line || [[ -n "$line" ]]; do
        ((line_num++))
        
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        
        # Count parameters
        local param_array=($line)
        if [[ ${#param_array[@]} -ne 8 ]]; then
            echo "Error: Line $line_num has ${#param_array[@]} parameters, expected 8" >&2
            echo "       Line: $line" >&2
            return 1
        fi
        
        # Validate each parameter is numeric
        local param_names=("a_start" "a_step" "a_num" "n_rel" "Ns_start" "Ns_end" "n" "t_num")
        for i in "${!param_array[@]}"; do
            if ! validate_numeric "${param_array[$i]}" "${param_names[$i]}" false; then
                echo "Error: Line $line_num, parameter ${param_names[$i]}" >&2
                return 1
            fi
        done
    done < "$param_file"
    
    echo "Parameter file validated successfully: $line_num parameter sets found"
    return 0
}

# -------------------------------------------------------------------------------------------------------------

# Function to count non-empty, non-comment lines
count_parameter_sets() {
    local param_file="$1"
    local count=0
    
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        ((count++))
    done < "$param_file"
    
    echo "$count"
}

# -------------------------------------------------------------------------------------------------------------

# Main execution
main() {
    # Initialize variables
    local param_file=""
    local array_spec=""
    local override_time=""
    local override_mem=""
    local override_cpu=""
    local job_name_prefix="vqmc_array"
    
    # Set default paths - CUSTOMIZE THESE FOR YOUR ENVIRONMENT
    declare -A defaults
    defaults[BASE_DIR]="/home/makkli4548/CODES/QuantumEigenSolver/"
    defaults[RUN_DIR]="/home/makkli4548/CODES/QuantumEigenSolver/Python/projects/2025_um_evolotion"
    defaults[LUSTRE_DIR]="/home/makkli4548/mylustre/DATA_EVO_2025_UM"
    defaults[LOG_DIR]="${defaults[RUN_DIR]}/LOG/RANDOM_MODELS_EVO_2025_UM"
    defaults[SLURM_DIR]="${defaults[RUN_DIR]}/SLURM"
    defaults[QES_PACKAGE_DIR]="${defaults[BASE_DIR]}/Python/QES"
    
    # Parse command line arguments
    if [ $# -eq 0 ]; then
        show_usage
        exit 1
    fi
    
    # Check for help
    if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
        show_usage
        exit 0
    fi
    
    # First argument should be parameter file
    param_file="$1"
    shift
    
    # Parse remaining options
    if ! parse_slurm_options override_time override_mem override_cpu "$@"; then
        case $? in
            1) show_usage; exit 1 ;;
            2) show_usage; exit 0 ;;
        esac
    fi
    
    # Validate parameter file
    echo "Validating parameter file: $param_file"
    validate_parameter_file "$param_file" || exit 1
    
    # Count parameter sets and set default array spec if not provided
    param_count=$(count_parameter_sets "$param_file")
    if [[ -z "$array_spec" ]]; then
        array_spec="1-${param_count}"
        echo "Auto-detected array specification: $array_spec"
    fi
    
    # Set default resources
    TIM="${override_time:-24:00:00}"
    MEM="${override_mem:-8}"
    CPU="${override_cpu:-4}"
    
    # Parse and validate memory
    if [[ "$MEM" =~ [gG][bB]?$ ]]; then
        MEM=$(parse_memory "$MEM") || exit 1
    fi
    
    # Validate resources
    validate_numeric "$CPU" "CPU" false || exit 1
    validate_numeric "$MEM" "MEM" false || exit 1
    TIM=$(validate_time "$TIM") || exit 1
    
    # Validate memory distribution with minimum 2GB per worker
    mem_per_worker=$(validate_memory_distribution "$MEM" "$CPU" 2) || exit 1
    
    # Validate QES package directory exists
    validate_file_exists "${defaults[QES_PACKAGE_DIR]}/setup.py" "QES setup.py file" || exit 1
    
    # Create and validate directories
    dirs_to_create=("${defaults[LUSTRE_DIR]}" "${defaults[LOG_DIR]}" "${defaults[SLURM_DIR]}")
    create_directories "${dirs_to_create[@]}" || exit 1
    
    # Validate access to run directory
    validate_directories "${defaults[RUN_DIR]}" || exit 1
    
    # Change to run directory
    cd "${defaults[RUN_DIR]}" || exit 1
    
    # Display configuration
    echo "=== VQMC Job Array Configuration ==="
    echo "  BASE_DIR=${defaults[BASE_DIR]}"
    echo "  RUN_DIR=${defaults[RUN_DIR]}"
    echo "  LUSTRE_DIR=${defaults[LUSTRE_DIR]}"
    echo "  QES_PACKAGE_DIR=${defaults[QES_PACKAGE_DIR]}"
    echo "  Parameter file: $param_file"
    echo "  Parameter sets: $param_count"
    echo "  Array specification: $array_spec"
    echo "  Resources per job: ${TIM}, ${MEM}gb, ${CPU} CPUs"
    echo "  mem_per_worker=${mem_per_worker}gb"
    echo "===================================="
    
    # Generate job script
    job_name="${job_name_prefix}_$(date +%Y%m%d_%H%M%S)"
    script_file="${defaults[SLURM_DIR]}/${job_name}.sh"
    
    # Create SLURM job array script
    {
        create_slurm_array_header "$CPU" "$MEM" "$TIM" "${defaults[SLURM_DIR]}" "$job_name" "$array_spec"
        
        echo "# Parameter file path"
        echo "PARAM_FILE=\"$param_file\""
        echo ""
        
        echo "# Get parameters for this array task"
        echo "PARAMS=\$(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" \"\$PARAM_FILE\" | grep -v '^[[:space:]]*#' | head -1)"
        echo "if [[ -z \"\$PARAMS\" ]]; then"
        echo "    echo \"Error: No parameters found for task ID \${SLURM_ARRAY_TASK_ID}\" >&2"
        echo "    exit 1"
        echo "fi"
        echo ""
        
        echo "# Parse parameters"
        echo "read -r a_start a_step a_num n_rel Ns_start Ns_end n t_num <<< \"\$PARAMS\""
        echo ""
        
        echo "# Validate parameters are set"
        echo "if [[ -z \"\$a_start\" ]] || [[ -z \"\$t_num\" ]]; then"
        echo "    echo \"Error: Failed to parse parameters: \$PARAMS\" >&2"
        echo "    exit 1"
        echo "fi"
        echo ""
        
        echo "# Log the job information"
        echo "echo \"Array task \${SLURM_ARRAY_TASK_ID}: \$PARAMS\" >> ${defaults[RUN_DIR]}/array_jobs_run.log"
        echo "echo -e \"\$(pwd)\t\$(date)\tTask_\${SLURM_ARRAY_TASK_ID}\" >> ${defaults[RUN_DIR]}/slurm_ids_run.log"
        echo ""
        
        add_module_section "Python/3.10.4-GCCcore-11.3.0" "HDF5"
        
        setup_qes_environment "${defaults[BASE_DIR]}" "qes_vqmc_env"
        
        echo "# Change to working directory"
        echo "cd ${defaults[RUN_DIR]}"
        echo ""
        
        echo "# Verify QES is available"
        echo "python3 -c \"import QES; print('QES version:', getattr(QES, '__version__', 'unknown'))\""
        echo ""
        
        echo "# Generate unique job identifier for this task"
        echo "job_id=\"task_\${SLURM_ARRAY_TASK_ID}_Ns_\${Ns_start}-\${Ns_end}_a_\${a_start}-\${a_step}-\${a_num}\""
        echo ""
        
        echo "# Run the main computation"
        echo "python3 ${defaults[RUN_DIR]}/evolution.py \\"
        echo "    ${defaults[LUSTRE_DIR]} \\"
        echo "    \$a_start \$a_step \$a_num \\"
        echo "    \$n_rel \$Ns_start \$Ns_end \\"
        echo "    \$n \$t_num ${mem_per_worker} \\"
        echo "    > ${defaults[LOG_DIR]}/log_\${job_id}.log 2>&1"
        echo ""
        
        echo "echo \"Array task \${SLURM_ARRAY_TASK_ID} completed successfully\""
    } > "$script_file"
    
    # Submit the job array
    echo "Submitting job array: $job_name"
    submit_slurm_job "$script_file" false || exit 1
    
    # Log job information
    echo -e "${job_name}\t${array_spec}\t$(date)" >> "${defaults[RUN_DIR]}/submitted_array_jobs.log"
    
    echo "Job array submitted successfully!"
    echo "Monitor with: squeue -u \$USER"
    echo "Cancel with: scancel \$JOB_ID (where JOB_ID is from squeue output)"
}

# Run main function with all arguments
main "$@"

# -----------------------------------------------------------------------------------------------------------
# End of script