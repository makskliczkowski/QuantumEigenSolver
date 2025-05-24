#SBATCH -N1
#SBATCH -c1
#SBATCH --mem=1gb
#SBATCH --time=1:00:00
#SBATCH --job-name=vqmc-loop

# Source the reusable library
HOME_DIR=${HOME:-$HOME}
CODES_DIR="${HOME_DIR}/Codes/"
PACKAGE_DIR="${CODES_DIR}/QuantumEigenSolver/"
source "${PACKAGE_DIR}/slurm/scripts/slurm_lib.sh"

# Function to display usage
show_usage() {
    show_usage_template "$0" "a_start a_step a_num n_rel Ns_start Ns_end n t_num"
    
    cat << EOF

Examples:
    $0 "0.68 0.06 1 5 16 16 1 100000" --time=24:00:00 --mem=16 -c16
    $0 0.68 0.06 1 5 16 16 1 100000 --time=133:19:59 --mem=16gb -c16
Using:
    ./evolution_makkli.sh "0.68 0.06 1 5 16 16 1 100000" --time=24:00:00 --mem=16gb -c16
EOF
}

# ---------------------------------------------------------------------------------------------------------

# Function to validate VQMC-specific parameters
validate_vqmc_params() {
    local a_start="$1" a_step="$2" a_num="$3" n_rel="$4"
    local Ns_start="$5" Ns_end="$6" n="$7" t_num="$8"
    
    # Validate all parameters are provided
    local params=("$a_start" "$a_step" "$a_num" "$n_rel" "$Ns_start" "$Ns_end" "$n" "$t_num")
    local param_names=("a_start" "a_step" "a_num" "n_rel" "Ns_start" "Ns_end" "n" "t_num")
    
    for i in "${!params[@]}"; do
        if [[ -z "${params[$i]}" ]]; then
            echo "Error: Missing required parameter: ${param_names[$i]}" >&2
            return 1
        fi
        
        # Validate numeric parameters
        validate_numeric "${params[$i]}" "${param_names[$i]}" false || return 1
    done
    
    # Additional domain-specific validations
    if (( $(echo "$Ns_start > $Ns_end" | bc -l) )); then
        echo "Error: Ns_start ($Ns_start) must be <= Ns_end ($Ns_end)" >&2
        return 1
    fi
    
    return 0
}

# ----------------------------------------------------------------------------------------------------------

# Function to parse VQMC parameters from string or arguments
parse_vqmc_params() {
    local -n params_ref=$1    # use nameref so we actually write into the caller’s array
    shift

    # First argument contains spaces, treat as parameter string
    if [[ "$1" == *" "* ]]; then
        read -r params_ref[a_start] params_ref[a_step] params_ref[a_num] params_ref[n_rel] \
                params_ref[Ns_start] params_ref[Ns_end] params_ref[n] params_ref[t_num] <<< "$1"
        return 1   
    else                        # Signal that we used parameter string format
        if [ $# -lt 8 ]; then
            echo "Error: Insufficient parameters. Need: a_start a_step a_num n_rel Ns_start Ns_end n t_num"
            return 2
        fi
        
        params_ref[a_start]="$1"
        params_ref[a_step]="$2"
        params_ref[a_num]="$3"
        params_ref[n_rel]="$4"
        params_ref[Ns_start]="$5"
        params_ref[Ns_end]="$6"
        params_ref[n]="$7"
        params_ref[t_num]="$8"
        
        return 0  # Signal that we used individual parameter format
    fi
}

# ----------------------------------------------------------------------------------------------------------

################

# Main execution
main() {
    # Initialize parameter array
    declare -A params
    declare -A defaults
    
    # Set default paths - CUSTOMIZE THESE FOR YOUR ENVIRONMENT
    defaults[BASE_DIR]="${PACKAGE_DIR}"
    defaults[RUN_DIR]="${PACKAGE_DIR}/Python/projects/2025_um_evolotion"
    defaults[LUSTRE_DIR]="${HOME_DIR}/mylustre/DATA_EVO_2025_UM"
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
    
    # Parse VQMC parameters
    parse_return=0
    if parse_vqmc_params params "$@"; then
        # Used individual parameters
        if [ $# -lt 11 ]; then
            echo "Error: Insufficient parameters. Need: a_start a_step a_num n_rel Ns_start Ns_end n t_num TIM MEM CPU"
            show_usage
            exit 1
        fi
        shift 8
    else
        parse_result=$?
        if [[ $parse_result -eq 2 ]]; then
            show_usage
            exit 1
        fi
        # Used parameter string format, shift by 1
        shift 1
    fi
    
    # Set defaults from parameter string if provided
    TIM="${params[param_time]:-01:00:00}"
    MEM="${params[param_mem]:-1}"
    CPU="${params[param_cpu]:-1}"
    
    # Ensure all critical parameters are defined
    if [[ -z "${params[n]:-}" ]] || [[ -z "${params[t_num]:-}" ]]; then
        echo "Error: Critical parameters 'n' and 't_num' must be provided" >&2
        show_usage
        exit 1
    fi
    
    # Parse memory if it came from parameters
    if [[ -n "${params[param_mem]:-}" ]]; then
        MEM=$(parse_memory "${params[param_mem]}") || exit 1
    fi
    
    # Parse remaining SLURM options
    override_time=""
    override_mem=""
    override_cpu=""

    if ! parse_slurm_options override_time override_mem override_cpu "$@"; then
        case $? in
            1) show_usage; exit 1 ;;
            2) show_usage; exit 0 ;;
        esac
    fi
    echo "Parsed SLURM options: TIM=$override_time, MEM=$override_mem, CPU=$override_cpu"

    # Apply resource overrides
    apply_resource_overrides TIM MEM CPU "$override_time" "$override_mem" "$override_cpu" || exit 1
    # Set resource parameters, using overrides if provided, otherwise defaults


    # Validate VQMC parameters
    validate_vqmc_params "${params[a_start]}" "${params[a_step]}" "${params[a_num]}" "${params[n_rel]}" \
                         "${params[Ns_start]}" "${params[Ns_end]}" "${params[n]}" "${params[t_num]}" || exit 1
    
    # Validate final resource parameters
    validate_numeric "$CPU" "CPU" false || exit 1
    validate_numeric "$MEM" "MEM" false || exit 1
    TIM=$(validate_time "$TIM") || exit 1
    
    # Validate memory distribution with minimum 2GB per worker
    # mem_per_worker=$(validate_memory_distribution "$MEM" "$CPU" 2) || exit 1
    
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
    echo "=== VQMC Job Configuration ==="
    echo "  BASE_DIR=${defaults[BASE_DIR]}"
    echo "  RUN_DIR=${defaults[RUN_DIR]}"
    echo "  LUSTRE_DIR=${defaults[LUSTRE_DIR]}"
    echo "  QES_PACKAGE_DIR=${defaults[QES_PACKAGE_DIR]}"
    echo "  Parameters:"
    echo "    Ns_start=${params[Ns_start]}, Ns_end=${params[Ns_end]}"
    echo "    a_start=${params[a_start]}, a_step=${params[a_step]}, a_num=${params[a_num]}"
    echo "    n_rel=${params[n_rel]}, n=${params[n]}, t_num=${params[t_num]}"
    echo "  Resources: ${TIM}, ${MEM}gb, ${CPU} CPUs"
    echo "  mem_per_worker=${mem_per_worker}gb"
    echo "=============================="
    
    # Generate job identifiers
    job_params="fun=2025_um_evo,Ns=${params[Ns_start]}-${params[Ns_end]},a=${params[a_start]}-${params[a_step]}-${params[a_num]}"
    script_file="${defaults[SLURM_DIR]}/${job_params}.sh"
    
    # Create SLURM job script
    {
        create_slurm_header "$CPU" "$MEM" "$TIM" "${defaults[SLURM_DIR]}" "$job_params"
        
        echo "# Log the job directory"
        echo "echo -e \"\$(pwd)\t\$(date)\" >> ${defaults[RUN_DIR]}/slurm_ids_run.log"
        echo ""
        
        add_module_section "Python/3.10.4-GCCcore-11.3.0" "HDF5"

        setup_qes_environment "${defaults[QES_PACKAGE_DIR]}" "requirements/requirements.txt" "qes_venv" "${CODES_DIR}/qes_venv"

        echo "# Change to working directory"
        echo "cd ${defaults[RUN_DIR]}"
        echo ""
        
        echo "# Verify QES is available"
        echo "python3 -c \"import QES; print('QES version:', getattr(QES, '__version__', 'unknown'))\""
        echo ""
        
        echo "# Run the main computation"
        echo "python3 ${defaults[RUN_DIR]}/evolution.py \\"
        echo "    ${defaults[LUSTRE_DIR]} \\"
        echo "    ${params[a_start]} ${params[a_step]} ${params[a_num]} \\"
        echo "    ${params[n_rel]} ${params[Ns_start]} ${params[Ns_end]} \\"
        echo "    ${params[n]} ${params[t_num]} ${mem_per_worker} \\"
        echo "    > ${defaults[LOG_DIR]}/log_${job_params}.log 2>&1"
        echo ""
        
        echo "echo \"Job completed successfully\""
    } > "$script_file"
    
    # Submit the job
    submit_slurm_job "$script_file" true || exit 1
    
    # Log job information
    log_job_info "${defaults[RUN_DIR]}/submitted_jobs.log" "$job_params"
}

# Run main function with all arguments
main "$@"

# -----------------------------------------------------------------------------------------------------------
# End of script
