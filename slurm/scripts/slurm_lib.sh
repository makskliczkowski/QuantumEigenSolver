#!/bin/bash
# slurm_lib.sh - Reusable SLURM job management functions

# Function to display usage template
show_usage_template() {
    local script_name="$1"
    local param_desc="$2"
    cat << EOF
Usage   : $script_name "param_string" [OPTIONS]
    or  : $script_name [individual parameters] [OPTIONS]

Parameter string format: "$param_desc"

Options:
    --time=TIME             Override time allocation (format: HH:MM:SS)
    --mem=MEM               Override memory allocation (in GB, format: Xgb or X)
    -c CPU, --cpu=CPU       Override CPU count
    -h, --help              Show this help message
EOF
}

# Function to parse memory value and convert to GB integer
#   It removes any 'gb' suffix and ensures the value is a positive integer.
#   Returns the memory value in GB or an error message if invalid.
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

# ----------------------------------------------------------------------

# Function to validate time format
#   It checks if the time is in HH:MM:SS format and returns the validated time.
#   Returns the time string or an error message if invalid.
validate_time() {
    local time_input="$1"
    if [[ ! "$time_input" =~ ^[0-9]{1,3}:[0-5]?[0-9]:[0-5]?[0-9]$ ]]; then
        echo "Error: Invalid time format: $time_input (expected HH:MM:SS)" >&2
        return 1
    fi
    echo "$time_input"
}

# ----------------------------------------------------------------------

# Function to validate numeric parameter
#   It checks if the value is numeric and optionally allows negative values.
#   Returns 0 if valid, 1 if invalid, and prints an error message to stderr.
validate_numeric() {
    local value="$1"
    local name="$2"
    local allow_negative="${3:-false}"
    
    if ! [[ "$value" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
        echo "Error: $name must be numeric, got: $value" >&2
        return 1
    fi
    
    if [[ "$allow_negative" == "false" ]] && (( $(echo "$value <= 0" | bc -l) )); then
        echo "Error: $name must be positive, got: $value" >&2
        return 1
    fi
    
    return 0
}

# ----------------------------------------------------------------------

# Function to parse common SLURM options
#   It extracts time, memory, and CPU options from the command line arguments.
#   It updates the provided references with the parsed values.
#   Returns 0 on success, 1 on error, and 2 if help is requested.
# Function to parse common SLURM options
#   $1 = name of variable to receive time override
#   $2 = name of variable to receive mem override
#   $3 = name of variable to receive cpu override
#   returns: 0=ok, 1=error, 2=help
parse_slurm_options() {
    local time_var=$1 mem_var=$2 cpu_var=$3
    shift 3

    # Temporaries for the values we’ll collect
    local _ot _om _oc

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                return 2
                ;;
            -t)
                _ot=$2; shift 2
                ;;
            -t*)
                _ot=${1#-t}; shift
                ;;
            --time=*)
                _ot=${1#*=}; shift
                ;;
            -m)
                _om=$2; shift 2
                ;;
            -m*)
                _om=${1#-m}; shift
                ;;
            --mem=*)
                _om=${1#*=}; shift
                ;;
            -c)
                _oc=$2; shift 2
                ;;
            -c*)
                _oc=${1#-c}; shift
                ;;
            --cpu=*)
                _oc=${1#*=}; shift
                ;;
            *)
                echo "Error: Unknown option: $1" >&2
                return 1
                ;;
        esac
    done

    # Write them back into the caller's variables
    eval "$time_var=\"\$_ot\""
    eval "$mem_var=\"\$_om\""
    eval "$cpu_var=\"\$_oc\""

    echo "Parsed SLURM options:"
    echo "  Time: ${!time_var:-not set}"
    echo "  Memory: ${!mem_var:-not set}"
    echo "  CPU: ${!cpu_var:-not set}"
    return 0
}

# ----------------------------------------------------------------------

# Function to apply resource overrides
#   It updates the provided references with validated overrides for time, memory, and CPU.
#   Returns 0 on success, 1 on error.
apply_resource_overrides() {
    local time_var=$1 mem_var=$2 cpu_var=$3
    local override_time=$4 override_mem=$5 override_cpu=$6
    local newval

    # 1) Time override
    if [ -n "$override_time" ]; then
        newval=$(validate_time "$override_time") || return 1
        # newval=$override_time
        eval "$time_var=\"\$newval\""
    fi

    # 2) Memory override
    if [ -n "$override_mem" ]; then
        newval=$(parse_memory "$override_mem") || return 1
        # newval=$override_mem
        eval "$mem_var=\"\$newval\""
    fi

    # 3) CPU override
    if [ -n "$override_cpu" ]; then
        validate_numeric "$override_cpu" "CPU" false || return 1
        # override_cpu=$(echo "$override_cpu" | sed 's/[^0-9]//g')
        eval "$cpu_var=\"\$override_cpu\""
    fi
    # Print the final values
    echo "Final resource parameters:"
    echo "  Time: ${!time_var:-not set}"
    echo "  Memory: ${!mem_var:-not set}"
    echo "  CPU: ${!cpu_var:-not set}"
    return 0
}


# ----------------------------------------------------------------------

# Function to create directory structure
#   It takes a list of directories and creates them if they do not exist.
#   Returns 0 on success, 1 if any directory cannot be created.
create_directories() {
    local dirs=("$@")
    for dir in "${dirs[@]}"; do
        if ! mkdir -p "$dir"; then
            echo "Error: Cannot create directory: $dir" >&2
            return 1
        fi
    done
}

# ----------------------------------------------------------------------

# Function to validate directory access
#   It checks if the directories exist and if they are writable.
#   Returns 0 if all directories are valid, 1 if any directory is invalid.
validate_directories() {
    local dirs=("$@")
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            echo "Error: Directory does not exist: $dir" >&2
            return 1
        fi
        if [[ ! -w "$dir" ]]; then
            echo "Error: No write permission for directory: $dir" >&2
            return 1
        fi
    done
}

# ----------------------------------------------------------------------

# Function to create SLURM script header
#   It generates a SLURM script header with specified resources and output directories.
#   Parameters:
#     - CPU count
#     - Memory in GB
#     - Time allocation in HH:MM:SS format
#     - Output directory for logs
#     - Job name
create_slurm_header() {
    local cpu="$1"
    local mem="$2"
    local time="$3"
    local output_dir="$4"
    local job_name="$5"
    
    cat << EOF
#!/bin/bash
#SBATCH -N1
#SBATCH -c${cpu}
#SBATCH --mem=${mem}gb
#SBATCH --time=${time}
#SBATCH -o ${output_dir}/out-%j-${job_name}.out
#SBATCH -e ${output_dir}/err-%j-${job_name}.err
#SBATCH --job-name=${job_name}

# Export job ID for use in script
export SLURM_JOB_ID=\${SLURM_JOB_ID}

# Set up error handling
set -e
set -u
set -o pipefail

# Function to clean up on exit
cleanup() {
    local exit_code=\$?
    echo "Job finished with exit code: \$exit_code"
    # Add any cleanup logic here
    exit \$exit_code
}
trap cleanup EXIT

EOF
}

# ----------------------------------------------------------------------

# Function to add module loading section
#   It generates a section to load required modules in the SLURM script.
#   It takes a list of module names as arguments.
add_module_section() {
    local modules=("$@")
    
    echo "# Load required modules"
    echo "source /usr/local/sbin/modules.sh"    # Ensure module command is available
    echo ""
    
    for module in "${modules[@]}"; do
        echo "module load $module"
    done
    echo ""
}

# ----------------------------------------------------------------------

# Function to setup Python environment (only create if doesn't exist)
#   It generates a section to create and activate a Python virtual environment.
#   It takes the run directory, requirements file, and optional environment name.
setup_python_environment() {
    local run_dir="$1"
    local requirements_file="$2" 
    local env_name="${3:-vqmc_env}"
    local env_path="${run_dir}/${env_name}"
    
    # Validate requirements file exists
    if [[ ! -f "$requirements_file" ]]; then
        echo "Error: Requirements file not found: $requirements_file" >&2
        return 1
    fi
    
    echo "# Python Environment Setup"
    echo "ENV_PATH=\"${env_path}\""
    echo "REQUIREMENTS_FILE=\"${requirements_file}\""
    echo ""
    echo "# Create virtual environment only if it doesn't exist"
    echo "if [[ ! -d \"\$ENV_PATH\" ]]; then"
    echo "    echo \"Creating new Python environment: \$ENV_PATH\""
    echo "    python3 -m venv \"\$ENV_PATH\" || {"
    echo "        echo \"Error: Failed to create virtual environment\" >&2"
    echo "        exit 1"
    echo "    }"
    echo "    echo \"Virtual environment created successfully\""
    echo "else"
    echo "    echo \"Using existing Python environment: \$ENV_PATH\""
    echo "fi"
    echo ""
    echo "# Activate environment"
    echo "source \"\$ENV_PATH/bin/activate\" || {"
    echo "    echo \"Error: Failed to activate virtual environment\" >&2"
    echo "    exit 1"
    echo "}"
    echo ""
    echo "# Upgrade pip and install/update requirements"
    echo "echo \"Upgrading pip and installing requirements...\""
    echo "pip install --upgrade pip"
    echo "pip install -r \"\$REQUIREMENTS_FILE\" || {"
    echo "    echo \"Error: Failed to install requirements\" >&2"
    echo "    exit 1"
    echo "}"
    echo ""
    echo "echo \"Python environment ready\""
    echo ""
}

# ----------------------------------------------------------------------

# Function to validate memory calculation
#   It checks if the total memory divided by CPU count meets a minimum requirement.
#   Parameters:
#     - Total memory in GB
#     - CPU count
#     - Minimum memory per worker in GB (default: 1)
validate_memory_distribution() {
    local total_mem="$1"
    local cpu_count="$2"
    local min_mem_per_worker="${3:-1}"
    
    if ! validate_numeric "$total_mem" "total memory" false || 
       ! validate_numeric "$cpu_count" "CPU count" false; then
        return 1
    fi
    
    local mem_per_worker=$((total_mem / cpu_count))
    
    if [[ $mem_per_worker -lt $min_mem_per_worker ]]; then
        echo "Error: Memory per worker ($mem_per_worker GB) is below minimum ($min_mem_per_worker GB)" >&2
        echo "       Total memory: ${total_mem}GB, CPUs: ${cpu_count}" >&2
        echo "       Consider increasing memory or decreasing CPU count" >&2
        return 1
    fi
    
    echo "$mem_per_worker"
}

# ----------------------------------------------------------------------

# Function to submit SLURM job
#   It submits a SLURM job script and optionally cleans up the script file after submission.
#   Parameters:
#     - Script file path
#     - Cleanup script file after submission (default: true)
submit_slurm_job() {
    local script_file="$1"
    local cleanup_script="${2:-true}"
    
    # Make the script executable
    chmod +x "$script_file"
    
    echo "Submitting job script: $script_file"
    
    # Submit the job
    if sbatch_output=$(sbatch "$script_file" 2>&1); then
        echo "Job submitted successfully:"
        echo "$sbatch_output"
        job_id=$(echo "$sbatch_output" | grep -o '[0-9]\+$')
        echo "Job ID: $job_id"
        
        # Clean up the temporary script file if requested
        if [[ "$cleanup_script" == "true" ]]; then
            rm -f "$script_file"
        fi
        
        return 0
    else
        echo "Job submission failed:" >&2
        echo "$sbatch_output" >&2
        return 1
    fi
}

# ----------------------------------------------------------------------

# Function to create job array script header
create_slurm_array_header() {
    local cpu="$1"
    local mem="$2"
    local time="$3"
    local output_dir="$4"
    local job_name="$5"
    local array_spec="$6"  # e.g., "1-10" or "1-10:2"
    
    cat << EOF
#!/bin/bash
#SBATCH -N1
#SBATCH -c${cpu}
#SBATCH --mem=${mem}gb
#SBATCH --time=${time}
#SBATCH --array=${array_spec}
#SBATCH -o ${output_dir}/out-%A_%a-${job_name}.out
#SBATCH -e ${output_dir}/err-%A_%a-${job_name}.err
#SBATCH --job-name=${job_name}

# Export job ID and array task ID for use in script
export SLURM_JOB_ID=\${SLURM_JOB_ID}
export SLURM_ARRAY_JOB_ID=\${SLURM_ARRAY_JOB_ID}
export SLURM_ARRAY_TASK_ID=\${SLURM_ARRAY_TASK_ID}

# Set up error handling
set -e
set -u
set -o pipefail

# Function to clean up on exit
cleanup() {
    local exit_code=\$?
    echo "Array task \${SLURM_ARRAY_TASK_ID} finished with exit code: \$exit_code"
    # Add any cleanup logic here
    exit \$exit_code
}
trap cleanup EXIT

EOF
}

# ----------------------------------------------------------------------

# Function to validate file exists and is readable
validate_file_exists() {
    local file="$1"
    local description="${2:-File}"
    
    if [[ ! -f "$file" ]]; then
        echo "Error: $description not found: $file" >&2
        return 1
    fi
    
    if [[ ! -r "$file" ]]; then
        echo "Error: $description not readable: $file" >&2
        return 1
    fi
    
    return 0
}

# -----------------------------------------------------------------------
# Setup the QES environment
# -----------------------------------------------------------------------

setup_qes_environment() {
    local base_dir="$1"
    local req_file="$2"
    local venv_name="$3"
    local venv_path="$4"

    echo "# Setup QES Python environment"
    echo "export QES_BASE_DIR=\"${base_dir}\""
    echo "export QES_PACKAGE_DIR=\"\${QES_BASE_DIR}\""
    echo ""
    
    echo "# Create virtual environment if it doesn't exist"
    echo "if [ ! -d \"\${venv_path}/\${venv_name}\" ]; then"
    echo "    echo \"Creating virtual environment: \${venv_name}\""
    echo "    python3 -m venv \"\${venv_path}/\${venv_name}\""
    echo "fi"
    echo ""
    
    echo "# Activate virtual environment"
    echo "source \"\${venv_path}/\${venv_name}/bin/activate\""
    echo "echo \"Activated virtual environment: \${venv_name}\""
    echo ""
    
    echo "# Upgrade pip"
    echo "pip install --upgrade pip"
    echo ""
    
    echo "# Install QES package in development mode"
    echo "echo \"Installing QES package from: \${QES_PACKAGE_DIR}\""
    echo "cd \"\${QES_PACKAGE_DIR}\""
    echo "pip install -e .[all]  # Install with all optional dependencies"
    echo ""
    
    echo "# Verify QES installation"
    echo "python3 -c \"import QES; print(f'QES package successfully imported from: {QES.__file__}')\""
    echo ""
}

# -----------------------------------------------------------------------

# Function to log job information
#   It appends job information and timestamp to a log file.
log_job_info() {
    local log_file="$1"
    local job_info="$2"
    local timestamp=$(date)
    
    echo -e "${job_info}\t${timestamp}" >> "$log_file"
}

# -----------------------------------------------------------------------
# End of slurm_lib.sh
# -----------------------------------------------------------------------