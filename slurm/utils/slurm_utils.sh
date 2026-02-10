#!/bin/bash
# slurm_utils.sh - Reusable SLURM job management functions

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
    -p PART, --partition=PART Partition to use
    -h, --help              Show this help message
EOF
}

# Function to parse memory value and convert to GB integer.
#   It strictly requires the input to be a number (assumed GB) or a number with 'gb'/'GB' suffix.
#   It rejects other suffixes like 'mb' to avoid interpretation errors.
#   Returns the memory value in GB (integer) or an error message if invalid.
parse_memory() {
    local mem_input="$1"
    local mem_value
    
    # Check if input has invalid suffix (anything other than nothing, g, gb, G, GB)
    # We use grep to check if the string ends with something else
    if echo "$mem_input" | grep -Eq '[^0-9gGbB]'; then
         echo "Error: Invalid memory format: $mem_input (Only GB supported, use 'gb' suffix or integer)" >&2
         return 1
    fi

    # Check specifically for 'm' or 'mb' or 'k' etc to be helpful
    if echo "$mem_input" | grep -iq '[mk]b\?$'; then
         echo "Error: Invalid memory format: $mem_input (Only GB supported)" >&2
         return 1
    fi

    # Remove 'gb' or 'g' suffix (case insensitive)
    mem_value=$(echo "$mem_input" | sed -E 's/[gG][bB]?$//')
    
    # Check if result is a positive integer
    if [[ ! "$mem_value" =~ ^[0-9]+$ ]] || [ "$mem_value" -le 0 ]; then
        echo "Error: Invalid memory value: $mem_input" >&2
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

# Function to check if sbatch is available
check_sbatch_available() {
    if ! command -v sbatch &> /dev/null; then
        echo "Error: sbatch command not found. Are you on a login node?" >&2
        return 1
    fi
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

# Function to log error messages
#   It appends error messages with timestamps to a specified log file.
#   Parameters:
#     - Log file path
#     - Error message
#     - Optional job parameters for context
log_error() {
    local logfile="$1"
    local message="$2"
    local ts
    ts=$(date +"%Y-%m-%d %H:%M:%S")
    echo -e "[ERROR] $ts | ${job_params:-"N/A"} | $message" >> "$logfile"
    echo "Error: $message (logged to $logfile)" >&2
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
#     - Optional partition name
create_slurm_header() {
    local cpu="$1"
    local mem="$2"
    local time="$3"
    local output_dir="$4"
    local job_name="$5"
    local partition="${6:-}"    # optional
    local gres="${7:-}"         # optional

    mkdir -p "${output_dir}" || {
        echo "Error: Failed to create output directory '${output_dir}'" >&2
        exit 1
    }

    cat << EOF
#!/bin/bash
#SBATCH -N1
#SBATCH -c${cpu}
#SBATCH --mem=${mem}gb
#SBATCH --time=${time}
#SBATCH -o ${output_dir}/out-%j-${job_name}.out
#SBATCH -e ${output_dir}/err-%j-${job_name}.err
#SBATCH --job-name=${job_name}
${partition:+#SBATCH -p ${partition}}
${gres:+#SBATCH --gres=${gres}}

# -----------------------------------------------------------------------------
# Environment setup and error handling
# -----------------------------------------------------------------------------

# Fail on errors, unset vars, and failed pipes
set -euo pipefail

# set -x

# Export job ID and metadata
export SLURM_JOB_ID=\${SLURM_JOB_ID:-unknown}
export JOB_NAME="${job_name}"
export JOB_START_TIME=\$(date '+%Y-%m-%d %H:%M:%S')

echo "============================================================"
echo " Job started: \${JOB_NAME}"
echo " Job ID: \${SLURM_JOB_ID}"
echo " Start time: \${JOB_START_TIME}"
echo " Node list : \${SLURM_NODELIST:-N/A}"
echo "============================================================"

# -----------------------------------------------------------------------------
# Cleanup on exit
# -----------------------------------------------------------------------------
cleanup() {
    local exit_code=\$?
    local end_time=\$(date '+%Y-%m-%d %H:%M:%S')

    echo "------------------------------------------------------------"
    echo " Job finished."
    echo " Job name   : \${JOB_NAME}"
    echo " Job ID     : \${SLURM_JOB_ID}"
    echo " Exit code  : \$exit_code"
    echo " End time   : \${end_time}"
    echo "------------------------------------------------------------"

    if [[ \$exit_code -ne 0 ]]; then
        echo "Error: Job \${JOB_NAME} exited with code \$exit_code" >&2
    fi
    exit \$exit_code
}
trap cleanup EXIT
EOF
}

# ----------------------------------------------------------------------

# Function to add module loading section.
#   It generates a section to load required modules in the SLURM script.
#   It takes a list of module names as arguments.
#   If no modules are provided, it generates a comment but no load commands.
add_module_section() {
    local modules=("$@")
    
    echo "# Load required modules"
    echo 'if [ -f /usr/local/sbin/modules.sh ]; then'
    echo '    source /usr/local/sbin/modules.sh'
    echo 'elif [ -f /etc/profile.d/modules.sh ]; then'
    echo '    source /etc/profile.d/modules.sh'
    echo 'elif [ -f /usr/share/modules/init/bash ]; then'
    echo '    source /usr/share/modules/init/bash'
    echo 'else'
    echo '    echo "Warning: No module initialization script found." >&2'
    echo 'fi'
    
    echo 'module purge || echo "Module purge failed, proceeding..."'
    echo ""
    
    if [ ${#modules[@]} -eq 0 ]; then
        echo "# No specific modules requested."
        return 0
    fi

    for module in "${modules[@]}"; do
        cat <<EOF
if module avail "$module" 2>&1 | grep -q "$module"; then
    module load "$module"
else
    base="\$(echo "$module" | cut -d'/' -f1)"   # take family before '/'
    echo "Module '$module' not found, falling back to '\$base'" >&2
    module load "\$base" || echo "Fallback '\$base' not available either." >&2
fi

EOF
    done
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
    echo "pip install --upgrade pip setuptools wheel -q"
    echo "pip install -r \"\$REQUIREMENTS_FILE\" -q || {"
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
    
    # Check if sbatch is available
    if ! command -v sbatch &> /dev/null; then
        echo "Error: sbatch not found. Skipping submission." >&2
        echo "Script generated at: $script_file"
        return 1
    fi

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

# Function to log job information
#   It appends job information and timestamp to a log file.
log_job_info() {
    local log_file="$1"
    local job_info="$2"
    local timestamp=$(date)
    
    echo -e "${job_info}\t${timestamp}" >> "$log_file"
}

# -----------------------------------------------------------------------
# End of slurm_utils.sh
# -----------------------------------------------------------------------
