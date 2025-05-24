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

# Function to validate numeric parameter
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

# Function to parse common SLURM options
parse_slurm_options() {
    local -n override_time_ref=$1
    local -n override_mem_ref=$2
    local -n override_cpu_ref=$3
    
    while [[ $# -gt 3 ]]; do
        shift 3  # Skip the reference parameters
        case $1 in
            --time=*)
                override_time_ref="${1#*=}"
                shift
                ;;
            --mem=*)
                override_mem_ref="${1#*=}"
                shift
                ;;
            -c*)
                if [ "${1#-c}" != "$1" ]; then
                    # Format: -c16
                    override_cpu_ref="${1#-c}"
                else
                    # Format: -c 16
                    shift
                    override_cpu_ref="$1"
                fi
                shift
                ;;
            --cpu=*)
                override_cpu_ref="${1#*=}"
                shift
                ;;
            -h|--help)
                return 2  # Signal help requested
                ;;
            *)
                echo "Error: Unknown option: $1" >&2
                return 1
                ;;
        esac
    done
    
    return 0
}

# Function to apply resource overrides
apply_resource_overrides() {
    local -n tim_ref=$1
    local -n mem_ref=$2
    local -n cpu_ref=$3
    local override_time="$4"
    local override_mem="$5"
    local override_cpu="$6"
    
    # Apply overrides with validation
    if [ -n "$override_time" ]; then
        tim_ref=$(validate_time "$override_time") || return 1
    fi
    if [ -n "$override_mem" ]; then
        mem_ref=$(parse_memory "$override_mem") || return 1
    fi
    if [ -n "$override_cpu" ]; then
        validate_numeric "$override_cpu" "CPU" false || return 1
        cpu_ref="$override_cpu"
    fi
    
    return 0
}

# Function to create directory structure
create_directories() {
    local dirs=("$@")
    for dir in "${dirs[@]}"; do
        if ! mkdir -p "$dir"; then
            echo "Error: Cannot create directory: $dir" >&2
            return 1
        fi
    done
}

# Function to validate directory access
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

# Function to create SLURM script header
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

# Function to add module loading section
add_module_section() {
    local modules=("$@")
    
    echo "# Load required modules"
    echo "source /usr/local/sbin/modules.sh"
    echo ""
    
    for module in "${modules[@]}"; do
        echo "module load $module"
    done
    echo ""
}

# Function to add Python environment setup
add_python_env_section() {
    local run_dir="$1"
    local requirements_file="$2"
    
    cat << EOF
# Set up Python environment
virtualenv ${run_dir}/venv
source ${run_dir}/venv/bin/activate
pip install -r ${requirements_file}

EOF
}

# Function to submit SLURM job
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

# Function to log job information
log_job_info() {
    local log_file="$1"
    local job_info="$2"
    local timestamp=$(date)
    
    echo -e "${job_info}\t${timestamp}" >> "$log_file"
}