#!/bin/bash
# run_py.sh - Submit Python QES jobs

# Get the directory of this script
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
source "$SCRIPT_DIR/utils/slurm_utils.sh"

# Default values
JOB_NAME="qes_py_job"
TIME="01:00:00"
MEM="4" # GB
CPU="1"
PARTITION=""
SCRIPT_PATH=""
VENV_PATH="./venv"
REQUIREMENTS_FILE="requirements.txt"
ARGS=""
OUTPUT_DIR="./logs"

# Usage info
usage() {
    show_usage_template "$0" "script_args"
    echo "    --script=PATH           Path to the python script (required)"
    echo "    --venv=PATH             Path to the virtual environment (default: ./venv)"
    echo "    --requirements=PATH     Path to requirements.txt (default: requirements.txt)"
    echo "    --job-name=NAME         Job name (default: qes_py_job)"
    echo "    --output-dir=DIR        Directory for logs (default: ./logs)"
    echo ""
    echo "Example: $0 --time=2:00:00 --script=my_script.py --venv=./myenv -- \"--arg1 val1\""
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -t)
            TIME=$(validate_time "$2") || exit 1; shift 2
            ;;
        --time=*)
            TIME=$(validate_time "${1#*=}") || exit 1; shift
            ;;
        -m)
            MEM=$(parse_memory "$2") || exit 1; shift 2
            ;;
        --mem=*)
            MEM=$(parse_memory "${1#*=}") || exit 1; shift
            ;;
        -c)
            CPU="$2"; validate_numeric "$CPU" "CPU" || exit 1; shift 2
            ;;
        --cpu=*)
            CPU="${1#*=}"; validate_numeric "$CPU" "CPU" || exit 1; shift
            ;;
        -p)
            PARTITION="$2"; shift 2
            ;;
        --partition=*)
            PARTITION="${1#*=}"; shift
            ;;
        --script=*)
            SCRIPT_PATH="${1#*=}"; shift
            ;;
        --venv=*)
            VENV_PATH="${1#*=}"; shift
            ;;
        --requirements=*)
            REQUIREMENTS_FILE="${1#*=}"; shift
            ;;
        --job-name=*)
            JOB_NAME="${1#*=}"; shift
            ;;
        --output-dir=*)
            OUTPUT_DIR="${1#*=}"; shift
            ;;
        --)
            shift
            ARGS="$*"
            break
            ;;
        *)
            echo "Error: Unknown option: $1" >&2
            usage
            ;;
    esac
done

# Validation
if [[ -z "$SCRIPT_PATH" ]]; then
    echo "Error: Script path (--script) is required." >&2
    usage
fi

validate_file_exists "$SCRIPT_PATH" "Python script" || exit 1

# Generate Script
SCRIPT_FILE="${OUTPUT_DIR}/submit_${JOB_NAME}_$(date +%s).sh"
create_directories "$OUTPUT_DIR"

# Write header
create_slurm_header "$CPU" "$MEM" "$TIME" "$OUTPUT_DIR" "$JOB_NAME" "$PARTITION" > "$SCRIPT_FILE"

# Append body
cat << EOF >> "$SCRIPT_FILE"

# Load modules
# Uncomment the following line if you need a system python module
# $(add_module_section "python")

# Python Environment
VENV_PATH="$VENV_PATH"
REQUIREMENTS_FILE="$REQUIREMENTS_FILE"

# Create venv if it doesn't exist and requirements are present
if [[ ! -d "\$VENV_PATH" ]]; then
    echo "Virtual environment not found at \$VENV_PATH"
    if [[ -f "\$REQUIREMENTS_FILE" ]]; then
        echo "Creating venv and installing requirements from \$REQUIREMENTS_FILE"
        python3 -m venv "\$VENV_PATH"
        source "\$VENV_PATH/bin/activate"
        pip install --upgrade pip
        pip install -r "\$REQUIREMENTS_FILE"
    else
        echo "Error: Cannot create venv. Requirements file not found: \$REQUIREMENTS_FILE" >&2
        # Fallback to just running python3 if user expects system python
        echo "Attempting to run with system python..."
    fi
else
    echo "Activating virtual environment: \$VENV_PATH"
    source "\$VENV_PATH/bin/activate"
fi

# Run Script
echo "Running: python3 $SCRIPT_PATH $ARGS"
python3 "$SCRIPT_PATH" $ARGS

EOF

# Submit
submit_slurm_job "$SCRIPT_FILE"
