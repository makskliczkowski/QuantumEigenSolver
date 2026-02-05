#!/bin/bash
# run_cpp.sh - Submit C++ QES jobs

# Get the directory of this script
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
UTILS_SCRIPT="$SCRIPT_DIR/utils/slurm_utils.sh"

if [[ ! -f "$UTILS_SCRIPT" ]]; then
    echo "Error: Utilities script not found at $UTILS_SCRIPT" >&2
    exit 1
fi

source "$UTILS_SCRIPT"

# Default values
JOB_NAME="qes_cpp_job"
TIME="01:00:00"
MEM="4" # GB
CPU="1"
PARTITION=""
EXE_PATH=""
ARGS=""
OUTPUT_DIR="./logs"

# Usage info
usage() {
    show_usage_template "$0" "argument_string"
    echo "    --exe=PATH              Path to the executable (required)"
    echo "    --job-name=NAME         Job name (default: $JOB_NAME)"
    echo "    --output-dir=DIR        Directory for logs (default: $OUTPUT_DIR)"
    echo ""
    echo "Example: $0 --time=2:00:00 --exe=./build/qsolver -- \"d=1 lx=10\""
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
        --exe=*)
            EXE_PATH="${1#*=}"; shift
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
if [[ -z "$EXE_PATH" ]]; then
    echo "Error: Executable path (--exe) is required." >&2
    usage
fi

validate_file_exists "$EXE_PATH" "Executable" || exit 1

# Convert EXE_PATH to absolute path to avoid issues if running from different directory
EXE_PATH=$(realpath "$EXE_PATH")

# Generate Script
SCRIPT_FILE="${OUTPUT_DIR}/submit_${JOB_NAME}_$(date +%s).sh"
create_directories "$OUTPUT_DIR"

# Write header
create_slurm_header "$CPU" "$MEM" "$TIME" "$OUTPUT_DIR" "$JOB_NAME" "$PARTITION" > "$SCRIPT_FILE"

# Append body
cat << EOF >> "$SCRIPT_FILE"

# Load modules
$(add_module_section "intel" "hdf5")

# Run Executable
echo "Running: $EXE_PATH $ARGS"
"$EXE_PATH" $ARGS

EOF

# Submit
submit_slurm_job "$SCRIPT_FILE"
