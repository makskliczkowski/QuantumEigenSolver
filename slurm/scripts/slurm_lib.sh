#!/bin/bash
# =============================================================================
# slurm_lib.sh — Reusable SLURM library for QES project submissions
# =============================================================================
#
# Provides:
#   - All functions from slurm_utils.sh (create_slurm_header, add_module_section, etc.)
#   - INI file parsing:
#       parse_ini_file()    — read [section]/key=value into INI_DATA[]
#       ini_get()           — lookup with fallback default
#       parse_range()       — expand "start:end:step" to values
#       parse_csv()         — split comma-separated values
#       parse_ini_sizes()   — parse semicolon-separated "lx,ly" pairs
#       split_size()        — split a single "lx,ly" token into variables
#   - setup_qes_environment()  — concurrent-safe venv setup with flock
#   - submit_job_array()       — SLURM array-job helper
#
# Usage:
#   source "${QES_SLURMPATH}/scripts/slurm_lib.sh"
# =============================================================================

# Resolve the directory of THIS script so we can find slurm_utils.sh
_SLURM_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SLURM_UTILS="${_SLURM_LIB_DIR}/../utils/slurm_utils.sh"

if [[ -f "$_SLURM_UTILS" ]]; then
    source "$_SLURM_UTILS"
else
    echo "Error: slurm_utils.sh not found at $_SLURM_UTILS" >&2
    exit 1
fi

# =============================================================================
# INI File Parsing — Generic .ini reader for SLURM submission scripts
# =============================================================================
#
# Provides:
#   parse_ini_file FILE             — reads [section] / key=value pairs into INI_DATA[]
#   ini_get SECTION KEY [DEFAULT]   — lookup with fallback
#   parse_range "start:end:step"    — expand range to one-value-per-line
#   parse_csv "a,b,c"               — split CSV to one-value-per-line
#
# IMPORTANT: The caller must declare the associative array BEFORE calling
#   parse_ini_file:
#     declare -A INI_DATA
#     parse_ini_file "my_config.ini"
#
# After parsing, SECTIONS[] holds section names in order, and
# INI_DATA["section::key"] holds values.
# =============================================================================

# -----------------------------------------------------------------------------
# parse_ini_file — Read a standard .ini file into INI_DATA[section::key]=val
# -----------------------------------------------------------------------------
#   - Lines starting with # or ; are comments (also inline # comments)
#   - Sections are [name]
#   - Key/value separated by first '='
#   - Leading/trailing whitespace is stripped from keys and values
#   - Populates global arrays: SECTIONS=() and INI_DATA[section::key]=value
# -----------------------------------------------------------------------------
parse_ini_file() {
    local ini_file="$1"
    local current_section=""

    # Reset the section list (caller owns INI_DATA)
    SECTIONS=()

    while IFS= read -r line || [[ -n "$line" ]]; do
        # Strip comments and leading/trailing whitespace
        line="${line%%#*}"
        line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        [[ -z "$line" ]] && continue

        # Section header
        if [[ "$line" =~ ^\[([^]]+)\]$ ]]; then
            current_section="${BASH_REMATCH[1]}"
            SECTIONS+=("$current_section")
            continue
        fi

        # Key = value (split only on first '=')
        if [[ "$line" =~ ^([^=]+)=(.*)$ && -n "$current_section" ]]; then
            local key val
            key="$(echo "${BASH_REMATCH[1]}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
            val="$(echo "${BASH_REMATCH[2]}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
            INI_DATA["${current_section}::${key}"]="$val"
        fi
    done < "$ini_file"
}

# -----------------------------------------------------------------------------
# ini_get SECTION KEY [DEFAULT] — Retrieve a value with optional fallback
# -----------------------------------------------------------------------------
ini_get() {
    local section="$1"
    local key="$2"
    local default="${3:-}"
    local lookup="${section}::${key}"
    if [[ -v "INI_DATA[$lookup]" ]]; then
        echo "${INI_DATA[$lookup]}"
    else
        echo "$default"
    fi
}

# -----------------------------------------------------------------------------
# parse_range "start:end:step" — Expand to one value per line (or echo scalar)
# -----------------------------------------------------------------------------
#   "0.0:2.0:0.5" → 0.000000\n0.500000\n1.000000\n1.500000\n2.000000
#   "1.5"          → 1.5
# -----------------------------------------------------------------------------
parse_range() {
    local range_str="$1"
    if [[ "$range_str" == *":"* ]]; then
        local start end step
        IFS=':' read -r start end step <<< "$range_str"
        seq -f "%.6f" "$start" "$step" "$end" 2>/dev/null || echo "$start"
    else
        echo "$range_str"
    fi
}

# -----------------------------------------------------------------------------
# parse_csv "a,b,c" — Split comma-separated values to one per line
# -----------------------------------------------------------------------------
parse_csv() {
    local csv_str="$1"
    echo "$csv_str" | tr ',' '\n' | sed 's/^ *//;s/ *$//'
}

# =============================================================================
# parse_ini_sizes — Parse semicolon-separated "lx,ly" size pairs
# =============================================================================
#   "2,2;3,3;4,3" → array of "2,2" "3,3" "4,3"
#   Stores result in the nameref array passed as $2.
# =============================================================================
parse_ini_sizes() {
    local sizes_str="$1"
    local -n _out_sizes="$2"
    _out_sizes=()
    IFS=';' read -ra _raw_sizes <<< "$sizes_str"
    for _s in "${_raw_sizes[@]}"; do
        _s="$(echo "$_s" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        [[ -n "$_s" ]] && _out_sizes+=("$_s")
    done
}

# =============================================================================
# split_size "lx,ly" — Read a single size token into lx/ly variables
# =============================================================================
#   Handles both "lx,ly" and "lx ly" formats.
#   Usage: local lx ly; split_size "$size" lx ly
# =============================================================================
split_size() {
    local size="$1"
    local -n _lx="$2"
    local -n _ly="$3"
    if [[ "$size" == *","* ]]; then
        IFS=',' read -r _lx _ly <<< "$size"
    else
        read -r _lx _ly <<< "$size"
    fi
}

# =============================================================================
# setup_qes_environment — Emit concurrent-safe venv setup code
# =============================================================================
#
# Generates bash code that, AT JOB RUNTIME:
#   1. Checks if the venv + stamp file are up to date → just activates
#   2. If not, acquires an exclusive flock, installs/updates, writes stamp
#   3. Cleans any corrupted ~* package dirs left by previous concurrent pip runs
#
# Arguments (positional):
#   $1  package_dir       — root of QuantumEigenSolver checkout
#   $2  requirements_file — path to requirements.txt
#   $3  venv_name         — logical name (for logging only)
#   $4  venv_path         — absolute path to the virtual-env directory
#
# Example call (at submission time, echoed into SLURM script):
#   setup_qes_environment "${PACKAGE_DIR}" "reqs.txt" "qes_venv" "/path/to/venv"
# =============================================================================
setup_qes_environment() {
    local package_dir="$1"
    local requirements_file="$2"
    local venv_name="${3:-qes_venv}"
    local venv_path="$4"

    if [[ -z "$venv_path" ]]; then
        venv_path="${package_dir}/${venv_name}"
    fi

    # Emit runtime bash code into the SLURM script
    # Variables prefixed with $ (no backslash) are expanded NOW (submission time).
    # Variables prefixed with \$ are expanded at JOB RUNTIME.
    cat <<EOF

# ===========================================================================
# QES Python Environment Setup  (concurrent-safe with flock)
# ===========================================================================
_QES_VENV="${venv_path}"
_QES_REQ="${requirements_file}"
_QES_LOCK="\${_QES_VENV}.lock"
_QES_STAMP="\${_QES_VENV}/.install_stamp"
_QES_PKG="${package_dir}"

# ---------- helper: does the env need (re-)installation? ----------
_qes_needs_install() {
    [[ ! -d "\$_QES_VENV" ]]                         && return 0
    [[ ! -f "\$_QES_STAMP" ]]                        && return 0
    [[ -f "\$_QES_REQ" && "\$_QES_REQ" -nt "\$_QES_STAMP" ]] && return 0
    return 1
}

# ---------- helper: remove corrupted ~* package dirs ----------
_qes_clean_corrupted() {
    local sp_dirs=(\${_QES_VENV}/lib/python*/site-packages)
    for sp in "\${sp_dirs[@]}"; do
        [[ -d "\$sp" ]] || continue
        local bad
        bad=\$(find "\$sp" -maxdepth 1 -name '~*' -type d 2>/dev/null || true)
        if [[ -n "\$bad" ]]; then
            echo "[venv] Cleaning corrupted packages in \$sp ..."
            echo "\$bad" | while IFS= read -r d; do
                echo "  rm -rf \$d"
                rm -rf "\$d"
            done
        fi
    done
}

# ---------- helper: perform the actual install ----------
_qes_do_install() {
    echo "[venv] Installing / updating QES environment ..."

    # create venv if missing
    if [[ ! -d "\$_QES_VENV" ]]; then
        echo "[venv] Creating virtual environment: \$_QES_VENV"
        python3 -m venv "\$_QES_VENV" || { echo "Error: venv creation failed" >&2; return 1; }
    fi

    source "\${_QES_VENV}/bin/activate"

    # clean leftover corrupted dirs before installing
    _qes_clean_corrupted

    pip install --upgrade pip setuptools wheel -q 2>&1 | tail -1
    pip install -r "\$_QES_REQ" -q 2>&1 | tail -5

    # stamp so other jobs skip install
    date '+%Y-%m-%d %H:%M:%S' > "\$_QES_STAMP"
    echo "[venv] Installation complete."
}

# ---------- main logic ----------
if _qes_needs_install; then
    echo "[venv] Environment needs install/update — acquiring lock ..."
    mkdir -p "\$(dirname "\$_QES_LOCK")"
    exec 9>"\$_QES_LOCK"

    # wait up to 15 min for lock (another job may be installing)
    if flock -w 900 9; then
        # double-check under lock (another job may have finished)
        if _qes_needs_install; then
            _qes_do_install || { flock -u 9; exec 9>&-; exit 1; }
        else
            echo "[venv] Another job completed installation while we waited."
        fi
        flock -u 9
    else
        echo "[venv] WARNING: Could not acquire lock after 900 s — using existing env" >&2
    fi
    exec 9>&-
else
    echo "[venv] QES environment is up to date (stamp found)."
fi

# Always activate (may have skipped install branch)
source "\${_QES_VENV}/bin/activate" || {
    echo "Error: Failed to activate QES venv at \$_QES_VENV" >&2
    exit 1
}

# Final paranoia: clean corrupted dirs (harmless if none)
_qes_clean_corrupted

echo "[venv] Python: \$(python3 --version), venv: \$_QES_VENV"
# ===========================================================================
EOF
}

# =============================================================================
# submit_job_array — Submit a SLURM array job driven by a parameter file
# =============================================================================
#
# Named arguments:
#   --job-name    NAME         SLURM job name
#   --script      PATH         Python script to run
#   --param-file  PATH         File with one set of args per line
#   --venv        PATH         Virtual-env path to activate
#   --time        HH:MM:SS     Wall-time limit
#   --mem         GB           Memory in GB
#   --cpu         N            CPU cores
#   --partition   NAME         SLURM partition
#   --concurrent  N            Max simultaneous array tasks (default 100)
#   --log-dir     PATH         Directory for stdout/stderr logs
#   --env         CMD_STRING   Extra env setup (module loads, exports, …)
#   --gres        SPEC         GPU resource spec (optional)
# =============================================================================
submit_job_array() {
    local job_name="" script="" param_file="" venv_path="" 
    local time_limit="04:00:00" mem="8" cpu="1" partition="" concurrent="100"
    local log_dir="./logs" env_setup="" gres=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --job-name)   job_name="$2";   shift 2 ;;
            --script)     script="$2";     shift 2 ;;
            --param-file) param_file="$2"; shift 2 ;;
            --venv)       venv_path="$2";  shift 2 ;;
            --time)       time_limit="$2"; shift 2 ;;
            --mem)        mem="$2";        shift 2 ;;
            --cpu)        cpu="$2";        shift 2 ;;
            --partition)  partition="$2";  shift 2 ;;
            --concurrent) concurrent="$2"; shift 2 ;;
            --log-dir)    log_dir="$2";    shift 2 ;;
            --env)        env_setup="$2";  shift 2 ;;
            --gres)       gres="$2";       shift 2 ;;
            *) echo "Warning: unknown arg to submit_job_array: $1" >&2; shift ;;
        esac
    done

    # Validate
    if [[ -z "$job_name" || -z "$script" || -z "$param_file" ]]; then
        echo "Error: submit_job_array requires --job-name, --script, --param-file" >&2
        return 1
    fi
    if [[ ! -f "$param_file" ]]; then
        echo "Error: Parameter file not found: $param_file" >&2
        return 1
    fi

    local n_tasks
    n_tasks=$(wc -l < "$param_file" | tr -d ' ')
    if [[ "$n_tasks" -eq 0 ]]; then
        echo "Error: Parameter file is empty: $param_file" >&2
        return 1
    fi

    mkdir -p "$log_dir"

    local array_script="${log_dir}/${job_name}_array.sh"

    cat > "$array_script" <<ARRAY_EOF
#!/bin/bash
#SBATCH -N1
#SBATCH -c${cpu}
#SBATCH --mem=${mem}gb
#SBATCH --time=${time_limit}
#SBATCH --array=1-${n_tasks}%${concurrent}
#SBATCH -o ${log_dir}/out-%A_%a-${job_name}.out
#SBATCH -e ${log_dir}/err-%A_%a-${job_name}.err
#SBATCH --job-name=${job_name}
${partition:+#SBATCH -p ${partition}}
${gres:+#SBATCH --gres=${gres}}

set -euo pipefail

echo "============================================================"
echo " Array Job: ${job_name}"
echo " Task ID: \${SLURM_ARRAY_TASK_ID} / ${n_tasks}"
echo " Job ID:  \${SLURM_JOB_ID}"
echo " Start:   \$(date)"
echo "============================================================"

# Environment setup
${env_setup}

# Activate venv
if [[ -d "${venv_path}" ]]; then
    source "${venv_path}/bin/activate"
else
    echo "Warning: venv not found at ${venv_path}, using system python" >&2
fi

# Read this task's parameters (line number = array task ID)
PARAMS=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "${param_file}")
if [[ -z "\$PARAMS" ]]; then
    echo "Error: No parameters at line \${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

echo "Running: python3 ${script} \$PARAMS"
python3 ${script} \$PARAMS

echo "Task \${SLURM_ARRAY_TASK_ID} completed at \$(date)"
ARRAY_EOF

    echo "================================================="
    echo "Submitting SLURM array job: ${job_name}"
    echo "  Tasks:      ${n_tasks}"
    echo "  Concurrent: ${concurrent}"
    echo "  Script:     ${array_script}"
    echo "================================================="

    submit_slurm_job "$array_script" false
}

# =============================================================================
# End of slurm_lib.sh
# =============================================================================
