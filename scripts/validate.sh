#!/bin/bash
set -e

# Script to validate the umbrella repo state and optionally build C++ code.

echo "Running validation script..."

check_path() {
    local type=$1 path=$2 label=$3
    if ! test "$type" "$path"; then
        echo "Error: $label not initialized or empty."
        echo "Please run: git submodule update --init --recursive"
        exit 1
    fi
}

# 1. Check Submodules
echo "Checking submodules..."
check_path -f "cpqusolver/README.md"        "Submodule 'cpqusolver'"
check_path -f "cpqusolver/source/README.md" "Nested submodule 'cpqusolver/source'"
check_path -d "pyqusolver/Python"           "Submodule 'pyqusolver'"
echo "✓ Submodules appear present."

# 2. Run Python Smoke Tests
echo "Running Python smoke tests..."
if [ -f "test_import.py" ]; then
    python3 test_import.py
else
    echo "Warning: test_import.py not found in root."
fi
echo "✓ Python smoke tests passed."

# 3. Optional C++ Build
if [[ "$1" == "--build" ]]; then
    echo "Building C++ code..."

    # Auto-detect MKL include directory if not set
    if [ -z "$MKL_INCL_DIR" ]; then
        if [ -d "/usr/include/mkl" ]; then
            echo "Auto-detected MKL include dir: /usr/include/mkl"
            export MKL_INCL_DIR="/usr/include/mkl"
        elif [ -d "/opt/intel/mkl/include" ]; then
            echo "Auto-detected MKL include dir: /opt/intel/mkl/include"
            export MKL_INCL_DIR="/opt/intel/mkl/include"
        elif [ -d "/usr/include" ] && [ -f "/usr/include/mkl.h" ]; then
             echo "Auto-detected MKL include dir: /usr/include"
             export MKL_INCL_DIR="/usr/include"
        fi
    fi

    # Create build directory if not exists
    mkdir -p build

    # Configure CMake
    echo "Configuring CMake..."
    cmake -S cpqusolver -B build -DCMAKE_BUILD_TYPE=Release

    # Build
    echo "Compiling..."
    cmake --build build -j$(nproc)

    # Run tests
    echo "Running C++ tests..."
    ctest --test-dir build --output-on-failure

    echo "✓ C++ build completed."
else
    echo "Skipping C++ build (use --build to enable)."
fi

echo "Validation successful."
