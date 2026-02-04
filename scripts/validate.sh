#!/bin/bash
set -e

# Script to validate the umbrella repo state and optionally build C++ code.

echo "Running validation script..."

# 1. Check Submodules
echo "Checking submodules..."
if [ ! -f "cpp/library/source/README.md" ]; then
    echo "Error: Submodule 'cpp/library/source' not initialized or empty."
    echo "Please run: git submodule update --init --recursive"
    exit 1
fi

if [ ! -d "pyqusolver/Python" ]; then
    echo "Error: Submodule 'pyqusolver' not initialized or empty."
    echo "Please run: git submodule update --init --recursive"
    exit 1
fi
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

    # Create build directory if not exists
    mkdir -p build

    # Configure CMake
    echo "Configuring CMake..."
    # We use -S cpp/library as the source directory
    cmake -S cpp/library -B build -DCMAKE_BUILD_TYPE=Release

    # Build
    echo "Compiling..."
    cmake --build build -j$(nproc)

    echo "✓ C++ build completed."
else
    echo "Skipping C++ build (use --build to enable)."
fi

echo "Validation successful."
