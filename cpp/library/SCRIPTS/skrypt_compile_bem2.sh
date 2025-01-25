#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=8gb
#SBATCH --time=1:00:00

# Load necessary modules
source /usr/local/sbin/modules.sh
module load intel/2022b
module load HDF5

# Define directories and variables
CPPSOURCE=./source/cpp
ARMADIR=/home/klimak97/LIBRARIES_CPP/armadillo-14.0.2
BUILD_DIR=/home/klimak97/CODES/QuantumEigenSolver/cpp/library/

# Navigate to the build directory
cd ${BUILD_DIR} || { echo "Failed to change directory to ${BUILD_DIR}"; exit 1; }

# Compile with Intel oneAPI icpx
icpx ./main.cpp                                                 \
        ./operators.cpp                                         \
        ${CPPSOURCE}/common.cpp                                 \
        ${CPPSOURCE}/directories.cpp                            \
        ${CPPSOURCE}/exceptions.cpp                             \
        ${CPPSOURCE}/str.cpp                                    \
        ${CPPSOURCE}/ui.cpp                                     \
        ${CPPSOURCE}/time.cpp                                   \
        ${CPPSOURCE}/maths.cpp                                  \
        ${CPPSOURCE}/Random/montecarlo.cpp                      \
        ${CPPSOURCE}/Lattices/lattices.cpp                      \
        ${CPPSOURCE}/Lattices/hexagonal.cpp                     \
        ${CPPSOURCE}/Lattices/honeycomb.cpp                     \
        ${CPPSOURCE}/Lattices/square.cpp                        \
        ${CPPSOURCE}/LinearAlgebra/solvers.cpp                  \
        ${CPPSOURCE}/LinearAlgebra/Solvers/solvers_arnoldi.cpp  \
        ${CPPSOURCE}/LinearAlgebra/Solvers/solvers_cg.cpp       \
        ${CPPSOURCE}/LinearAlgebra/Solvers/solvers_arma.cpp     \
        ${CPPSOURCE}/LinearAlgebra/Solvers/solvers_direct.cpp   \
        ${CPPSOURCE}/LinearAlgebra/Solvers/solvers_pseudo.cpp   \
        ${CPPSOURCE}/LinearAlgebra/Solvers/solvers_minres.cpp   \
        ${CPPSOURCE}/LinearAlgebra/Solvers/solvers_minresqlp.cpp\
        ${CPPSOURCE}/LinearAlgebra/preconditioners.cpp          \
        ${CPPSOURCE}/LinearAlgebra/pfaffian.cpp                 \
        ${CPPSOURCE}/LinearAlgebra/ode.cpp                      \
        ./user_interface.cpp                                    \
        ./src/nqs/nqs_def.cpp                                   \
        ./src/nqs/nqs.cpp                                       \
        ./src/nqs/nqs_quench.cpp                                \
        ./src/nqs/nqs_opt.cpp                                   \
        ./src/nqs/nqs_lower.cpp                                 \
        ./src/nqs/nqs_sampling.cpp                              \
        ./src/nqs/nqs_evo.cpp                                   \
        ./src/nqs/rbm/rbm.cpp                                   \
        ./src/nqs/rbm/rbm_spin.cpp                              \
        ./src/nqs/rbmpp/rbmpp.cpp                               \
        ./src/nqs/operators/nqs_operator.cpp                    \
        ./src/operators/operator_parser.cpp                     \
        ./src/user_interface/ui_check_eth.cpp                   \
        ./src/user_interface/ui_check_nqs.cpp                   \
        ./src/user_interface/ui_check_quadratic.cpp             \
        ./src/user_interface/ui_check_symmetries.cpp            \
        -o qsolver                                              \
        -I${ARMADIR}/include/                                   \
        -pthread -lhdf5                                         \
        -Wformat=0 -lmkl_intel_lp64 -lmkl_core -lmkl_core       \
        -liomp5 -lpthread -lm -ldl -lmkl_sequential             \
        -lstdc++fs -std=c++2a -O3 -fopenmp                      \
        > compilation.txt 2>&1

# Notify completion
if [[ $? -eq 0 ]]; then
    echo "Compilation successful. Executable: qsolver"
else
    echo "Compilation failed. Check compilation.txt for details."
    exit 1
fi