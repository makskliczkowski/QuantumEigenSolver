#!/bin/bash
#SBATCH -N1
#SBATCH -c1
#SBATCH --mem=8gb
#SBATCH --time=1:00:00

# source /usr/local/sbin/modules.sh

module load intel
module load HDF5


cd ~/CODES/QuantumEigenSolver/cpp/library
ls

CPPSOURCE=./source/cpp
ARMADIR=/home/kliczkowski/LIBRARY/armadillo-12.8.0


icpx ./main.cpp ./operators.cpp ./user_interface.cpp            \
        ./src/user_interface/ui_check_eth.cpp                   \
        ./src/user_interface/ui_check_eth_time_evo.cpp          \
        ./src/user_interface/ui_check_nqs.cpp                   \
        ./src/user_interface/ui_check_quadratic.cpp             \
        ./src/user_interface/ui_check_symmetries.cpp            \
        ./src/nsq/nqs.cpp                                       \
        ./src/nqs/nqs_opt.cpp                                   \
        ./src/nqs/nqs_lower.cpp                                 \
        ./src/nqs/nqs_sampling.cpp                              \
        ./src/nqs/rbm/rbm.cpp                                   \
        ./src/nqs/rbm/rbm_spin.cpp                             \
        ./src/nqs/operators/nqs_operator.cpp                    \
        ./src/operator_parser.cpp                               \
        ${CPPSOURCE}/common.cpp ${CPPSOURCE}/directories.cpp    \
        ${CPPSOURCE}/exceptions.cpp ${CPPSOURCE}/str.cpp        \
        ${CPPSOURCE}/ui.cpp                                     \
        ${CPPSOURCE}/time.cpp                                   \
        ${CPPSOURCE}/Lattices/hexagonal.cpp                     \
        ${CPPSOURCE}/LinearAlgebra/solvers.cpp                  \
        ${CPPSOURCE}/LinearAlgebra/Solvers/solvers_cg.cpp       \
        ${CPPSOURCE}/LinearAlgebra/Solvers/solvers_minres.cpp   \
        ${CPPSOURCE}/LinearAlgebra/Solvers/solvers_minresqlp.cpp\
        ${CPPSOURCE}/LinearAlgebra/preconditioners.cpp          \
        ${CPPSOURCE}/LinearAlgebra/pfaffian.cpp                 \
        ${CPPSOURCE}/Lattices/square.cpp -o ./qsolver.o         \
        -I${ARMADIR}/include/ -pthread -lhdf5 -Wall             \
        -Wformat=0 -lmkl_intel_lp64 -lmkl_core -lmkl_core       \
        -liomp5 -lpthread -lm -ldl -lmkl_sequential             \
        -lstdc++fs -std=c++17                                   \
        -fopenmp -std=c++2a -O3 >& compilation.txt

echo "end compile"
