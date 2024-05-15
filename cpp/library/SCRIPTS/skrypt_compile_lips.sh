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


icpx ./main.cpp ./operators.cpp ./user_interface.cpp \
        ./src/user_interface/ui_check_eth.cpp \
        ${CPPSOURCE}/common.cpp ${CPPSOURCE}/directories.cpp \
        ${CPPSOURCE}/exceptions.cpp ${CPPSOURCE}/str.cpp \
        ${CPPSOURCE}/ui.cpp ${CPPSOURCE}/Lattices/hexagonal.cpp \
        ${CPPSOURCE}/Lattices/square.cpp -o ./qsolver.o \
        -I${ARMADIR}/include/  \
        -pthread -lhdf5 -Wall \
        -Wformat=0 -lmkl_intel_lp64 -lmkl_core -lmkl_core \
        -liomp5 -lpthread -lm -ldl -lmkl_sequential \
        -lstdc++fs -std=c++17 \
        -fopenmp -std=c++2a -O3 >& compilation.txt

echo "end compile"
