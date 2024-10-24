#!/bin/bash
#SBATCH -N1
#SBATCH -c1
#SBATCH --mem=8gb
#SBATCH --time=1:00:00

source /usr/local/sbin/modules.sh

module load intel/2022b
module load HDF5


CPPSOURCE=./source/cpp
ARMADIR=/home/klimak97/LIBRARIES_CPP/armadillo-12.2.0

cd /home/klimak97/CODES/QuantumEigenSolver/cpp/library/
ls

icpx ./main.cpp ./operators.cpp ./user_interface.cpp ${CPPSOURCE}/common.cpp ${CPPSOURCE}/directories.cpp ${CPPSOURCE}/exceptions.cpp ${CPPSOURCE}/str.cpp ${CPPSOURCE}/ui.cpp ${CPPSOURCE}/Lattices/hexagonal.cpp ${CPPSOURCE}/Lattices/square.cpp -o ./qsolver.o -I${ARMADIR}/include/ -pthread -lhdf5 -Wall -Wformat=0 -lmkl_intel_lp64 -lmkl_core -lmkl_core -liomp5 -lpthread -lm -ldl -lmkl_sequential -lstdc++fs -std=c++17 -fopenmp -std=c++2a -O3 >& compilation.txt

echo "end compile"
