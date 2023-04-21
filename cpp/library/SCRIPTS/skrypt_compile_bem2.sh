#!/bin/bash
#SBATCH -N1
#SBATCH -c24
#SBATCH --mem=192gb
#SBATCH --time=299:59:59

source /usr/local/sbin/modules.sh

module load intel
module load OpenMPI
module load HDF5


CPPSOURCE=./source/cpp

cd /home/klimak97/Codes/QuantumEigenSolver/cpp/library/

icpx ./main.cpp ./operators.cpp ./user_interface.cpp ${CPPSOURCE}/common.cpp ${CPPSOURCE}/directories.cpp ${CPPSOURCE}/exceptions.cpp ${CPPSOURCE}/str.cpp ${CPPSOURCE}/ui.cpp ${CPPSOURCE}/Lattices/hexagonal.cpp ${CPPSOURCE}/Lattices/square.cpp -o ./vqmc.o -I./armadillo-11.4.0/include/ -pthread -lhdf5 -Wall -Wformat=0 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -lmkl_sequential -lstdc++fs -std=c++11 -std=c++17 -std=c++14 -fopenmp -std=c++2a -O3 >& compilation.txt

echo "end compile"