#!/bin/bash
#SBATCH -N1
#SBATCH -c24
#SBATCH --mem=192gb
#SBATCH --time=299:59:59

source /usr/local/sbin/modules.sh

module load intel
module load OpenMPI
module load HDF5

cd /home/klimak97/VQMC_sub/HamiltonianSolver_ed_vqmc/cpp/VQMC_S/

icpx ./main.cpp ./rbm.cpp ./user_interface.cpp ./source/cpp/statistical.cpp ./source/cpp/common.cpp ./source/cpp/ui.cpp ./source/cpp/Lattices/square.cpp ./source/cpp/Lattices/hexagonal.cpp -o ./vqmc.o -I./armadillo-11.4.0/include/ -pthread -lhdf5 -Wall -Wformat=0 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -lmkl_sequential -lstdc++fs -std=c++11 -std=c++17 -std=c++14 -fopenmp -std=c++2a -O3 >& compil.txt

echo "end compile"