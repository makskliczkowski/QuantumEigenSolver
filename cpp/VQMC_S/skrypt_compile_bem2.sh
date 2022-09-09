#!/bin/bash
#SBATCH -N1
#SBATCH -c1
#SBATCH --mem=1gb
#SBATCH --time=1:00:00
#SBATCH --job-name=vqmc-c


source /usr/local/sbin/modules.sh
#source /etc/profile.d/modules.sh
#source /etc/profile.d/modules-local.sh

module load intel/2021a
module unload GCCcore
module load OpenMPI/4.1.1-GCC-10.3.0
#module load MKL
#module load openmpi

dir="/home/klimak97/VQMC/HamiltonianSolver_ed_vqmc/cpp/VQMC_S/"
echo $dir
cd /home/klimak97/VQMC/HamiltonianSolver_ed_vqmc/cpp/VQMC_S/

#g++ main.cpp rbm.cpp statistical.cpp user_interface.cpp str.cpp common.cpp lattices/square.cpp lattices/hexagonal.cpp -o vqmc.o -I./armadillo-11.2.2/include/ -ggdb3 -pthread -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lstdc++fs -llapack -fcx-fortran-rules -fomit-frame-pointer -lblas -lstdc++ -fopenmp -std=c++17 -O3 >& compil.txt
#dpcpp
#icpx  
icpx ./main.cpp ./rbm.cpp ./statistical.cpp ./user_interface.cpp ./str.cpp ./common.cpp ./lattices/square.cpp ./lattices/hexagonal.cpp -o ./vqmc.o -I./armadillo-11.2.2/include/ -pthread -Wall -Wformat=0 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lstdc++fs -std=c++11 -std=c++17 -std=c++14 -fopenmp -std=c++2a -O3 >& compil.txt
echo "End compile"