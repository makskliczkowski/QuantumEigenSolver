#PBS -S /bin/bash
#PBS -q main
#PBS -l walltime=00:10:59
#PBS -l software=gcc
#PBS -l select=1:ncpus=1:mem=1GB


export MKL_SERIAL=yes
export CPATH="/usr/include/hdf5/serial"

echo "Start compile"

module load mkl
module load openmpi
module load intel

cd $PBS_O_WORKDIR/
g++ main.cpp rbm.cpp statistical.cpp user_interface.cpp str.cpp common.cpp lattices/square.cpp lattices/hexagonal.cpp -o vqmc.o -I./armadillo-11.2.2/include/ -ggdb3 -pthread -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lstdc++fs -llapack -fcx-fortran-rules -fomit-frame-pointer -lblas -lstdc++ -fopenmp -std=c++17 -O3 >& compil.txt

#g++ main.cpp rbm.cpp statistical.cpp user_interface.cpp str.cpp common.cpp lattices/square.cpp lattices/hexagonal.cpp -o DQMC_14x14.o -I./armadillo-10.6.2/include/ -ggdb3 -pthread -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lstdc++fs -llapack -fcx-fortran-rules -fomit-frame-pointer -lblas -lstdc++ -fopenmp -std=c++17 -O3 >& compil.txt
#icpc main.cpp hubbard_dqmc_qr.cpp hubbard.cpp user_interface.cpp general_model.cpp cpp/statistical.cpp cpp/common.cpp cpp/Lattices/square.cpp cpp/Lattices/hexagonal.cpp -o DQMC.o -I./armadillo-10.6.2/include/ -pthread -Wall -Wformat=0 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lstdc++fs -llapack -lblas -std=c++11 -std=c++17 -std=c++14 -fopenmp -std=c++2a -O3 -fp-model fast=2 >& log_compilation.txt


echo "End compile"