#!/bin/bash
# set OMP_NUM_THREADS=16
echo "Start compile"

source /opt/intel/oneapi/setvars.sh

#g++ main.cpp rbm.cpp common.cpp statistical.cpp str.cpp user_interface.cpp lattices/hexagonal.cpp lattices/square.cpp -o rbm.o -pthread -Wall -Wformat=0 -I../../../../LIBRARIES_CPP/armadillo-11.0.1/include/ -fopenmp -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lstdc++fs -llapack -fcx-fortran-rules -fomit-frame-pointer -lblas -std=c++11 -O3 >& log_compilation.txt
dpcpp main.cpp rbm.cpp common.cpp statistical.cpp str.cpp user_interface.cpp lattices/hexagonal.cpp lattices/square.cpp -o rbm.o -pthread -Wall -Wformat=0 -I../../../../LIBRARIES_CPP/armadillo-11.0.1/include/ -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lstdc++fs -llapack -lblas -std=c++11 -std=c++17 -std=c++14 -fopenmp -std=c++2a -O3 -fp-model fast=2 >& log_compilation.txt


echo "End compile"