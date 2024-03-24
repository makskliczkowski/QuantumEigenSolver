cd ../
set CPPSOURCE=./source/cpp
set ARMADIR=%ARMADILLO_INCL_DIR%

"%CXX%" ./main.cpp ./operators.cpp ./user_interface.cpp "%CPPSOURCE%"/common.cpp "%CPPSOURCE%"/directories.cpp "%CPPSOURCE%"/exceptions.cpp ^
        "%CPPSOURCE%"/str.cpp "%CPPSOURCE%"/ui.cpp "%CPPSOURCE%"/Lattices/hexagonal.cpp "%CPPSOURCE%"/Lattices/square.cpp -o ./qsolver.exe ^
        -I"%ARMADILLO_INCL_DIR%"/ -I"%MKL_INCL_DIR%" -L"%MKL_LIB_DIR%" ^
        -L"%HDF5_LIB_DIR%" -I"%HDF5_INCL_DIR%" ^
        -I"%ONE_API_COMPILER_INCL%"/ -L"%ONE_API_COMPILER_LIB%" ^
        -DH5_BUILT_AS_DYNAMIC_LIB ^ 
        -llibhdf5_tools -llibhdf5_hl -llibhdf5_hl_cpp ^
        -lmkl_intel_lp64 -lmkl_core ^
        -lmkl_sequential -fopenmp -std=c++2a -O3 > compilation_windows.log 2>&1