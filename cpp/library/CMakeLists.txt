cmake_minimum_required(VERSION 3.5)
project(QuantumEigenSolver)

# set the compilers

set(CMAKE_C_COMPILER "$ENV{CC}")
set(CMAKE_CXX_COMPILER "$ENV{CXX}")
set(COMPILER_LIBS_WIN "$ENV{ONE_API_COMPILER_LIB}")

message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler libraries: ${COMPILER_LIBS_WIN}")

# set variables necessary for the project
set(CPPSOURCE "${CMAKE_SOURCE_DIR}/source/cpp")

# Find the path of the MKL include directory using an environment variable
find_path(MKL_INCL_DIR
    NAMES "mkl.h"
    PATHS "$ENV{MKL_INCL_DIR}"
    DOC "Path to MKL include directory"
)

# Find the path of the MKL library directory using an environment variable
find_path(MKL_LIB_DIR
    NAMES "mkl_core.lib"
    PATHS "$ENV{MKL_LIB_DIR}"
    DOC "Path to MKL library directory"
)

find_path(HDF5_LIB_DIR
    NAMES "hdf5.lib"
    PATHS "$ENV{HDF5_LIB_DIR}"
    DOC "Path to HDF5 library directory"
)

find_path(ARMADILLO_INCL_DIR
    NAMES "armadillo"
    PATHS "$ENV{ARMADILLO_INCL_DIR}"
    DOC "Path to Armadillo include directory"
)


# Print the paths found
message(STATUS "MKL include directory: ${MKL_INCL_DIR}")
message(STATUS "MKL library directory: ${MKL_LIB_DIR}")
message(STATUS "Armadillo include directory: ${ARMADILLO_INCL_DIR}")
message(STATUS "HDF5 library directory: ${HDF5_LIB_DIR}")

# Set the source files
set(SOURCES 
    main.cpp 
    operators.cpp 
    user_interface.cpp 
    ${CPPSOURCE}/common.cpp 
    ${CPPSOURCE}/directories.cpp 
    ${CPPSOURCE}/exceptions.cpp 
    ${CPPSOURCE}/str.cpp 
    ${CPPSOURCE}/ui.cpp 
    ${CPPSOURCE}/Lattices/hexagonal.cpp 
    ${CPPSOURCE}/Lattices/square.cpp
)

# Set the include directories
set(INCLUDE_DIRS 
    ${ARMADILLO_INCL_DIR}
    ${CPPSOURCE}
    ${MKL_INCL_DIR}
    $ENV{HDF5_INCL_DIR}
    $ENV{ONE_API_COMPILER_INCL}
)

######################### Set the libraries #########################

# Set the libraries directories
set(LIB_DIRS ${MKL_LIB_DIR})
set(LIB_DIRS ${HDF5_LIB_DIR})

# Set the libraries to link against
set(LIBRARIES 
    hdf5
)

# Check the pthread
find_library(PTHREAD_LIBRARY pthread)
if(PTHREAD_LIBRARY)
    list(APPEND LIBRARIES ${PTHREAD_LIBRARY})
endif()

# Chech the dl
find_library(DL_LIBRARY dl)
if(DL_LIBRARY)
    list(APPEND LIBRARIES ${DL_LIBRARY})
    message(STATUS "dl found")
endif()

# Check the stdc++fs
find_library(STDCPPFS_LIBRARY stdc++fs)
if(STDCPPFS_LIBRARY)
    list(APPEND LIBRARIES ${STDCPPFS_LIBRARY})
    message(STATUS "stdc++fs found")
endif()

# Check for Intel MKL libraries
# find_library(MKL_INTEL_LP64_LIBRARY mkl_intel_lp64)
find_library(MKL_CORE_LIBRARY mkl_core HINTS ${MKL_LIB_DIR})
find_library(MKL_SEQUENTIAL_LIBRARY mkl_sequential HINTS ${MKL_LIB_DIR})
find_library(MKL_RT_LIBRARY mkl_rt HINTS ${MKL_LIB_DIR})
find_library(MKL_INTEL_ILP64_LIBRARY mkl_intel_ilp64 HINTS ${MKL_LIB_DIR})
if(MKL_CORE_LIBRARY)
    list(APPEND LIBRARIES ${MKL_CORE_LIBRARY})
    message(STATUS "Intel MKL libraries found: ${MKL_CORE_LIBRARY}")
endif()
if (MKL_SEQUENTIAL_LIBRARY)
    list(APPEND LIBRARIES ${MKL_SEQUENTIAL_LIBRARY})
    message(STATUS "Intel MKL libraries found: ${MKL_SEQUENTIAL_LIBRARY}")
endif()
if (MKL_RT_LIBRARY)
    list(APPEND LIBRARIES ${MKL_RT_LIBRARY})
    message(STATUS "Intel MKL libraries found: ${MKL_RT_LIBRARY}")
endif()
if (MKL_INTEL_ILP64_LIBRARY)
    list(APPEND LIBRARIES ${MKL_INTEL_ILP64_LIBRARY})
    message(STATUS "Intel MKL libraries found: ${MKL_INTEL_ILP64_LIBRARY}")
endif()

######################### HDF5 #########################


# Check for libhdf5 and libhdf5_cpp (if necessary)
find_library(LIBHDF5_HL_LIBRARY libhdf5_hl HINTS ${HDF5_LIB_DIR})
find_library(LIBHDF5_CPP_HL_LIBRARY libhdf5_hl_cpp HINTS ${HDF5_LIB_DIR})
find_library(LIBHDF5_HL_LIBRARY libhdf5_tools HINTS ${HDF5_LIB_DIR})

if(LIBHDF5_HL_LIBRARY AND LIBHDF5_CPP_HL_LIBRARY)
    list(APPEND LIBRARIES ${LIBHDF5_HL_LIBRARY} ${LIBHDF5_CPP_HL_LIBRARY})
    message(STATUS "additional HDF5 libraries found ${LIBHDF5_HL_LIBRARY} ${LIBHDF5_CPP_HL_LIBRARY}")
endif()

if(LIBHDF5_HL_LIBRARY)
    list(APPEND LIBRARIES ${LIBHDF5_HL_LIBRARY})
    message(STATUS "additional HDF5 libraries found ${LIBHDF5_HL_LIBRARY}")
endif()

######################### OMP #########################    

# omp
find_package(OpenMP REQUIRED)
list(APPEND LIBRARIES OpenMP::OpenMP_CXX)

######################### Set the executable #########################

# Add the executable
add_executable(qsolver ${SOURCES})

# Set include directories for the target
target_include_directories(qsolver PRIVATE ${INCLUDE_DIRS})

# Set libraries directories for the target
target_link_directories(qsolver PRIVATE ${LIB_DIRS})

# Link the libraries to the target
target_link_libraries(qsolver PRIVATE ${LIBRARIES})

# Compiler flags
set(CMAKE_CXX_STANDARD 20)
set_target_properties(qsolver   PROPERTIES
                                CXX_STANDARD 20
                                CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W0 -fopenmp -O3 -DH5_BUILT_AS_DYNAMIC_LIB -Qiopenmp")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a -pthread -lmkl_intel_lp64 -lmkl_core -liomp5 -lpthread -lm -ldl -lmkl_sequential -lstdc++fs -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /FS /bigobj /DMKL_ILP64 /Qiopenmp")

# Redirect output to compilation.txt
set(OUTPUT_FILE "compilation.txt")
file(WRITE ${OUTPUT_FILE} "Compilation output will be redirected to ${OUTPUT_FILE}")

# Custom target to redirect output
add_custom_target(redirect_output ALL DEPENDS ${OUTPUT_FILE})