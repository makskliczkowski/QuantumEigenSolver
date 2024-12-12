cmake_minimum_required(VERSION 3.23.0)
# set version number
set (QuantumEigenSolver_VERSION_MAJOR 1)
project(QuantumEigenSolver VERSION ${QuantumEigenSolver_VERSION_MAJOR}.0.1)

# Detect platform
if (WIN32)
    set(OS_PLATFORM "windows")
    message(STATUS "Compiling on Windows")
elseif (UNIX AND NOT APPLE)
    set(OS_PLATFORM "linux")
    message(STATUS "Compiling on Linux")
elseif (APPLE)  # adding apple support for macOS
    set(OS_PLATFORM "macos")
    message(STATUS "Compiling on macOS")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

##########################################################################

# Set compilers
if (OS_PLATFORM STREQUAL "windows")
    set(CMAKE_C_COMPILER "$ENV{CC}")
    set(CMAKE_CXX_COMPILER "$ENV{CXX}")
    set(COMPILER_LIBS_WIN "$ENV{ONE_API_COMPILER_LIB}")
    message(STATUS "Compiler libraries: ${COMPILER_LIBS_WIN}")
elseif (OS_PLATFORM STREQUAL "linux")
    find_program(ICPX_COMPILER icpx)
    if (ICPX_COMPILER)
        set(CMAKE_C_COMPILER icx)
        set(CMAKE_CXX_COMPILER icpx)
        message(STATUS "Using Intel compilers")
    else()
        set(CMAKE_C_COMPILER "/usr/bin/gcc")
        set(CMAKE_CXX_COMPILER "/usr/bin/g++")
        message(STATUS "Using GCC compilers")
    endif()
else()
    # macOS
    find_program(GCC_COMPILER g++)
    if (GCC_COMPILER)
        set(CMAKE_C_COMPILER gcc)
        set(CMAKE_CXX_COMPILER g++)
        message(STATUS "Using GCC compilers")
    else()
        find_program(CLANG_COMPILER clang++)
        if (CLANG_COMPILER)
            set(CMAKE_C_COMPILER clang)
            set(CMAKE_CXX_COMPILER clang++)
            message(STATUS "Using Clang compilers")
        else()
            message(FATAL_ERROR "No suitable compiler found")
        endif()
    endif()
endif()

message(STATUS "\tC compiler: ${CMAKE_C_COMPILER}")
message(STATUS "\tC++ compiler: ${CMAKE_CXX_COMPILER}")

##########################################################################

# Set variables necessary for the project - this is additional library path comming from the source code (https://github.com/makskliczkowski/general_cpp)
set(CPPSOURCE "${CMAKE_SOURCE_DIR}/source/cpp")

##########################################################################

if (WIN32 OR (UNIX AND NOT APPLE) OR DEFINED ENV{USE_MKL})
    find_path(MKL_INCL_DIR
        NAMES "mkl.h"
        PATHS "$ENV{MKL_INCL_DIR}"
        DOC "Path to MKL include directory")

    # Find the path of the MKL library directory using an environment variable
    find_path(MKL_LIB_DIR
        NAMES "mkl_core.lib" "libmkl_core.so"
        PATHS "$ENV{MKL_LIB_DIR}" "/usr/lib" "/usr/local/lib" "/opt/intel/mkl/lib"
        DOC "Path to MKL library directory"
    )
endif()

# Find the path of the HDF5 include directory using an environment variable or common paths
find_path(HDF5_INCL_DIR
    NAMES "hdf5.h"
    PATHS "$ENV{HDF5_INCL_DIR}" "/usr/include/hdf5/serial" "/usr/local/include/hdf5" "/opt/homebrew/include"
    DOC "Path to HDF5 include directory"
)

# Find the path of the HDF5 library directory using an environment variable or common paths
find_path(HDF5_LIB_DIR
    NAMES "hdf5.lib" "libhdf5.so" "libhdf5.dylib" "libhdf5.a"
    PATHS "$ENV{HDF5_LIB_DIR}" "/usr/lib" "/usr/local/lib" "/usr/lib/x86_64-linux-gnu/hdf5/serial" "/opt/homebrew/lib"
    DOC "Path to HDF5 library directory"
)

# Find the path of the HDF5 library directory
find_library(HDF5_LIBRARIES NAMES hdf5 HINTS "$ENV{HDF5_LIB_DIR}" "/usr/lib" "/usr/local/lib" "/usr/lib/x86_64-linux-gnu/hdf5/serial")
find_library(HDF5_HL_LIBRARIES NAMES hdf5_hl HINTS "$ENV{HDF5_LIB_DIR}" "/usr/lib" "/usr/local/lib" "/usr/lib/x86_64-linux-gnu/hdf5/serial")
find_library(HDF5_CPP_LIBRARIES NAMES hdf5_cpp HINTS "$ENV{HDF5_LIB_DIR}" "/usr/lib" "/usr/local/lib" "/usr/lib/x86_64-linux-gnu/hdf5/serial")
find_package(HDF5 COMPONENTS C HL REQUIRED)

# Print the paths found
message(STATUS "HDF5:")
message(STATUS "\tHDF5 include directory: ${HDF5_INCL_DIR}")
message(STATUS "\tHDF5 library directory: ${HDF5_LIB_DIR}")
message(STATUS "\tHDF5 libraries: ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES}")
message(STATUS "\tHDF5 C++ library: ${HDF5_CPP_LIBRARIES}")

# for macos include BLAS and LAPACK
if (OS_PLATFORM STREQUAL "macos")
    # find_library(BLAS_LIBRARY blas REQUIRED)
    find_library(LAPACK_LIBRARY lapack REQUIRED)
    find_library(OPENBLAS_LIBRARY
                NAMES openblas
                HINTS /usr/local/lib /usr/local/opt/openblas/lib /opt/homebrew/lib /opt/homebrew/opt/openblas/lib
                REQUIRED)
    # if (BLAS_LIBRARY AND LAPACK_LIBRARY AND OPENBLAS_LIBRARY)
    # message(STATUS "BLAS and LAPACK libraries found ${BLAS_LIBRARY} ${LAPACK_LIBRARY} ${OPENBLAS_LIBRARY}")
    # list(APPEND LIBRARIES ${BLAS_LIBRARY} ${LAPACK_LIBRARY} ${OPENBLAS_LIBRARY})
    if (OPENBLAS_LIBRARY AND LAPACK_LIBRARY)
        message(STATUS "OpenBLAS library found ${OPENBLAS_LIBRARY} ${LAPACK_LIBRARY}")
    endif()
endif()

# Specify HDF5 library path directly if needed
link_directories("${HDF5_LIB_DIR}")

##########################################################################

# Find the path of the Armadillo include directory
find_path(ARMADILLO_INCL_DIR
    NAMES "armadillo"
    PATHS "$ENV{ARMADILLO_INCL_DIR}" "/usr/include" "/usr/local/include"
    DOC "Path to Armadillo include directory"
)
# Print the paths found
message(STATUS "ARMA:")
message(STATUS "\tArmadillo include directory: ${ARMADILLO_INCL_DIR}")

message(STATUS "MKL:")
if (WIN32 OR UNIX AND NOT APPLE)
    message(STATUS "\tMKL include directory: ${MKL_INCL_DIR}")
    message(STATUS "\tMKL library directory: ${MKL_LIB_DIR}")
else()
    message(STATUS "\tMKL not searched on macOS")
endif()

##########################################################################

# find the OpenMPI package and set the include directories
find_package(MPI QUIET)
# If MPI is found, include its directories and link the libraries
if(MPI_FOUND)
    message(STATUS "MPI")
    message(STATUS "\tMPI include directory: ${MPI_INCLUDE_PATH}")
    message(STATUS "\tMPI C libraries: ${MPI_C_LIBRARIES}")
    message(STATUS "\tMPI C++ libraries: ${MPI_CXX_LIBRARIES}")
    message("\tMPI definitions added: -DENABLE_MPI")
    include_directories(${MPI_INCLUDE_PATH})
    set(LIBRARIES ${LIBRARIES} ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
    add_definitions(-DENABLE_MPI)
else()
    message(STATUS "MPI not found, proceeding without MPI support")
endif()

##########################################################################

# Find all .cpp files recursively in the CPPSOURCE directory
file(GLOB_RECURSE SOURCES 
    ${CPPSOURCE}/*.cpp
)

# Set the source files
set(SOURCES 
    main.cpp 
    ${CPPSOURCE}/common.cpp 
    ${CPPSOURCE}/directories.cpp 
    ${CPPSOURCE}/exceptions.cpp 
    ${CPPSOURCE}/str.cpp 
    ${CPPSOURCE}/ui.cpp 
    ${CPPSOURCE}/time.cpp
    ${CPPSOURCE}/maths.cpp
    ${CPPSOURCE}/Random/montecarlo.cpp
    ${CPPSOURCE}/Lattices/lattices.cpp 
    ${CPPSOURCE}/Lattices/hexagonal.cpp 
    ${CPPSOURCE}/Lattices/square.cpp
    ${CPPSOURCE}/Lattices/honeycomb.cpp
    ${CPPSOURCE}/LinearAlgebra/solvers.cpp
    ${CPPSOURCE}/LinearAlgebra/Solvers/solvers_arnoldi.cpp
    ${CPPSOURCE}/LinearAlgebra/Solvers/solvers_cg.cpp
    ${CPPSOURCE}/LinearAlgebra/Solvers/solvers_arma.cpp
    ${CPPSOURCE}/LinearAlgebra/Solvers/solvers_direct.cpp
    ${CPPSOURCE}/LinearAlgebra/Solvers/solvers_pseudo.cpp
    ${CPPSOURCE}/LinearAlgebra/Solvers/solvers_minres.cpp
    ${CPPSOURCE}/LinearAlgebra/Solvers/solvers_minresqlp.cpp
    ${CPPSOURCE}/LinearAlgebra/preconditioners.cpp
    ${CPPSOURCE}/LinearAlgebra/pfaffian.cpp
    ${CPPSOURCE}/LinearAlgebra/ode.cpp
    operators.cpp 
    user_interface.cpp 
    src/nqs/nqs.cpp
    src/nqs/nqs_opt.cpp
    src/nqs/nqs_lower.cpp
    src/nqs/nqs_sampling.cpp
    src/nqs/nqs_evo.cpp
    src/nqs/rbm/rbm.cpp
    src/nqs/rbm/rbm_spin.cpp
    src/nqs/rbmpp/rbmpp.cpp
    src/nqs/operators/nqs_operator.cpp
    src/user_interface/ui_check_eth.cpp
    src/user_interface/ui_check_nqs.cpp
    src/user_interface/ui_check_quadratic.cpp
    src/user_interface/ui_check_symmetries.cpp
    src/operator_parser.cpp
)

# Set the include directories
set(INCLUDE_DIRS 
    ${ARMADILLO_INCL_DIR}
    ${CPPSOURCE}
    ${MKL_INCL_DIR}
    ${HDF5_INCL_DIR}
)

######################### Set the libraries #########################

# Set the libraries directories
set(LIB_DIRS ${MKL_LIB_DIR} ${HDF5_LIB_DIR})

# Set the libraries to link against
set(LIBRARIES hdf5)
list(APPEND LIBRARIES ${OPENBLAS_LIBRARY} ${LAPACK_LIBRARY})

# Check for Intel MKL libraries
if (WIN32 OR (UNIX AND NOT APPLE))
    message(STATUS "\tMKL library directory: ${MKL_LIB_DIR}")
    if (OS_PLATFORM STREQUAL "windows")
        find_library(MKL_CORE_LIBRARY mkl_core HINTS ${MKL_LIB_DIR})
        find_library(MKL_SEQUENTIAL_LIBRARY mkl_sequential HINTS ${MKL_LIB_DIR})
        find_library(MKL_RT_LIBRARY mkl_rt HINTS ${MKL_LIB_DIR})
        find_library(MKL_INTEL_ILP64_LIBRARY mkl_intel_ilp64 HINTS ${MKL_LIB_DIR})
    elseif (OS_PLATFORM STREQUAL "linux")
        message(STATUS "Searching for Intel MKL libraries")
        find_library(MKL_CORE_LIBRARY mkl_core HINTS ${MKL_LIB_DIR})
        find_library(MKL_SEQUENTIAL_LIBRARY libmkl_sequential.so HINTS ${MKL_LIB_DIR})
        find_library(MKL_RT_LIBRARY mkl_rt HINTS ${MKL_LIB_DIR})
        find_library(MKL_INTEL_ILP64_LIBRARY libmkl_intel_ilp64.so HINTS ${MKL_LIB_DIR})
    endif()

    # Add MKL libraries to the list if found
    if(MKL_CORE_LIBRARY)
        list(APPEND LIBRARIES ${MKL_CORE_LIBRARY})
        message(STATUS "\tIntel MKL library found: ${MKL_CORE_LIBRARY}")
    endif()
    if (MKL_SEQUENTIAL_LIBRARY)
        list(APPEND LIBRARIES ${MKL_SEQUENTIAL_LIBRARY})
        message(STATUS "\tIntel MKL library found: ${MKL_SEQUENTIAL_LIBRARY}")
    endif()
    if (MKL_RT_LIBRARY)
        list(APPEND LIBRARIES ${MKL_RT_LIBRARY})
        message(STATUS "\tIntel MKL library found: ${MKL_RT_LIBRARY}")
    endif()
    if (MKL_INTEL_ILP64_LIBRARY)
        list(APPEND LIBRARIES ${MKL_INTEL_ILP64_LIBRARY})
        message(STATUS "\tIntel MKL library found: ${MKL_INTEL_ILP64_LIBRARY}")
    endif()
    message(STATUS "\tMKL libraries: ${LIBRARIES}")
    include_directories(${MKL_INCL_DIR})
    message(STATUS "\tMKL include directory: ${MKL_INCL_DIR}")
endif()

######################### Set the libraries #########################

# Add platform-specific libraries
if (UNIX AND NOT APPLE)
    find_library(PTHREAD_LIBRARY pthread)
    if(PTHREAD_LIBRARY)
        list(APPEND LIBRARIES ${PTHREAD_LIBRARY})
    endif()

    find_library(DL_LIBRARY dl)
    if(DL_LIBRARY)
        list(APPEND LIBRARIES ${DL_LIBRARY})
        message(STATUS "dl found")
    endif()

    find_library(STDCPPFS_LIBRARY stdc++fs)
    if(STDCPPFS_LIBRARY)
        list(APPEND LIBRARIES ${STDCPPFS_LIBRARY})
        message(STATUS "stdc++fs found")
    endif()
elseif (APPLE)
    find_library(PTHREAD_LIBRARY pthread)
    if(PTHREAD_LIBRARY)
        list(APPEND LIBRARIES ${PTHREAD_LIBRARY})
    endif()

    find_library(DL_LIBRARY dl)
    if(DL_LIBRARY)
        list(APPEND LIBRARIES ${DL_LIBRARY})
        message(STATUS "dl found")
    endif()
elseif (OS_PLATFORM STREQUAL "windows")
    list(APPEND LIBRARIES ${COMPILER_LIBS_WIN})
endif()

######################### HDF5 #########################

# Check for libhdf5 and libhdf5_cpp (if necessary)
if (APPLE)
    find_library(LIBHDF5_HL_LIBRARY hdf5_hl HINTS ${HDF5_LIB_DIR})
    find_library(LIBHDF5_CPP_HL_LIBRARY hdf5_cpp HINTS ${HDF5_LIB_DIR})
    find_library(LIBHDF5_TOOLS_LIBRARY hdf5_tools HINTS ${HDF5_LIB_DIR})
else()
    find_library(LIBHDF5_HL_LIBRARY libhdf5_hl HINTS ${HDF5_LIB_DIR})
    find_library(LIBHDF5_CPP_HL_LIBRARY libhdf5_cpp HINTS ${HDF5_LIB_DIR})
    find_library(LIBHDF5_TOOLS_LIBRARY libhdf5_tools HINTS ${HDF5_LIB_DIR})
endif()

if(LIBHDF5_HL_LIBRARY AND LIBHDF5_CPP_HL_LIBRARY)
    list(APPEND LIBRARIES ${LIBHDF5_HL_LIBRARY} ${LIBHDF5_CPP_HL_LIBRARY})
    message(STATUS "additional HDF5 libraries found ${LIBHDF5_HL_LIBRARY} ${LIBHDF5_CPP_HL_LIBRARY}")
endif()

if(LIBHDF5_HL_LIBRARY)
    list(APPEND LIBRARIES ${LIBHDF5_HL_LIBRARY})
    message(STATUS "additional HDF5 libraries found ${LIBHDF5_HL_LIBRARY}")
endif()

if(LIBHDF5_TOOLS_LIBRARY)
    list(APPEND LIBRARIES ${LIBHDF5_TOOLS_LIBRARY})
    message(STATUS "additional HDF5 tools library found ${LIBHDF5_TOOLS_LIBRARY}")
endif()

######################### OMP #########################    

# Find OpenMP package
find_package(OpenMP)

######################### Set the executable #########################

# Add the executable first
add_executable(qsolver ${SOURCES})

# Set include directories for the target
target_include_directories(qsolver PRIVATE ${INCLUDE_DIRS})

# Link libraries to the executable
# add linker flags for GNU ld
# if (UNIX AND NOT APPLE)
#     set(CMAKE_EXE_LINKER_FLAGS_S "${CMAKE_EXE_LINKER_FLAGS} ,--start-group")
#     set(CMAKE_EXE_LINKER_FLAGS_E "${CMAKE_EXE_LINKER_FLAGS} ,--end-group")
# endif()

# print the libraries
message(STATUS "Libraries: ${CMAKE_EXE_LINKER_FLAGS_S}\n\t${MKL_CORE_LIBRARY}\n\t${MKL_SEQUENTIAL_LIBRARY}\n\t${MKL_RT_LIBRARY}\n\t${MKL_INTEL_ILP64_LIBRARY}\n\t${CMAKE_EXE_LINKER_FLAGS_E}\n\t${PTHREAD_LIBRARY}\n\t${DL_LIBRARY}\n\t${LIBHDF5_HL_LIBRARY}")
message(STATUS "HDF5 libraries: ${LIBHDF5_HL_LIBRARY} ${LIBHDF5_CPP_HL_LIBRARY} ${LIBHDF5_TOOLS_LIBRARY}")
message(STATUS "All: ${LIBRARIES}")
target_link_libraries(qsolver
    -Wl${CMAKE_EXE_LINKER_FLAGS_S}
    ${MKL_CORE_LIBRARY}
    ${MKL_SEQUENTIAL_LIBRARY}
    ${MKL_RT_LIBRARY}
    ${MKL_INTEL_ILP64_LIBRARY}
    -Wl${CMAKE_EXE_LINKER_FLAGS_E}
    ${LIBRARIES}
    # OpenMP::OpenMP_CXX  # Uncomment if you are using OpenMP
)

# Compiler flags
set(CMAKE_CXX_STANDARD 20)
set_target_properties(qsolver PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
    OUTPUT_NAME_DEBUG "qsolver_debug"
    OUTPUT_NAME_RELEASE "qsolver"
)

# Compiler flags for different build types
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Additional flags for specific OS platforms
message(STATUS "Using C++ Compiler: ${CMAKE_CXX_COMPILER}")
if (OS_PLATFORM STREQUAL "linux")
    if (CMAKE_CXX_COMPILER STREQUAL "icpx")
        message (STATUS "\tUsing Intel compilers")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} -qopenmp")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -qopenmp -D_DEBUG")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -qopenmp")

        # linker
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -qopenmp")
        # Replace any instance of -fopenmp with -qopenmp if necessary
        string(REPLACE "-fopenmp" "-qopenmp" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REPLACE "-fopenmp" "-qopenmp" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
        string(REPLACE "-fopenmp" "-qopenmp" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    else()
        message (STATUS "\tUsing GCC C++ compilers")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS} -fopenmp")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fopenmp -D_DEBUG")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp")
    endif()
elseif (OS_PLATFORM STREQUAL "windows")
    message (STATUS "\tUsing MSVC C++ compilers")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} /openmp")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /FS /bigobj /DMKL_ILP64 /Qiopenmp /D_DEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /FS /bigobj /DMKL_ILP64 /Qiopenmp")
elseif (OS_PLATFORM STREQUAL "macos")
    message (STATUS "\tUsing Clang C++ compilers")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")
endif()

# Redirect output to compilation.txt
set(OUTPUT_FILE "compilation.txt")
file(WRITE ${OUTPUT_FILE} "Compilation output will be redirected to ${OUTPUT_FILE}")

# Custom target to redirect output
add_custom_target(redirect_output ALL DEPENDS ${OUTPUT_FILE})

add_custom_target(build_all
    COMMAND rm -rf CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ..
    COMMAND ${CMAKE_COMMAND} --build .
    COMMAND rm -rf CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ..
    COMMAND ${CMAKE_COMMAND} --build .
)

add_custom_target(build_deb
    COMMAND rm -rf CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ..
    COMMAND ${CMAKE_COMMAND} --build .
)

add_custom_target(build_deb_reg
    COMMAND rm -rf CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ..
    COMMAND ${CMAKE_COMMAND} -DNQS_SREG
    COMMAND ${CMAKE_COMMAND} --build .
)

add_custom_target(build_rel
    COMMAND rm -rf CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ..
    COMMAND ${CMAKE_COMMAND} --build .
)

add_custom_target(build_rel_reg
    COMMAND rm -rf CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ..
    COMMAND ${CMAKE_COMMAND} -DNQS_SREG
    COMMAND ${CMAKE_COMMAND} --build .
)

# Clear target to remove build artifacts
add_custom_target(clear
    COMMAND ${CMAKE_COMMAND} --build . --target clean
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/Makefile
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/cmake_install.cmake
    COMMENT "Removing all build files and resetting the build directory."
)

###################################################################################################

# Export the targets to a config file
install(TARGETS qsolver
        EXPORT qsolverTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

# Create a qsolver-config.cmake file
install(EXPORT qsolverTargets
        FILE qsolver-config.cmake
        NAMESPACE qsolver::
        DESTINATION lib/cmake/qsolver)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/INPUTS/
        DESTINATION inputs
        FILES_MATCHING PATTERN "*.txt")
# Additional install commands (e.g., installing documentation, etc.)
# install(DIRECTORY docs/ DESTINATION share/qsolver/docs)

# Set installation prefix (default: /usr/local)
if(NOT CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Default installation prefix" FORCE)
endif()

# Output a message indicating the installation directory
message(STATUS "Installing qsolver to ${CMAKE_INSTALL_PREFIX}")

# add_custom_command(TARGET budiscovery POST_BUILD
# COMMAND ${CMAKE_COMMAND} -E copy
# ${CMAKE_BINARY_DIR}/libbulxi.so     # Replace with your built library path
# ${CMAKE_BINARY_DIR}/../some/other/location/libbulxi.so)  # Destination path