cmake_minimum_required(VERSION 3.23.0)
project(QuantumEigenSolver)

# Detect platform
if (WIN32)
    set(OS_PLATFORM "windows")
    message(STATUS "Compiling on Windows")
elseif (UNIX)
    set(OS_PLATFORM "linux")
    message(STATUS "Compiling on Linux")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(OS_PLATFORM "linux")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(OS_PLATFORM "windows")
else()
    message(FATAL_ERROR "Unsupported OS: ${CMAKE_SYSTEM_NAME}")
endif()

##########################################################################

# Set compilers
if (OS_PLATFORM STREQUAL "windows")
    set(CMAKE_C_COMPILER "$ENV{CC}")
    set(CMAKE_CXX_COMPILER "$ENV{CXX}")
    set(COMPILER_LIBS_WIN "$ENV{ONE_API_COMPILER_LIB}")
    message(STATUS "Compiler libraries: ${COMPILER_LIBS_WIN}")
else()
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
endif()

message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")

##########################################################################

# Set variables necessary for the project
set(CPPSOURCE "${CMAKE_SOURCE_DIR}/source/cpp")

##########################################################################

# Find the path of the MKL include directory using an environment variable
find_path(MKL_INCL_DIR
    NAMES "mkl.h"
    PATHS "$ENV{MKL_INCL_DIR}"
    DOC "Path to MKL include directory"
)

# Find the path of the MKL library directory using an environment variable
find_path(MKL_LIB_DIR
    NAMES "mkl_core.lib" "libmkl_core.so"
    PATHS "$ENV{MKL_LIB_DIR}" "/usr/lib" "/usr/local/lib" "/opt/intel/mkl/lib"
    DOC "Path to MKL library directory"
)

# Find the path of the HDF5 include directory using an environment variable or common paths
find_path(HDF5_INCL_DIR
    NAMES "hdf5.h"
    PATHS "$ENV{HDF5_INCL_DIR}" "/usr/include/hdf5/serial" "/usr/local/include/hdf5"
    DOC "Path to HDF5 include directory"
)

# Find the path of the HDF5 library directory using an environment variable or common paths
find_path(HDF5_LIB_DIR
    NAMES "hdf5.lib" "libhdf5.so"
    PATHS "$ENV{HDF5_LIB_DIR}" "/usr/lib" "/usr/local/lib" "/usr/lib/x86_64-linux-gnu/hdf5/serial"
    DOC "Path to HDF5 library directory"
)

# Find the path of the HDF5 library directory
find_library(HDF5_LIBRARIES NAMES hdf5 HINTS "$ENV{HDF5_LIB_DIR}" "/usr/lib" "/usr/local/lib" "/usr/lib/x86_64-linux-gnu/hdf5/serial")
find_library(HDF5_HL_LIBRARIES NAMES hdf5_hl HINTS "$ENV{HDF5_LIB_DIR}" "/usr/lib" "/usr/local/lib" "/usr/lib/x86_64-linux-gnu/hdf5/serial")

# Print the paths found
message(STATUS "HDF5 include directory: ${HDF5_INCL_DIR}")
message(STATUS "HDF5 libraries: ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES}")

# Find the path of the Armadillo include directory
find_path(ARMADILLO_INCL_DIR
    NAMES "armadillo"
    PATHS "$ENV{ARMADILLO_INCL_DIR}" "/usr/include" "/usr/local/include"
    DOC "Path to Armadillo include directory"
)

##########################################################################

# Print the paths found
message(STATUS "MKL include directory: ${MKL_INCL_DIR}")
message(STATUS "MKL library directory: ${MKL_LIB_DIR}")
message(STATUS "Armadillo include directory: ${ARMADILLO_INCL_DIR}")
message(STATUS "HDF5 include directory: ${HDF5_INCL_DIR}")
message(STATUS "HDF5 library directory: ${HDF5_LIB_DIR}")

# Specify HDF5 library path directly if needed
link_directories("${HDF5_LIB_DIR}")

##########################################################################

# Set the source files
set(SOURCES 
    main.cpp 
    operators.cpp 
    user_interface.cpp 
    src/user_interface/ui_check_eth.cpp
    src/user_interface/ui_check_nqs.cpp
    src/user_interface/ui_check_quadratic.cpp
    src/user_interface/ui_check_symmetries.cpp
    src/nqs.cpp
    src/operator_parser.cpp
    ${CPPSOURCE}/common.cpp 
    ${CPPSOURCE}/directories.cpp 
    ${CPPSOURCE}/exceptions.cpp 
    ${CPPSOURCE}/str.cpp 
    ${CPPSOURCE}/ui.cpp 
    ${CPPSOURCE}/time.cpp
    ${CPPSOURCE}/Lattices/hexagonal.cpp 
    ${CPPSOURCE}/Lattices/square.cpp
    ${CPPSOURCE}/LinearAlgebra/solvers.cpp
    ${CPPSOURCE}/LinearAlgebra/Solvers/solvers_cg.cpp
    ${CPPSOURCE}/LinearAlgebra/Solvers/solvers_minres.cpp
    ${CPPSOURCE}/LinearAlgebra/Solvers/solvers_minresqlp.cpp
    ${CPPSOURCE}/LinearAlgebra/preconditioners.cpp
    ${CPPSOURCE}/LinearAlgebra/pfaffian.cpp
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

message(STATUS "MKL library directory: ${MKL_LIB_DIR}")
# Check for Intel MKL libraries
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
    message(STATUS "Intel MKL library found: ${MKL_CORE_LIBRARY}")
endif()
if (MKL_SEQUENTIAL_LIBRARY)
    list(APPEND LIBRARIES ${MKL_SEQUENTIAL_LIBRARY})
    message(STATUS "Intel MKL library found: ${MKL_SEQUENTIAL_LIBRARY}")
endif()
if (MKL_RT_LIBRARY)
    list(APPEND LIBRARIES ${MKL_RT_LIBRARY})
    message(STATUS "Intel MKL library found: ${MKL_RT_LIBRARY}")
endif()
if (MKL_INTEL_ILP64_LIBRARY)
    list(APPEND LIBRARIES ${MKL_INTEL_ILP64_LIBRARY})
    message(STATUS "Intel MKL library found: ${MKL_INTEL_ILP64_LIBRARY}")
endif()
message(STATUS "MKL libraries: ${LIBRARIES}")
include_directories(${MKL_INCL_DIR})
message(STATUS "MKL include directory: ${MKL_INCL_DIR}")

######################### Set the libraries #########################

# Add platform-specific libraries
if (OS_PLATFORM STREQUAL "linux")
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
elseif (OS_PLATFORM STREQUAL "windows")
    list(APPEND LIBRARIES ${COMPILER_LIBS_WIN})
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

# Find OpenMP package
find_package(OpenMP REQUIRED)

######################### Set the executable #########################

# Add the executable first
add_executable(qsolver ${SOURCES})

# Set include directories for the target
target_include_directories(qsolver PRIVATE ${INCLUDE_DIRS})

# Link libraries to the executable
target_link_libraries(qsolver
    -Wl,--start-group
    ${MKL_CORE_LIBRARY}
    ${MKL_SEQUENTIAL_LIBRARY}
    ${MKL_RT_LIBRARY}
    ${MKL_INTEL_ILP64_LIBRARY}
    -Wl,--end-group
    ${PTHREAD_LIBRARY}
    ${DL_LIBRARY}
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
message(STATUS "Using Compiler: ${CMAKE_CXX_COMPILER}")
if (OS_PLATFORM STREQUAL "linux")
    if (CMAKE_CXX_COMPILER STREQUAL "icpx")
        message (STATUS "\tUsing Intel compilers")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG} -qopenmp")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -qopenmp -D_DEBUG")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -qopenmp")

        # linker
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -qopenmp")
        # Replace any instance of -fopenmp with -qopenmp if necessary
        string(REPLACE "-fopenmp" "-qopenmp" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        string(REPLACE "-fopenmp" "-qopenmp" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
        string(REPLACE "-fopenmp" "-qopenmp" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    else()
        message (STATUS "\tUsing GCC compilers")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fopenmp -D_DEBUG")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp")
    endif()
elseif (OS_PLATFORM STREQUAL "windows")
    message (STATUS "\tUsing MSVC compilers")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /FS /bigobj /DMKL_ILP64 /Qiopenmp /D_DEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /FS /bigobj /DMKL_ILP64 /Qiopenmp")
endif()

# Redirect output to compilation.txt
set(OUTPUT_FILE "compilation.txt")
file(WRITE ${OUTPUT_FILE} "Compilation output will be redirected to ${OUTPUT_FILE}")

# Custom target to redirect output
add_custom_target(redirect_output ALL DEPENDS ${OUTPUT_FILE})

add_custom_target(build_all
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ..
    COMMAND ${CMAKE_COMMAND} --build .
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ..
    COMMAND ${CMAKE_COMMAND} --build .
)

add_custom_target(build_deb
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ..
    COMMAND ${CMAKE_COMMAND} --build .
)

add_custom_target(build_deb_reg
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ..
    COMMAND ${CMAKE_COMMAND} -DNQS_SREG
    COMMAND ${CMAKE_COMMAND} --build .
)

add_custom_target(build_rel
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ..
    COMMAND ${CMAKE_COMMAND} --build .
)

add_custom_target(build_rel_reg
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