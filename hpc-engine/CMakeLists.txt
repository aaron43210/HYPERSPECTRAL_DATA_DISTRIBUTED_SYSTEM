cmake_minimum_required(VERSION 3.16)
project(hyperspatial_hpc CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Portability: the QGIS plugin ships a *prebuilt* bin/hyperspatial_cli, and a
# binary compiled with -march=native emits instructions specific to the build
# machine's CPU — running it on an older/different CPU aborts with SIGILL
# (illegal instruction). Default to a portable -O3 build. For a machine-local
# build you can opt back in with -DNATIVE_ARCH=ON.
option(NATIVE_ARCH "Optimise for the local CPU (-march=native); NOT portable" OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
if(NATIVE_ARCH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

find_package(GDAL REQUIRED)

# Better OpenMP detection on macOS
find_package(OpenMP)
if(APPLE AND NOT OpenMP_CXX_FOUND)
    message(STATUS "OpenMP not found by default on macOS, checking Homebrew...")
    execute_process(COMMAND brew --prefix libomp
                    OUTPUT_VARIABLE HOMEBREW_LIBOMP
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(EXISTS "${HOMEBREW_LIBOMP}")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP}/include")
        set(OpenMP_CXX_LIB_NAMES omp)
        set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP}/lib/libomp.dylib")
        set(OpenMP_CXX_FOUND TRUE)
        
        # Create the imported target so target_link_libraries works
        if(NOT TARGET OpenMP::OpenMP_CXX)
            add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
            set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS "-Xpreprocessor" "-fopenmp")
            set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${HOMEBREW_LIBOMP}/include")
            set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_LIBRARIES "${HOMEBREW_LIBOMP}/lib/libomp.dylib")
        endif()
        
        message(STATUS "Found Homebrew OpenMP: ${HOMEBREW_LIBOMP}")
    endif()
endif()


# Cross-vendor GPU backend (OpenCL). One API covers Apple, NVIDIA, Intel and AMD
# GPUs via CL_DEVICE_TYPE_GPU, so there are no separate CUDA/Metal/oneAPI paths.
# Optional: if OpenCL isn't present the engine builds CPU-only and the GPU path
# is compiled out. Opt out explicitly with -DUSE_OPENCL=OFF.
option(USE_OPENCL "Build the OpenCL GPU backend if an OpenCL SDK is available" ON)
set(HPC_CORE_SOURCES src/RasterReader.cpp src/TileWriter.cpp)
if(USE_OPENCL)
    find_package(OpenCL)
    if(OpenCL_FOUND)
        list(APPEND HPC_CORE_SOURCES src/gpu/gpu_backend_opencl.cpp)
        message(STATUS "OpenCL found (${OpenCL_LIBRARIES}); GPU backend enabled.")
    else()
        message(STATUS "OpenCL not found; building CPU-only (GPU backend disabled).")
    endif()
endif()

add_library(hpc_core STATIC ${HPC_CORE_SOURCES})
target_include_directories(hpc_core PUBLIC include)
target_link_libraries(hpc_core PUBLIC GDAL::GDAL)
if(USE_OPENCL AND OpenCL_FOUND)
    target_link_libraries(hpc_core PUBLIC OpenCL::OpenCL)
    target_compile_definitions(hpc_core PUBLIC HSI_HAVE_OPENCL=1)
endif()

add_executable(hyperspatial_cli src/main.cpp)
target_link_libraries(hyperspatial_cli PRIVATE hpc_core)

if(OpenMP_CXX_FOUND)
    target_link_libraries(hyperspatial_cli PRIVATE OpenMP::OpenMP_CXX)
endif()

find_package(MPI)
if(MPI_CXX_FOUND)
    add_executable(hyperspatial_cli_mpi src/main_mpi.cpp)
    target_link_libraries(hyperspatial_cli_mpi PRIVATE hpc_core MPI::MPI_CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(hyperspatial_cli_mpi PRIVATE OpenMP::OpenMP_CXX)
    endif()
else()
    message(WARNING "MPI not found. hyperspatial_cli_mpi will not be built.")
endif()
