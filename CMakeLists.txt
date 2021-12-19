project(cgbn_math)

find_package(CUDA)

include_directories(./)
include_directories(./include)
include_directories(./samples)
include_directories(./cub-1.8.0)

set(CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_70)

file(GLOB_RECURSE CURRENT_HEADERS *.h *.cuh)
file(GLOB CURRENT_SOURCES *.cu)

source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

cuda_add_library(cgbn_math SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
#cuda_add_library(cgbn_math STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})

