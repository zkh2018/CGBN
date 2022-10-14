project(cgbn_math)

find_package(CUDA)

include_directories(./)
include_directories(./include)
include_directories(./cub-1.8.0)


set(CUDA_NVCC_FLAGS -arch=sm_60 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -Xcompiler -fPIC)

file(GLOB_RECURSE CURRENT_HEADERS *.h *.cuh)
file(GLOB CURRENT_SOURCES *.cu)

source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

#cuda_add_library(cgbn_math SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
cuda_add_library(cgbn_math STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})

