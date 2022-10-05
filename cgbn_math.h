#ifndef ZION_CUDA_MATH_H
#define ZION_CUDA_MATH_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "cgbn_mem.h"

#include <cuda_runtime.h>
#include "gpu_support.h"

#define BlockDepth 64
const int BITS = 256;
namespace gpu{

const int BASE_BITS = 64;
const int N = BITS / BASE_BITS;
typedef uint64_t Int;
typedef uint64_t Int256[N];
  const int BUCKET_INSTANCES = 64;
  const int BUCKET_INSTANCES_G2 = 64;


typedef cudaStream_t CudaStream;
void create_stream(CudaStream* stream);
void release_stream(CudaStream& stream);
void sync(CudaStream stream);
void sync_device();
void gpu_reset();

void gpu_malloc(void** ptr, size_t size);
void gpu_set_zero(void* ptr, size_t size, CudaStream stream = 0);
void gpu_free(void*ptr);
void copy_cpu_to_gpu(void* dst, const void* src, size_t size, CudaStream stream = 0);
void copy_gpu_to_cpu(void* dst, const void* src, size_t size, CudaStream stream = 0);
void copy_gpu_to_gpu(void* dst, const void* src, size_t size, CudaStream stream = 0);

struct gpu_meta{
  void *ptr;
  size_t size = 0;
  void resize(const size_t _size);
  void release();
  gpu_meta(){
    ptr = nullptr;
  }
};

struct gpu_buffer{
  int total_n;
  int n;
  cgbn_mem_t<BITS>* ptr;
  void resize(int new_n);
  void resize_host(int new_n);
  void release();
  void release_host();

  gpu_buffer(){
    total_n = 0;
    n = 0;
    ptr = nullptr;
  }

  void copy_from_host(gpu_buffer& buf, CudaStream stream = 0);
  void copy_to_host(gpu_buffer& buf, CudaStream stream = 0);

  void copy_from_host(const uint32_t* data, const uint32_t n, CudaStream stream = 0);
  void copy_to_host(uint32_t* data, const uint32_t n, CudaStream stream = 0);
};


}//gpu

#endif
