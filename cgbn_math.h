#ifndef ZION_CUDA_MATH_H
#define ZION_CUDA_MATH_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <gmp.h>
#include "cgbn/cgbn_mem.h"

#include <cuda_runtime.h>
#include "cgbn/cgbn.h"
#include "gpu_support.h"

const int BITS = 256;
const int BITS_PER_NUM = 32;
const int NUM = BITS/BITS_PER_NUM; 
#define BlockDepth 64

namespace gpu{
#define TPI 4
typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;
#define max_threads_per_block  (256/TPI)

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

int add_two_num(cgbn_mem_t<BITS>* c, cgbn_mem_t<BITS>* const a, cgbn_mem_t<BITS>* const b, uint32_t* carry, const uint32_t count);
int add_1(cgbn_mem_t<BITS>* c, cgbn_mem_t<BITS>* const a, uint32_t b, uint32_t* carry, const uint32_t count);

int sub_two_num(cgbn_mem_t<BITS>* c, cgbn_mem_t<BITS>* const a, cgbn_mem_t<BITS>* const b, uint32_t* carry, const uint32_t count);
int sub_1(cgbn_mem_t<BITS>* c, cgbn_mem_t<BITS>* const a, uint32_t b, uint32_t* carry, const uint32_t count);

int add(
    cgbn_mem_t<BITS>* x1, cgbn_mem_t<BITS>* y1, cgbn_mem_t<BITS>* z1, 
    cgbn_mem_t<BITS>* x2, cgbn_mem_t<BITS>* y2, cgbn_mem_t<BITS>* z2, 
    cgbn_mem_t<BITS>* x_out, cgbn_mem_t<BITS>* y_out, cgbn_mem_t<BITS>* z_out);


int mul_two_num(
    cgbn_mem_t<BITS>* c_low, 
    cgbn_mem_t<BITS>* c_high, 
    cgbn_mem_t<BITS>* const a, cgbn_mem_t<BITS>* const b, const uint32_t count);

int mul_1(cgbn_mem_t<BITS>* c, cgbn_mem_t<BITS>* const a, uint32_t b, uint32_t* carry, const uint32_t count);


}//gpu

#endif
