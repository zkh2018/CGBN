#include "math.h"
#include <cuda_runtime.h>
#include <cuda.h>

#include "cgbn/cgbn.h"
#include "utility/cpu_support.h"
#include "utility/cpu_simple_bn_math.h"
#include "utility/gpu_support.h"

#define TPI 32
typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

namespace gpu{

void gpu_malloc(void** ptr, size_t size){
  CUDA_CHECK(cudaMalloc(ptr, size));
}
void gpu_free(void* ptr){
  CUDA_CHECK(cudaFree(ptr));
}
void copy_cpu_to_gpu(void* dst, void* src, size_t size){
  CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}
void copy_gpu_to_cpu(void* dst, void* src, size_t size){
  CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}


void gpu_buffer::resize(int new_n){
  if(total_n == 0){
    total_n = new_n;
    n = new_n;
    gpu_malloc((void**)&ptr, n * sizeof(cgbn_mem_t<BITS>));
  }else if(total_n < new_n){
    gpu_free((void*)ptr);
    total_n = new_n;
    n = new_n;
    gpu_malloc((void**)&ptr, n * sizeof(cgbn_mem_t<BITS>));
  }else{
    n = new_n;
  }
}
void gpu_buffer::resize_host(int new_n){
  if(total_n == 0){
    total_n = new_n;
    n = new_n;
    ptr = (cgbn_mem_t<BITS>*)malloc(n * sizeof(cgbn_mem_t<BITS>)); 
  }else if(total_n < n){
    free(ptr);
    total_n = n;
    n = new_n;
    ptr = (cgbn_mem_t<BITS>*)malloc(n * sizeof(cgbn_mem_t<BITS>));
  }else{
    n = new_n;
  }
}

void gpu_buffer::copy_from_host(gpu_buffer& buf){
  copy_cpu_to_gpu(ptr, buf.ptr, n * sizeof(cgbn_mem_t<BITS>));
}
void gpu_buffer::copy_to_host(gpu_buffer& buf){
  copy_gpu_to_cpu(buf.ptr, ptr, n * sizeof(cgbn_mem_t<BITS>));
}

__global__ void kernel_add(cgbn_error_report_t* report, cgbn_mem_t<BITS>* c, cgbn_mem_t<BITS>* const a, cgbn_mem_t<BITS>* const b, uint32_t *carry, const uint32_t count){
  int instance = instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance >= count) return;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
  env_t::cgbn_t  ta, tb, tc;                                             // define a, b, r as 1024-bit bignums
  cgbn_load(bn_env, ta, a);      // load my instance's a value
  cgbn_load(bn_env, tb, b);      // load my instance's b value
  *carry = cgbn_add(bn_env, tc, ta, tb);                           // r=a+b
  cgbn_store(bn_env, c, tc);   // store r into sum
}

__global__ void kernel_sub(cgbn_error_report_t* report, cgbn_mem_t<BITS>* c, cgbn_mem_t<BITS>* const a, cgbn_mem_t<BITS>* const b, uint32_t* carry){
  int instance = threadIdx.x;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
  env_t::cgbn_t  ta, tb, tr;                                             // define a, b, r as 1024-bit bignums
  cgbn_load(bn_env, ta, a);      // load my instance's a value
  cgbn_load(bn_env, tb, b);      // load my instance's b value
  *carry = cgbn_sub(bn_env, tr, ta, tb);                           // r=a+b
  cgbn_store(bn_env, c, tr);   // store r into sum
}

__global__ void kernel_sub_1(cgbn_error_report_t* report, cgbn_mem_t<BITS>* c, cgbn_mem_t<BITS>* const a, const uint32_t b, uint32_t* carry){
  int instance = threadIdx.x;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
  env_t::cgbn_t  ta, tr;                                             // define a, b, r as 1024-bit bignums
  cgbn_load(bn_env, ta, a);      // load my instance's a value
  //cgbn_load(bn_env, tb, b);      // load my instance's b value
  *carry = cgbn_sub_ui32(bn_env, tr, ta, b);                           // r=a+b
  cgbn_store(bn_env, c, tr);   // store r into sum
}

int add_two_num(cgbn_mem_t<BITS>* c, cgbn_mem_t<BITS>* const a, cgbn_mem_t<BITS>* const b, uint32_t* carry, const uint32_t count){
  cgbn_error_report_t *report;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_add<<<1, TPI>>>(report, c, a, b, carry, count);
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  return 0;
}

int sub_two_num(cgbn_mem_t<BITS>* c, cgbn_mem_t<BITS>* const a, cgbn_mem_t<BITS>* const b, uint32_t* carry){
  cgbn_error_report_t *report;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_sub<<<1, 1>>>(report, c, a, b, carry);
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  return 0;
}

int sub_1(cgbn_mem_t<BITS>* c, cgbn_mem_t<BITS>* const a, const uint32_t b, uint32_t* carry){
  cgbn_error_report_t *report;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_sub_1<<<1, 1>>>(report, c, a, b, carry);
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  return 0;
}

//int main(){
//  cgbn_mem_t<BITS> a, b, c;
//  add_two_num(&c, &a, &b);
//  return 0;
//}
}
