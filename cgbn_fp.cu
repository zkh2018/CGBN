#include "cgbn_fp.h"
#include "cgbn_fp.cuh"

#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>

#include "cgbn/cgbn.h"
#include "utility/cpu_support.h"
#include "utility/cpu_simple_bn_math.h"
#include "utility/gpu_support.h"

namespace gpu{

Fp_model::Fp_model(const int count){
  init(count);
}

void Fp_model::init(const int count){
  _count = count;
  gpu_malloc((void**)&mont_repr_data, count * sizeof(cgbn_mem_t<BITS>));
  //gpu_malloc((void**)&modulus_data, count * sizeof(cgbn_mem_t<BITS>));
}
void Fp_model::init_host(const int count){
  _count = count;
  mont_repr_data = (cgbn_mem_t<BITS>*)malloc(count * sizeof(cgbn_mem_t<BITS>));
  //modulus_data = (cgbn_mem_t<BITS>*)malloc(count * sizeof(cgbn_mem_t<BITS>));
}
void Fp_model::resize(const int count){
  if(_count < count){
    if(_count > 0){
      gpu_free(mont_repr_data);
    }
    _count = count;
    gpu_malloc((void**)&mont_repr_data, count * sizeof(cgbn_mem_t<BITS>));
  }
  //gpu_malloc((void**)&modulus_data, count * sizeof(cgbn_mem_t<BITS>));
}
void Fp_model::resize_host(const int count){
  if(_count < count){
    if(_count > 0){
      free(mont_repr_data);
    }
    _count = count;
    mont_repr_data = (cgbn_mem_t<BITS>*)malloc(count * sizeof(cgbn_mem_t<BITS>));
  }
  //modulus_data = (cgbn_mem_t<BITS>*)malloc(count * sizeof(cgbn_mem_t<BITS>));
}
void Fp_model::release(){
  gpu_free(mont_repr_data);
  //gpu_free(modulus_data);
}
void Fp_model::release_host(){
  free(mont_repr_data);
  //free(modulus_data);
}

void Fp_model::copy_from_cpu(const Fp_model& fp){
  copy_cpu_to_gpu(mont_repr_data, fp.mont_repr_data, sizeof(cgbn_mem_t<BITS>) * _count);
  //copy_cpu_to_gpu(modulus_data, fp.modulus_data, sizeof(cgbn_mem_t<BITS>) * _count);
}
void Fp_model::copy_from_gpu(const Fp_model& fp){
  copy_gpu_to_gpu(mont_repr_data, fp.mont_repr_data, sizeof(cgbn_mem_t<BITS>) * _count);
}
void Fp_model::copy_to_cpu(Fp_model& fp){
  copy_gpu_to_cpu(mont_repr_data, fp.mont_repr_data, sizeof(cgbn_mem_t<BITS>) * _count);
  //copy_gpu_to_cpu(fp.modulus_data, modulus_data, sizeof(cgbn_mem_t<BITS>) * _count);
}

void Fp_model::clear(CudaStream stream){
  gpu_set_zero(this->mont_repr_data, _count * sizeof(cgbn_mem_t<BITS>), stream);
}

__global__ void kernel_fp_add(cgbn_error_report_t* report, cgbn_mem_t<BITS>* const in1, cgbn_mem_t<BITS>* const in2, cgbn_mem_t<BITS>* module_data, cgbn_mem_t<BITS>* max_value, const uint32_t count){
  int instance = (blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance >= count) return;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());
  device_fp_add(bn_env, in1 + instance, in2 + instance, module_data + instance, max_value);
}

__global__ void kernel_fp_sub(cgbn_error_report_t* report, cgbn_mem_t<BITS>* const in1, cgbn_mem_t<BITS>* const in2, cgbn_mem_t<BITS>* module_data, cgbn_mem_t<BITS>* max_value, const uint32_t count){
  int instance = (blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance >= count) return;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());
  device_fp_sub(bn_env, in1 + instance, in2 + instance, module_data + instance, max_value);
}


__global__ void kernel_mul_reduce(cgbn_error_report_t* report, uint32_t* res,cgbn_mem_t<BITS>* const in1, cgbn_mem_t<BITS>* const in2, cgbn_mem_t<BITS>* module_data, cgbn_mem_t<BITS>* tmp_buffer, uint64_t inv, const uint32_t count){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int instance = tid / TPI;
  if(instance >= count) return;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  device_mul_reduce(bn_env, res + instance * 3 * BITS/32, in1 + instance, in2 + instance, module_data + instance, tmp_buffer + instance, inv);
}

int fp_add(cgbn_mem_t<BITS>* in1, cgbn_mem_t<BITS>* in2, cgbn_mem_t<BITS>* module_data, cgbn_mem_t<BITS>* max_value, const uint32_t count){
  cgbn_error_report_t *report;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  uint32_t instances = std::min(count, (uint32_t)max_threads_per_block);
  uint32_t threads = instances * TPI;
  uint32_t blocks = (count + instances - 1) / instances;
  kernel_fp_add<<<blocks, threads>>>(report, in1, in2, module_data, max_value, count);
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  CUDA_CHECK(cgbn_error_report_free(report));
  return 0;
}

int fp_sub(cgbn_mem_t<BITS>* in1, cgbn_mem_t<BITS>* in2, cgbn_mem_t<BITS>* module_data, cgbn_mem_t<BITS>* max_value, const uint32_t count){
  cgbn_error_report_t *report;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  uint32_t instances = std::min(count, (uint32_t)max_threads_per_block);
  uint32_t threads = instances * TPI;
  uint32_t blocks = (count + instances - 1) / instances;
  kernel_fp_sub<<<blocks, threads>>>(report, in1, in2, module_data, max_value, count);
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  CUDA_CHECK(cgbn_error_report_free(report));
  return 0;
}

int fp_mul_reduce(cgbn_mem_t<BITS>* in1, cgbn_mem_t<BITS>* in2, uint64_t inv, cgbn_mem_t<BITS>* module_data, cgbn_mem_t<BITS>* tmp_buffer, uint32_t* res, const uint32_t count){
  cgbn_error_report_t *report;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 

  uint32_t instances = std::min(count, (uint32_t)max_threads_per_block);
  uint32_t threads = instances * TPI;
  uint32_t blocks = (count + instances - 1) / instances;

  kernel_mul_reduce<<<blocks, threads>>>(report, res, in1, in2, module_data, tmp_buffer, inv, count);
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  CUDA_CHECK(cgbn_error_report_free(report));
  return 0;
}

}//gpu
