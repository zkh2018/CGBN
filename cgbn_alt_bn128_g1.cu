
#include "cgbn_alt_bn128_g1.h"
#include "cgbn_alt_bn128_g1.cuh"

#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>

#include "cgbn/cgbn.h"
#include "utility/cpu_support.h"
#include "utility/cpu_simple_bn_math.h"
#include "utility/gpu_support.h"

namespace gpu{

alt_bn128_g1::alt_bn128_g1(const int count){
  init(count);
}
void alt_bn128_g1::init(const int count){
  x.init(count);
  y.init(count);
  z.init(count);
}
void alt_bn128_g1::init_host(const int count){
  x.init_host(count);
  y.init_host(count);
  z.init_host(count);
}
void alt_bn128_g1::release(){
  x.release();
  y.release();
  z.release();
}
void alt_bn128_g1::release_host(){
  x.release_host();
  y.release_host();
  z.release_host();
}
void alt_bn128_g1::copy_from_cpu(const alt_bn128_g1& g1){
  x.copy_from_cpu(g1.x);
  y.copy_from_cpu(g1.y);
  z.copy_from_cpu(g1.z);
}
void alt_bn128_g1::copy_to_cpu(alt_bn128_g1& g1){
  g1.x.copy_to_cpu(x);
  g1.y.copy_to_cpu(y);
  g1.z.copy_to_cpu(z);
}
void alt_bn128_g1::clear(CudaStream stream ){
  this->x.clear(stream);
  this->y.clear(stream);
  this->z.clear(stream);
}

__global__ void kernel_alt_bn128_g1_add(cgbn_error_report_t* report, alt_bn128_g1 a, alt_bn128_g1 b, alt_bn128_g1 c, const uint32_t count, cgbn_mem_t<BITS>* max_value, cgbn_mem_t<BITS>* modulus, const uint64_t inv){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int instance = tid / TPI;
  int local_instance = threadIdx.x / TPI;
  if(instance >= count) return;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  

  DevAltBn128G1 dev_a, dev_b;
  dev_a.load(bn_env, a, instance);
  dev_b.load(bn_env, b, instance);

  //const int n = BITS / 32;
  __shared__ uint32_t cache[64 * 8 * 3];
  uint32_t *res = &cache[local_instance * 8 * 3];
  //uint32_t *res = tmp_res + instance * 3 * n;
  //cgbn_mem_t<BITS>* buffer = tmp_buffer + instance;
  __shared__ uint32_t cache_buffer[64 * 8];
  uint32_t *buffer = &cache_buffer[local_instance * 8];
  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);

  DevAltBn128G1 dev_c;
  dev_alt_bn128_g1_add(bn_env, dev_a, dev_b, &dev_c, res, buffer, local_max_value, local_modulus, inv);
  dev_c.store(bn_env, c, instance);
}

__global__ void kernel_alt_bn128_g1_reduce_sum(
    cgbn_error_report_t* report, 
    alt_bn128_g1 values, 
    Fp_model scalars,
    const size_t *index_it,
    alt_bn128_g1 partial, 
    uint32_t* counters, 
    const int ranges_size, 
    const uint32_t *firsts,
    const uint32_t *seconds,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    alt_bn128_g1 t_one,
    Fp_model field_zero,
    Fp_model field_one,
    char *density,
    cgbn_mem_t<BITS>* bn_exponents,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv
    ){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int instance = tid / TPI;
  if(instance >= ranges_size) return;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  

  const int n = BITS / 32;
  //uint32_t *res = tmp_res + instance * 3 * n;
  __shared__ uint32_t cache[64 * 3 * BITS/32];
  uint32_t *res = &cache[instance * 3 * n];
  //cgbn_mem_t<BITS>* buffer = tmp_buffer + instance;
  __shared__ uint32_t cache_buffer[64 * BITS/32];
  uint32_t *buffer = &cache_buffer[instance * BITS/32];
  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);

  DevAltBn128G1 result, dev_t_zero;
  DevFp dev_field_zero, dev_field_one;
  dev_t_zero.load(bn_env, t_zero, 0);
  dev_field_zero.load(bn_env, field_zero, 0);
  dev_field_one.load(bn_env, field_one, 0);
  result.copy_from(bn_env, dev_t_zero);
  int count = 0;
  for(int i = firsts[instance]; i < seconds[instance]; i++){
    const int j = index_it[i];
    DevFp scalar;
    scalar.load(bn_env, scalars, j);
    if(scalar.isequal(bn_env, dev_field_zero)){
    }
    else if(scalar.isequal(bn_env, dev_field_one)){
      DevAltBn128G1 dev_b;
      dev_b.load(bn_env, values, i);
      dev_alt_bn128_g1_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv);
    }
    else{
      const int group_thread = threadIdx.x & (TPI-1);
      if(group_thread == 0){
        density[i] = 1;
      }
      //DevFp a = scalar.as_bigint(bn_env, res, buffer, local_modulus, inv);
      //a.store(bn_env, bn_exponents, i);
      count += 1;
    }
  }  result.store(bn_env, partial, instance);
  const int group_thread = threadIdx.x & (TPI-1);
  if(group_thread == 0)
    counters[instance] = count;
}

__global__ void kernel_alt_bn128_g1_reduce_sum_one_range_pre(
    cgbn_error_report_t* report, 
    Fp_model scalars,
    const size_t *index_it,
    uint32_t* counters, 
    char* flags,
    const int ranges_size, 
    const uint32_t* firsts,
    const uint32_t* seconds,
    cgbn_mem_t<BITS>* max_value,
    Fp_model field_zero,
    Fp_model field_one,
    char *density,
    cgbn_mem_t<BITS>* bn_exponents,
    const uint64_t inv,
    cgbn_mem_t<BITS>* field_modulus, const uint64_t field_inv
    ){
  int local_instance = threadIdx.x / TPI;//0~63
  int local_instances = 64;
  int instance = blockIdx.x * local_instances + local_instance;

  int range_offset = blockIdx.y * gridDim.x * local_instances;
  int first = firsts[blockIdx.y];
  int second = seconds[blockIdx.y];
  int reduce_depth = second - first;//30130

  context_t bn_context(cgbn_report_monitor, report, range_offset + instance);
  env_t          bn_env(bn_context.env<env_t>());  

  __shared__ uint32_t cache_res[64 * 24];
  uint32_t *res = &cache_res[local_instance * 24];
  __shared__ uint32_t cache_buffer[512];
  uint32_t *buffer = &cache_buffer[local_instance * 8];
  env_t::cgbn_t local_field_modulus;
  cgbn_load(bn_env, local_field_modulus, field_modulus);

  DevFp dev_field_zero, dev_field_one;
  dev_field_zero.load(bn_env, field_zero, 0);
  dev_field_one.load(bn_env, field_one, 0);
  int count = 0;
  for(int i = first + instance; i < first + reduce_depth; i+= gridDim.x * local_instances){
    const int j = index_it[i];
    DevFp scalar;
    scalar.load(bn_env, scalars, j);
    if(scalar.isequal(bn_env, dev_field_zero)){
    }
    else if(scalar.isequal(bn_env, dev_field_one)){
      flags[j] = 1;
    }
    else{
      const int group_thread = threadIdx.x & (TPI-1);
      if(group_thread == 0){
        density[i] = 1;
      }
      DevFp a = scalar.as_bigint(bn_env, res, buffer, local_field_modulus, field_inv);
      a.store(bn_env, bn_exponents, i);
      count += 1;
    }
  }
  __shared__ int cache_counters[64];
  const int group_thread = threadIdx.x & (TPI-1);
  if(group_thread == 0)
    cache_counters[local_instance] = count;
  __syncthreads();
  if(local_instance == 0){
    for(int i = 1; i < local_instances; i++){
      count += cache_counters[i];
    }
    if(group_thread == 0){
      counters[blockIdx.y * gridDim.x + blockIdx.x] = count;
    }
  }
}
__global__ void kernel_alt_bn128_g1_reduce_sum_one_range(
    cgbn_error_report_t* report, 
    alt_bn128_g1 values, 
    Fp_model scalars,
    const size_t *index_it,
    alt_bn128_g1 partial, 
    const int ranges_size, 
    const uint32_t* firsts,
    const uint32_t* seconds,
    const char* flags,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv
    ){
  int local_instance = threadIdx.x / TPI;//0~63
  int local_instances = 64;
  int instance = blockIdx.x * local_instances + local_instance;

  int range_offset = blockIdx.y * gridDim.x * local_instances;
  int first = firsts[blockIdx.y];
  int second = seconds[blockIdx.y];
  int reduce_depth = second - first;//30130

  context_t bn_context(cgbn_report_monitor, report, range_offset + instance);
  env_t          bn_env(bn_context.env<env_t>());  

  __shared__ uint32_t cache_res[64 * 24];
  uint32_t *res = &cache_res[local_instance * 24];
  __shared__ uint32_t cache_buffer[512];
  uint32_t *buffer = &cache_buffer[local_instance * 8];
  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);

  DevAltBn128G1 result;
  DevFp dev_field_zero, dev_field_one;
  result.load(bn_env, t_zero, 0);
  for(int i = first + instance; i < first + reduce_depth; i+= gridDim.x * local_instances){
    const int j = index_it[i];
    if(flags[j] == 1){
      DevAltBn128G1 dev_b;
      dev_b.load(bn_env, values, i);
      dev_alt_bn128_g1_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv);
    }
  }
  result.store(bn_env, partial, range_offset + instance);
  __syncthreads();
  if(local_instance == 0){
    for(int i = 1; i < local_instances; i++){
      DevAltBn128G1 dev_b;
      dev_b.load(bn_env, partial, range_offset + instance + i);
      dev_alt_bn128_g1_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv);
    }
    result.store(bn_env, partial, range_offset + instance);
  }
}

__global__ void kernel_alt_bn128_g1_reduce_sum_one_range2(
    cgbn_error_report_t* report, 
    alt_bn128_g1 values, 
    Fp_model scalars,
    const size_t *index_it,
    alt_bn128_g1 partial, 
    const int ranges_size, 
    const uint32_t* firsts,
    const uint32_t* seconds,
    const char* flags,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv
    ){
  int local_instance = threadIdx.x / TPI;//0~63
  int local_instances = 64;
  int instance = blockIdx.x * local_instances + local_instance;

  int range_offset = blockIdx.y * gridDim.x * local_instances;
  int first = firsts[blockIdx.y];
  int second = seconds[blockIdx.y];
  int reduce_depth = second - first;//30130

  context_t bn_context(cgbn_report_monitor, report, range_offset + instance);
  env_t          bn_env(bn_context.env<env_t>());  

  __shared__ uint32_t cache_res[64 * 24];
  uint32_t *res = &cache_res[local_instance * 24];
  __shared__ uint32_t cache_buffer[512];
  uint32_t *buffer = &cache_buffer[local_instance * 8];
  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);

  DevAltBn128G1 result;
  result.load(bn_env, t_zero, 0);
  for(int i = first + instance; i < first + reduce_depth; i+= gridDim.x * local_instances){
    const int j = index_it[i];
    if(flags[j] == 1){
      DevAltBn128G1 dev_b;
      dev_b.load(bn_env, values, i);
      dev_alt_bn128_g1_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv);
    }
  }
  result.store(bn_env, partial, range_offset + instance);
}

__global__ void kernel_alt_bn128_g1_reduce_sum_one_range3(
    cgbn_error_report_t* report, 
    alt_bn128_g1 partial, 
    const int n, 
    const int range_instances,
    cgbn_mem_t<BITS>* max_value,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv
    ){
  int local_instance = threadIdx.x / TPI;//0~63
  int local_instances = blockDim.x / TPI;
  int instance = blockIdx.x * local_instances + local_instance;

  int range_offset = blockIdx.y * range_instances;

  context_t bn_context(cgbn_report_monitor, report, range_offset + instance);
  env_t          bn_env(bn_context.env<env_t>());  

  __shared__ uint32_t cache_res[64 * 24];
  uint32_t *res = &cache_res[local_instance * 24];
  __shared__ uint32_t cache_buffer[512];
  uint32_t *buffer = &cache_buffer[local_instance * 8];
  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);

  DevAltBn128G1 result;
  result.load(bn_env, partial, range_offset + instance);
  int i = instance + gridDim.x * local_instances;

  if(i < n){
    DevAltBn128G1 dev_b;
    dev_b.load(bn_env, partial, range_offset + i);
    dev_alt_bn128_g1_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv);
  }
  result.store(bn_env, partial, range_offset + instance);
}

__global__ void kernel_alt_bn128_g1_reduce_sum_one_range4(
    cgbn_error_report_t* report, 
    alt_bn128_g1 partial, 
    const int n, 
    const int range_offset,
    cgbn_mem_t<BITS>* max_value,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv
    ){
  int instance = threadIdx.x / TPI;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  

  __shared__ uint32_t res[24];
  __shared__ uint32_t buffer[8];
  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);

  DevAltBn128G1 result;
  result.load(bn_env, partial, 0);

  for(int i = 1; i < n; i++){
    DevAltBn128G1 dev_b;
    dev_b.load(bn_env, partial, i * range_offset);
    dev_alt_bn128_g1_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv);
  }
  result.store(bn_env, partial, 0);
}

__global__ void kernel_alt_bn128_g1_reduce_sum(
    cgbn_error_report_t* report, 
    alt_bn128_g1 partial_in, 
    const uint32_t* counters_in, 
    alt_bn128_g1 partial_out, 
    uint32_t* counters_out, 
    const int ranges_size, 
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    int depth, int step,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv
    ){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int instance = tid / TPI;
  if(instance >= ranges_size) return;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  

  //const int n = BITS / 32;
  __shared__ uint32_t cache_res[64 * 3 * BITS/32];
  uint32_t *res = &cache_res[instance * BITS/32 * 3];
  //uint32_t *res = tmp_res + instance * 3 * n;
  //cgbn_mem_t<BITS>* buffer = tmp_buffer + instance;
  __shared__ uint32_t cache_buffer[64 * BITS/32];
  uint32_t *buffer = &cache_buffer[instance * BITS/32];
  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);

  DevAltBn128G1 result;
  result.load(bn_env, t_zero, 0);
  //result.copy_from(bn_env, dev_t_zero);
  int count = 0;
  for(int i = 0; i < depth; i++){
    DevAltBn128G1 dev_b;
    dev_b.load(bn_env, partial_in, instance * depth * step + i * step);
    dev_alt_bn128_g1_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv);
    count += counters_in[instance * depth + i];
  }
  result.store(bn_env, partial_out, instance);
  const int group_thread = threadIdx.x & (TPI-1);
  if(group_thread == 0){
    counters_out[instance] = count;
  }
}

__global__ void kernel_alt_bn128_g1_reduce_sum2(
    cgbn_error_report_t* report, 
    alt_bn128_g1 data, 
    alt_bn128_g1 out, 
    const int n, 
    cgbn_mem_t<BITS>* max_value,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv
    ){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int instance = tid / TPI;
  int local_instance = threadIdx.x / TPI;
  int local_instances = blockDim.x / TPI;
  int offset = gridDim.x * local_instances;
  if(instance >= n) return;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  

  __shared__ uint32_t cache_res[64 * 24];
  uint32_t *res = &cache_res[local_instance * 24];
  __shared__ uint32_t cache_buffer[64 * 8];
  uint32_t *buffer = &cache_buffer[local_instance * 8];
  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);

  DevAltBn128G1 result;
  result.load(bn_env, data, instance);
  for(int i = instance + offset; i < n; i+=offset){
    DevAltBn128G1 dev_b;
    dev_b.load(bn_env, data, i);
    dev_alt_bn128_g1_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv);
  }
  result.store(bn_env, out, instance);
  __syncthreads();
  if(local_instance == 0){
    for(int i = 1; i < local_instances; i++){
      DevAltBn128G1 dev_b;
      dev_b.load(bn_env, out, instance + i);
      dev_alt_bn128_g1_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv);
    }
    result.store(bn_env, out, instance);
  }
}

int alt_bn128_g1_add(alt_bn128_g1 a, alt_bn128_g1 b, alt_bn128_g1 c, const uint32_t count, cgbn_mem_t<BITS>* max_value, cgbn_mem_t<BITS>* modulus, const uint64_t inv){
  cgbn_error_report_t *report;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 

  uint32_t instances = std::min(count, (uint32_t)max_threads_per_block);
  uint32_t threads = instances * TPI;
  uint32_t blocks = (count + instances - 1) / instances;

  kernel_alt_bn128_g1_add<<<blocks, threads>>>(report, a, b, c, count, max_value, modulus, inv);

  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  CUDA_CHECK(cgbn_error_report_free(report));
  return 0;
}

int alt_bn128_g1_reduce_sum(
    alt_bn128_g1 values, 
    Fp_model scalars, 
    const size_t *index_it,
    alt_bn128_g1 partial, 
    uint32_t *counters,
    const uint32_t ranges_size,
    const uint32_t *firsts,
    const uint32_t *seconds,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    alt_bn128_g1 t_one,
    Fp_model field_zero,
    Fp_model field_one,
    char *density,
    cgbn_mem_t<BITS>* bn_exponents,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv
    ){
  cgbn_error_report_t *report;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 

  uint32_t instances = std::min(ranges_size, (uint32_t)max_threads_per_block);
  uint32_t threads = instances * TPI;
  uint32_t blocks = (ranges_size + instances - 1) / instances;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  kernel_alt_bn128_g1_reduce_sum<<<blocks, threads>>>(report, values, scalars, index_it, partial, counters, ranges_size, firsts, seconds, max_value, t_zero, t_one, field_zero, field_one, density, bn_exponents, modulus, inv);

  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(start); 
  cudaEventSynchronize(stop);   
  float costtime;
  cudaEventElapsedTime(&costtime, start, stop);
  printf("kernel time = %fms\n", costtime);
  CGBN_CHECK(report);
  CUDA_CHECK(cgbn_error_report_free(report));
  return 0;
}

int alt_bn128_g1_reduce_sum_one_range(
    alt_bn128_g1 values, 
    Fp_model scalars, 
    const size_t *index_it,
    alt_bn128_g1 partial, 
    uint32_t *counters,
    char* flags,
    const uint32_t ranges_size,
    const uint32_t *firsts,
    const uint32_t *seconds,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    Fp_model field_zero,
    Fp_model field_one,
    char *density,
    cgbn_mem_t<BITS>* bn_exponents,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    cgbn_mem_t<BITS>* field_modulus, const uint64_t field_inv,
    const int max_reduce_depth
    ){
  cgbn_error_report_t *report = get_error_report();

  uint32_t threads = 512;
  //const int reduce_depth = 30130;//second - first;
  const int local_instances = 64 * BlockDepth;
  uint32_t block_x =  (max_reduce_depth + local_instances - 1) / local_instances;
  dim3 blocks(block_x, ranges_size, 1);
  kernel_alt_bn128_g1_reduce_sum_one_range_pre<<<blocks, threads>>>(report, scalars, index_it, counters, flags, ranges_size, firsts, seconds, max_value, field_zero, field_one, density, bn_exponents, inv, field_modulus, field_inv);

//*********test
  const int local_instances2 = 64 * BlockDepth;
  uint32_t block_x2 =  (max_reduce_depth + local_instances2 - 1) / local_instances2;
  dim3 blocks2(block_x2, ranges_size, 1);
  kernel_alt_bn128_g1_reduce_sum_one_range2<<<blocks2, threads>>>(report, values, scalars, index_it, partial, ranges_size, firsts, seconds, flags, max_value, t_zero, modulus, inv);
  int n = block_x2 * 64;
  int range_offset = n;
  while(n >= 2){
    int instances = std::min(64, n/2);
    int threads = instances * TPI; 
    int blockx = n / (instances * 2);
    kernel_alt_bn128_g1_reduce_sum_one_range3<<<dim3(blockx, ranges_size, 1), threads>>>(report, partial, n, range_offset, max_value, modulus, inv);
    n /= 2;
  }
  kernel_alt_bn128_g1_reduce_sum_one_range4<<<1, TPI>>>(report, partial, ranges_size, range_offset, max_value, modulus, inv);
//********test

  //kernel_alt_bn128_g1_reduce_sum_one_range<<<blocks, threads>>>(report, values, scalars, index_it, partial, ranges_size, firsts, seconds, flags, max_value, t_zero, modulus, inv);
  //CUDA_CHECK(cudaDeviceSynchronize());
  //CGBN_CHECK(report);
  return 0;
}
void alt_bn128_g1_reduce_sum(
    alt_bn128_g1 partial_in, 
    const uint32_t *counters_in,
    alt_bn128_g1 partial_out, 
    uint32_t *counters_out,
    const uint32_t ranges_size,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    const int max_reduce_depth
    ){
  cgbn_error_report_t *report = get_error_report();
  uint32_t instances = std::min(ranges_size, (uint32_t)max_threads_per_block);
  uint32_t threads = instances * TPI;
  uint32_t blocks = (ranges_size + instances - 1) / instances;
  //int reduce_depth = 30130;
  const int local_instances = 64 * BlockDepth;
  uint32_t depth =  (max_reduce_depth + local_instances - 1) / local_instances;
  int step = 64;
  kernel_alt_bn128_g1_reduce_sum<<<blocks, threads>>>(report, partial_in, counters_in, partial_out, counters_out, ranges_size, max_value, t_zero, depth, step, modulus, inv);
  //CUDA_CHECK(cudaDeviceSynchronize());
  //CGBN_CHECK(report);
}

void alt_bn128_g1_reduce_sum_one_instance(
    alt_bn128_g1 partial_in, 
    const uint32_t *counters_in,
    alt_bn128_g1 partial_out, 
    uint32_t *counters_out,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    const int max_reduce_depth
    ){
  cgbn_error_report_t *report = get_error_report();
  uint32_t instances = 1;
  uint32_t threads = instances * TPI;
  uint32_t blocks = 1;
  kernel_alt_bn128_g1_reduce_sum<<<blocks, threads>>>(report, partial_in, counters_in, partial_out, counters_out, 1, max_value, t_zero, max_reduce_depth, 1, modulus, inv);
  CUDA_CHECK(cudaDeviceSynchronize());
  //CGBN_CHECK(report);
}

template<int BlockSize, int BlockNum>
__global__ void test(
    cgbn_error_report_t* report, 
    alt_bn128_g1 data, 
    alt_bn128_g1 out, 
    int n,
    cgbn_mem_t<BITS>* max_value,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv
    ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int instance = tid / TPI;
  int local_instance = threadIdx.x / TPI;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  DevAltBn128G1 a;
  a.load(bn_env, data, instance);
  __shared__ uint32_t cache_buffer[BlockSize*8];
  __shared__ uint32_t cache_res[BlockSize*24];
  uint32_t *buffer = &cache_buffer[local_instance * 8];
  uint32_t *res = &cache_res[local_instance * 24];
  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);
  for(int i = instance + BlockNum*BlockSize; i < n; i+=BlockNum*BlockSize){
    DevAltBn128G1 b;
    b.load(bn_env, data, i);
    dev_alt_bn128_g1_add(bn_env, a, b, &a, res, buffer, local_max_value, local_modulus, inv);
  }
  a.store(bn_env, out, instance);
  //__syncthreads();
  //if(local_instance == 0){
  //  for(int i = 1; i < BlockSize && i < n; i++){
  //    DevAltBn128G1 b;
  //    b.load(bn_env, out, instance + i);
  //    dev_alt_bn128_g1_add(bn_env, a, b, &a, res, buffer, local_max_value, local_modulus, inv);
  //  }
  //  a.store(bn_env, out, blockIdx.x);
  //}
}
void alt_bn128_g1_reduce_sum2(
    alt_bn128_g1 data, 
    alt_bn128_g1 out, 
    const uint32_t n,
    cgbn_mem_t<BITS>* max_value,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv, 
    CudaStream stream){
  cgbn_error_report_t *report = get_error_report();
  uint32_t threads = 512;
  uint32_t local_instances = threads / TPI;//64
  uint32_t instances = std::min(n, (uint32_t)(local_instances * BlockDepth));
  //uint32_t blocks = (n + instances - 1) / instances;
  //kernel_alt_bn128_g1_reduce_sum2<<<blocks, threads>>>(report, data, out, n, max_value, modulus, inv);
  test<64, 64><<<64, 512, 0, stream>>>(report, data, out, n-1, max_value, modulus, inv);
  const int tmp_n = 64*64; 
  test<64, 8><<<8, 512, 0, stream>>>(report, out, data, tmp_n, max_value, modulus, inv);
  test<16, 4><<<4, 128, 0, stream>>>(report, data, out, 64*8, max_value, modulus, inv);
  test<8, 1><<<1, 64, 0, stream>>>(report, out, data, 64, max_value, modulus, inv);
  test<1, 1><<<1, 8, 0, stream>>>(report, data, out, 8, max_value, modulus, inv);
  //CUDA_CHECK(cudaDeviceSynchronize());
}


void init_error_report(){
  get_error_report();
}

__global__ void kernel_warmup(){
  int sum = 0;
  for(int i = 0; i < 1000; i++){
    sum += i;
  }
  printf("warm up : %d\n", sum);
}
void warm_up(){
  kernel_warmup<<<1, 1>>>();
  CUDA_CHECK(cudaDeviceSynchronize());
}

} //gpu
