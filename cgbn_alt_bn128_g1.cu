
#include "cgbn_alt_bn128_g1.h"
#include "cgbn_alt_bn128_g1.cuh"
#include "bigint_256.cuh"

#include <algorithm>


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
void alt_bn128_g1::resize(const int count){
  x.resize(count);
  y.resize(count);
  z.resize(count);
}
void alt_bn128_g1::resize_host(const int count){
  x.resize_host(count);
  y.resize_host(count);
  z.resize_host(count);
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
void alt_bn128_g1::copy_from_cpu(const alt_bn128_g1& g1, CudaStream stream){
  x.copy_from_cpu(g1.x, stream);
  y.copy_from_cpu(g1.y, stream);
  z.copy_from_cpu(g1.z, stream);
}
void alt_bn128_g1::copy_from_gpu(const alt_bn128_g1& g1, CudaStream stream){
  x.copy_from_gpu(g1.x, stream);
  y.copy_from_gpu(g1.y, stream);
  z.copy_from_gpu(g1.z, stream);
}
void alt_bn128_g1::copy_to_cpu(alt_bn128_g1& g1, CudaStream stream){
  g1.x.copy_to_cpu(x, stream);
  g1.y.copy_to_cpu(y, stream);
  g1.z.copy_to_cpu(z, stream);
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
__global__ void kernel_alt_bn128_g1_reduce_sum_one_range6(
    cgbn_error_report_t* report, 
    alt_bn128_g1 partial, 
    const int n, 
    const uint32_t* firsts,
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
  result.load(bn_env, partial, firsts[0]);

  for(int i = 1; i < n; i++){
    DevAltBn128G1 dev_b;
    dev_b.load(bn_env, partial, firsts[i]);
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

template<int BlockInstances>
__global__ void kernel_alt_bn128_g1_reduce_sum_one_range5(
    cgbn_error_report_t* report, 
    alt_bn128_g1 values, 
    Fp_model scalars,
    const size_t *index_it,
    alt_bn128_g1 partial, 
    const int ranges_size, 
    const int range_id_offset,
    const uint32_t* firsts,
    const uint32_t* seconds,
    char* flags,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv
    ){
  const int local_instance = threadIdx.x / TPI;//0~63
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int instance = tid / TPI;

  const int instance_offset = (range_id_offset + blockIdx.y) * gridDim.x * BlockInstances;
  const int first = firsts[range_id_offset + blockIdx.y];
  const int second = seconds[range_id_offset + blockIdx.y];
  const int reduce_depth = second - first;//30130
  if(reduce_depth <= 1) return;
  const int half_depth = (reduce_depth + 1) / 2;

  context_t bn_context(cgbn_report_monitor, report, instance_offset + instance);
  env_t          bn_env(bn_context.env<env_t>());  
  if(instance >= half_depth) return;

  __shared__ uint32_t cache_res[BlockInstances * 24];
  uint32_t *res = &cache_res[local_instance * 24];
  __shared__ uint32_t cache_buffer[BlockInstances * 8];
  uint32_t *buffer = &cache_buffer[local_instance * 8];
  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);

  DevAltBn128G1 result;
  if(flags[index_it[first + instance]] == 1){
	  result.load(bn_env, values, first+instance);
  }else{
	  result.load(bn_env, t_zero, 0);
  }
  for(int i = first + instance+half_depth; i < first + reduce_depth; i+= half_depth){
    const int j = index_it[i];
    if(flags[j] == 1){
      DevAltBn128G1 dev_b;
      dev_b.load(bn_env, values, i);
      dev_alt_bn128_g1_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv);
    }
  }
  result.store(bn_env, partial, first + instance);
}

template<int BlockInstances>
__global__ void kernel_alt_bn128_g1_reduce_sum_one_range7(
    cgbn_error_report_t* report, 
    alt_bn128_g1 values, 
    Fp_model scalars,
    const size_t *index_it,
    alt_bn128_g1 partial, 
    const int ranges_size, 
    const int range_id_offset,
    const uint32_t* firsts,
    const uint32_t* seconds,
    char* flags,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv
    ){
  const int local_instance = threadIdx.x / TPI;//0~63
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int instance = tid / TPI;

  const int instance_offset = (range_id_offset + blockIdx.y) * gridDim.x * BlockInstances;
  const int first = firsts[range_id_offset + blockIdx.y];
  const int second = seconds[range_id_offset + blockIdx.y];
  const int reduce_depth = second - first;//30130
  if(reduce_depth <= 1) return;
  const int half_depth = (reduce_depth + 1) / 2;

  context_t bn_context(cgbn_report_monitor, report, instance_offset + instance);
  env_t          bn_env(bn_context.env<env_t>());  
  if(instance >= half_depth) return;

  __shared__ uint32_t cache_res[BlockInstances * 24];
  uint32_t *res = &cache_res[local_instance * 24];
  __shared__ uint32_t cache_buffer[BlockInstances * 8];
  uint32_t *buffer = &cache_buffer[local_instance * 8];
  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);

  DevAltBn128G1 result;
  //if(flags[index_it[first + instance]] == 1){
	  result.load(bn_env, values, first+instance);
  //}else{
  //        result.load(bn_env, t_zero, 0);
  //}
  for(int i = first + instance+half_depth; i < first + reduce_depth; i+= half_depth){
    //const int j = index_it[i];
    //if(flags[j] == 1){
      DevAltBn128G1 dev_b;
      dev_b.load(bn_env, values, i);
      dev_alt_bn128_g1_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv);
    //}
  }
  result.store(bn_env, partial, first + instance);
}

__global__ void kernel_update_seconds(const uint32_t *firsts, uint32_t* seconds, const int range_size){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < range_size){
		int first = firsts[tid];
		int second = seconds[tid];
		seconds[tid] = first + (second - first + 1) / 2;
	}
}
int alt_bn128_g1_reduce_sum_one_range5(
    alt_bn128_g1 values, 
    Fp_model scalars, 
    const size_t *index_it,
    alt_bn128_g1 partial, 
    uint32_t *counters,
    char* flags,
    const uint32_t ranges_size,
    const uint32_t *firsts,
    uint32_t *seconds,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    Fp_model field_zero,
    Fp_model field_one,
    char *density,
    cgbn_mem_t<BITS>* bn_exponents,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    cgbn_mem_t<BITS>* field_modulus, const uint64_t field_inv,
    const int max_reduce_depth, cudaStream_t stream
    ){
  cgbn_error_report_t *report = get_error_report();

  uint32_t threads = 512;
  const int local_instances = 64 * BlockDepth;
  uint32_t block_x =  (max_reduce_depth + local_instances - 1) / local_instances;
  dim3 blocks(block_x, ranges_size, 1);
  kernel_alt_bn128_g1_reduce_sum_one_range_pre<<<blocks, threads, 0, stream>>>(report, scalars, index_it, counters, flags, ranges_size, firsts, seconds, max_value, field_zero, field_one, density, bn_exponents, inv, field_modulus, field_inv);

  int n = max_reduce_depth;
  const int local_instances2 = 32;
  threads = local_instances2 * TPI;
  uint32_t block_x2 =  ((n+1)/2 + local_instances2 - 1) / local_instances2;
  dim3 blocks2(block_x2, ranges_size, 1);
  kernel_alt_bn128_g1_reduce_sum_one_range5<local_instances2><<<blocks2, dim3(threads, 1, 1), 0, stream>>>(report, values, scalars, index_it, partial, ranges_size, 0, firsts, seconds, flags, max_value, t_zero, modulus, inv);
  const int update_threads = 64;
  const int update_blocks = (ranges_size + update_threads - 1) / update_threads;
  kernel_update_seconds<<<update_blocks, update_threads, 0, stream>>>(firsts, seconds, ranges_size);
  //CUDA_CHECK(cudaDeviceSynchronize());
  n = (n+1)/2;
  while(n>=2){
	  uint32_t block_x2 =  ((n+1)/2 + local_instances2 - 1) / local_instances2;
	  dim3 blocks2(block_x2, ranges_size, 1);
	  kernel_alt_bn128_g1_reduce_sum_one_range7<local_instances2><<<blocks2, dim3(threads, 1, 1), 0, stream>>>(report, partial, scalars, index_it, partial, ranges_size, 0, firsts, seconds, flags, max_value, t_zero, modulus, inv);
	  //CUDA_CHECK(cudaDeviceSynchronize());
	  kernel_update_seconds<<<update_blocks, update_threads, 0, stream>>>(firsts, seconds, ranges_size);
	  //CUDA_CHECK(cudaDeviceSynchronize());
	  n = (n+1)/2;
  }
  kernel_alt_bn128_g1_reduce_sum_one_range6<<<1, TPI, 0, stream>>>(report, partial, ranges_size, firsts, max_value, modulus, inv);
  //CUDA_CHECK(cudaDeviceSynchronize());
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
    threads = instances * TPI; 
    int blockx = n < instances * 2 ? 1 : (n + instances*2-1) / (instances * 2);
    kernel_alt_bn128_g1_reduce_sum_one_range3<<<dim3(blockx, ranges_size, 1), threads>>>(report, partial, n, range_offset, max_value, modulus, inv);
    n = blockx * instances;
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
  //CUDA_CHECK(cudaDeviceSynchronize());
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

template<int BlockInstances>
__global__ void kernel_reduce_sum(
    cgbn_error_report_t* report, 
    alt_bn128_g1 data, 
    alt_bn128_g1 out, 
    const int half_n,
    const int n,
    cgbn_mem_t<BITS>* max_value,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv
    ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int instance = tid / TPI;
  if(instance >= half_n) return;
  int local_instance = threadIdx.x / TPI;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  __shared__ uint32_t cache_buffer[BlockInstances*8];
  __shared__ uint32_t cache_res[BlockInstances*24];
  uint32_t *buffer = &cache_buffer[local_instance * 8];
  uint32_t *res = &cache_res[local_instance * 24];
  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);

  DevAltBn128G1 a;
  a.load(bn_env, data, instance);
  for(int i = instance + half_n; i < n; i+= half_n){
    DevAltBn128G1 b;
    b.load(bn_env, data, i);
    dev_alt_bn128_g1_add(bn_env, a, b, &a, res, buffer, local_max_value, local_modulus, inv);
  }
  a.store(bn_env, out, instance);
}

void alt_bn128_g1_reduce_sum2(
    alt_bn128_g1 data, 
    alt_bn128_g1 out, 
    const uint32_t n,
    cgbn_mem_t<BITS>* max_value,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv, 
    CudaStream stream){
  cgbn_error_report_t *report = get_error_report();
  if(true){
    int len = n-1;
    const int instances = 64;
    int threads = instances * TPI;
    int half_len = (len + 1) / 2;
    int blocks = (half_len + instances - 1) / instances;
    kernel_reduce_sum<instances><<<blocks, threads, 0, stream>>>(report, data, out, half_len, len, max_value, modulus, inv);
    len = half_len;
    while(len > 1){
        int half_len = (len + 1) / 2;
        int blocks = (half_len + instances - 1) / instances;
        kernel_reduce_sum<instances><<<blocks, threads, 0, stream>>>(report, out, out, half_len, len, max_value, modulus, inv);
        len = half_len;
    }
  }
  if(false){
      uint32_t threads = 512;
      uint32_t local_instances = threads / TPI;//64
      uint32_t instances = std::min(n, (uint32_t)(local_instances * BlockDepth));
      test<64, 64><<<64, 512, 0, stream>>>(report, data, out, n-1, max_value, modulus, inv);
      const int tmp_n = 64*64; 
      test<64, 8><<<8, 512, 0, stream>>>(report, out, data, tmp_n, max_value, modulus, inv);
      test<16, 4><<<4, 128, 0, stream>>>(report, data, out, 64*8, max_value, modulus, inv);
      test<8, 1><<<1, 64, 0, stream>>>(report, out, data, 64, max_value, modulus, inv);
      test<1, 1><<<1, 8, 0, stream>>>(report, data, out, 8, max_value, modulus, inv);
      //CUDA_CHECK(cudaDeviceSynchronize());
  }
}

template<int BlockInstances>
__global__ void kernel_elementwise_mul_scalar(
    cgbn_error_report_t* report,
    Fp_model datas, 
    Fp_model sconst, 
    const uint32_t n,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv){
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int instance = tid / TPI;
  const int local_instance = threadIdx.x / TPI;
  const int local_instances = blockDim.x / TPI;
  if(instance >= n) return;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  

  __shared__ uint32_t cache[BlockInstances * 3 * BITS/32];
  uint32_t *res = &cache[local_instance * 3 * BITS/32];
  __shared__ uint32_t cache_buffer[BlockInstances * BITS/32];
  uint32_t *buffer = &cache_buffer[local_instance * BITS/32];

  env_t::cgbn_t local_modulus;
  cgbn_load(bn_env, local_modulus, modulus);
  DevFp local_sconst;
  local_sconst.load(bn_env, sconst, 0);
  for(int i = instance; i < n; i += gridDim.x * local_instances){
    DevFp a;
    a.load(bn_env, datas, i);
    a = a.mul(bn_env, local_sconst, res, buffer, local_modulus, inv);
    a.store(bn_env, datas, i);
  }
}

void alt_bn128_g1_elementwise_mul_scalar(
    Fp_model datas, 
    Fp_model sconst, 
    const uint32_t n,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv){
  //cgbn_error_report_t *report;
  //CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  cgbn_error_report_t *report = get_error_report();
  const int instances = 64;
  const int threads = instances * TPI;
  const int blocks = (n + instances - 1) / instances;
  //printf("blocks = %d, threads=%d\n", blocks, threads);

  kernel_elementwise_mul_scalar<instances><<<blocks, threads>>>(report, datas, sconst, n, modulus, inv); 
  //cuda_check(cudaDeviceSynchronize());
}


inline __device__ void dev_butterfly_2(
        env_t& bn_env, 
        Fp_model& twiddles, 
        const int twiddle_offset,
        const uint32_t stride, 
        const uint32_t stage_length, 
        uint32_t out_offset, 
        const env_t::cgbn_t& max_value,
        const env_t::cgbn_t& modulus,
        const uint64_t inv,
        uint32_t *res,
        uint32_t *buffer,
        Fp_model& out){
    unsigned int out_offset2 = out_offset + stage_length;
    DevFp out1, out2;
    //FieldT t = out[out_offset2];
    out2.load(bn_env, out, out_offset2);
    out1.load(bn_env, out, out_offset);
    //out[out_offset2] = out[out_offset] - t;
    DevFp tmp_out2 = out1.sub(bn_env, out2, max_value, modulus);
    tmp_out2.store(bn_env, out, out_offset2);
    //out[out_offset] += t;
    DevFp tmp_out = out1.add(bn_env, out2, max_value, modulus); 
    tmp_out.store(bn_env, out, out_offset);
    out_offset2++;
    out_offset++;
    for (unsigned int k = 1; k < stage_length; k++){
        //FieldT t = twiddles[k] * out[out_offset2];
        out2.load(bn_env, out, out_offset2);
        out1.load(bn_env, out, out_offset);
        DevFp twiddle;
        twiddle.load(bn_env, twiddles, twiddle_offset + k);
        DevFp t = twiddle.mul(bn_env, out2, res, buffer, modulus, inv);
        //out[out_offset2] = out[out_offset] - t;
        tmp_out2 = out1.sub(bn_env, t, max_value, modulus);
        tmp_out2.store(bn_env, out, out_offset2);
        //out[out_offset] += t;
        tmp_out = out1.add(bn_env, t, max_value, modulus);
        tmp_out.store(bn_env, out, out_offset);
        out_offset2++;
        out_offset++;
    }
}

inline __device__ void dev_butterfly_4(
        env_t& bn_env, 
        Fp_model& twiddles, 
        const int twiddles_len,
        const int twiddle_offset,
        const uint32_t stride, 
        const uint32_t stage_length, 
        uint32_t out_offset, 
        const env_t::cgbn_t& max_value,
        const env_t::cgbn_t& modulus,
        const uint64_t inv,
        uint32_t *res,
        uint32_t *buffer,
        Fp_model& out){
    DevFp j;
    j.load(bn_env, twiddles, twiddle_offset + twiddles_len-1);
    uint32_t tw = 0;
    /* Case twiddle == one */
    {
		const unsigned i0  = out_offset;
        const unsigned i1  = out_offset + stage_length;
        const unsigned i2  = out_offset + stage_length*2;
        const unsigned i3  = out_offset + stage_length*3;

		DevFp z0, z1, z2, z3;
        //const FieldT z0  = out[i0];
        z0.load(bn_env, out, i0);
        //const FieldT z1  = out[i1];
        z1.load(bn_env, out, i1);
        //const FieldT z2  = out[i2];
        z2.load(bn_env, out, i2);
        //const FieldT z3  = out[i3];
        z3.load(bn_env, out, i3);

        DevFp t1, t2, t3, t4, t4j;
        //const FieldT t1  = z0 + z2;
        t1 = z0.add(bn_env, z2, max_value, modulus);
        //const FieldT t2  = z1 + z3;
        t2 = z1.add(bn_env, z3, max_value, modulus);
        //const FieldT t3  = z0 - z2;
        t3 = z0.sub(bn_env, z2, max_value, modulus);
        //const FieldT t4j = j * (z1 - z3);
        t4 = z1.sub(bn_env, z3, max_value, modulus);
        t4j = j.mul(bn_env, t4, res, buffer, modulus, inv);

        DevFp out0, out1, out2, out3;
        //out[i0] = t1 + t2;
        out0 = t1.add(bn_env, t2, max_value, modulus);
        out0.store(bn_env, out, i0);
        //out[i1] = t3 - t4j;
        out1 = t3.sub(bn_env, t4j, max_value, modulus);
        out1.store(bn_env, out, i1);
        //out[i2] = t1 - t2;
        out2 = t1.sub(bn_env, t2, max_value, modulus);
        out2.store(bn_env, out, i2);
        //out[i3] = t3 + t4j;
        out3 = t3.add(bn_env, t4j, max_value, modulus);
        out3.store(bn_env, out, i3);

        out_offset++;
        tw += 3;
    }

	for (unsigned int k = 1; k < stage_length; k++)
	{
		const unsigned i0  = out_offset;
		const unsigned i1  = out_offset + stage_length;
		const unsigned i2  = out_offset + stage_length*2;
		const unsigned i3  = out_offset + stage_length*3;

        DevFp z0, z1, z2, z3;
        DevFp out0, out1, out2, out3;
		//const FieldT z0  = out[i0];
        z0.load(bn_env, out, i0);
        out1.load(bn_env, out, i1);
        out2.load(bn_env, out, i2);
        out3.load(bn_env, out, i3);
        DevFp tw0, tw1, tw2;
        tw0.load(bn_env, twiddles, twiddle_offset + tw);
        tw1.load(bn_env, twiddles, twiddle_offset + tw+1);
        tw2.load(bn_env, twiddles, twiddle_offset + tw+2);
		//const FieldT z1  = out[i1] * twiddles[tw];
        z1 = out1.mul(bn_env, tw0, res, buffer, modulus, inv);
		//const FieldT z2  = out[i2] * twiddles[tw+1];
        z2 = out2.mul(bn_env, tw1, res, buffer, modulus, inv);
		//const FieldT z3  = out[i3] * twiddles[tw+2];
        z3 = out3.mul(bn_env, tw2, res, buffer, modulus, inv);

        DevFp t1, t2, t3, t4, t4j;
		//const FieldT t1  = z0 + z2;
        t1 = z0.add(bn_env, z2, max_value, modulus);
		//const FieldT t2  = z1 + z3;
        t2 = z1.add(bn_env, z3, max_value, modulus);
		//const FieldT t3  = z0 - z2;
        t3 = z0.sub(bn_env, z2, max_value, modulus);
		//const FieldT t4j = j * (z1 - z3);
        t4 = z1.sub(bn_env, z3, max_value, modulus);
        t4j = j.mul(bn_env, t4, res, buffer, modulus, inv);

		//out[i0] = t1 + t2;
        out0 = t1.add(bn_env, t2, max_value, modulus);
        out0.store(bn_env, out, i0);
		//out[i1] = t3 - t4j;
        out1 = t3.sub(bn_env, t4j, max_value, modulus);
        out1.store(bn_env, out, i1);
		//out[i2] = t1 - t2;
        out2 = t1.sub(bn_env, t2, max_value, modulus);
        out2.store(bn_env, out, i2);
		//out[i3] = t3 + t4j;
        out3 = t3.add(bn_env, t4j, max_value, modulus);
        out3.store(bn_env, out, i3);

		out_offset++;
		tw += 3;
	}

}

template<int BlockInstances>
__global__ void kernel_fft_internal(
    cgbn_error_report_t* report,
    Fp_model in,
    const int n,
    Fp_model twiddles,
    const int twiddles_len,
    const int twiddle_offset,
    const int *in_offsets,
    const int *out_offsets,
    const int *stage_lengths,
    const int *radixs,
    const int *strides,
    cgbn_mem_t<BITS>* max_value, 
    cgbn_mem_t<BITS>* modulus, 
    const uint64_t inv,
    Fp_model out){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int instance = tid / TPI;
    int local_instance = threadIdx.x / TPI;
    if(instance >= n) return;

    context_t bn_context(cgbn_report_monitor, report, instance);
    env_t          bn_env(bn_context.env<env_t>());  

    __shared__ uint32_t cache[BlockInstances * 3 * BITS/32];
    uint32_t *res = &cache[local_instance * 3 * BITS/32];
    __shared__ uint32_t cache_buffer[BlockInstances * BITS/32];
    uint32_t *buffer = &cache_buffer[local_instance * BITS/32];

    env_t::cgbn_t local_max_value, local_modulus;
    cgbn_load(bn_env, local_max_value, max_value);
    cgbn_load(bn_env, local_modulus, modulus);

    int in_offset = in_offsets[instance];
    int out_offset = out_offsets[instance];
    int stage_length = stage_lengths[instance]; 
    int radix = radixs[instance];
    int stride = strides[instance];
    if(stage_length == 1){
        for(int k = 0; k < radix; k++){
            DevFp dev_in;
            dev_in.load(bn_env, in, in_offset + k * stride);
            dev_in.store(bn_env, out, out_offset + k); 
        }
    }
    switch(radix)
    {
        case 2:
            dev_butterfly_2(bn_env, twiddles, twiddle_offset, 0, stage_length, out_offset, local_max_value, local_modulus, inv, res, buffer, out);
            break;
        case 4:
            dev_butterfly_4(bn_env, twiddles, twiddles_len, twiddle_offset, 0, stage_length, out_offset, local_max_value, local_modulus, inv, res, buffer, out);
            break;
        default:
            printf("error\n");
    }
}

void fft_internal(
    Fp_model in,
    const int n,
    Fp_model twiddles,
    const int twiddles_len,
    const int twiddle_offset,
    const int *in_offsets,
    const int *out_offsets,
    const int *stage_lengths,
    const int *radixs,
    const int *strides,
    cgbn_mem_t<BITS>* max_value, 
    cgbn_mem_t<BITS>* modulus, 
    const uint64_t inv,
    Fp_model out){
  //cgbn_error_report_t *report;
  //CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  cgbn_error_report_t *report = get_error_report();
    
  const int instances = 64;
  int threads = instances * TPI;
  int blocks = (n + instances - 1) / instances;
  kernel_fft_internal<instances><<<blocks, threads>>>(
    report,
    in, n,
    twiddles,
    twiddles_len,
    twiddle_offset,
    in_offsets,
    out_offsets,
    stage_lengths,
    radixs,
    strides,
    max_value,
    modulus,
    inv,
    out);
  //CUDA_CHECK(cudaDeviceSynchronize());
}


__global__ void kernel_fft_copy(
    Fp_model in,
    Fp_model out,
    const int *in_offsets,
    const int *out_offsets,
    const int *strides,
    const int n,
    const int radix){
    const int instance = threadIdx.x + blockIdx.x * blockDim.x;
    if(instance >= n) return;

    const int in_offset = in_offsets[instance];
    const int out_offset = out_offsets[instance];
    const int stride = strides[instance];
    using namespace BigInt256;

    for(int i = 0; i < radix; i++){
        Fp a;
        a.load((Int*)(in.mont_repr_data + in_offset + i * stride)); 
        a.store((Int*)(out.mont_repr_data + out_offset + i));
    }
}

void fft_copy(
    Fp_model in,
    Fp_model out,
    const int *in_offsets,
    const int *out_offsets,
    const int *strides,
    const int n,
    const int radix){
  //cgbn_error_report_t *report;
  //CUDA_CHECK(cgbn_error_report_alloc(&report)); 
    cgbn_error_report_t *report = get_error_report();
  const int instances = 64;
  int threads = instances * TPI;
  int blocks = (n + instances - 1) / instances;
  kernel_fft_copy<<<blocks, threads>>>(in, out, in_offsets, out_offsets, strides, n, radix);
  //CUDA_CHECK(cudaDeviceSynchronize());
}

template<int BlockInstances>
__global__ void kernel_butterfly_2_new(
        Fp_model out,
        Fp_model twiddles, 
        const int twiddle_offset,
        const int *strides, 
        const uint32_t stage_length, 
        const int* out_offsets, 
        const int n,
        cgbn_mem_t<BITS>* max_value, 
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv){
    const int instance = threadIdx.x + blockIdx.x * blockDim.x;
    if(instance >= n) return;

    using namespace BigInt256;
    Fp local_max_value, local_modulus;
    local_max_value.load((Int*)max_value);
    local_modulus.load((Int*)modulus);

    uint32_t out_offset = out_offsets[instance];
    uint32_t out_offset2 = out_offset + stage_length;
    Fp out1, out2;
    //FieldT t = out[out_offset2];
    out2.load((Int*)(out.mont_repr_data + out_offset2));
    out1.load((Int*)(out.mont_repr_data + out_offset));
    //out[out_offset2] = out[out_offset] - t;
    Fp tmp_out2 = out1.sub(out2, local_modulus);
    tmp_out2.store((Int*)(out.mont_repr_data + out_offset2));
    //out[out_offset] += t;
    Fp tmp_out = out1.add(out2, local_modulus); 
    tmp_out.store((Int*)(out.mont_repr_data + out_offset));
    out_offset2++;
    out_offset++;
    for (unsigned int k = 1; k < stage_length; k++){
        //FieldT t = twiddles[k] * out[out_offset2];
        out2.load((Int*)(out.mont_repr_data + out_offset2));
        out1.load((Int*)(out.mont_repr_data + out_offset));
        Fp twiddle;
        twiddle.load((Int*)(twiddles.mont_repr_data + twiddle_offset + k));
        Fp t = twiddle.mul(out2, local_modulus, inv);
        //out[out_offset2] = out[out_offset] - t;
        tmp_out2 = out1.sub(t, local_modulus);
        tmp_out2.store((Int*)(out.mont_repr_data + out_offset2));
        //out[out_offset] += t;
        tmp_out = out1.add(t, local_modulus);
        tmp_out.store((Int*)(out.mont_repr_data + out_offset));
        out_offset2++;
        out_offset++;
    }
}


void butterfly_2(
        Fp_model out,
        Fp_model twiddles, 
        const int twiddle_offset,
        const int *strides, 
        const uint32_t stage_length, 
        const int* out_offsets, 
        const int n,
        cgbn_mem_t<BITS>* max_value, 
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv){
    const int instances = 64;
    int threads = instances;
    int blocks = (n + instances - 1) / instances;
    kernel_butterfly_2_new<instances><<<blocks, threads>>>(
            out, 
            twiddles, 
            twiddle_offset,
            strides, 
            stage_length, 
            out_offsets, n, max_value, modulus, inv);
}

template<int BlockInstances>
__global__ void kernel_butterfly_4_new(
        Fp_model out,
        Fp_model twiddles, 
        const int twiddles_len,
        const int twiddle_offset,
        const int* strides, 
        const uint32_t stage_length, 
        const int* out_offsets, 
        const int n,
        cgbn_mem_t<BITS>* max_value, 
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv){
    const int instance = threadIdx.x + blockIdx.x * blockDim.x;
    if(instance >= n) return;
    using namespace BigInt256;
    Fp local_modulus;
    local_modulus.load((Int*)modulus);

    const int real_instance = instance / stage_length;
    const int stage_instance = instance % stage_length;
    uint32_t out_offset = out_offsets[real_instance] + stage_instance;
    //uint32_t stride = strides[instance];

    Fp j;
    j.load((Int*)(twiddles.mont_repr_data + twiddle_offset + twiddles_len-1));
    uint32_t tw = stage_instance * 3;
    /* Case twiddle == one */
    if(false){
		const unsigned i0  = out_offset;
        const unsigned i1  = out_offset + stage_length;
        const unsigned i2  = out_offset + stage_length*2;
        const unsigned i3  = out_offset + stage_length*3;

		Fp z0, z1, z2, z3;
        //const FieldT z0  = out[i0];
        z0.load((Int*)(out.mont_repr_data + i0));
        //const FieldT z1  = out[i1];
        z1.load((Int*)(out.mont_repr_data + i1));
        //const FieldT z2  = out[i2];
        z2.load((Int*)(out.mont_repr_data + i2));
        //const FieldT z3  = out[i3];
        z3.load((Int*)(out.mont_repr_data + i3));

        Fp t1, t2, t3, t4, t4j;
        //const FieldT t1  = z0 + z2;
        t1 = z0.add(z2, local_modulus);
        //const FieldT t2  = z1 + z3;
        t2 = z1.add(z3, local_modulus);
        //const FieldT t3  = z0 - z2;
        t3 = z0.sub(z2, local_modulus);
        //const FieldT t4j = j * (z1 - z3);
        t4 = z1.sub(z3, local_modulus);
        t4j = j.mul(t4, local_modulus, inv);

        Fp out0, out1, out2, out3;
        //out[i0] = t1 + t2;
        out0 = t1.add(t2, local_modulus);
        out0.store((Int*)(out.mont_repr_data + i0));
        //out[i1] = t3 - t4j;
        out1 = t3.sub(t4j, local_modulus);
        out1.store((Int*)(out.mont_repr_data + i1));
        //out[i2] = t1 - t2;
        out2 = t1.sub(t2, local_modulus);
        out2.store((Int*)(out.mont_repr_data + i2));
        //out[i3] = t3 + t4j;
        out3 = t3.add(t4j, local_modulus);
        out3.store((Int*)(out.mont_repr_data + i3));

        out_offset++;
        tw += 3;
    }

	//for (unsigned int k = 0; k < stage_length; k++)
	{
		const unsigned i0  = out_offset;
		const unsigned i1  = out_offset + stage_length;
		const unsigned i2  = out_offset + stage_length*2;
		const unsigned i3  = out_offset + stage_length*3;

        Fp z0, z1, z2, z3;
        Fp out0, out1, out2, out3;
		//const FieldT z0  = out[i0];
        z0.load((Int*)(out.mont_repr_data + i0));
        out1.load((Int*)(out.mont_repr_data + i1));
        out2.load((Int*)(out.mont_repr_data + i2));
        out3.load((Int*)(out.mont_repr_data + i3));
        Fp tw0, tw1, tw2;
        tw0.load((Int*)(twiddles.mont_repr_data + twiddle_offset + tw));
        tw1.load((Int*)(twiddles.mont_repr_data + twiddle_offset + tw+1));
        tw2.load((Int*)(twiddles.mont_repr_data + twiddle_offset + tw+2));
		//const FieldT z1  = out[i1] * twiddles[tw];
        z1 = out1.mul(tw0, local_modulus, inv);
		//const FieldT z2  = out[i2] * twiddles[tw+1];
        z2 = out2.mul(tw1, local_modulus, inv);
		//const FieldT z3  = out[i3] * twiddles[tw+2];
        z3 = out3.mul(tw2, local_modulus, inv);

        Fp t1, t2, t3, t4, t4j;
		//const FieldT t1  = z0 + z2;
        t1 = z0.add(z2, local_modulus);
		//const FieldT t2  = z1 + z3;
        t2 = z1.add(z3, local_modulus);
		//const FieldT t3  = z0 - z2;
        t3 = z0.sub(z2, local_modulus);
		//const FieldT t4j = j * (z1 - z3);
        t4 = z1.sub(z3, local_modulus);
        t4j = j.mul(t4, local_modulus, inv);

		//out[i0] = t1 + t2;
        out0 = t1.add(t2, local_modulus);
        out0.store((Int*)(out.mont_repr_data + i0));
		//out[i1] = t3 - t4j;
        out1 = t3.sub(t4j, local_modulus);
        out1.store((Int*)(out.mont_repr_data + i1));
		//out[i2] = t1 - t2;
        out2 = t1.sub(t2, local_modulus);
        out2.store((Int*)(out.mont_repr_data + i2));
		//out[i3] = t3 + t4j;
        out3 = t3.add(t4j, local_modulus);
        out3.store((Int*)(out.mont_repr_data + i3));

		//out_offset++;
		//tw += 3;
	}

}

void butterfly_4(
        Fp_model out,
        Fp_model twiddles, 
        const int twiddles_len,
        const int twiddle_offset,
        const int* strides, 
        const uint32_t stage_length, 
        const int* out_offsets, 
        const int n,
        cgbn_mem_t<BITS>* max_value, 
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv){
    const int instances = 64;
    int threads = instances;
    const int total_n = n * stage_length;
    int blocks = (total_n + instances - 1) / instances;
    kernel_butterfly_4_new<instances><<<blocks, threads>>>(
            out, 
            twiddles, 
            twiddles_len,
            twiddle_offset,
            strides, 
            stage_length, 
            out_offsets, total_n, max_value, modulus, inv);
    //CUDA_CHECK(cudaDeviceSynchronize());
}

template<int BlockInstances>
__global__ void kernel_multiply_by_coset_and_constant_new(
        Fp_model inputs,
        const int n,
        Fp_model g,
        Fp_model c, 
        Fp_model one,
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv,
        const int gmp_num_bits){
    const int instance = threadIdx.x + blockIdx.x * blockDim.x;
    if(instance >= n) return;

    using namespace BigInt256;
    Fp local_modulus;
    local_modulus.load((Int*)modulus);

    Fp dev_c, dev_g, dev_one;
    dev_c.load((Int*)c.mont_repr_data);
    dev_g.load((Int*)g.mont_repr_data);
    dev_one.load((Int*)one.mont_repr_data);
    if(instance == 0){
        Fp a0;
        a0.load((Int*)inputs.mont_repr_data);
        a0 = a0.mul(dev_c, local_modulus, inv);
        a0.store((Int*)(inputs.mont_repr_data));
    }else{
        Fp tmp = dev_g.power(dev_one, instance, local_modulus, inv, gmp_num_bits);
        Fp u = dev_c.mul(tmp, local_modulus, inv);
        Fp ai;
        ai.load((Int*)(inputs.mont_repr_data + instance));
        ai = ai.mul(u, local_modulus, inv);
        ai.store((Int*)(inputs.mont_repr_data + instance));
    }
}

void multiply_by_coset_and_constant(
        Fp_model inputs,
        const int n,
        Fp_model g,
        Fp_model c, 
        Fp_model one,
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv,
        const int gmp_num_bits){
    //CUDA_CHECK(cgbn_error_report_alloc(&report)); 
    const int instances = 64;
    int threads = instances;
    int blocks = (n + instances - 1) / instances;
    kernel_multiply_by_coset_and_constant_new<instances><<<blocks, threads>>>(inputs, n, g, c, one, modulus, inv, gmp_num_bits); 
    //CUDA_CHECK(cudaDeviceSynchronize());
}

template<int BlockInstances>
__global__ void kernel_calc_xor_new(
        Fp_model xor_results,
        const int n,
        const int offset,
        Fp_model g,
        Fp_model one,
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv,
        const int gmp_num_bits){
    const int instance = threadIdx.x + blockIdx.x * blockDim.x;
    if(instance >= n-1) return;

    using namespace BigInt256;
    Fp local_modulus;
    local_modulus.load((Int*)modulus);

    Fp dev_g, dev_one;
    dev_g.load((Int*)(g.mont_repr_data));
    dev_one.load((Int*)one.mont_repr_data);
    Fp xor_result = dev_g.power(dev_one, instance + offset, local_modulus, inv, gmp_num_bits);
    xor_result.store((Int*)(xor_results.mont_repr_data + instance+offset));
}

//xor_result = g^i
void calc_xor(
        Fp_model xor_results,
        const int n,
        const int offset,
        Fp_model g,
        Fp_model one,
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv,
        const int gmp_num_bits){
    const int instances = 64;
    int threads = instances;
    int blocks = (n + instances - 1) / instances;
    kernel_calc_xor_new<instances><<<blocks, threads>>>(xor_results, n, offset, g, one, modulus, inv, gmp_num_bits); 
}

template<int BlockInstances>
__global__ void kernel_multiply_new(
        Fp_model inputs,
        Fp_model xor_results,
        const int n,
        const int offset,
        Fp_model c, 
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv){
    const int instance = threadIdx.x + blockIdx.x * blockDim.x;
    if(instance >= n) return;

    using namespace BigInt256;
    Fp local_modulus;
    local_modulus.load((Int*)modulus);

    Fp dev_c;
    dev_c.load((Int*)(c.mont_repr_data));
    if(instance == 0){
        Fp a0;
        a0.load((Int*)inputs.mont_repr_data);
        a0 = a0.mul(dev_c, local_modulus, inv);
        a0.store((Int*)inputs.mont_repr_data);
    }else{
        Fp xor_result;
        xor_result.load((Int*)(xor_results.mont_repr_data + instance));
        Fp u = dev_c.mul(xor_result, local_modulus, inv);
        Fp ai;
        ai.load((Int*)(inputs.mont_repr_data + instance));
        ai = ai.mul(u, local_modulus, inv);
        ai.store((Int*)(inputs.mont_repr_data + instance));
    }
}

//inputs[0] *= c
//inputs[i] *= xor_results[i]
void multiply(
        Fp_model inputs,
        Fp_model xor_results,
        const int n,
        const int offset,
        Fp_model c, 
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv){
    const int instances = 64;
    int threads = instances;
    int blocks = (n + instances - 1) / instances;
    kernel_multiply_new<instances><<<blocks, threads>>>(inputs, xor_results, n, offset, c, modulus, inv); 
}


template<int BlockInstances>
__global__ void kernel_calc_H_new(
        Fp_model A,
        Fp_model B,
        Fp_model C,
        Fp_model out,
        Fp_model Z_inverse_at_coset,
        const int n,
        cgbn_mem_t<BITS>* max_value, 
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv){
    const int instance = threadIdx.x + blockIdx.x * blockDim.x;
    if(instance >= n) return;

    using namespace BigInt256;
    Fp local_modulus; 
    local_modulus.load((Int*)modulus);

    Fp dev_a, dev_b, dev_c, dev_out, dev_coset;
    dev_coset.load((Int*)Z_inverse_at_coset.mont_repr_data);
    dev_a.load((Int*)(A.mont_repr_data + instance));
    dev_b.load((Int*)(B.mont_repr_data + instance));
    dev_c.load((Int*)(C.mont_repr_data + instance));
    Fp tmp = dev_a.mul(dev_b, local_modulus, inv);
    dev_out = tmp.sub(dev_c, local_modulus);
    dev_out = dev_out.mul(dev_coset, local_modulus, inv);
    dev_out.store((Int*)(out.mont_repr_data + instance));
}

//out[i] = ((A[i] * B[i]) - C[i]) * Z_inverse_at_coset
void calc_H(
        Fp_model A,
        Fp_model B,
        Fp_model C,
        Fp_model out,
        Fp_model Z_inverse_at_coset,
        const int n,
        cgbn_mem_t<BITS>* max_value, 
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv){
    const int instances = 64;
    int threads = instances;
    int blocks = (n + instances - 1) / instances;
    kernel_calc_H_new<instances><<<blocks, threads>>>(A, B, C, out, Z_inverse_at_coset, n, max_value, modulus, inv); 
}


void init_error_report(){
  get_error_report();
}

__global__ void kernel_warmup(){
  int sum = 0;
  for(int i = 0; i < 1000; i++){
    sum += i;
  }
}
void warm_up(){
  //kernel_warmup<<<1, 1>>>();
  //cuda_check(cudadevicesynchronize());
  cudaSetDevice(0);
  cudaFree(0);
}

} //gpu
