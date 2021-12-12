#include "cgbn_alt_bn128_g2.h"
#include "cgbn_alt_bn128_g2.cuh"

#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>

#include "cgbn/cgbn.h"
#include "utility/cpu_support.h"
#include "utility/cpu_simple_bn_math.h"
#include "utility/gpu_support.h"

namespace gpu{

alt_bn128_g2::alt_bn128_g2(const int count){
  init(count);
}
void alt_bn128_g2::init(const int count){
  x.init(count);
  y.init(count);
  z.init(count);
}
void alt_bn128_g2::init_host(const int count){
  x.init_host(count);
  y.init_host(count);
  z.init_host(count);
}
void alt_bn128_g2::resize(const int count){
  x.resize(count);
  y.resize(count);
  z.resize(count);
}
void alt_bn128_g2::resize_host(const int count){
  x.resize_host(count);
  y.resize_host(count);
  z.resize_host(count);
}
void alt_bn128_g2::release(){
  x.release();
  y.release();
  z.release();
}
void alt_bn128_g2::release_host(){
  x.release_host();
  y.release_host();
  z.release_host();
}
void alt_bn128_g2::copy_from_cpu(const alt_bn128_g2& host){
  x.copy_from_cpu(host.x);
  y.copy_from_cpu(host.y);
  z.copy_from_cpu(host.z);
}
void alt_bn128_g2::copy_from_gpu(const alt_bn128_g2& gpu){
  x.copy_from_gpu(gpu.x);
  y.copy_from_gpu(gpu.y);
  z.copy_from_gpu(gpu.z);
}
void alt_bn128_g2::copy_to_cpu(alt_bn128_g2& host){
  host.x.copy_to_cpu(x);
  host.y.copy_to_cpu(y);
  host.z.copy_to_cpu(z);
}
void alt_bn128_g2::clear(CudaStream stream ){
  this->x.clear(stream);
  this->y.clear(stream);
  this->z.clear(stream);
}

__global__ void kernel_alt_bn128_g2_reduce_sum_pre(
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

__global__ void kernel_alt_bn128_g1_reduce_sum_after(
    cgbn_error_report_t* report, 
    alt_bn128_g2 partial, 
    const int n, 
    cgbn_mem_t<BITS>* max_value,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    Fp_model non_residue
    ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int local_instance = threadIdx.x / TPI;
  int instance = tid / TPI;
  if(instance >= n) return;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  

  __shared__ uint32_t cache_res[24*32];
  __shared__ uint32_t cache_buffer[8*32];
  uint32_t* res = &cache_res[local_instance*24];
  uint32_t* buffer = &cache_buffer[local_instance * 8]; 
  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);
  DevFp dev_non_residue;
  dev_non_residue.load(bn_env, non_residue, 0);

  DevAltBn128G2 result;
  result.load(bn_env, partial, instance);
  DevAltBn128G2 dev_b;
  dev_b.load(bn_env, partial, instance + n);
  dev_alt_bn128_g2_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv, dev_non_residue);
  result.store(bn_env, partial, instance);
}

__global__ void kernel_alt_bn128_g2_reduce_sum(
    cgbn_error_report_t* report, 
    const int range_id,
    const int range_offset,
    alt_bn128_g2 values, 
    Fp_model scalars,
    const size_t *index_it,
    alt_bn128_g2 partial, 
    const int ranges_size, 
    const uint32_t* firsts,
    const uint32_t* seconds,
    const char* flags,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g2 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    Fp_model non_residue
    ){
  int local_instance = threadIdx.x / TPI;//0~63
  int local_instances = blockDim.x / TPI;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int instance = tid / TPI;
  int first = firsts[blockIdx.y];
  int second = seconds[blockIdx.y];
  int reduce_depth = second - first;//30130
  int offset = blockIdx.y * gridDim.x * local_instances;

  context_t bn_context(cgbn_report_monitor, report, offset + instance);
  env_t          bn_env(bn_context.env<env_t>());  

  __shared__ uint32_t cache_res[32 * 24];
  uint32_t *res = &cache_res[local_instance * 24];
  __shared__ uint32_t cache_buffer[256];
  uint32_t *buffer = &cache_buffer[local_instance * 8];
  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);
  DevFp dev_non_residue;
  dev_non_residue.load(bn_env, non_residue, 0);

  DevAltBn128G2 result;
  result.load(bn_env, t_zero, 0);
  for(int i = first + instance; i < first + reduce_depth; i+= gridDim.x * local_instances){
    const int j = index_it[i];
    if(flags[j] == 1){
      DevAltBn128G2 dev_b;
      dev_b.load(bn_env, values, i);
      dev_alt_bn128_g2_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv, dev_non_residue);
    }
  }
  result.store(bn_env, partial, offset + instance);
}

int alt_bn128_g2_reduce_sum_one_range(
    alt_bn128_g2 values, 
    Fp_model scalars, 
    const size_t *index_it,
    alt_bn128_g2 partial, 
    uint32_t *counters,
    char* flags,
    const uint32_t ranges_size,
    const uint32_t *firsts,
    const uint32_t *seconds,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g2 t_zero,
    Fp_model field_zero,
    Fp_model field_one,
    Fp_model non_residue,
    char *density,
    cgbn_mem_t<BITS>* bn_exponents,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    cgbn_mem_t<BITS>* field_modulus, const uint64_t field_inv,
    const int max_reduce_depth){
  cgbn_error_report_t *report = get_error_report();

  uint32_t threads = 512;
  const int local_instances = 64 * BlockDepth;
  uint32_t block_x =  (max_reduce_depth + local_instances - 1) / local_instances;
  dim3 blocks(block_x, ranges_size, 1);
  kernel_alt_bn128_g2_reduce_sum_pre<<<blocks, threads>>>(report, scalars, index_it, counters, flags, ranges_size, firsts, seconds, max_value, field_zero, field_one, density, bn_exponents, inv, field_modulus, field_inv);
  CUDA_CHECK(cudaDeviceSynchronize());

  const int blocks_per_range = REDUCE_BLOCKS_PER_RANGE;
  const int threads_per_block = TPI * INSTANCES_PER_BLOCK;
  kernel_alt_bn128_g2_reduce_sum<<<dim3(blocks_per_range, ranges_size, 1), threads_per_block>>>(report, 0, 0, values, scalars, index_it, partial, ranges_size, firsts, seconds, flags, max_value, t_zero, modulus, inv, non_residue);
  CUDA_CHECK(cudaDeviceSynchronize());

  int n = blocks_per_range * INSTANCES_PER_BLOCK * ranges_size;
  while(n>=2){
    int half_n = n / 2;
    int blocks = (half_n + INSTANCES_PER_BLOCK-1) / INSTANCES_PER_BLOCK;
    kernel_alt_bn128_g1_reduce_sum_after<<<blocks, threads_per_block>>>(report, partial, half_n, max_value, modulus, inv, non_residue);
    CUDA_CHECK(cudaDeviceSynchronize());
    n /= 2;
  }
  return 0;
}

template<int BlockSize, int BlockNum>
__global__ void test_g2(
    cgbn_error_report_t* report, 
    alt_bn128_g2 data, 
    alt_bn128_g2 out, 
    int n,
    cgbn_mem_t<BITS>* max_value,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    Fp_model non_residue
    ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int instance = tid / TPI;
  int local_instance = threadIdx.x / TPI;
  if(instance >= n) return;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  __shared__ uint32_t cache_buffer[BlockSize*8];
  __shared__ uint32_t cache_res[BlockSize*24];
  uint32_t *buffer = &cache_buffer[local_instance * 8];
  uint32_t *res = &cache_res[local_instance * 24];
  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);
  DevFp dev_non_residue;
  dev_non_residue.load(bn_env, non_residue, 0);

  DevAltBn128G2 a;
  if(instance < n)
    a.load(bn_env, data, instance);
  for(int i = instance + BlockNum*BlockSize; i < n; i+=BlockNum*BlockSize){
    DevAltBn128G2 b;
    b.load(bn_env, data, i);
    dev_alt_bn128_g2_add(bn_env, a, b, &a, res, buffer, local_max_value, local_modulus, inv, dev_non_residue);
  }
  if(instance < n)
    a.store(bn_env, out, instance);
}
void alt_bn128_g2_reduce_sum2(
    alt_bn128_g2 data, 
    alt_bn128_g2 out, 
    const uint32_t n,
    cgbn_mem_t<BITS>* max_value,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv, 
    Fp_model non_residue, 
    CudaStream stream){
  cgbn_error_report_t *report = get_error_report();
  //uint32_t threads = 512;
  //uint32_t local_instances = threads / TPI;//64
  //uint32_t instances = std::min(n, (uint32_t)(local_instances * BlockDepth));
  //uint32_t blocks = (n + instances - 1) / instances;
  //kernel_alt_bn128_g1_reduce_sum2<<<blocks, threads>>>(report, data, out, n, max_value, modulus, inv);

  test_g2<32, 128><<<128, 256>>>(report, data, out, n-1, max_value, modulus, inv, non_residue);
  CUDA_CHECK(cudaDeviceSynchronize());
  int tmp_n = 32*128; 
  test_g2<32, 16><<<16, 256>>>(report, out, data, tmp_n, max_value, modulus, inv, non_residue);
  CUDA_CHECK(cudaDeviceSynchronize());
  test_g2<16, 4><<<4, 128>>>(report, data, out, 32*16, max_value, modulus, inv, non_residue);
  CUDA_CHECK(cudaDeviceSynchronize());
  test_g2<8, 1><<<1, 64>>>(report, out, data, 64, max_value, modulus, inv, non_residue);
  CUDA_CHECK(cudaDeviceSynchronize());
  test_g2<1, 1><<<1, 8>>>(report, data, out, 8, max_value, modulus, inv, non_residue);
  CUDA_CHECK(cudaDeviceSynchronize());
}
}//namespace gpu
