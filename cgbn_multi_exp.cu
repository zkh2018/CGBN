#include "cgbn_alt_bn128_g1.h"
#include "cgbn_alt_bn128_g1.cuh"
#include "cgbn_multi_exp.h"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>

#include <time.h>
#include <vector>
#include <algorithm>


namespace gpu{
inline __device__ size_t dev_get_id(const size_t c, const size_t bitno, uint64_t* data){
  const uint64_t one = 1;
  const uint64_t mask = (one << c) - one;
  const size_t limb_num_bits = 64;//sizeof(mp_limb_t) * 8;

  const size_t part = bitno / limb_num_bits;
  const size_t bit = bitno % limb_num_bits;
  size_t id = (data[part] & (mask << bit)) >> bit;
  //const mp_limb_t next_data = (bit + c >= limb_num_bits && part < 3) ? bn_exponents[i].data[part+1] : 0;
  //id |= (next_data & (mask >> (limb_num_bits - bit))) << (limb_num_bits - bit);
  id |= (((bit + c >= limb_num_bits && part < 3) ? data[part+1] : 0) & (mask >> (limb_num_bits - bit))) << (limb_num_bits - bit);

  return id;
}

__global__ void kernel_get_instances_and_bucket_id(
    const int *starts, const int *ends, const int bucket_num,
    int *instances, int *instance_bucket_ids,
    const int left, const int right){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= bucket_num) return;

  int bucket_size = ends[tid] - starts[tid];
  if(bucket_size >= left && bucket_size < right){
    int half_bucket_size = bucket_size / 2;
    int i = atomicAdd(instances, half_bucket_size);
    for(int j = 0; j < half_bucket_size; j++){
      instance_bucket_ids[i + j] = tid;
    }
  }
}

__global__ void kernel_update_ends(const int *instance_bucket_ids, const int *starts, int* ends, const int total_instances){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for(int instance = tid; instance < total_instances; instance += gridDim.x * blockDim.x){
    int bucket_id = instance_bucket_ids[instance];
    int start = starts[bucket_id];
    int bucket_size = ends[bucket_id] - start;
    int half_bucket_size = bucket_size / 2;
    int bucket_instance = instance % half_bucket_size;
    if(bucket_instance == 0)
      ends[bucket_id] = start + half_bucket_size;
  }
}

template<int left, int right>
__global__ void kernel_get_bucket(
    int *starts, int *ends,
    int *out_starts, int *out_ends, int* out_ids,
    int *num,
    int bucket_num
    ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for(int i = tid; i < bucket_num; i+=gridDim.x * blockDim.x){
    int start = starts[i];
    int end = ends[i];
    int len = end-start;
    if(len > left && len <= right){
      int index = atomicAdd(num, 1);
      out_starts[index] = start;
      out_ends[index] = end;
      out_ids[index] = i;
    }
  }
}

//with_density = false
__global__ void kernel_bucket_counter(
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    int* bucket_counters){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for(int i = tid; i < data_length; i+= gridDim.x * blockDim.x){
    size_t id = dev_get_id(c, k*c, (uint64_t*)bn_exponents[i]._limbs);
    if(id != 0){
      atomicAdd(&bucket_counters[id], 1);
    }
  }
}
//with_density = true
__global__ void kernel_bucket_counter(
    const char* density,
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    int* bucket_counters){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for(int i = tid; i < data_length; i+= gridDim.x * blockDim.x){
    if(density[i]){
      size_t id = dev_get_id(c, k*c, (uint64_t*)bn_exponents[i]._limbs);
      if(id != 0){
        atomicAdd(&bucket_counters[id], 1);
      }
    }
  }
}

//with_density = false
__global__ void kernel_split_to_bucket(
    alt_bn128_g1 data, 
    alt_bn128_g1 out, 
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    int *indexs, int*tids){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < data_length; i+= gridDim.x * blockDim.x){
    size_t id = dev_get_id(c, k*c, (uint64_t*)bn_exponents[i]._limbs);
    if(id != 0){
      int index = atomicAdd(&indexs[id], 1);
      tids[index] = i;
    }
  }
}
//with_density = true
__global__ void kernel_split_to_bucket(
    alt_bn128_g1 data, 
    alt_bn128_g1 out, 
    const char* density,
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    int *indexs, int*tids){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < data_length; i+= gridDim.x * blockDim.x){
    if(density[i]){
      size_t id = dev_get_id(c, k*c, (uint64_t*)bn_exponents[i]._limbs);
      if(id != 0){
        int index = atomicAdd(&indexs[id], 1);
        tids[index] = i;
      }
    }
  }
}

__global__ void kernel_split_to_bucket(
    alt_bn128_g1 data, 
    alt_bn128_g1 out, 
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    int *starts, int *indexs, int*tids, int *ids){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < data_length; i+= gridDim.x * blockDim.x){
    size_t id = dev_get_id(c, k*c, (uint64_t*)bn_exponents[i]._limbs);
    if(id != 0){
      int start = starts[id];
      int end = indexs[id];
      for(int j = start; j<end; j++){
        if(i == tids[j]){
          ids[j] = j;
          break;
        }
      }
    }
  }
}
__global__ void kernel_split_to_bucket(
    alt_bn128_g1 data, 
    alt_bn128_g1 out, 
    const char* density,
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    int *starts, int *indexs, int*tids, int *ids){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < data_length; i+= gridDim.x * blockDim.x){
    if(density[i]){
      size_t id = dev_get_id(c, k*c, (uint64_t*)bn_exponents[i]._limbs);
      if(id != 0){
        int start = starts[id];
        int end = indexs[id];
        for(int j = start; j<end; j++){
          if(i == tids[j]){
            ids[j] = j;
            break;
          }
        }
      }
    }
  }
}

__global__ void kernel_split_to_bucket(
    alt_bn128_g1 data, 
    alt_bn128_g1 out, 
    const int c, const int k,
    const int data_length,
    int *tids){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < data_length; i+= gridDim.x * blockDim.x){
    int real_i = tids[i];
    //size_t bucket_id = dev_get_id(c, k*c, (uint64_t*)bn_exponents[real_i]._limbs);
    int index = i;// + starts[bucket_id];
//#pragma unroll
    for(int j = 0; j < 4; j++){
      ((uint64_t*)out.x.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.x.mont_repr_data[real_i]._limbs)[j];
      ((uint64_t*)out.y.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.y.mont_repr_data[real_i]._limbs)[j];
      ((uint64_t*)out.z.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.z.mont_repr_data[real_i]._limbs)[j];
    }
  }
}

__global__ void kernel_split_to_bucket(
    alt_bn128_g1 data, 
    alt_bn128_g1 out, 
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    int *indexs){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < data_length; i+= gridDim.x * blockDim.x){
    size_t id = dev_get_id(c, k*c, (uint64_t*)bn_exponents[i]._limbs);
    if(id != 0){
      int index = atomicAdd(&indexs[id], 1);
#pragma unroll
      for(int j = 0; j < 4; j++){
        ((uint64_t*)out.x.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.x.mont_repr_data[i]._limbs)[j];
        ((uint64_t*)out.y.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.y.mont_repr_data[i]._limbs)[j];
        ((uint64_t*)out.z.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.z.mont_repr_data[i]._limbs)[j];
      }
    }
  }
}
__global__ void kernel_split_to_bucket(
    alt_bn128_g1 data, 
    alt_bn128_g1 out, 
    const char* density,
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    int *indexs){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < data_length; i+= gridDim.x * blockDim.x){
    if(density[i]){
      size_t id = dev_get_id(c, k*c, (uint64_t*)bn_exponents[i]._limbs);
      if(id != 0){
        int index = atomicAdd(&indexs[id], 1);
#pragma unroll
        for(int j = 0; j < 4; j++){
          ((uint64_t*)out.x.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.x.mont_repr_data[i]._limbs)[j];
          ((uint64_t*)out.y.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.y.mont_repr_data[i]._limbs)[j];
          ((uint64_t*)out.z.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.z.mont_repr_data[i]._limbs)[j];
        }
      }
    }
  }
}
template<int BS, int Offset>
__global__ void kernel_bucket_reduce_sum_one_instance(
    cgbn_error_report_t* report, 
    alt_bn128_g1 data,
    int* starts, int* ends,
    const int bucket_num,
    alt_bn128_g1 buckets,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv){
  int tid = threadIdx.x;
  int bucket_id = blockIdx.x;
  int instance = tid / TPI;
  int local_instance = instance;//threadIdx.x / TPI;
  int start = starts[bucket_id];
  int end = ends[bucket_id];
  int n = end - start;
  //if(n <= 1) return;

  context_t bn_context(cgbn_report_monitor, report, bucket_id * BS + instance);
  env_t          bn_env(bn_context.env<env_t>());  
  __shared__ uint32_t cache_res[BS * 3 * BITS/32];
  uint32_t *res = &cache_res[local_instance * BITS/32 * 3];
  __shared__ uint32_t cache_buffer[BS * BITS/32];
  uint32_t *buffer = &cache_buffer[local_instance * BITS/32];

  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);

  DevAltBn128G1 result;
  result.load(bn_env, t_zero, 0);
  for(int i = start + instance; i < end; i+=BS){
    DevAltBn128G1 other;
    other.load(bn_env, data, i);
    dev_alt_bn128_g1_add(bn_env, result, other, &result, res, buffer, local_max_value, local_modulus, inv);
  }
  if(instance < n)
	  result.store(bn_env, data, start + instance);
  __syncthreads();
  if(instance == 0){
	 for(int i = 1; i < BS && i < n; i++){
		 DevAltBn128G1 other;
		 other.load(bn_env, data, start + i);
		 dev_alt_bn128_g1_add(bn_env, result, other, &result, res, buffer, local_max_value, local_modulus, inv);
	 } 
	 result.store(bn_env, buckets, bucket_id * Offset);
  }
  //result.store(bn_env, data, start);
}

template<int BS>
__global__ void kernel_bucket_reduce_sum(
    cgbn_error_report_t* report, 
    alt_bn128_g1 data,
    int* starts, int* ends,
    alt_bn128_g1 buckets,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv){
  int bid = blockIdx.x;

  int start = starts[bid];
  int n = ends[bid] - start;
  if(n <= 0) return;

  int tid = threadIdx.x;
  int instance = tid / TPI;
  int instances = blockDim.x / TPI;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  __shared__ uint32_t cache_res[BS * 3 * BITS/32];
  uint32_t *res = &cache_res[instance * BITS/32 * 3];
  __shared__ uint32_t cache_buffer[BS * BITS/32];
  uint32_t *buffer = &cache_buffer[instance * BITS/32];

  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);

  DevAltBn128G1 result;
  result.load(bn_env, t_zero, 0);
  for(int i = instance; i < n; i += instances){
    int j = start + i;
    DevAltBn128G1 dev_b;
    if(i == instance){
      result.load(bn_env, data, j);
    }else{
      dev_b.load(bn_env, data, j);
      dev_alt_bn128_g1_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv);
    }
  }
  result.store(bn_env, buckets, bid * instances + instance);
  __syncthreads();
  if(instance == 0){
    for(int i = 1; i < instances; i++){
      DevAltBn128G1 dev_b;
      dev_b.load(bn_env, buckets, bid * instances + i);
      dev_alt_bn128_g1_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv);
    }
    result.store(bn_env, buckets, bid * instances);
  }
}


template<int Offset>
__global__ void kernel_one_bucket_reduce_sum(
    alt_bn128_g1 data,
    int* starts, int* ids,
    alt_bn128_g1 buckets,
    const int bucket_num){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for(int i = tid; i < bucket_num; i+=gridDim.x * blockDim.x){
    int bid = ids[i];
    int start = starts[i];
    for(int j = 0; j < 4; j++){
      ((uint64_t*)buckets.x.mont_repr_data[bid * Offset]._limbs)[j] = ((uint64_t*)data.x.mont_repr_data[start]._limbs)[j];
      ((uint64_t*)buckets.y.mont_repr_data[bid * Offset]._limbs)[j] = ((uint64_t*)data.y.mont_repr_data[start]._limbs)[j];
      ((uint64_t*)buckets.z.mont_repr_data[bid * Offset]._limbs)[j] = ((uint64_t*)data.z.mont_repr_data[start]._limbs)[j];
    }
  }

}
template<int BS, int Offset>
__global__ void kernel_small_bucket_reduce_sum(
    cgbn_error_report_t* report, 
    alt_bn128_g1 data,
    int* starts, int* ends, int* ids,
    alt_bn128_g1 buckets,
    const int bucket_num,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int instance = tid / TPI;
  if(instance >= bucket_num) return;
  int local_instance = threadIdx.x / TPI;
  int real_bid = ids[instance];
  int start = starts[instance];
  int n = ends[instance] - start;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  __shared__ uint32_t cache_res[BS * 24];
  uint32_t *res = &cache_res[local_instance * 24];
  __shared__ uint32_t cache_buffer[BS * 8];
  uint32_t *buffer = &cache_buffer[local_instance * 8];

  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);

  DevAltBn128G1 result;
  result.load(bn_env, data, start);
  for(int i = 1; i < n; i++){
    int j = start + i;
    DevAltBn128G1 dev_b;
    dev_b.load(bn_env, data, j);
    dev_alt_bn128_g1_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv);
  }
  result.store(bn_env, buckets, real_bid * Offset);
}

template<int BS, int Offset>
__global__ void kernel_medium_bucket_reduce_sum(
    cgbn_error_report_t* report, 
    alt_bn128_g1 data,
    int* starts, int* ends, int* ids,
    alt_bn128_g1 buckets,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv){
  int bid = blockIdx.x;
  int real_bid = ids[bid];

  int start = starts[bid];
  int n = ends[bid] - start;
  if(n <= 0) return;

  int tid = threadIdx.x;
  int instance = tid / TPI;
  int instances = blockDim.x / TPI;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  __shared__ uint32_t cache_res[BS * 24];
  uint32_t *res = &cache_res[instance * 24];
  __shared__ uint32_t cache_buffer[BS * 8];
  uint32_t *buffer = &cache_buffer[instance * 8];

  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);

  DevAltBn128G1 result;
  result.load(bn_env, t_zero, 0);
  for(int i = instance; i < n; i += instances){
    int j = start + i;
    DevAltBn128G1 dev_b;
    if(i == instance){
      result.load(bn_env, data, j);
    }else{
      dev_b.load(bn_env, data, j);
      dev_alt_bn128_g1_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv);
    }
  }
  result.store(bn_env, buckets, real_bid * Offset + instance);
  __syncthreads();
  if(instance == 0){
    for(int i = 1; i < instances && i < n; i++){
      DevAltBn128G1 dev_b;
      dev_b.load(bn_env, buckets, real_bid * Offset + i);
      dev_alt_bn128_g1_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv);
    }
    result.store(bn_env, buckets, real_bid * Offset);
  }
}

__global__ void kernel_reverse(alt_bn128_g1 data, alt_bn128_g1 out, int n, int offset){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += gridDim.x * blockDim.x){
    int in_i = i * offset;
    int out_i = n - i - 1;
#pragma unroll
    for(int j = 0; j < 4; j++){
      ((uint64_t*)out.x.mont_repr_data[out_i]._limbs)[j] = ((uint64_t*)data.x.mont_repr_data[in_i]._limbs)[j];
      ((uint64_t*)out.y.mont_repr_data[out_i]._limbs)[j] = ((uint64_t*)data.y.mont_repr_data[in_i]._limbs)[j];
      ((uint64_t*)out.z.mont_repr_data[out_i]._limbs)[j] = ((uint64_t*)data.z.mont_repr_data[in_i]._limbs)[j];
    }
  }
}

__device__ void dev_prefix_sum(env_t& bn_env, alt_bn128_g1 in, const int n, const int Instances, uint32_t* res, uint32_t* buffer, env_t::cgbn_t local_max_value, env_t::cgbn_t local_modulus, const int inv){
  int tid = threadIdx.x;
  int local_instance = tid / TPI;
  int local_instances = Instances;
  int offset = blockIdx.x * local_instances;

  for(int stride = 1; stride <= Instances; stride *= 2){
    __syncthreads();
    int index = (local_instance+1)*stride*2 - 1; 
    if(index < Instances && index < n){
      DevAltBn128G1 dev_a, dev_b;
      dev_a.load(bn_env, in, offset + index);
      dev_b.load(bn_env, in, offset + index - stride);
      if(offset + index == 1){
        dev_a.x.print(bn_env, buffer);
        dev_b.x.print(bn_env, buffer);
      }
      dev_alt_bn128_g1_add(bn_env, dev_a, dev_b, &dev_a, res, buffer, local_max_value, local_modulus, inv);
      dev_a.store(bn_env, in, offset + index);
      if(offset + index == 1){
        dev_a.x.print(bn_env, buffer);
      }
    }
    __syncthreads();
  }
  for (unsigned int stride = Instances/2; stride > 0 ; stride /= 2) {
    __syncthreads();
    int index = (local_instance+1)*stride*2 - 1;
    if(index + stride < Instances && index + stride < n){
      DevAltBn128G1 dev_a, dev_b;
      dev_a.load(bn_env, in, offset + index + stride);
      dev_b.load(bn_env, in, offset + index);
      dev_alt_bn128_g1_add(bn_env, dev_a, dev_b, &dev_a, res, buffer, local_max_value, local_modulus, inv);
      dev_a.store(bn_env, in, offset + index + stride);
    }
  }
  __syncthreads();
}

template<int Instances, int RealInstances, bool SaveBlockSum>
__global__ void kernel_prefix_sum(
    cgbn_error_report_t* report, 
    alt_bn128_g1 data, 
    alt_bn128_g1 block_sums, 
    const int n,
    cgbn_mem_t<BITS>* max_value,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int instance = tid / TPI;
  int local_instance = threadIdx.x / TPI;
  int local_instances = Instances;
  if(instance >= n) return;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  __shared__ uint32_t cache_res[RealInstances * 24];
  uint32_t *res = &cache_res[local_instance * 24];
  __shared__ uint32_t cache_buffer[RealInstances * 8];
  uint32_t *buffer = &cache_buffer[local_instance * 8];
  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);
  int offset = blockIdx.x * local_instances;
  for(int stride = 1; stride <= RealInstances; stride *= 2){
    __syncthreads();
    int index = (local_instance+1)*stride*2 - 1; 
    if(index < Instances && offset + index < n){
      DevAltBn128G1 dev_a, dev_b;
      dev_a.load(bn_env, data, offset + index);
      dev_b.load(bn_env, data, offset + index - stride);
      dev_alt_bn128_g1_add(bn_env, dev_a, dev_b, &dev_a, res, buffer, local_max_value, local_modulus, inv);
      dev_a.store(bn_env, data, offset + index);
    }
    __syncthreads();
  }
  for (unsigned int stride = (Instances >> 1); stride > 0 ; stride>>=1) {
    __syncthreads();
    int index = (local_instance+1)*stride*2 - 1;
    if(index + stride < Instances && offset + index + stride < n){
      DevAltBn128G1 dev_a, dev_b;
      dev_a.load(bn_env, data, offset + index + stride);
      dev_b.load(bn_env, data, offset + index);
      dev_alt_bn128_g1_add(bn_env, dev_a, dev_b, &dev_a, res, buffer, local_max_value, local_modulus, inv);
      dev_a.store(bn_env, data, offset + index + stride);
    }
  }
  __syncthreads();
  if(SaveBlockSum && local_instance == 0){
    DevAltBn128G1 dev_a;
    dev_a.load(bn_env, data, blockIdx.x * local_instances + local_instances-1);
    dev_a.store(bn_env, block_sums, blockIdx.x);
  }
}

template<int Instances>
__global__ void kernel_add_block_sum(
    cgbn_error_report_t* report, 
    alt_bn128_g1 data, 
    alt_bn128_g1 block_sums, 
    const int n,
    cgbn_mem_t<BITS>* max_value,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv){
  const int instances = Instances;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int instance = i / TPI;
  int local_instance = threadIdx.x / TPI;
  if(instances + instance >= n) return;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  __shared__ uint32_t cache_res[Instances * 24];
  uint32_t *res = &cache_res[local_instance * 24];
  __shared__ uint32_t cache_buffer[Instances * 8];
  uint32_t *buffer = &cache_buffer[local_instance * 8];
  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);

  DevAltBn128G1 dev_block_sum, dev_a;
  dev_block_sum.load(bn_env, block_sums, blockIdx.x);
  dev_a.load(bn_env, data, instance + instances);//offset = instances
  dev_alt_bn128_g1_add(bn_env, dev_a, dev_block_sum, &dev_a, res, buffer, local_max_value, local_modulus, inv);
  dev_a.store(bn_env, data, instance + instances);
}

void bucket_counter(
    const bool with_density,
    const char* density,
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    const int bucket_nums,
    int* bucket_counters,
    CudaStream stream){
  int threads = 512;
  int blocks = (data_length + threads-1) / threads;
  if(with_density){
    kernel_bucket_counter<<<blocks, threads>>>(density, bn_exponents, c, k, data_length, bucket_counters);
  }else{
    kernel_bucket_counter<<<blocks, threads>>>(bn_exponents, c, k, data_length, bucket_counters);
  }
  //CUDA_CHECK(cudaDeviceSynchronize());
}

void prefix_sum(const int *in, int *out, const int n, CudaStream stream){
  //thrust::exclusive_scan(thrust::cuda::par.on(stream), in, in + n, out);
  thrust::exclusive_scan(thrust::device, in, in + n, out);
  //CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void kernel_get_bid_and_counter(
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    const int bucket_num,
    int* bucket_counters,
    int* bucket_ids, int* value_ids){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for(int i = tid; i < data_length; i+= gridDim.x * blockDim.x){
    size_t id = dev_get_id(c, k*c, (uint64_t*)bn_exponents[i]._limbs);
    if(id >= bucket_num) printf("error bucket_id\n");
    if(id != 0){
      atomicAdd(&bucket_counters[id], 1);
      bucket_ids[i] = id;
      value_ids[i] = i;
    }else{
        bucket_ids[i] = bucket_num+1;
        value_ids[i] = i;
    }
  }
}
__global__ void kernel_calc_bucket_index(
		const int *bucket_ids, 
		const int *starts, 
		const int *ends,
		const int bucket_num, int* bucket_index){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for(int i = tid; i < bucket_num; i+= gridDim.x * blockDim.x){
	int start = starts[i]; 
	int end = ends[i];
	for(int j = start; j < end; j++){
		bucket_index[j] = j - start;
	}
  }
}

__global__ void kernel_split_to_bucket(
		alt_bn128_g1 data,
		alt_bn128_g1 buckets,
		const int data_length,
		const int bucket_num,
		const int* starts,
		const int* value_ids,
		const int* bucket_ids,
		const int *bucket_index){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= data_length) return;
	int bucket_id = bucket_ids[tid];
	if(bucket_id > 0 && bucket_id < bucket_num){
		int src_i = value_ids[tid];
		int dst_i = tid;//starts[bucket_id] + bucket_index[tid];
#pragma unroll
		for(int j = 0; j < 4; j++){
			((int64_t*)(buckets.x.mont_repr_data[dst_i]._limbs))[j] = ((int64_t*)(data.x.mont_repr_data[src_i]._limbs))[j];
			((int64_t*)(buckets.y.mont_repr_data[dst_i]._limbs))[j] = ((int64_t*)(data.y.mont_repr_data[src_i]._limbs))[j];
			((int64_t*)(buckets.z.mont_repr_data[dst_i]._limbs))[j] = ((int64_t*)(data.z.mont_repr_data[src_i]._limbs))[j];
		}

	}
}

void split_to_bucket(
    alt_bn128_g1 data, 
    alt_bn128_g1 out, 
    const bool with_density,
    const char* density,
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    int *starts,
    int *indexs, CudaStream stream){
  int threads = 512;
  int blocks = (data_length + threads-1) / threads;

  if(true){
	  const int bucket_num = (1<<c);
	  int *bucket_ids, *value_ids, *bucket_index;
	  CUDA_CHECK(cudaMalloc((void**)&bucket_ids, sizeof(int) * data_length));
	  CUDA_CHECK(cudaMalloc((void**)&value_ids, sizeof(int) * data_length));
	  kernel_get_bid_and_counter<<<blocks, threads>>>(bn_exponents, c, k, data_length, bucket_num, indexs, bucket_ids, value_ids); 
	  //CUDA_CHECK(cudaDeviceSynchronize());
	  thrust::sort_by_key(thrust::device, bucket_ids, bucket_ids + data_length, value_ids); 
	  //CUDA_CHECK(cudaDeviceSynchronize());
      
	  kernel_split_to_bucket<<<blocks, threads>>>(data, out, data_length, bucket_num, starts, value_ids, bucket_ids, bucket_index);
	  cudaFree(bucket_ids);
	  cudaFree(value_ids);
  }

  if(false){
    const int bucket_num = 1<<c;
    int *tids, *sorted_tids;
    cudaMalloc((void**)&tids, data_length * sizeof(int));
    cudaMalloc((void**)&sorted_tids, data_length * sizeof(int));
    //1 get tids
    if(with_density){
      kernel_split_to_bucket<<<blocks, threads>>>(data, out, density, bn_exponents, c, k, data_length, indexs, tids);
    }else{
      kernel_split_to_bucket<<<blocks, threads>>>(data, out, bn_exponents, c, k, data_length, indexs, tids);
    }
    //2. sort
    int real_n = 0;
    cudaMemcpy(&real_n, indexs + bucket_num -1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(starts + bucket_num, indexs + bucket_num - 1, sizeof(int), cudaMemcpyDeviceToDevice);
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, tids, sorted_tids,
        real_n, 1<<c, starts, starts + 1);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sorting operation
    cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, tids, sorted_tids,
            real_n, 1<<c, starts, starts+ 1);
    
    //std::vector<int> ids1(data_length), ids2(data_length);
    //cudaMemcpy(ids1.data(), value_ids, data_length*sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(ids2.data(), sorted_tids, data_length*sizeof(int), cudaMemcpyDeviceToHost);
    //for(int i = 0; i < real_n; i++){
    //    if(ids1[i] != ids2[i]){
    //        printf("compare id failed %d\n", i);
    //        break;
    //    }
    //}
    //printf("compare id success\n");

    //3. split 
    blocks = (real_n + threads-1)/threads;
    kernel_split_to_bucket<<<blocks, threads>>>(data, out, c, k, real_n, sorted_tids);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(tids);
    cudaFree(sorted_tids);
    cudaFree(d_temp_storage);
  }
  if(false){
    if(with_density){
      kernel_split_to_bucket<<<blocks, threads>>>(data, out, density, bn_exponents, c, k, data_length, indexs);
    }else{
      kernel_split_to_bucket<<<blocks, threads>>>(data, out, bn_exponents, c, k, data_length, indexs);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  }

}

#define LITTLE_BUCKET_REDUCE(left, right){\
  cudaMemset(tmp_num, 0, sizeof(int));\
  kernel_get_bucket<left, right><<<(bucket_num + 511) / 512, 512>>>(starts, ends, tmp_starts, tmp_ends, tmp_ids, tmp_num, bucket_num);\
  cudaMemcpy(&little_num, tmp_num, sizeof(int), cudaMemcpyDeviceToHost);\
  if(little_num > 0){\
    blocks = (little_num + local_instances-1) / local_instances;\
    kernel_small_bucket_reduce_sum<local_instances, BUCKET_INSTANCES><<<blocks, threads2>>>(report, data, tmp_starts, tmp_ends, tmp_ids, buckets, little_num, max_value, t_zero, modulus, inv);\
    CUDA_CHECK(cudaDeviceSynchronize());\
  }\
}
#define LARGE_BUCKET_REDUCE(left, right, instances){\
    cudaMemset(tmp_num, 0, sizeof(int));\
    kernel_get_bucket<left, right><<<(bucket_num + 511) / 512, 512>>>(starts, ends, tmp_starts, tmp_ends, tmp_ids, tmp_num, bucket_num);\
    cudaMemcpy(&large_num, tmp_num, sizeof(int), cudaMemcpyDeviceToHost);\
    if(large_num > 0){ \
      kernel_medium_bucket_reduce_sum<instances, BUCKET_INSTANCES><<<large_num, instances*8>>>(report, data, tmp_starts, tmp_ends, tmp_ids, buckets, max_value, t_zero, modulus, inv);\
      CUDA_CHECK(cudaDeviceSynchronize());\
    }\
}

__global__ void kernel_bucket_reduce_by_certain_instances(
    cgbn_error_report_t* report, 
    alt_bn128_g1 data,
    const int *starts, 
    int *ends, 
    const int *instance_bucket_ids,
    const int total_instances,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int instance = tid / TPI;
  int local_instance = threadIdx.x / TPI;
  int local_instances = blockDim.x / TPI;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  __shared__ uint32_t cache_res[64 * 24];
  uint32_t *res = &cache_res[local_instance * 24];
  __shared__ uint32_t cache_buffer[64 * 8];
  uint32_t *buffer = &cache_buffer[local_instance * 8];
  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);

  //if(instance >= total_instances) return;
  for(int real_instance = instance; real_instance < total_instances; real_instance += gridDim.x * local_instances){
    int bucket_id = instance_bucket_ids[real_instance];
    int start = starts[bucket_id];
    int bucket_size = ends[bucket_id] - start;
    int half_bucket_size = bucket_size / 2;
    int bucket_instance = real_instance % half_bucket_size;

    DevAltBn128G1 result;
    result.load(bn_env, data, start + bucket_instance);
    for(int i = bucket_instance + half_bucket_size; i < bucket_size; i+= half_bucket_size){
      DevAltBn128G1 other;
      other.load(bn_env, data, start + i);
      dev_alt_bn128_g1_add(bn_env, result, other, &result, res, buffer, local_max_value, local_modulus, inv);
    }
    result.store(bn_env, data, start + bucket_instance);
    //update ends
    //ends[bucket_id] = start + half_bucket_size;
  }
}

template<int BlockInstances, int Offset>
__global__ void kernel_bucket_reduce(
    cgbn_error_report_t* report, 
    alt_bn128_g1 data,
    alt_bn128_g1 buckets,
    const int *starts, 
    int *ends, 
    const int bucket_id_offset,
    const int bucket_num,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv, const int instances){
    const int bucket_id = bucket_id_offset + blockIdx.y;
    if(bucket_id >= bucket_num) return;
    const int start = starts[bucket_id];
    const int end = ends[bucket_id];
    const int bucket_size = end - start;
    if(bucket_size <= 1){
	    return;
    }
    if(instances >= bucket_size) return;

    const int half_bucket_size = (bucket_size + 1) / 2;
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int instance = tid / TPI; 
    const int local_instance = threadIdx.x / TPI; 
    if(instance >= instances) return;

    context_t bn_context(cgbn_report_monitor, report, bucket_id * gridDim.x * BlockInstances + instance);
    env_t          bn_env(bn_context.env<env_t>());  

    __shared__ uint32_t cache_res[BlockInstances * 24];
    uint32_t *res = &cache_res[local_instance * 24];
    __shared__ uint32_t cache_buffer[BlockInstances * 8];
    uint32_t *buffer = &cache_buffer[local_instance * 8];
    env_t::cgbn_t local_max_value, local_modulus;
    cgbn_load(bn_env, local_max_value, max_value);
    cgbn_load(bn_env, local_modulus, modulus);

    if(instance + half_bucket_size < bucket_size){
        DevAltBn128G1 result;
        result.load(bn_env, data, start + instance);
        DevAltBn128G1 other;
        other.load(bn_env, data, start + instance + half_bucket_size);
        dev_alt_bn128_g1_add(bn_env, result, other, &result, res, buffer, local_max_value, local_modulus, inv);
        result.store(bn_env, data, start + instance);
    }
}
__global__ void kernel_update_bucket_ends(const int* starts, int* ends, const int bucket_num, const int instances){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < bucket_num){
		const int start = starts[tid];
		int bucket_size = ends[tid] - start;
        //int half_bucket_size = (bucket_size + 1) / 2;
		if(instances < bucket_size){
			ends[tid] = instances;
		}
	}
}	

template<int Offset>
__global__ void kernel_copy(
    cgbn_error_report_t* report, 
    alt_bn128_g1 data,
    int* starts, 
    int* ends, 
    alt_bn128_g1 buckets,
    alt_bn128_g1 zero,
    const int bucket_num){
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int instance = tid / TPI;
  if(instance >= bucket_num) return;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  int bid = instance;
  int start = starts[bid];
  int end = ends[bid];
  if(end - start == 0){
      DevAltBn128G1 dev_zero;
      dev_zero.load(bn_env, zero, 0);
      dev_zero.store(bn_env, buckets, bid * Offset);
  }else{
      DevAltBn128G1 a;
      a.load(bn_env, data, start);
      a.store(bn_env, buckets, bid * Offset);
  }
}

__global__ void kernel_get_max_bucket_size(const int *starts, const int *ends, int* max_bucket_size, const int bucket_num){
	int tid = threadIdx.x;
	__shared__ int buffer[512];
	if(tid < bucket_num){
		buffer[tid] = ends[tid] - starts[tid];
	}else{
		buffer[tid] = 0;
	}
	for(int i = tid + 512; i < bucket_num; i+=512){
		int tmp = ends[i] - starts[i];
		if(buffer[tid] < tmp){
			buffer[tid] = tmp;
		}
	}
	__syncthreads();
	int n = 512;
	while(n > 1){
		n /= 2;
		if(tid < n){
			if(buffer[tid] < buffer[tid + n]){
				buffer[tid] = buffer[tid + n];
			}
		}
		__syncthreads();
	}
	if(tid == 0){
		max_bucket_size[0] = buffer[0];
	}
}

__global__ void kernel_reverse(
      alt_bn128_g1 data,
      int* starts,
      alt_bn128_g1 out,
      const int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = tid; i < n; i += gridDim.x * blockDim.x){
      int in_i = starts[i];
      int out_i = n - i - 1;
#pragma unroll
      for(int j = 0; j < 4; j++){
        ((uint64_t*)out.x.mont_repr_data[out_i]._limbs)[j] = ((uint64_t*)data.x.mont_repr_data[in_i]._limbs)[j];
        ((uint64_t*)out.y.mont_repr_data[out_i]._limbs)[j] = ((uint64_t*)data.y.mont_repr_data[in_i]._limbs)[j];
        ((uint64_t*)out.z.mont_repr_data[out_i]._limbs)[j] = ((uint64_t*)data.z.mont_repr_data[in_i]._limbs)[j];
      }
    }
}

__global__ void kernel_init_buckets(
    cgbn_error_report_t* report, 
    alt_bn128_g1 data,
    alt_bn128_g1 zero,
    const int bucket_num){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int instance = tid / TPI;
	if(instance >= bucket_num) return;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  DevAltBn128G1 dev_zero;
  dev_zero.load(bn_env, zero, 0);
  dev_zero.store(bn_env, data, instance);
}

__global__ void kernel_calc_bucket_half_size(const int *starts, const int *ends, int* sizes, const int bucket_num){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < bucket_num){
        sizes[tid] = (ends[tid] - starts[tid])/2;
    }
}
__global__ void kernel_get_bucket_tids(const int* half_sizes, const int bucket_num, int* bucket_tids, int*bucket_ids){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < bucket_num){
        int start = 0;
        if(tid > 0) start = half_sizes[tid-1];
        for(int i = start; i < half_sizes[tid]; i++){
            bucket_tids[i] = i - start;
            bucket_ids[i] = tid;
        }
    }
}
__global__ void kernel_update_ends2(
        const int *starts, 
        int* half_sizes, 
        int* ends, 
        const int bucket_num){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid < bucket_num){
    int start = starts[tid];
    int end = ends[tid];
    int half_bucket_size = (end-start)/2;
    if(end-start > 1) ends[tid] = start + half_bucket_size;
    half_sizes[tid] = half_bucket_size / 2;
  }
}

template<int BlockInstances>
__global__ void kernel_bucket_reduce_g1(
    cgbn_error_report_t* report, 
    alt_bn128_g1 data,
    const int *starts, 
    const int *ends,
    const int *bucket_ids,
    const int *bucket_tids,
    const int total_instances,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int instance = tid / TPI;
  int local_instance = threadIdx.x / TPI;
  if(instance >= total_instances) return;
  int bucket_id = bucket_ids[instance];
  int start = starts[bucket_id];
  int bucket_size = ends[bucket_id] - start;
  if(bucket_size <= 1) return;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  __shared__ uint32_t cache_res[BlockInstances * 24];
  uint32_t *res = &cache_res[local_instance * 24];
  __shared__ uint32_t cache_buffer[BlockInstances * 8];
  uint32_t *buffer = &cache_buffer[local_instance * 8];
  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);

  int half_bucket_size = bucket_size / 2;
  int bucket_instance = bucket_tids[instance];

  DevAltBn128G1 result;
  result.load(bn_env, data, start + bucket_instance);
  for(int i = bucket_instance + half_bucket_size; i < bucket_size; i+= half_bucket_size){
      DevAltBn128G1 other;
      other.load(bn_env, data, start + i);
      dev_alt_bn128_g1_add(bn_env, result, other, &result, res, buffer, local_max_value, local_modulus, inv);
  }
  result.store(bn_env, data, start + bucket_instance);
}


void bucket_reduce_sum(
    alt_bn128_g1 data,
    int* starts, int* ends, int* ids,
    int *d_instance_bucket_ids,
    alt_bn128_g1 buckets,
    const int bucket_num,
    const int data_size,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    CudaStream stream){
  cgbn_error_report_t *report = get_error_report();
  if(false){
    const int local_instances = 16;
    const int threads = local_instances * TPI;
    const int blocks = (bucket_num + local_instances - 1) / local_instances;
    kernel_bucket_reduce_sum_one_instance<local_instances, BUCKET_INSTANCES><<<bucket_num, threads>>>(report, data, starts, ends, bucket_num, buckets, max_value, t_zero, modulus, inv);
  }
  if(true){
    int *half_sizes, *bucket_ids;
    CUDA_CHECK(cudaMalloc((void**)&half_sizes, bucket_num * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&bucket_ids, data_size * sizeof(int)));
    int* bucket_tids = d_instance_bucket_ids;
    int threads = 256;
    int blocks = (bucket_num + threads-1) / threads;
    kernel_calc_bucket_half_size<<<blocks, threads>>>(starts, ends, half_sizes, bucket_num);
    CUDA_CHECK(cudaDeviceSynchronize());
    while(1){
        thrust::inclusive_scan(thrust::device, half_sizes, half_sizes + bucket_num, half_sizes);
        //CUDA_CHECK(cudaDeviceSynchronize());
        threads = 256;
        blocks = (bucket_num + threads-1) / threads;
        kernel_get_bucket_tids<<<blocks, threads>>>(half_sizes, bucket_num, bucket_tids, bucket_ids);
        //CUDA_CHECK(cudaDeviceSynchronize());
        int total_instances = 0;
        CUDA_CHECK(cudaMemcpy(&total_instances, half_sizes + bucket_num-1, sizeof(int), cudaMemcpyDeviceToHost)); 
        if(total_instances == 0) break;
        const int local_instances = 64;
        threads = local_instances * TPI;
        blocks = (total_instances + local_instances - 1) / local_instances;
        kernel_bucket_reduce_g1<local_instances><<<blocks, threads>>>(report, data, starts, ends, bucket_ids, bucket_tids, total_instances, max_value, t_zero, modulus, inv); 
        //CUDA_CHECK(cudaDeviceSynchronize());
        threads = 256;
        blocks = (bucket_num + threads-1) / threads;
        kernel_update_ends2<<<blocks, threads>>>(starts, half_sizes, ends, bucket_num);
        //CUDA_CHECK(cudaDeviceSynchronize());
    }
    threads = 512;
    int local_instances = 64;
    blocks = (bucket_num + local_instances-1) / local_instances;
    kernel_copy<BUCKET_INSTANCES><<<blocks, threads>>>(report, data, starts, ends, buckets, t_zero, bucket_num);
    cudaFree(half_sizes);
    cudaFree(bucket_ids);
  }

  if(false){
	  const int init_blocks = (bucket_num * BUCKET_INSTANCES) / 64;

  std::vector<std::vector<int>> sections = {
                    {128, 10240000}
  };
  int *d_instances = d_instance_bucket_ids + data_size;
  for(int i = 0; i < sections.size(); i++){
    int left = sections[i][0];
    int right = sections[i][1];
    while(1){
      cudaMemset(d_instances, 0, sizeof(int));
      int threads = 512;
      int blocks = (bucket_num + threads-1) / threads;
      kernel_get_instances_and_bucket_id<<<blocks, threads>>>(starts, ends, bucket_num, d_instances, d_instance_bucket_ids, left, right);
      int total_instances = 0;
      cudaMemcpy(&total_instances, d_instances, sizeof(int), cudaMemcpyDeviceToHost);
      if(total_instances > 0){
        int local_instances = 64;
        int threads = local_instances * TPI;
        int blocks = (total_instances + local_instances - 1) / local_instances;
        //blocks = 32;
        kernel_bucket_reduce_by_certain_instances<<<blocks, threads>>>(report, data, starts, ends, d_instance_bucket_ids, total_instances, max_value, t_zero, modulus, inv);
        kernel_update_ends<<<blocks, threads>>>(d_instance_bucket_ids, starts, ends, total_instances);
      }else{
        break;
      }
    }
  }

    int one_num = 0, little_num = 0, large_num = 0;
    int *tmp_starts = starts + bucket_num;
    int *tmp_ends = ends + bucket_num;
    int *tmp_ids = ids;
    int *tmp_num = ids + bucket_num;

    cudaMemset(tmp_num, 0, sizeof(int));
    kernel_get_bucket<0, 1><<<(bucket_num + 511) / 512, 512>>>(starts, ends, tmp_starts, tmp_ends, tmp_ids, tmp_num, bucket_num);
    cudaMemcpy(&one_num, tmp_num, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaDeviceSynchronize());

    int threads = 512;
    int blocks = (one_num + 511) / 512;
    if(one_num > 0){
      kernel_one_bucket_reduce_sum<BUCKET_INSTANCES><<<blocks, threads>>>(data, tmp_starts, tmp_ids, buckets, one_num);
    }

    ///////////////////////////////////////////
    const int threads2 = 256;
    const int local_instances = threads2/TPI;

    LITTLE_BUCKET_REDUCE(1, 2);
    LITTLE_BUCKET_REDUCE(2, 3);
    LITTLE_BUCKET_REDUCE(3, 4);
    LITTLE_BUCKET_REDUCE(4, 5);
    LITTLE_BUCKET_REDUCE(5, 6);
    LITTLE_BUCKET_REDUCE(6, 7);
    LITTLE_BUCKET_REDUCE(7, 8);
    LITTLE_BUCKET_REDUCE(8, 16);
    ///////////////////////////////////////////

    LARGE_BUCKET_REDUCE(16, 32, 4);
    LARGE_BUCKET_REDUCE(32, 64, 8);
    LARGE_BUCKET_REDUCE(64, 128, 16);
    //LARGE_BUCKET_REDUCE(128, 1024, 16);
    //LARGE_BUCKET_REDUCE(1024, 10240000, 64);

    ///////////////////////////////////////////
    //const int threads = 128;
    //const int blocks = bucket_num;
    //kernel_bucket_reduce_sum<threads / TPI><<<blocks, threads, 0, stream>>>(report, data, starts, ends, buckets, max_value, t_zero, modulus, inv);
    //CUDA_CHECK(cudaDeviceSynchronize());
  }
}

void reverse(alt_bn128_g1 in, alt_bn128_g1 out, const int n, const int offset, CudaStream stream){
  const int threads = 512;
  int reverse_blocks = (n + threads - 1) / threads;
  kernel_reverse<<<reverse_blocks, threads>>>(in, out, n, offset);
  //CUDA_CHECK(cudaDeviceSynchronize());
}

//64*64, 16, 1
//64, 64, 16
//1, 64, 64*16
template<int Instances, int ReduceDepthPerBlock>
__global__ void kernel_prefix_sum_pre(
      cgbn_error_report_t* report, 
      alt_bn128_g1 data, 
      const int n,
      cgbn_mem_t<BITS>* max_value,
      cgbn_mem_t<BITS>* modulus, const uint64_t inv,
      int stride){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int instance = tid / TPI;
  int local_instance = threadIdx.x / TPI;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  __shared__ uint32_t cache_res[Instances * 24];
  uint32_t *res = &cache_res[local_instance * 24];
  __shared__ uint32_t cache_buffer[Instances * 8];
  uint32_t *buffer = &cache_buffer[local_instance * 8];
  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);

  int offset = blockIdx.x * ReduceDepthPerBlock;
  int index = (local_instance + 1) * stride * 2 - 1;
  if(index < ReduceDepthPerBlock && offset + index < n){
    DevAltBn128G1 dev_a, dev_b;
    dev_a.load(bn_env, data, offset + index);
    dev_b.load(bn_env, data, offset + index - stride);
    dev_alt_bn128_g1_add(bn_env, dev_a, dev_b, &dev_a, res, buffer, local_max_value, local_modulus, inv);
    dev_a.store(bn_env, data, offset + index);
  }
}
template<int Instances, int ReduceDepthPerBlock>
__global__ void kernel_prefix_sum_post(
      cgbn_error_report_t* report, 
      alt_bn128_g1 data, 
      alt_bn128_g1 block_sums, 
      const int n,
      cgbn_mem_t<BITS>* max_value,
      cgbn_mem_t<BITS>* modulus, const uint64_t inv,
      int stride, bool save_block_sum){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int instance = tid / TPI;
  int local_instance = threadIdx.x / TPI;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  __shared__ uint32_t cache_res[Instances * 24];
  uint32_t *res = &cache_res[local_instance * 24];
  __shared__ uint32_t cache_buffer[Instances * 8];
  uint32_t *buffer = &cache_buffer[local_instance * 8];
  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);

  int offset = blockIdx.x * ReduceDepthPerBlock;
  int index = (local_instance + 1) * stride * 2 - 1;
  if(index + stride < ReduceDepthPerBlock && offset + index + stride < n){
    DevAltBn128G1 dev_a, dev_b;
    dev_a.load(bn_env, data, offset + index + stride);
    dev_b.load(bn_env, data, offset + index);
    dev_alt_bn128_g1_add(bn_env, dev_a, dev_b, &dev_a, res, buffer, local_max_value, local_modulus, inv);
    dev_a.store(bn_env, data, offset + index + stride);
  }
  if(save_block_sum && local_instance == 0){
    DevAltBn128G1 dev_a;
    dev_a.load(bn_env, data, blockIdx.x * ReduceDepthPerBlock + ReduceDepthPerBlock - 1);
    dev_a.store(bn_env, block_sums, blockIdx.x);
  }
}

//64*64, 16, 1
//63 * 64, 1, 64*16
template<int Instances, int ReduceDepthPerInstance, int Step>
__global__ void kernel_add_block_sum2(
      cgbn_error_report_t* report, 
      alt_bn128_g1 data, 
      const int n,
      cgbn_mem_t<BITS>* max_value,
      cgbn_mem_t<BITS>* modulus, const uint64_t inv){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int instance = tid / TPI;
  int local_instance = threadIdx.x / TPI;

  DevAltBn128G1 current_block_sum;
  //current_block_sum.load(bn_env, data, instance * ReduceDepthPerInstance);
  for(int i = 0; i < ReduceDepthPerInstance - 1; i++){
    DevAltBn128G1 next_block_data;
    //next_block_data.load(bn_env, data, )
  }
}

void prefix_sum(
    alt_bn128_g1 data, 
    alt_bn128_g1 block_sums, 
    alt_bn128_g1 block_sums2, 
    const int n,//2^16
    cgbn_mem_t<BITS>* max_value,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv, CudaStream stream){
  cgbn_error_report_t *report = get_error_report();
  const int threads = 512;
  int instances = threads / TPI;//64
  int prefix_sum_blocks = (n + instances - 1) / instances;//2^10
  int prefix_sum_blocks2 = (prefix_sum_blocks + instances-1) / instances;//2^4

  for(int stride = 1; stride <= 32; stride *= 2){
    int instances = 32 / stride;
    int threads = instances * TPI;
    kernel_prefix_sum_pre<32, 64><<<prefix_sum_blocks, threads>>>(report, data, n, max_value, modulus, inv, stride);
  }
  for(int stride = 32; stride > 0; stride /= 2){
    int instances = 32 / stride;
    int threads = instances * TPI;
    bool save_block_sum = (stride == 1);
    kernel_prefix_sum_post<32, 64><<<prefix_sum_blocks, threads>>>(report, data, block_sums, n, max_value, modulus, inv, stride, save_block_sum);
  }

  for(int stride = 1; stride <= 32; stride *= 2){
    int instances = 32 / stride;
    int threads = instances * TPI;
    kernel_prefix_sum_pre<32, 64><<<prefix_sum_blocks2, threads>>>(report, block_sums, prefix_sum_blocks, max_value, modulus, inv, stride);
  }
  for(int stride = 32; stride > 0; stride /= 2){
    int instances = 32 / stride;
    int threads = instances * TPI;
    bool save_block_sum = (stride == 1);
    kernel_prefix_sum_post<32, 64><<<prefix_sum_blocks2, threads>>>(report, block_sums, block_sums2, prefix_sum_blocks, max_value, modulus, inv, stride, save_block_sum);
  }
  
  //kernel_prefix_sum<64, 32, true><<<prefix_sum_blocks2, threads/2, 0, stream>>>(report, block_sums, block_sums2, prefix_sum_blocks, max_value, modulus, inv);
  kernel_prefix_sum<16, 8, false><<<1, 128/2>>>(report, block_sums2, block_sums2, prefix_sum_blocks2, max_value, modulus, inv);
  kernel_add_block_sum<64><<<prefix_sum_blocks2-1, threads>>>(report, block_sums, block_sums2, prefix_sum_blocks, max_value, modulus, inv);
  kernel_add_block_sum<64><<<prefix_sum_blocks-1, threads>>>(report, data, block_sums, n, max_value, modulus, inv);
  //CUDA_CHECK(cudaDeviceSynchronize());
}
}// namespace gpu
