#include "cgbn_alt_bn128_g2.h"
#include "cgbn_alt_bn128_g2.cuh"
#include "cgbn_multi_exp.h"
#include "cgbn_multi_exp_g2.h"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>

#include <time.h>
#include <vector>
#include <algorithm>


namespace gpu{
inline __device__ size_t dev_get_id_g2(const size_t c, const size_t bitno, uint64_t* data){
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

__global__ void kernel_get_instances_and_bucket_id_g2(
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

__global__ void kernel_update_ends_g2(const int *instance_bucket_ids, const int *starts, int* ends, const int total_instances){
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
__global__ void kernel_get_bucket_g2(
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

__global__ void kernel_split_to_bucket_g2(
    alt_bn128_g2 data, 
    alt_bn128_g2 out, 
    const char* density,
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    int *indexs, int* tids){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < data_length; i+= gridDim.x * blockDim.x){
    if(density[i]){
      size_t id = dev_get_id_g2(c, k*c, (uint64_t*)bn_exponents[i]._limbs);
      if(id != 0){
        int index = atomicAdd(&indexs[id], 1);
        tids[index] = i;
      }
    }
  }
}

__global__ void kernel_split_to_bucket_g2(
    alt_bn128_g2 data, 
    alt_bn128_g2 out, 
    const int c, const int k,
    const int data_length,
    int *tids){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < data_length; i+= gridDim.x * blockDim.x){
    int real_i = tids[i];
    int index = i;// + starts[bucket_id];
    //#pragma unroll
    for(int j = 0; j < 4; j++){
      ((uint64_t*)out.x.c0.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.x.c0.mont_repr_data[real_i]._limbs)[j];
      ((uint64_t*)out.y.c0.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.y.c0.mont_repr_data[real_i]._limbs)[j];
      ((uint64_t*)out.z.c0.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.z.c0.mont_repr_data[real_i]._limbs)[j];

      ((uint64_t*)out.x.c1.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.x.c1.mont_repr_data[real_i]._limbs)[j];
      ((uint64_t*)out.y.c1.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.y.c1.mont_repr_data[real_i]._limbs)[j];
      ((uint64_t*)out.z.c1.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.z.c1.mont_repr_data[real_i]._limbs)[j];
    }
  }
}

__global__ void kernel_split_to_bucket_g2(
    alt_bn128_g2 data, 
    alt_bn128_g2 out, 
    const char* density,
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    int *indexs){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < data_length; i+= gridDim.x * blockDim.x){
    if(density[i]){
      size_t id = dev_get_id_g2(c, k*c, (uint64_t*)bn_exponents[i]._limbs);
      if(id != 0){
        int index = atomicAdd(&indexs[id], 1);
//#pragma unroll
        for(int j = 0; j < 4; j++){
          ((uint64_t*)out.x.c0.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.x.c0.mont_repr_data[i]._limbs)[j];
          ((uint64_t*)out.y.c0.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.y.c0.mont_repr_data[i]._limbs)[j];
          ((uint64_t*)out.z.c0.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.z.c0.mont_repr_data[i]._limbs)[j];

          ((uint64_t*)out.x.c1.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.x.c1.mont_repr_data[i]._limbs)[j];
          ((uint64_t*)out.y.c1.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.y.c1.mont_repr_data[i]._limbs)[j];
          ((uint64_t*)out.z.c1.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.z.c1.mont_repr_data[i]._limbs)[j];
        }
      }
    }
  }
}

void split_to_bucket_g2(
    alt_bn128_g2 data, 
    alt_bn128_g2 out, 
    const char* density,
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    int *starts, int *indexs, CudaStream stream){
  int threads = 512;
  int blocks = (data_length + threads-1) / threads;

  if(false){
    kernel_split_to_bucket_g2<<<blocks, threads, 0, stream>>>(data, out, density, bn_exponents, c, k, data_length, indexs);
    //CUDA_CHECK(cudaDeviceSynchronize());
  }else{
    const int bucket_num = 1<<c;
    int *tids, *ids, *sorted_tids;
    cudaMalloc((void**)&tids, data_length * sizeof(int));
    cudaMalloc((void**)&sorted_tids, data_length * sizeof(int));
    //1 get tids
    kernel_split_to_bucket_g2<<<blocks, threads, 0, stream>>>(data, out, density, bn_exponents, c, k, data_length, indexs, tids);
    CUDA_CHECK(cudaDeviceSynchronize());
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

    //3. split 
    blocks = (real_n + threads-1)/threads;
    kernel_split_to_bucket_g2<<<blocks, threads, 0, stream>>>(data, out, c, k, real_n, sorted_tids);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(tids);
    cudaFree(d_temp_storage);
  }
}

__global__ void kernel_bucket_reduce_by_certain_instances_g2(
    cgbn_error_report_t* report, 
    alt_bn128_g2 data,
    const int *starts, 
    int *ends, 
    const int *instance_bucket_ids,
    const int total_instances,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g2 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    Fp_model non_residue){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int instance = tid / TPI;
  int local_instance = threadIdx.x / TPI;
  int local_instances = blockDim.x / TPI;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  __shared__ uint32_t cache_res[32 * 24];
  uint32_t *res = &cache_res[local_instance * 24];
  __shared__ uint32_t cache_buffer[32 * 8];
  uint32_t *buffer = &cache_buffer[local_instance * 8];
  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);
  DevFp dev_non_residue;
  dev_non_residue.load(bn_env, non_residue, 0);

  for(int real_instance = instance; real_instance < total_instances; real_instance += gridDim.x * local_instances){
    int bucket_id = instance_bucket_ids[real_instance];
    int start = starts[bucket_id];
    int bucket_size = ends[bucket_id] - start;
    int half_bucket_size = bucket_size / 2;
    int bucket_instance = real_instance % half_bucket_size;

    DevAltBn128G2 result;
    result.load(bn_env, data, start + bucket_instance);
    for(int i = bucket_instance + half_bucket_size; i < bucket_size; i+= half_bucket_size){
      DevAltBn128G2 other;
      other.load(bn_env, data, start + i);
      dev_alt_bn128_g2_add(bn_env, result, other, &result, res, buffer, local_max_value, local_modulus, inv, dev_non_residue);
    }
    result.store(bn_env, data, start + bucket_instance);
    //update ends
  }
}

template<int Offset>
__global__ void kernel_one_bucket_reduce_sum_g2(
    alt_bn128_g2 data,
    int* starts, int* ids,
    alt_bn128_g2 buckets,
    const int bucket_num){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for(int i = tid; i < bucket_num; i+=gridDim.x * blockDim.x){
    int bid = ids[i];
    int start = starts[i];
    for(int j = 0; j < 4; j++){
      ((uint64_t*)buckets.x.c0.mont_repr_data[bid * Offset]._limbs)[j] = ((uint64_t*)data.x.c0.mont_repr_data[start]._limbs)[j];
      ((uint64_t*)buckets.y.c0.mont_repr_data[bid * Offset]._limbs)[j] = ((uint64_t*)data.y.c0.mont_repr_data[start]._limbs)[j];
      ((uint64_t*)buckets.z.c0.mont_repr_data[bid * Offset]._limbs)[j] = ((uint64_t*)data.z.c0.mont_repr_data[start]._limbs)[j];

      ((uint64_t*)buckets.x.c1.mont_repr_data[bid * Offset]._limbs)[j] = ((uint64_t*)data.x.c1.mont_repr_data[start]._limbs)[j];
      ((uint64_t*)buckets.y.c1.mont_repr_data[bid * Offset]._limbs)[j] = ((uint64_t*)data.y.c1.mont_repr_data[start]._limbs)[j];
      ((uint64_t*)buckets.z.c1.mont_repr_data[bid * Offset]._limbs)[j] = ((uint64_t*)data.z.c1.mont_repr_data[start]._limbs)[j];
    }
  }

}

template<int BS, int Offset>
__global__ void kernel_small_bucket_reduce_sum_g2(
    cgbn_error_report_t* report, 
    alt_bn128_g2 data,
    int* starts, int* ends, int* ids,
    alt_bn128_g2 buckets,
    const int bucket_num,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g2 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    Fp_model non_residue){
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
  DevFp dev_non_residue;
  dev_non_residue.load(bn_env, non_residue, 0);

  DevAltBn128G2 result;
  result.load(bn_env, data, start);
  for(int i = 1; i < n; i++){
    int j = start + i;
    DevAltBn128G2 dev_b;
    dev_b.load(bn_env, data, j);
    dev_alt_bn128_g2_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv, dev_non_residue);
  }
  result.store(bn_env, buckets, real_bid * Offset);
}

#define LITTLE_BUCKET_REDUCE_G2(left, right){\
    cudaMemset(tmp_num, 0, sizeof(int));\
    kernel_get_bucket_g2<left, right><<<(bucket_num + 511) / 512, 512>>>(starts, ends, tmp_starts, tmp_ends, tmp_ids, tmp_num, bucket_num);\
    cudaMemcpy(&little_num, tmp_num, sizeof(int), cudaMemcpyDeviceToHost);\
    if(little_num > 0){\
      blocks = (little_num + local_instances-1) / local_instances;\
      kernel_small_bucket_reduce_sum_g2<local_instances, BUCKET_INSTANCES_G2><<<blocks, threads2, 0, stream>>>(report, data, tmp_starts, tmp_ends, tmp_ids, buckets, little_num, max_value, t_zero, modulus, inv, non_residue);\
    }\
}

template<int BS, int Offset>
__global__ void kernel_medium_bucket_reduce_sum_g2(
    cgbn_error_report_t* report, 
    alt_bn128_g2 data,
    int* starts, int* ends, int* ids,
    alt_bn128_g2 buckets,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g2 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    Fp_model non_residue){
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
  DevFp dev_non_residue;
  dev_non_residue.load(bn_env, non_residue, 0);

  DevAltBn128G2 result;
  result.load(bn_env, t_zero, 0);
  for(int i = instance; i < n; i += instances){
    int j = start + i;
    DevAltBn128G2 dev_b;
    if(i == instance){
      result.load(bn_env, data, j);
    }else{
      dev_b.load(bn_env, data, j);
      dev_alt_bn128_g2_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv, dev_non_residue);
    }
  }
  result.store(bn_env, buckets, real_bid * Offset + instance);
  __syncthreads();
  if(instance == 0){
    for(int i = 1; i < instances && i < n; i++){
      DevAltBn128G2 dev_b;
      dev_b.load(bn_env, buckets, real_bid * Offset + i);
      dev_alt_bn128_g2_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv, dev_non_residue);
    }
    result.store(bn_env, buckets, real_bid * Offset);
  }
}

#define LARGE_BUCKET_REDUCE_G2(left, right, instances){\
    cudaMemset(tmp_num, 0, sizeof(int));\
    kernel_get_bucket_g2<left, right><<<(bucket_num + 511) / 512, 512>>>(starts, ends, tmp_starts, tmp_ends, tmp_ids, tmp_num, bucket_num);\
    cudaMemcpy(&large_num, tmp_num, sizeof(int), cudaMemcpyDeviceToHost);\
    if(large_num > 0){ \
      kernel_medium_bucket_reduce_sum_g2<instances, BUCKET_INSTANCES_G2><<<large_num, instances*8, 0, stream>>>(report, data, tmp_starts, tmp_ends, tmp_ids, buckets, max_value, t_zero, modulus, inv, non_residue);\
    }\
}

template<int BS>
__global__ void kernel_bucket_reduce_sum_g2(
    cgbn_error_report_t* report, 
    alt_bn128_g2 data,
    int* starts, int* ends,
    alt_bn128_g2 buckets,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g2 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    Fp_model non_residue){
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
  DevFp dev_non_residue;
  dev_non_residue.load(bn_env, non_residue, 0);

  DevAltBn128G2 result;
  result.load(bn_env, t_zero, 0);
  for(int i = instance; i < n; i += instances){
    int j = start + i;
    DevAltBn128G2 dev_b;
    if(i == instance){
      result.load(bn_env, data, j);
    }else{
      dev_b.load(bn_env, data, j);
      dev_alt_bn128_g2_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv, dev_non_residue);
    }
  }
  result.store(bn_env, buckets, bid * BS + instance);
  __syncthreads();
  if(instance == 0){
    for(int i = 1; i < instances; i++){
      DevAltBn128G2 dev_b;
      dev_b.load(bn_env, buckets, bid * BS + i);
      dev_alt_bn128_g2_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv, dev_non_residue);
    }
    result.store(bn_env, buckets, bid * BS);
  }
}

template<int BS, int Offset>
__global__ void kernel_bucket_reduce_sum_g2_test(
    cgbn_error_report_t* report, 
    alt_bn128_g2 data,
    int* starts, int* ends,
    alt_bn128_g2 buckets,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g2 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    Fp_model non_residue,
    const int bucket_num){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int instance = tid / TPI;
  if(instance >= bucket_num) return;

  int start = starts[instance];
  int n = ends[instance] - start;
  if(n <= 0) return;

  int local_instance = threadIdx.x / TPI;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  __shared__ uint32_t cache_res[BS * 3 * BITS/32];
  uint32_t *res = &cache_res[local_instance * BITS/32 * 3];
  __shared__ uint32_t cache_buffer[BS * BITS/32];
  uint32_t *buffer = &cache_buffer[local_instance * BITS/32];

  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);
  DevFp dev_non_residue;
  dev_non_residue.load(bn_env, non_residue, 0);

  DevAltBn128G2 result;
  result.load(bn_env, t_zero, 0);
  for(int i = 0; i < n; i += 1){
    int j = start + i;
    DevAltBn128G2 dev_b;
    if(i == 0){
      result.load(bn_env, data, j);
    }else{
      dev_b.load(bn_env, data, j);
      dev_alt_bn128_g2_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv, dev_non_residue);
    }
  }
  //result.store(bn_env, buckets, instance * Offset);
}

void bucket_reduce_sum_g2(
    alt_bn128_g2 data,
    int* starts, int* ends, int* ids,
    int *d_instance_bucket_ids,
    alt_bn128_g2 buckets,
    const int bucket_num,
    const int data_size,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g2 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    Fp_model non_residue, 
    CudaStream stream){
  cgbn_error_report_t *report = get_error_report();
  std::vector<std::vector<int>> sections = {
                    {8, 32},
                    {128, 10240000}
  };
  int *d_instances = d_instance_bucket_ids + data_size - 1;
  for(int i = 0; i < sections.size(); i++){
    int left = sections[i][0];
    int right = sections[i][1];
    while(1){
      cudaMemset(d_instances, 0, sizeof(int));
      int threads = 512;
      int blocks = (bucket_num + threads-1) / threads;
      kernel_get_instances_and_bucket_id_g2<<<blocks, threads>>>(starts, ends, data_size, d_instances, d_instance_bucket_ids, left, right);
      int total_instances = 0;
      cudaMemcpy(&total_instances, d_instances, sizeof(int), cudaMemcpyDeviceToHost);
      if(total_instances > 0){
        int local_instances = 32;
        int threads = local_instances * TPI;
        int blocks = (total_instances + local_instances - 1) / local_instances;
        //blocks = 32;
        kernel_bucket_reduce_by_certain_instances_g2<<<blocks, threads>>>(report, data, starts, ends, d_instance_bucket_ids, total_instances, max_value, t_zero, modulus, inv, non_residue);
        kernel_update_ends_g2<<<blocks, threads>>>(d_instance_bucket_ids, starts, ends, total_instances);
      }else{
        break;
      }
    }
  }

  if(true){
    if(true){
      int one_num = 0, little_num = 0, large_num = 0;
      int *tmp_starts = starts + bucket_num;
      int *tmp_ends = ends + bucket_num;
      int *tmp_ids = ids;
      int *tmp_num = ids + bucket_num;

      cudaMemset(tmp_num, 0, sizeof(int));
      kernel_get_bucket_g2<0, 1><<<(bucket_num + 511) / 512, 512>>>(starts, ends, tmp_starts, tmp_ends, tmp_ids, tmp_num, bucket_num);
      cudaMemcpy(&one_num, tmp_num, sizeof(int), cudaMemcpyDeviceToHost);

      int threads = 512;
      int blocks = (one_num + 511) / 512;
      if(one_num > 0){
        kernel_one_bucket_reduce_sum_g2<BUCKET_INSTANCES_G2><<<blocks, threads>>>(data, tmp_starts, tmp_ids, buckets, one_num);
      }

      ///////////////////////////////////////////
      const int threads2 = 256;
      const int local_instances = threads2/TPI;

      LITTLE_BUCKET_REDUCE_G2(1, 2);
      LITTLE_BUCKET_REDUCE_G2(2, 3);
      LITTLE_BUCKET_REDUCE_G2(3, 4);
      LITTLE_BUCKET_REDUCE_G2(4, 5);
      LITTLE_BUCKET_REDUCE_G2(5, 6);
      LITTLE_BUCKET_REDUCE_G2(6, 7);
      LITTLE_BUCKET_REDUCE_G2(7, 8);
      LITTLE_BUCKET_REDUCE_G2(8, 12);
      LITTLE_BUCKET_REDUCE_G2(12, 16);
      /////////////////////////////////////////////

      LARGE_BUCKET_REDUCE_G2(16, 24, 4);
      LARGE_BUCKET_REDUCE_G2(24, 32, 4);
      LARGE_BUCKET_REDUCE_G2(32, 48, 8);
      LARGE_BUCKET_REDUCE_G2(48, 64, 8);
      LARGE_BUCKET_REDUCE_G2(64, 128, 16);
      LARGE_BUCKET_REDUCE_G2(128, 102400000, 32);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    //const int threads = BUCKET_INSTANCES_G2 * TPI;
    //const int blocks = (bucket_num + BUCKET_INSTANCES_G2-1) / BUCKET_INSTANCES_G2;
    //kernel_bucket_reduce_sum_g2_test<BUCKET_INSTANCES_G2, BUCKET_INSTANCES_G2><<<blocks, threads, 0, stream>>>(report, data, starts, ends, buckets, max_value, t_zero, modulus, inv, non_residue, bucket_num);
    ///////////////////////////////////////////
    if(false){
      const int threads = TPI;//BUCKET_INSTANCES_G2 * TPI;
      const int blocks = bucket_num;
      kernel_bucket_reduce_sum_g2<BUCKET_INSTANCES_G2><<<blocks, threads, 0, stream>>>(report, data, starts, ends, buckets, max_value, t_zero, modulus, inv, non_residue);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
}

__global__ void kernel_reverse_g2(alt_bn128_g2 data, alt_bn128_g2 out, int n, int offset){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = tid; i < n; i += gridDim.x * blockDim.x){
    int in_i = i * offset;
    int out_i = n - i - 1;
#pragma unroll
    for(int j = 0; j < 4; j++){
      ((uint64_t*)out.x.c0.mont_repr_data[out_i]._limbs)[j] = ((uint64_t*)data.x.c0.mont_repr_data[in_i]._limbs)[j];
      ((uint64_t*)out.y.c0.mont_repr_data[out_i]._limbs)[j] = ((uint64_t*)data.y.c0.mont_repr_data[in_i]._limbs)[j];
      ((uint64_t*)out.z.c0.mont_repr_data[out_i]._limbs)[j] = ((uint64_t*)data.z.c0.mont_repr_data[in_i]._limbs)[j];

      ((uint64_t*)out.x.c1.mont_repr_data[out_i]._limbs)[j] = ((uint64_t*)data.x.c1.mont_repr_data[in_i]._limbs)[j];
      ((uint64_t*)out.y.c1.mont_repr_data[out_i]._limbs)[j] = ((uint64_t*)data.y.c1.mont_repr_data[in_i]._limbs)[j];
      ((uint64_t*)out.z.c1.mont_repr_data[out_i]._limbs)[j] = ((uint64_t*)data.z.c1.mont_repr_data[in_i]._limbs)[j];
    }
  }
}

void reverse_g2(alt_bn128_g2 in, alt_bn128_g2 out, const int n, const int offset, CudaStream stream){
  const int threads = 512;
  int reverse_blocks = (n + threads - 1) / threads;
  kernel_reverse_g2<<<reverse_blocks, threads, 0, stream>>>(in, out, n, offset);
}

template<int Instances, int ReduceDepthPerBlock>
__global__ void kernel_prefix_sum_pre_g2(
      cgbn_error_report_t* report, 
      alt_bn128_g2 data, 
      const int n,
      cgbn_mem_t<BITS>* max_value,
      cgbn_mem_t<BITS>* modulus, const uint64_t inv,
      Fp_model non_residue,
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
  DevFp dev_non_residue;
  dev_non_residue.load(bn_env, non_residue, 0);

  int offset = blockIdx.x * ReduceDepthPerBlock;
  int index = (local_instance + 1) * stride * 2 - 1;
  if(index < ReduceDepthPerBlock && index < n){
    DevAltBn128G2 dev_a, dev_b;
    dev_a.load(bn_env, data, offset + index);
    dev_b.load(bn_env, data, offset + index - stride);
    dev_alt_bn128_g2_add(bn_env, dev_a, dev_b, &dev_a, res, buffer, local_max_value, local_modulus, inv, dev_non_residue);
    dev_a.store(bn_env, data, offset + index);
  }
}

template<int Instances, int ReduceDepthPerBlock>
__global__ void kernel_prefix_sum_post_g2(
      cgbn_error_report_t* report, 
      alt_bn128_g2 data, 
      alt_bn128_g2 block_sums, 
      const int n,
      cgbn_mem_t<BITS>* max_value,
      cgbn_mem_t<BITS>* modulus, const uint64_t inv,
      Fp_model non_residue, 
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
  DevFp dev_non_residue;
  dev_non_residue.load(bn_env, non_residue, 0);

  int offset = blockIdx.x * ReduceDepthPerBlock;
  int index = (local_instance + 1) * stride * 2 - 1;
  if(index + stride < ReduceDepthPerBlock && index + stride < n){
    DevAltBn128G2 dev_a, dev_b;
    dev_a.load(bn_env, data, offset + index + stride);
    dev_b.load(bn_env, data, offset + index);
    dev_alt_bn128_g2_add(bn_env, dev_a, dev_b, &dev_a, res, buffer, local_max_value, local_modulus, inv, dev_non_residue);
    dev_a.store(bn_env, data, offset + index + stride);
  }
  if(save_block_sum && local_instance == 0){
    DevAltBn128G2 dev_a;
    dev_a.load(bn_env, data, blockIdx.x * ReduceDepthPerBlock + ReduceDepthPerBlock - 1);
    dev_a.store(bn_env, block_sums, blockIdx.x);
  }
}
template<int Instances, int RealInstances, bool SaveBlockSum>
__global__ void kernel_prefix_sum_g2(
    cgbn_error_report_t* report, 
    alt_bn128_g2 data, 
    alt_bn128_g2 block_sums, 
    const int n,
    cgbn_mem_t<BITS>* max_value,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    Fp_model non_residue){
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
  DevFp dev_non_residue;
  dev_non_residue.load(bn_env, non_residue, 0);

  int offset = blockIdx.x * local_instances;
  for(int stride = 1; stride <= RealInstances; stride *= 2){
    __syncthreads();
    int index = (local_instance+1)*stride*2 - 1; 
    if(index < Instances && index < n){
      DevAltBn128G2 dev_a, dev_b;
      dev_a.load(bn_env, data, offset + index);
      dev_b.load(bn_env, data, offset + index - stride);
      dev_alt_bn128_g2_add(bn_env, dev_a, dev_b, &dev_a, res, buffer, local_max_value, local_modulus, inv, dev_non_residue);
      dev_a.store(bn_env, data, offset + index);
    }
    __syncthreads();
  }
  for (unsigned int stride = (Instances >> 1); stride > 0 ; stride>>=1) {
    __syncthreads();
    int index = (local_instance+1)*stride*2 - 1;
    if(index + stride < Instances && index + stride < n){
      DevAltBn128G2 dev_a, dev_b;
      dev_a.load(bn_env, data, offset + index + stride);
      dev_b.load(bn_env, data, offset + index);
      dev_alt_bn128_g2_add(bn_env, dev_a, dev_b, &dev_a, res, buffer, local_max_value, local_modulus, inv, dev_non_residue);
      dev_a.store(bn_env, data, offset + index + stride);
    }
  }
  __syncthreads();
  if(SaveBlockSum && local_instance == 0){
    DevAltBn128G2 dev_a;
    dev_a.load(bn_env, data, blockIdx.x * local_instances + local_instances-1);
    dev_a.store(bn_env, block_sums, blockIdx.x);
  }
}

template<int Instances>
__global__ void kernel_add_block_sum_g2(
    cgbn_error_report_t* report, 
    alt_bn128_g2 data, 
    alt_bn128_g2 block_sums, 
    const int n,
    cgbn_mem_t<BITS>* max_value,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    Fp_model non_residue){
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
  DevFp dev_non_residue;
  dev_non_residue.load(bn_env, non_residue, 0);

  DevAltBn128G2 dev_block_sum, dev_a;
  dev_block_sum.load(bn_env, block_sums, blockIdx.x);
  dev_a.load(bn_env, data, instance + instances);//offset = instances
  dev_alt_bn128_g2_add(bn_env, dev_a, dev_block_sum, &dev_a, res, buffer, local_max_value, local_modulus, inv, dev_non_residue);
  dev_a.store(bn_env, data, instance + instances);
}

__global__ void kernel_prefix_sum_g2_test(
    cgbn_error_report_t* report, 
    alt_bn128_g2 data, 
    const int n,
    cgbn_mem_t<BITS>* max_value,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    Fp_model non_residue){
  int local_instance = threadIdx.x / TPI;
  context_t bn_context(cgbn_report_monitor, report, local_instance);
  env_t          bn_env(bn_context.env<env_t>());  
  __shared__ uint32_t cache_res[24];
  uint32_t *res = &cache_res[local_instance * 24];
  __shared__ uint32_t cache_buffer[8];
  uint32_t *buffer = &cache_buffer[local_instance * 8];
  env_t::cgbn_t local_max_value, local_modulus;
  cgbn_load(bn_env, local_max_value, max_value);
  cgbn_load(bn_env, local_modulus, modulus);
  DevFp dev_non_residue;
  dev_non_residue.load(bn_env, non_residue, 0);

  DevAltBn128G2 runing_sum, dev_a;
  for(int i = 0; i < n; i++){
    dev_a.load(bn_env, data, i);
    dev_alt_bn128_g2_add(bn_env, runing_sum, dev_a, &runing_sum, res, buffer, local_max_value, local_modulus, inv, dev_non_residue);
    runing_sum.store(bn_env, data, i);
  }
}

void prefix_sum_g2(
    alt_bn128_g2 data, 
    alt_bn128_g2 block_sums, 
    alt_bn128_g2 block_sums2, 
    const int n,//2^16
    cgbn_mem_t<BITS>* max_value,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv, 
    Fp_model non_residue, 
    CudaStream stream){
  cgbn_error_report_t *report = get_error_report();
  if(true){
    const int threads = 512;
    int instances = threads / TPI;//64
    int prefix_sum_blocks = (n + instances - 1) / instances;//2^10
    int prefix_sum_blocks2 = (prefix_sum_blocks + instances-1) / instances;//2^4

    for(int stride = 1; stride <= 32; stride *= 2){
      int instances = 32 / stride;
      int threads = instances * TPI;
      kernel_prefix_sum_pre_g2<32, 64><<<prefix_sum_blocks, threads, 0, stream>>>(report, data, n, max_value, modulus, inv, non_residue, stride);
    }
    for(int stride = 32; stride > 0; stride /= 2){
      int instances = 32 / stride;
      int threads = instances * TPI;
      bool save_block_sum = (stride == 1);
      kernel_prefix_sum_post_g2<32, 64><<<prefix_sum_blocks, threads, 0, stream>>>(report, data, block_sums, n, max_value, modulus, inv, non_residue, stride, save_block_sum);
    }

    for(int stride = 1; stride <= 32; stride *= 2){
      int instances = 32 / stride;
      int threads = instances * TPI;
      kernel_prefix_sum_pre_g2<32, 64><<<prefix_sum_blocks2, threads, 0, stream>>>(report, block_sums, prefix_sum_blocks, max_value, modulus, inv, non_residue, stride);
    }
    for(int stride = 32; stride > 0; stride /= 2){
      int instances = 32 / stride;
      int threads = instances * TPI;
      bool save_block_sum = (stride == 1);
      kernel_prefix_sum_post_g2<32, 64><<<prefix_sum_blocks2, threads, 0, stream>>>(report, block_sums, block_sums2, prefix_sum_blocks, max_value, modulus, inv, non_residue, stride, save_block_sum);
    }

    //kernel_prefix_sum<64, 32, true><<<prefix_sum_blocks2, threads/2, 0, stream>>>(report, block_sums, block_sums2, prefix_sum_blocks, max_value, modulus, inv);
    kernel_prefix_sum_g2<16, 8, false><<<1, 128/2, 0, stream>>>(report, block_sums2, block_sums2, prefix_sum_blocks2, max_value, modulus, inv, non_residue);
    kernel_add_block_sum_g2<64><<<prefix_sum_blocks2-1, threads, 0, stream>>>(report, block_sums, block_sums2, prefix_sum_blocks, max_value, modulus, inv, non_residue);
    kernel_add_block_sum_g2<64><<<prefix_sum_blocks-1, threads, 0, stream>>>(report, data, block_sums, n, max_value, modulus, inv, non_residue);
    //CUDA_CHECK(cudaDeviceSynchronize());
  }
  //kernel_prefix_sum_g2_test<<<1, 8>>>(report, data, n, max_value, modulus, inv, non_residue);
}

}//namespace gpu
