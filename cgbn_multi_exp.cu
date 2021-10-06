#include "cgbn_alt_bn128_g1.h"
#include "cgbn_alt_bn128_g1.cuh"
#include "cgbn_multi_exp.h"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

namespace gpu{

  __device__ size_t dev_get_id(const size_t c, const size_t bitno, uint64_t* data){
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
          for(int j = 0; j < 4; j++){
            ((uint64_t*)out.x.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.x.mont_repr_data[i]._limbs)[j];
            ((uint64_t*)out.y.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.y.mont_repr_data[i]._limbs)[j];
            ((uint64_t*)out.z.mont_repr_data[index]._limbs)[j] = ((uint64_t*)data.z.mont_repr_data[i]._limbs)[j];
          }
        }
      }
    }
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

    DevAltBn128G1 result, dev_t_zero;
    dev_t_zero.load(bn_env, t_zero, 0);
    result.copy_from(bn_env, dev_t_zero);
    for(int i = instance; i < n; i += instances){
      int j = start + i;
      DevAltBn128G1 dev_b;
      dev_b.load(bn_env, data, j);
      dev_alt_bn128_g1_add(bn_env, result, dev_b, &result, res, buffer, local_max_value, local_modulus, inv);
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

  __global__ void kernel_reverse(alt_bn128_g1 data, alt_bn128_g1 out, int n, int offset){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = tid; i < n; i += gridDim.x * blockDim.x){
      int in_i = i * offset;
      int out_i = n - i - 1;
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

  template<int Instances, bool SaveBlockSum>
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
    __shared__ uint32_t cache_res[Instances * 24];
    uint32_t *res = &cache_res[local_instance * 24];
    __shared__ uint32_t cache_buffer[Instances * 8];
    uint32_t *buffer = &cache_buffer[local_instance * 8];
    env_t::cgbn_t local_max_value, local_modulus;
    cgbn_load(bn_env, local_max_value, max_value);
    cgbn_load(bn_env, local_modulus, modulus);
    int offset = blockIdx.x * local_instances;
    for(int stride = 1; stride <= Instances; stride *= 2){
      __syncthreads();
      int index = (local_instance+1)*stride*2 - 1; 
      if(index < Instances && index < n){
        DevAltBn128G1 dev_a, dev_b;
        dev_a.load(bn_env, data, offset + index);
        dev_b.load(bn_env, data, offset + index - stride);
        //if(offset + index == 1){
        //  dev_a.x.print(bn_env, buffer);
        //  dev_b.x.print(bn_env, buffer);
        //}
        dev_alt_bn128_g1_add(bn_env, dev_a, dev_b, &dev_a, res, buffer, local_max_value, local_modulus, inv);
        dev_a.store(bn_env, data, offset + index);
        //if(offset + index == 1){
        //  dev_a.x.print(bn_env, buffer);
        //}
      }
      __syncthreads();
    }
    for (unsigned int stride = Instances/2; stride > 0 ; stride /= 2) {
      __syncthreads();
      int index = (local_instance+1)*stride*2 - 1;
      if(index + stride < Instances && index + stride < n){
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
      const char* density,
      const cgbn_mem_t<BITS>* bn_exponents,
      const int c, const int k,
      const int data_length,
      const int bucket_nums,
      int* bucket_counters){
    int threads = 512;
    int blocks = (data_length + threads-1) / threads;
    kernel_bucket_counter<<<blocks, threads>>>(density, bn_exponents, c, k, data_length, bucket_counters);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void prefix_sum(const int *in, int *out, const int n){
    thrust::exclusive_scan(thrust::device, in, in + n, out);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void split_to_bucket(
      alt_bn128_g1 data, 
      alt_bn128_g1 out, 
      const char* density,
      const cgbn_mem_t<BITS>* bn_exponents,
      const int c, const int k,
      const int data_length,
      int *indexs){
    int threads = 512;
    int blocks = (data_length + threads-1) / threads;

    kernel_split_to_bucket<<<blocks, threads>>>(data, out, density, bn_exponents, c, k, data_length, indexs);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void bucket_reduce_sum(
      alt_bn128_g1 data,
      int* starts, int* ends,
      alt_bn128_g1 buckets,
      const int bucket_num,
      cgbn_mem_t<BITS>* max_value,
      alt_bn128_g1 t_zero,
      cgbn_mem_t<BITS>* modulus, const uint64_t inv){
    cgbn_error_report_t *report = get_error_report();
    const int threads = 128;
    const int blocks = bucket_num;
    kernel_bucket_reduce_sum<threads / TPI><<<blocks, threads>>>(report, data, starts, ends, buckets, max_value, t_zero, modulus, inv);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void reverse(alt_bn128_g1 in, alt_bn128_g1 out, const int n, const int offset){
    const int threads = 512;
    int reverse_blocks = (n + threads - 1) / threads;
    kernel_reverse<<<reverse_blocks, threads>>>(in, out, n, offset);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void prefix_sum(
      alt_bn128_g1 data, 
      alt_bn128_g1 block_sums, 
      alt_bn128_g1 block_sums2, 
      const int n,//2^16
      cgbn_mem_t<BITS>* max_value,
      cgbn_mem_t<BITS>* modulus, const uint64_t inv){
    cgbn_error_report_t *report = get_error_report();
    const int threads = 512;
    int instances = threads / TPI;//64
    int prefix_sum_blocks = (n + instances - 1) / instances;//2^10
    int prefix_sum_blocks2 = (prefix_sum_blocks + instances-1) / instances;//2^4

    kernel_prefix_sum<64, true><<<prefix_sum_blocks, threads>>>(report, data, block_sums, n, max_value, modulus, inv);
    kernel_prefix_sum<64, true><<<prefix_sum_blocks2, threads>>>(report, block_sums, block_sums2, prefix_sum_blocks, max_value, modulus, inv);
    kernel_prefix_sum<16, false><<<1, 128>>>(report, block_sums2, block_sums2, prefix_sum_blocks2, max_value, modulus, inv);
    kernel_add_block_sum<64><<<prefix_sum_blocks2-1, threads>>>(report, block_sums, block_sums2, prefix_sum_blocks, max_value, modulus, inv);
    kernel_add_block_sum<64><<<prefix_sum_blocks-1, threads>>>(report, data, block_sums, n, max_value, modulus, inv);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}// namespace gpu