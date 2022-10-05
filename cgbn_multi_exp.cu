#include "cgbn_multi_exp.h"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

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
    kernel_bucket_counter<<<blocks, threads, 0, stream>>>(density, bn_exponents, c, k, data_length, bucket_counters);
  }else{
    kernel_bucket_counter<<<blocks, threads, 0, stream>>>(bn_exponents, c, k, data_length, bucket_counters);
  }
  //CUDA_CHECK(cudaDeviceSynchronize());
}

void prefix_sum(const int *in, int *out, const int n, CudaStream stream){
  thrust::exclusive_scan(thrust::cuda::par.on(stream), in, in + n, out);
  //thrust::exclusive_scan(thrust::device, in, in + n, out);
  //CUDA_CHECK(cudaDeviceSynchronize());
}

}// namespace gpu
