#ifndef LOW_FUNC_CUH
#define LOW_FUNC_CUH

#include "cgbn_ect.cuh"
#include "low_func_gpu.h"
#include <stdint.h>
#include <thrust/scan.h>
#include "bigint_256.cuh"

namespace gpu{

__global__ void kernel_ect_add_new(
    mcl_bn128_g1 R, 
    mcl_bn128_g1 P,
    mcl_bn128_g1 Q,
    Fp_model one, 
    Fp_model p, 
    Fp_model a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
    using namespace BigInt256;
  Ect lP, lQ;
  Int256 lone, lp, la;
  load(lP, P, 0);
  load(lQ, Q, 0);

  load(lone, one, 0); 
  load(la, a, 0); 
  load(lp, p, 0); 

  add(lP, lP, lQ, lone, lp, specialA_, la, mode_, rp);  
  store(R, lP, 0);
}

__global__ void kernel_reduce_sum_pre_new(
    const Fp_model scalars,
    char* flags,
    const Fp_model field_zero,
    const Fp_model field_one,
    const int n,
    char *density,
    cgbn_mem_t<BITS>* bn_exponents,
    cgbn_mem_t<BITS>* const field_modulus, const uint64_t field_inv){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    using namespace BigInt256;
    Int256 local_zero, local_one;
    memcpy(local_zero, field_zero.mont_repr_data, sizeof(Int256));
    memcpy(local_one, field_one.mont_repr_data, sizeof(Int256));
    for(int i = tid; i < n; i+=gridDim.x * blockDim.x){
        //Int *scalar = (Int*)(scalars.mont_repr_data + i);
        Int256 scalar;
        memcpy(scalar, scalars.mont_repr_data + i, sizeof(Int256));
        if(BigInt256::dev_equal(scalar, local_zero)){
        }
        else if(BigInt256::dev_equal(scalar, local_one)){
            flags[i] = 1;
        }else{
            density[i] = 1;
            Int res[BigInt256::N];
            BigInt256::dev_as_bigint(scalar, (Int*)field_modulus, field_inv, res);
            memcpy(bn_exponents + i, res, BigInt256::N*sizeof(Int));
        }
    }
}

__global__ void kernel_mcl_bn128_g1_reduce_sum_pre_new(
    const Fp_model scalars,
    const size_t *index_it,
    uint32_t* counters, 
    char* flags,
    const int ranges_size, 
    const uint32_t* firsts,
    const uint32_t* seconds,
    const Fp_model field_zero,
    const Fp_model field_one,
    char *density,
    cgbn_mem_t<BITS>* bn_exponents,
    cgbn_mem_t<BITS>* const field_modulus, const uint64_t field_inv
    ){
  int instance = blockIdx.x * blockDim.x + threadIdx.x;

  //int range_offset = blockIdx.y * gridDim.x * blockDim.x;
  int first = firsts[blockIdx.y];
  int second = seconds[blockIdx.y];
  int reduce_depth = second - first;//30130
  using namespace BigInt256;
  Int256 local_zero, local_one;
  memcpy(local_zero, field_zero.mont_repr_data, sizeof(Int256));
  memcpy(local_one, field_one.mont_repr_data, sizeof(Int256));

  for(int i = first + instance; i < first + reduce_depth; i+= gridDim.x * blockDim.x){
    const int j = index_it[i];
    //Int *scalar = (Int*)(scalars.mont_repr_data + j);
    Int256 scalar;
    memcpy(scalar, scalars.mont_repr_data + j, sizeof(Int256));
    if(BigInt256::dev_equal(scalar, local_zero)){
    }
    else if(BigInt256::dev_equal(scalar, local_one)){
      flags[j] = 1;
    }
    else{
        density[i] = 1;
        Int res[N];
        dev_as_bigint(scalar, (Int*)field_modulus, field_inv, res);
        memcpy(bn_exponents + i, res, N*sizeof(Int));
    }
  }
}

__global__ void kernel_mcl_bn128_g1_reduce_sum_one_range5_new(
    mcl_bn128_g1 values, 
    const Fp_model scalars,
    const size_t *index_it,
    mcl_bn128_g1 partial, 
    const int ranges_size, 
    const int range_id_offset,
    const uint32_t* firsts,
    const uint32_t* seconds,
    char* flags,
    const mcl_bn128_g1 t_zero,
    const Fp_model one, 
    const Fp_model p, 
    const Fp_model a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
  const int instance = threadIdx.x + blockIdx.x * blockDim.x;

  const int first = firsts[range_id_offset + blockIdx.y];
  const int second = seconds[range_id_offset + blockIdx.y];
  const int reduce_depth = second - first;//30130
  if(reduce_depth <= 1) return;
  const int half_depth = (reduce_depth + 1) / 2;

  if(instance >= half_depth) return;

    using namespace BigInt256;
  Int256 lone, lp, la;
  load(lone, one, 0); 
  load(la, a, 0); 
  load(lp, p, 0); 

  Ect result;
  if(flags[index_it[first + instance]] == 1){
	  load(result, values, first+instance);
  }else{
	  load(result, t_zero, 0);
  }
  for(int i = first + instance+half_depth; i < first + reduce_depth; i+= half_depth){
    const int j = index_it[i];
    if(flags[j] == 1){
      Ect dev_b;
      load(dev_b, values, i);
      add(result, result, dev_b, lone, lp, specialA_, la, mode_, rp);  
    }
  }
  store(partial, result, first + instance);
}

__global__ void kernel_mcl_bn128_g1_reduce_sum_one_range7_new(
    mcl_bn128_g1 values, 
    const Fp_model scalars,
    const size_t *index_it,
    mcl_bn128_g1 partial, 
    const int ranges_size, 
    const int range_id_offset,
    const uint32_t* firsts,
    const uint32_t* seconds,
    char* flags,
    const mcl_bn128_g1 t_zero,
    const Fp_model one, 
    const Fp_model p, 
    const Fp_model a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
  const int instance = threadIdx.x + blockIdx.x * blockDim.x;

  const int first = firsts[range_id_offset + blockIdx.y];
  const int second = seconds[range_id_offset + blockIdx.y];
  const int reduce_depth = second - first;//30130
  if(reduce_depth <= 1) return;
  const int half_depth = (reduce_depth + 1) / 2;

  if(instance >= half_depth) return;

  using namespace BigInt256;
  Int256 lone, lp, la;
  load(lone, one, 0); 
  load(la, a, 0); 
  load(lp, p, 0); 

  Ect result;
  load(result, values, first+instance);
  for(int i = first + instance+half_depth; i < first + reduce_depth; i+= half_depth){
      Ect dev_b;
      load(dev_b, values, i);
      add(result, result, dev_b, lone, lp, specialA_, la, mode_, rp);  
  }
  store(partial, result, first + instance);
}

__global__ void kernel_mcl_update_seconds(const uint32_t *firsts, uint32_t* seconds, const int range_size){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < range_size){
		int first = firsts[tid];
		int second = seconds[tid];
        if(second - first <= 1){
            seconds[tid]=first;
        }else{
            seconds[tid] = first + (second - first + 1) / 2;
        }
	}
}

__global__ void kernel_mcl_bn128_g1_reduce_sum_one_range6_new(
    mcl_bn128_g1 partial, 
    const int n, 
    const uint32_t* firsts,
    const Fp_model one, 
    const Fp_model p, 
    const Fp_model a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
  using namespace BigInt256;
  Int256 lone, lp, la;
  load(lone, one, 0); 
  load(la, a, 0); 
  load(lp, p, 0); 

  Ect result;
  load(result, partial, firsts[0]);

  for(int i = 1; i < n; i++){
    Ect dev_b;
    load(dev_b, partial, firsts[i]);
    add(result, result, dev_b, lone, lp, specialA_, la, mode_, rp);  
  }
  store(partial, result, 0);
}

int mcl_bn128_g1_reduce_sum(
    mcl_bn128_g1 values, 
    Fp_model scalars, 
    const size_t *index_it,
    mcl_bn128_g1 partial, 
    uint32_t *counters,
    char* flags,
    const uint32_t ranges_size,
    const uint32_t *firsts,
    uint32_t *seconds,
    mcl_bn128_g1 t_zero,
    Fp_model field_zero,
    Fp_model field_one,
    char *density,
    cgbn_mem_t<BITS>* bn_exponents,
    cgbn_mem_t<BITS>* const field_modulus, const uint64_t field_inv,
    Fp_model one, Fp_model p, Fp_model a, const int specialA_, const int mode_, const uint64_t rp,
    const int max_reduce_depth, cudaStream_t stream
    ){
  uint32_t threads = 64;
  const int local_instances = 64 * BlockDepth;
  uint32_t block_x =  (max_reduce_depth + local_instances - 1) / local_instances;
  dim3 blocks(block_x, ranges_size, 1);
  kernel_mcl_bn128_g1_reduce_sum_pre_new<<<blocks, threads, 0, stream>>>(scalars, index_it, counters, flags, ranges_size, firsts, seconds, field_zero, field_one, density, bn_exponents, field_modulus, field_inv);

  if(true){
      int n = max_reduce_depth;
      const int local_instances2 = 64;
      threads = local_instances2;
      uint32_t block_x2 =  ((n+1)/2 + local_instances2 - 1) / local_instances2;
      dim3 blocks2(block_x2, ranges_size, 1);
      kernel_mcl_bn128_g1_reduce_sum_one_range5_new<<<blocks2, dim3(threads, 1, 1), 0, stream>>>(values, scalars, index_it, partial, ranges_size, 0, firsts, seconds, flags, t_zero, one, p, a, specialA_, mode_, rp);
      const int update_threads = 64;
      const int update_blocks = (ranges_size + update_threads - 1) / update_threads;
      kernel_mcl_update_seconds<<<update_blocks, update_threads, 0, stream>>>(firsts, seconds, ranges_size);
      //CUDA_CHECK(cudaDeviceSynchronize());
      n = (n+1)/2;
      while(n>=2){
          uint32_t block_x2 =  ((n+1)/2 + local_instances2 - 1) / local_instances2;
          dim3 blocks2(block_x2, ranges_size, 1);
          kernel_mcl_bn128_g1_reduce_sum_one_range7_new<<<blocks2, dim3(threads, 1, 1), 0, stream>>>(partial, scalars, index_it, partial, ranges_size, 0, firsts, seconds, flags, t_zero, one, p, a, specialA_, mode_, rp);
          //CUDA_CHECK(cudaDeviceSynchronize());
          kernel_mcl_update_seconds<<<update_blocks, update_threads, 0, stream>>>(firsts, seconds, ranges_size);
          //CUDA_CHECK(cudaDeviceSynchronize());
          n = (n+1)/2;
      }
  }

  kernel_mcl_bn128_g1_reduce_sum_one_range6_new<<<1, 1, 0, stream>>>(partial, ranges_size, firsts, one, p, a, specialA_, mode_, rp);
  //CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

inline __device__ size_t dev_mcl_get_id(const size_t c, const size_t bitno, uint64_t* data){
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

//with_density = false
__global__ void kernel_mcl_bucket_counter(
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    int* bucket_counters){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for(int i = tid; i < data_length; i+= gridDim.x * blockDim.x){
    size_t id = dev_mcl_get_id(c, k*c, (uint64_t*)bn_exponents[i]._limbs);
    if(id != 0){
      atomicAdd(&bucket_counters[id], 1);
    }
  }
}
//with_density = true
__global__ void kernel_mcl_bucket_counter(
    const char* density,
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    int* bucket_counters){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for(int i = tid; i < data_length; i+= gridDim.x * blockDim.x){
    if(density[i]){
      size_t id = dev_mcl_get_id(c, k*c, (uint64_t*)bn_exponents[i]._limbs);
      if(id != 0){
        atomicAdd(&bucket_counters[id], 1);
      }
    }
  }
}

void mcl_bucket_counter(
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
    kernel_mcl_bucket_counter<<<blocks, threads, 0, stream>>>(density, bn_exponents, c, k, data_length, bucket_counters);
  }else{
    kernel_mcl_bucket_counter<<<blocks, threads, 0, stream>>>(bn_exponents, c, k, data_length, bucket_counters);
  }
  //CUDA_CHECK(cudaDeviceSynchronize());
}

void mcl_prefix_sum(const int *in, int *out, const int n, CudaStream stream){
  thrust::exclusive_scan(thrust::cuda::par.on(stream), in, in + n, out);
}

__global__ void kernel_mcl_get_bid_and_counter(
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    const int bucket_num,
    int* bucket_counters,
    int* bucket_ids, int* value_ids){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for(int i = tid; i < data_length; i+= gridDim.x * blockDim.x){
    size_t id = dev_mcl_get_id(c, k*c, (uint64_t*)bn_exponents[i]._limbs);
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

__global__ void kernel_mcl_split_to_bucket(
		mcl_bn128_g1 data,
		mcl_bn128_g1 buckets,
		const int data_length,
		const int bucket_num,
		const int* starts,
		const int* value_ids,
		const int* bucket_ids){
	int instance = threadIdx.x + blockIdx.x * blockDim.x;
	if(instance >= data_length) return;
	int bucket_id = bucket_ids[instance];
    using namespace BigInt256;
	if(bucket_id > 0 && bucket_id < bucket_num){
		int src_i = value_ids[instance];
		int dst_i = instance;
        Ect a;
        load(a, data, src_i);
        store(buckets, a, dst_i);
	}
}

void mcl_split_to_bucket(
    mcl_bn128_g1 data, 
    mcl_bn128_g1 out, 
    const bool with_density,
    const char* density,
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    int *starts,
    int *indexs, 
    int* tmp,
    CudaStream stream){
  int threads = 512;
  int blocks = (data_length + threads-1) / threads;
  const int bucket_num = (1<<c);
  int *bucket_ids = tmp, *value_ids = tmp + data_length;
  kernel_mcl_get_bid_and_counter<<<blocks, threads, 0, stream>>>(bn_exponents, c, k, data_length, bucket_num, indexs, bucket_ids, value_ids); 
  thrust::sort_by_key(thrust::cuda::par.on(stream), bucket_ids, bucket_ids + data_length, value_ids); 

  blocks = (data_length + 63) / 64;
  threads = 64;
  kernel_mcl_split_to_bucket<<<blocks, threads, 0, stream>>>(data, out, data_length, bucket_num, starts, value_ids, bucket_ids);
}

__global__ void kernel_mcl_calc_bucket_half_size(const int *starts, const int *ends, int* sizes, const int bucket_num){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < bucket_num){
        const int len = ends[tid] - starts[tid];
        if(len <= 1) sizes[tid] = 0;
        else{
            sizes[tid] = (len + 1)/2;
        }
    }
}

__global__ void kernel_mcl_get_bucket_tids(const int* half_sizes, const int bucket_num, int* bucket_tids, int*bucket_ids){
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

__global__ void kernel_mcl_bucket_reduce_g1_new(
    mcl_bn128_g1 data,
    const int *starts, 
    const int *ends,
    const int *bucket_ids,
    const int *bucket_tids,
    const int total_instances,
    mcl_bn128_g1 t_zero,
    Fp_model one, 
    Fp_model p, 
    Fp_model a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
  int instance = threadIdx.x + blockIdx.x * blockDim.x;
  if(instance >= total_instances) return;
  int bucket_id = bucket_ids[instance];
  int start = starts[bucket_id];
  int bucket_size = ends[bucket_id] - start;
  if(bucket_size < 1) return;

  int half_bucket_size = (bucket_size + 1) / 2;
  int bucket_instance = bucket_tids[instance];

  using namespace BigInt256;

  Int256 lone, lp, la;
  load(lone, one, 0); 
  load(la, a, 0); 
  load(lp, p, 0); 

  Ect result;
  load(result, data, start + bucket_instance);
  for(int i = bucket_instance + half_bucket_size; i < bucket_size; i+= half_bucket_size){
      Ect other;
      load(other, data, start + i);
      add(result, result, other, lone, lp, specialA_, la, mode_, rp);  
  }
  store(data, result, start + bucket_instance);
}

__global__ void kernel_mcl_update_ends2(
        const int *starts, 
        int* half_sizes, 
        int* ends, 
        const int bucket_num){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid < bucket_num){
    int start = starts[tid];
    int end = ends[tid];
    if(end-start > 1) {
        //calc the new ends
        int half_bucket_size = (end-start+1)/2;
        ends[tid] = start + half_bucket_size;
        //calc the new half_sizes
        half_sizes[tid] = half_bucket_size > 1 ? (half_bucket_size + 1) / 2 : 0;
    }else{
        half_sizes[tid] = 0;
    }
  }
}

__global__ void kernel_mcl_copy(
    mcl_bn128_g1 data,
    int* starts, 
    int* ends, 
    mcl_bn128_g1 buckets,
    mcl_bn128_g1 zero,
    const int bucket_num){
  const int instance = threadIdx.x + blockIdx.x * blockDim.x;
  if(instance >= bucket_num) return;
  int bid = instance;
  int start = starts[bid];
  int end = ends[bid];
  using namespace BigInt256;
  if(end - start == 0){
      Ect dev_zero;
      load(dev_zero, zero, 0);
      store(buckets, dev_zero, bid);
  }else{
      Ect a;
      load(a, data, start);
      store(buckets, a, bid);
  }
}

void mcl_bucket_reduce_sum(
    mcl_bn128_g1 data,
    int* starts, int* ends, int* ids,
    int *d_instance_bucket_ids,
    mcl_bn128_g1 buckets,
    const int bucket_num,
    const int data_size,
    mcl_bn128_g1 t_zero,
    Fp_model one, 
    Fp_model p, 
    Fp_model a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp,
    CudaStream stream){
  int *half_sizes, *bucket_ids;
  int* bucket_tids = d_instance_bucket_ids;
  half_sizes = ids;
  bucket_ids = d_instance_bucket_ids + data_size;
  int threads = 256;
  int blocks = (bucket_num + threads-1) / threads;
  if(true){
      kernel_mcl_calc_bucket_half_size<<<blocks, threads, 0, stream>>>(starts, ends, half_sizes, bucket_num);
      //CUDA_CHECK(cudaDeviceSynchronize());
      while(1){
          thrust::inclusive_scan(thrust::cuda::par.on(stream), half_sizes, half_sizes + bucket_num, half_sizes);
          //CUDA_CHECK(cudaDeviceSynchronize());
          threads = 256;
          blocks = (bucket_num + threads-1) / threads;
          kernel_mcl_get_bucket_tids<<<blocks, threads, 0, stream>>>(half_sizes, bucket_num, bucket_tids, bucket_ids);
          //CUDA_CHECK(cudaDeviceSynchronize());
          int total_instances = 0;
          CUDA_CHECK(cudaMemcpyAsync(&total_instances, half_sizes + bucket_num-1, sizeof(int), cudaMemcpyDeviceToHost, stream)); 
          sync(stream); 
          if(total_instances == 0) break;
          const int local_instances = 64;
          threads = local_instances;
          blocks = (total_instances + local_instances - 1) / local_instances;
          kernel_mcl_bucket_reduce_g1_new<<<blocks, threads, 0, stream>>>(data, starts, ends, bucket_ids, bucket_tids, total_instances, t_zero, one, p, a, specialA_, mode_, rp); 
          //CUDA_CHECK(cudaDeviceSynchronize());
          threads = 256;
          blocks = (bucket_num + threads-1) / threads;
          kernel_mcl_update_ends2<<<blocks, threads, 0, stream>>>(starts, half_sizes, ends, bucket_num);
          //CUDA_CHECK(cudaDeviceSynchronize());
      }
  }

  int local_instances = 64;
  threads = local_instances;
  blocks = (bucket_num + local_instances-1) / local_instances;
  kernel_mcl_copy<<<blocks, threads, 0, stream>>>(data, starts, ends, buckets, t_zero, bucket_num);
}

__global__ void kernel_mcl_reverse(
      mcl_bn128_g1 data, mcl_bn128_g1 out, int n, int offset){
  int instance = blockIdx.x * blockDim.x + threadIdx.x;
  using namespace BigInt256;
  for(int i = instance; i < n; i += gridDim.x * blockDim.x){
    int in_i = i * offset;
    int out_i = n - i - 1;
    Ect a;
    load(a, data, in_i);
    store(out, a, out_i);
  }
}

void mcl_reverse(mcl_bn128_g1 in, mcl_bn128_g1 out, const int n, const int offset, CudaStream stream){
  const int threads = 64;
  int reverse_blocks = (n + 63) / 64;
  kernel_mcl_reverse<<<reverse_blocks, threads, 0, stream>>>(in, out, n, offset);
}


__global__ void kernel_mcl_reduce_sum_new(
    mcl_bn128_g1 data, 
    mcl_bn128_g1 out, 
    const int half_n,
    const int n,
    Fp_model one, 
    Fp_model p, 
    Fp_model a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
  int instance = threadIdx.x + blockIdx.x * blockDim.x;
  if(instance >= half_n) return;
  using namespace BigInt256;
  Int256 lone, lp, la;
  load(lone, one, 0); 
  load(la, a, 0); 
  load(lp, p, 0); 

  Ect dev_a;
  load(dev_a, data, instance);
  for(int i = instance + half_n; i < n; i+= half_n){
    Ect dev_b;
    load(dev_b, data, i);
    add(dev_a, dev_a, dev_b, lone, lp, specialA_, la, mode_, rp);  
  }
  store(out, dev_a, instance);
}

void mcl_bn128_g1_reduce_sum2(
    mcl_bn128_g1 data, 
    mcl_bn128_g1 out, 
    const uint32_t n,
    Fp_model one, 
    Fp_model p, 
    Fp_model a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp,
    CudaStream stream){
  int len = n-1;
  const int instances = 64;
  int threads = instances;
  int half_len = (len + 1) / 2;
  int blocks = (half_len + instances - 1) / instances;
  kernel_mcl_reduce_sum_new<<<blocks, threads, 0, stream>>>(data, out, half_len, len, one, p, a, specialA_, mode_, rp);
  len = half_len;
  while(len > 1){
      int half_len = (len + 1) / 2;
      int blocks = (half_len + instances - 1) / instances;
      kernel_mcl_reduce_sum_new<<<blocks, threads, 0, stream>>>(out, out, half_len, len, one, p, a, specialA_, mode_, rp);
      len = half_len;
  }
}

__global__ void kernel_mcl_bn128_g2_reduce_sum_one_range5(
    mcl_bn128_g2 values, 
    const Fp_model scalars,
    const size_t *index_it,
    mcl_bn128_g2 partial, 
    const int ranges_size, 
    const int range_id_offset,
    const uint32_t* firsts,
    const uint32_t* seconds,
    char* flags,
    const mcl_bn128_g2 t_zero,
    const Fp_model one, 
    const Fp_model p, 
    const Fp_model2 a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
  const int instance = threadIdx.x + blockIdx.x * blockDim.x;

  //const int instance_offset = (range_id_offset + blockIdx.y) * gridDim.x * blockDim.x;
  const int first = firsts[range_id_offset + blockIdx.y];
  const int second = seconds[range_id_offset + blockIdx.y];
  const int reduce_depth = second - first;//30130
  if(reduce_depth <= 1) return;
  const int half_depth = (reduce_depth + 1) / 2;

  if(instance >= half_depth) return;

  using namespace BigInt256;
  BigInt256::Point la;
  memcpy(la.c0, a.c0.mont_repr_data, sizeof(Int256));
  memcpy(la.c1, a.c1.mont_repr_data, sizeof(Int256));
  BigInt256::Ect2 result;
  if(flags[index_it[first + instance]] == 1){
      load(result, values, first+instance);
  }else{
      load(result, t_zero, 0);
  }
  for(int i = first + instance+half_depth; i < first + reduce_depth; i+= half_depth){
      const int j = index_it[i];
      if(flags[j] == 1){
          BigInt256::Ect2 dev_b;
          load(dev_b, values, i);
          BigInt256::add_g2(result, result, dev_b, (Int*)one.mont_repr_data, (Int*)p.mont_repr_data, specialA_, la, mode_, rp);  
      }
  }
  store(partial, result, first+instance);
}

__global__ void kernel_mcl_bn128_g2_reduce_sum_one_range7(
    mcl_bn128_g2 values, 
    Fp_model scalars,
    const size_t *index_it,
    mcl_bn128_g2 partial, 
    const int ranges_size, 
    const int range_id_offset,
    const uint32_t* firsts,
    const uint32_t* seconds,
    char* flags,
    const mcl_bn128_g2 t_zero,
    const Fp_model one, 
    const Fp_model p, 
    const Fp_model2 a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
  const int instance = threadIdx.x + blockIdx.x * blockDim.x;

  //const int instance_offset = (range_id_offset + blockIdx.y) * gridDim.x * blockDim.x;
  const int first = firsts[range_id_offset + blockIdx.y];
  const int second = seconds[range_id_offset + blockIdx.y];
  const int reduce_depth = second - first;//30130
  if(reduce_depth <= 1) return;
  const int half_depth = (reduce_depth + 1) / 2;

  if(instance >= half_depth) return;

  using namespace BigInt256;
  BigInt256::Point la;
  memcpy(la.c0, a.c0.mont_repr_data, sizeof(Int256));
  memcpy(la.c1, a.c1.mont_repr_data, sizeof(Int256));
  BigInt256::Ect2 result;
  load(result, values, first+instance);
  Int256 lone, lp;
  memcpy(lone, one.mont_repr_data, 32);
  memcpy(lp, p.mont_repr_data, 32);
  for(int i = first + instance+half_depth; i < first + reduce_depth; i+= half_depth){
      BigInt256::Ect2 dev_b;
      load(dev_b, values, i);
      BigInt256::add_g2(result, result, dev_b, lone, lp, specialA_, la, mode_, rp);  
  }
  store(partial, result, first+instance);
}

__global__ void kernel_mcl_bn128_g2_reduce_sum_one_range6(
    mcl_bn128_g2 partial, 
    const int n, 
    const uint32_t* firsts,
    const Fp_model one, 
    const Fp_model p, 
    const Fp_model2 a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
  //int instance = threadIdx.x;

  using namespace BigInt256;

  BigInt256::Point la;
  memcpy(la.c0, a.c0.mont_repr_data, sizeof(Int256));
  memcpy(la.c1, a.c1.mont_repr_data, sizeof(Int256));
  Int256 lone, lp;
  memcpy(lone, one.mont_repr_data, 32);
  memcpy(lp, p.mont_repr_data, 32);
  BigInt256::Ect2 result;
  load(result, partial, firsts[0]);
  for(int i = 1; i < n; i++){
      BigInt256::Ect2 dev_b;
      load(dev_b, partial, firsts[i]);
      BigInt256::add_g2(result, result, dev_b, lone, lp, specialA_, la, mode_, rp);  
  }
  store(partial, result, 0);
}

inline __device__ void printInt256(const BigInt256::Int256 x){
    for(int i = 0; i < 4; i++){
        printf("%lu ", x[i]);
    }
    printf("\n");
}
__global__ void kernel_mcl_bn128_g2_reduce_sum_new(
    const int range_id,
    const int range_offset,
    mcl_bn128_g2 values, 
    const Fp_model scalars,
    const size_t *index_it,
    mcl_bn128_g2 partial, 
    const int ranges_size, 
    const uint32_t* firsts,
    const uint32_t* seconds,
    const char* flags,
    const Fp_model one, 
    const Fp_model p, 
    const Fp_model2 a, 
    const int specialA_,
    const int mode_,
    mcl_bn128_g2 t_zero,
    const uint64_t rp){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int first = firsts[blockIdx.y];
  int second = seconds[blockIdx.y];
  int reduce_depth = second - first;//30130
  int offset = blockIdx.y * gridDim.x * blockDim.x;

  using BigInt256::Int;
  using BigInt256::Int256;

  BigInt256::Point la;
  memcpy(la.c0, a.c0.mont_repr_data, sizeof(Int256));
  memcpy(la.c1, a.c1.mont_repr_data, sizeof(Int256));
  BigInt256::Ect2 result;
  load(result, t_zero, 0);
  Int256 local_one, local_p;
  memcpy(local_one, one.mont_repr_data, sizeof(BigInt256::Int256));
  memcpy(local_p, p.mont_repr_data, sizeof(BigInt256::Int256));
  for(int i = first + tid; i < first + reduce_depth; i+= gridDim.x * blockDim.x){
    const int j = index_it[i];
    if(flags[j] == 1){
        BigInt256::Ect2 dev_b;
        load(dev_b, values, i);
        BigInt256::add_g2(result, result, dev_b, local_one, local_p, specialA_, la, mode_, rp);  
    }
  }
  store(partial, result, offset + tid);
}

__global__ void kernel_mcl_bn128_g2_reduce_sum_after_new(
    mcl_bn128_g2 partial, 
    const int half_n, 
    const int n, 
    const Fp_model one, 
    const Fp_model p, 
    const Fp_model2 a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= half_n) return;

  using BigInt256::Int;
  using BigInt256::Int256;
  BigInt256::Point la;
  memcpy(la.c0, a.c0.mont_repr_data, sizeof(Int256));
  memcpy(la.c1, a.c1.mont_repr_data, sizeof(Int256));

  BigInt256::Ect2 result;
  load(result, partial, tid);
  for(int i = half_n; i < n; i+= half_n){
        BigInt256::Ect2 dev_b;
        load(dev_b, partial, tid+i);
        BigInt256::add_g2(result, result, dev_b, (Int*)one.mont_repr_data, (Int*)p.mont_repr_data, specialA_, la, mode_, rp);  
  }
  store(partial, result, tid);
}

int mcl_bn128_g2_reduce_sum_new(
    mcl_bn128_g2 values, 
    Fp_model scalars, 
    const size_t *index_it,
    mcl_bn128_g2 partial, 
    uint32_t *counters,
    char* flags,
    const uint32_t ranges_size,
    const uint32_t *firsts,
    uint32_t *seconds,
    const mcl_bn128_g2 t_zero,
    const Fp_model field_zero,
    const Fp_model field_one,
    char *density,
    cgbn_mem_t<BITS>* bn_exponents,
    cgbn_mem_t<BITS>* const field_modulus, const uint64_t field_inv,
    const Fp_model one, const Fp_model p, const Fp_model2 a, const int specialA_, const int mode_, const uint64_t rp,
    const int max_reduce_depth, const int values_size, cudaStream_t stream
    ){

  if(true){
      uint32_t threads = 64;//TPI * 64;
      const int local_instances = 64 * BlockDepth;
      uint32_t block_x =  (max_reduce_depth + local_instances - 1) / local_instances;
      dim3 blocks(block_x, ranges_size, 1);
      kernel_mcl_bn128_g1_reduce_sum_pre_new<<<blocks, threads, 0, stream>>>(scalars, index_it, counters, flags, ranges_size, firsts, seconds, field_zero, field_one, density, bn_exponents, field_modulus, field_inv);
  }else{
      int threads = 256;
      int blocks = (values_size + threads - 1) / threads;
      kernel_reduce_sum_pre_new<<<blocks, threads, 0, stream>>>(scalars, flags, field_zero, field_one, values_size, density, bn_exponents, field_modulus, field_inv);
  }

  const int blocks_per_range = REDUCE_BLOCKS_PER_RANGE;
  const int threads_per_block = INSTANCES_PER_BLOCK;

  if(true){
      int n = max_reduce_depth;
      const int local_instances2 = 64;
      int threads = local_instances2;
      uint32_t block_x2 =  ((n+1)/2 + local_instances2 - 1) / local_instances2;
      dim3 blocks2(block_x2, ranges_size, 1);
      kernel_mcl_bn128_g2_reduce_sum_one_range5<<<blocks2, dim3(threads, 1, 1), 0, stream>>>(values, scalars, index_it, partial, ranges_size, 0, firsts, seconds, flags, t_zero, one, p, a, specialA_, mode_, rp);
      const int update_threads = 64;
      const int update_blocks = (ranges_size + update_threads - 1) / update_threads;
      kernel_mcl_update_seconds<<<update_blocks, update_threads, 0, stream>>>(firsts, seconds, ranges_size);
      //CUDA_CHECK(cudaDeviceSynchronize());
      n = (n+1)/2;
      while(n>=2){
          uint32_t block_x2 =  ((n+1)/2 + local_instances2 - 1) / local_instances2;
          dim3 blocks2(block_x2, ranges_size, 1);
          kernel_mcl_bn128_g2_reduce_sum_one_range7<<<blocks2, dim3(threads, 1, 1), 0, stream>>>(partial, scalars, index_it, partial, ranges_size, 0, firsts, seconds, flags, t_zero, one, p, a, specialA_, mode_, rp);
          kernel_mcl_update_seconds<<<update_blocks, update_threads, 0, stream>>>(firsts, seconds, ranges_size);
          n = (n+1)/2;
      }
      kernel_mcl_bn128_g2_reduce_sum_one_range6<<<1, 1, 0, stream>>>(partial, ranges_size, firsts, one, p, a, specialA_, mode_, rp);
  }else{
      kernel_mcl_bn128_g2_reduce_sum_new<<<dim3(blocks_per_range, ranges_size, 1), threads_per_block, 0, stream>>>(0, 0, values, scalars, index_it, partial, ranges_size, firsts, seconds, flags, one, p, a, specialA_, mode_, t_zero, rp);

      int n = blocks_per_range * INSTANCES_PER_BLOCK * ranges_size;
      while(n>=2){
          int half_n = n / 2;
          int blocks = (half_n + INSTANCES_PER_BLOCK-1) / INSTANCES_PER_BLOCK;
          kernel_mcl_bn128_g2_reduce_sum_after_new<<<blocks, threads_per_block, 0, stream>>>(partial, half_n, n, one, p, a, specialA_, mode_, rp);
          n /= 2;
      }
  }
  return 0;
}

__global__ void kernel_mcl_split_to_bucket_g2(
    mcl_bn128_g2 data,
    mcl_bn128_g2 buckets,
    const int data_length,
    const int bucket_num,
    const int* starts,
    const int* value_ids,
    const int* bucket_ids){
  int instance = threadIdx.x + blockIdx.x * blockDim.x;
  if(instance >= data_length) return;
  int bucket_id = bucket_ids[instance];
  using namespace BigInt256;
  if(bucket_id > 0 && bucket_id < bucket_num){
      int src_i = value_ids[instance];
      int dst_i = instance;
          Ect2 a;
          load(a, data, src_i);
          store(buckets, a, dst_i);
    }
}

void mcl_split_to_bucket_g2(
    mcl_bn128_g2 data, 
    mcl_bn128_g2 out, 
    const bool with_density,
    const char* density,
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    int *starts,
    int *indexs, 
    int* tmp,
    CudaStream stream){
  int threads = 512;
  int blocks = (data_length + threads-1) / threads;
  const int bucket_num = (1<<c);
  int *bucket_ids = tmp, *value_ids = tmp + data_length;
  kernel_mcl_get_bid_and_counter<<<blocks, threads, 0, stream>>>(bn_exponents, c, k, data_length, bucket_num, indexs, bucket_ids, value_ids); 
  thrust::sort_by_key(thrust::cuda::par.on(stream), bucket_ids, bucket_ids + data_length, value_ids); 

  blocks = (data_length + 63) / 64;
  threads = 64;
  kernel_mcl_split_to_bucket_g2<<<blocks, threads, 0, stream>>>(data, out, data_length, bucket_num, starts, value_ids, bucket_ids);
}


__global__ void kernel_mcl_bucket_reduce_g2_new(
    mcl_bn128_g2 data,
    const int *starts, 
    const int *ends,
    const int *bucket_ids,
    const int *bucket_tids,
    const int total_instances,
    mcl_bn128_g2 t_zero,
    Fp_model one, 
    Fp_model p, 
    Fp_model2 a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
  int instance = threadIdx.x + blockIdx.x * blockDim.x;
  if(instance >= total_instances) return;
  int bucket_id = bucket_ids[instance];
  int start = starts[bucket_id];
  int bucket_size = ends[bucket_id] - start;
  if(bucket_size <= 1) return;

  int half_bucket_size = (bucket_size + 1) / 2;
  int bucket_instance = bucket_tids[instance];

  using namespace BigInt256;
  BigInt256::Point la;
  memcpy(la.c0, a.c0.mont_repr_data, sizeof(Int256));
  memcpy(la.c1, a.c1.mont_repr_data, sizeof(Int256));
  BigInt256::Ect2 result;
  load(result, data, start + bucket_instance);
  Int256 local_one, local_p;
  memcpy(local_one, one.mont_repr_data, sizeof(Int256));
  memcpy(local_p, p.mont_repr_data, sizeof(Int256));
  for(int i = bucket_instance + half_bucket_size; i < bucket_size; i+= half_bucket_size){
    BigInt256::Ect2 other;
    load(other, data, start+i);
    BigInt256::add_g2(result, result, other, local_one, local_p, specialA_, la, mode_, rp);  
  }
  store(data, result, start + bucket_instance);
}

template<int Offset>
__global__ void kernel_mcl_copy_g2(
    mcl_bn128_g2 data,
    int* starts, 
    int* ends, 
    mcl_bn128_g2 buckets,
    mcl_bn128_g2 zero,
    const int bucket_num){
  const int instance = threadIdx.x + blockIdx.x * blockDim.x;
  if(instance >= bucket_num) return;
  int bid = instance;
  int start = starts[bid];
  int end = ends[bid];
  using namespace BigInt256;
  if(end - start == 0){
      Ect2 dev_zero;
      load(dev_zero, zero, 0);
      store(buckets, dev_zero, bid * Offset);
  }else{
      Ect2 a;
      load(a, data, start);
      store(buckets, a, bid * Offset);
  }
}

void mcl_bucket_reduce_sum_g2(
    mcl_bn128_g2 data,
    int* starts, int* ends, int* ids,
    int *d_instance_bucket_ids,
    mcl_bn128_g2 buckets,
    const int bucket_num,
    const int data_size,
    mcl_bn128_g2 t_zero,
    Fp_model one, 
    Fp_model p, 
    Fp_model2 a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp,
    CudaStream stream){
  int *half_sizes, *bucket_ids;
  int* bucket_tids = d_instance_bucket_ids;
  half_sizes = ids;
  bucket_ids = d_instance_bucket_ids + data_size;
  int threads = 256;
  int blocks = (bucket_num + threads-1) / threads;
  if(true){
      kernel_mcl_calc_bucket_half_size<<<blocks, threads, 0, stream>>>(starts, ends, half_sizes, bucket_num);
      //CUDA_CHECK(cudaDeviceSynchronize());
      while(1){
          thrust::inclusive_scan(thrust::cuda::par.on(stream), half_sizes, half_sizes + bucket_num, half_sizes);
          //CUDA_CHECK(cudaDeviceSynchronize());
          threads = 256;
          blocks = (bucket_num + threads-1) / threads;
          kernel_mcl_get_bucket_tids<<<blocks, threads>>>(half_sizes, bucket_num, bucket_tids, bucket_ids);
          //CUDA_CHECK(cudaDeviceSynchronize());
          int total_instances = 0;
          CUDA_CHECK(cudaMemcpyAsync(&total_instances, half_sizes + bucket_num-1, sizeof(int), cudaMemcpyDeviceToHost, stream)); 
          sync(stream); 
          if(total_instances == 0) break;
          const int local_instances = 64;
          if(false){
          }else{
              threads = local_instances;
              blocks = (total_instances + local_instances - 1) / local_instances;
              kernel_mcl_bucket_reduce_g2_new<<<blocks, threads, 0, stream>>>(data, starts, ends, bucket_ids, bucket_tids, total_instances, t_zero, one, p, a, specialA_, mode_, rp); 
          }
          //CUDA_CHECK(cudaDeviceSynchronize());
          threads = 256;
          blocks = (bucket_num + threads-1) / threads;
          kernel_mcl_update_ends2<<<blocks, threads, 0, stream>>>(starts, half_sizes, ends, bucket_num);
          //CUDA_CHECK(cudaDeviceSynchronize());
      }
  }else{
  }
  int local_instances = 64;
  threads = local_instances;
  blocks = (bucket_num + local_instances-1) / local_instances;
  kernel_mcl_copy_g2<BUCKET_INSTANCES><<<blocks, threads, 0, stream>>>(data, starts, ends, buckets, t_zero, bucket_num);
}

__global__ void kernel_mcl_reverse_g2(
    mcl_bn128_g2 data, mcl_bn128_g2 out, int n, int offset){
  int instance = blockIdx.x * blockDim.x + threadIdx.x;
  using namespace BigInt256;
  for(int i = instance; i < n; i += gridDim.x * blockDim.x){
      int in_i = i * offset;
      int out_i = n - i - 1;
      Ect2 a;
      load(a, data, in_i);
      store(out, a, out_i);
    }
}

void mcl_reverse_g2(mcl_bn128_g2 in, mcl_bn128_g2 out, const int n, const int offset, CudaStream stream){
  const int threads = 64;
  int reverse_blocks = (n + 63) / 64;
  kernel_mcl_reverse_g2<<<reverse_blocks, threads, 0, stream>>>(in, out, n, offset);
  //CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void kernel_mcl_reduce_sum_g2_new(
    mcl_bn128_g2 data, 
    mcl_bn128_g2 out, 
    const int half_n,
    const int n,
    Fp_model one, 
    Fp_model p, 
    Fp_model2 a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
  int instance = threadIdx.x + blockIdx.x * blockDim.x;
  if(instance >= half_n) return;

  using namespace BigInt256;
  BigInt256::Point la;
  memcpy(la.c0, a.c0.mont_repr_data, sizeof(Int256));
  memcpy(la.c1, a.c1.mont_repr_data, sizeof(Int256));
  Int256 local_one, local_p;
  memcpy(local_one, one.mont_repr_data, sizeof(BigInt256::Int256));
  memcpy(local_p, p.mont_repr_data, sizeof(BigInt256::Int256));

  BigInt256::Ect2 dev_a;
  load(dev_a, data, instance);

  for(int i = instance + half_n; i < n; i+= half_n){
      BigInt256::Ect2 dev_b;
      load(dev_b, data, i);
      BigInt256::add_g2(dev_a, dev_a, dev_b, local_one, local_p, specialA_, la, mode_, rp);  
  }
  store(out, dev_a, instance);
}

void mcl_bn128_g2_reduce_sum2(
    mcl_bn128_g2 data, 
    mcl_bn128_g2 out, 
    const uint32_t n,
    Fp_model one, 
    Fp_model p, 
    Fp_model2 a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp,
    CudaStream stream){
  int len = n-1;
  const int instances = 64;
  int threads = instances ;
  int half_len = (len + 1) / 2;
  int blocks = (half_len + instances - 1) / instances;
  kernel_mcl_reduce_sum_g2_new<<<blocks, threads, 0, stream>>>(data, out, half_len, len, one, p, a, specialA_, mode_, rp);
  len = half_len;
  while(len > 1){
      int half_len = (len + 1) / 2;
      int blocks = (half_len + instances - 1) / instances;
      kernel_mcl_reduce_sum_g2_new<<<blocks, threads, 0, stream>>>(out, out, half_len, len, one, p, a, specialA_, mode_, rp);
      len = half_len;
  }
}

}// namespace gpu

#endif
