#ifndef LOW_PREFIX_SUM_FUNC_CUH
#define LOW_PREFIX_SUM_FUNC_CUH

#include "cgbn_ect.cuh"
#include "low_func_gpu.h"
#include <stdint.h>
#include <thrust/scan.h>
#include "bigint_256.cuh"

namespace gpu{

template<int ReduceDepthPerBlock>
__global__ void kernel_mcl_prefix_sum_pre_new(
      mcl_bn128_g1 data, 
      const int n,
      int stride,
      Fp_model one, 
      Fp_model p, 
      Fp_model a, 
      const int specialA_,
      const int mode_,
    const uint64_t rp){
  int local_instance = threadIdx.x;

  using namespace BigInt256;
  Int256 lone, lp, la;
  load(lone, one, 0); 
  load(la, a, 0); 
  load(lp, p, 0); 

  int offset = blockIdx.x * ReduceDepthPerBlock;
  int index = (local_instance + 1) * stride * 2 - 1;
  if(index < ReduceDepthPerBlock && offset + index < n){
    Ect dev_a, dev_b;
    load(dev_a, data, offset + index);
    load(dev_b, data, offset + index - stride);
    add(dev_a, dev_a, dev_b, lone, lp, specialA_, la, mode_, rp);  
    store(data, dev_a, offset + index);
  }
}

template<int ReduceDepthPerBlock>
__global__ void kernel_mcl_prefix_sum_post_new(
      mcl_bn128_g1 data, 
      mcl_bn128_g1 block_sums, 
      const int n,
      int stride, bool save_block_sum,
      Fp_model one, 
      Fp_model p, 
      Fp_model a, 
      const int specialA_,
      const int mode_,
      const uint64_t rp){
  int local_instance = threadIdx.x;

  using namespace BigInt256;
  Int256 lone, lp, la;
  load(lone, one, 0); 
  load(la, a, 0); 
  load(lp, p, 0); 

  int offset = blockIdx.x * ReduceDepthPerBlock;
  int index = (local_instance + 1) * stride * 2 - 1;
  if(index + stride < ReduceDepthPerBlock && offset + index + stride < n){
    Ect dev_a, dev_b;
    load(dev_a, data, offset + index + stride);
    load(dev_b, data, offset + index);
    add(dev_a, dev_a, dev_b, lone, lp, specialA_, la, mode_, rp);  
    store(data, dev_a, offset + index + stride);
  }
  if(save_block_sum && local_instance == 0){
    Ect dev_a;
    load(dev_a, data, blockIdx.x * ReduceDepthPerBlock + ReduceDepthPerBlock - 1);
    store(block_sums, dev_a, blockIdx.x);
  }
}

template<int Instances, int RealInstances, bool SaveBlockSum>
__global__ void kernel_mcl_prefix_sum_new(
    mcl_bn128_g1 data, 
    mcl_bn128_g1 block_sums, 
    const int n,
    Fp_model one, 
    Fp_model p, 
    Fp_model a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
    int instance = threadIdx.x + blockIdx.x * blockDim.x;
    int local_instance = threadIdx.x;
    int local_instances = Instances;
    if(instance >= n) return;

    using namespace BigInt256;
    Int256 lone, lp, la;
    load(lone, one, 0); 
    load(la, a, 0); 
    load(lp, p, 0); 

    int offset = blockIdx.x * local_instances;
    for(int stride = 1; stride <= RealInstances; stride *= 2){
        __syncthreads();
        int index = (local_instance+1)*stride*2 - 1; 
        if(index < Instances && offset + index < n){
            Ect dev_a, dev_b;
            load(dev_a, data, offset + index);
            load(dev_b, data, offset + index - stride);
            add(dev_a, dev_a, dev_b, lone, lp, specialA_, la, mode_, rp, true);  
            store(data, dev_a, offset + index);
        }
        __syncthreads();
    }
    for (unsigned int stride = (Instances >> 1); stride > 0 ; stride>>=1) {
        __syncthreads();
        int index = (local_instance+1)*stride*2 - 1;
        if(index + stride < Instances && offset + index + stride < n){
            Ect dev_a, dev_b;
            load(dev_a, data, offset + index + stride);
            load(dev_b, data, offset + index);
            add(dev_a, dev_a, dev_b, lone, lp, specialA_, la, mode_, rp, true);  
            store(data, dev_a, offset + index + stride);
        }
    }
    __syncthreads();
    if(SaveBlockSum && local_instance == 0){
        Ect dev_a;
        load(dev_a, data, blockIdx.x * local_instances + local_instances-1);
        store(block_sums, dev_a, blockIdx.x);
    }
}


template<int BlockInstances>
__global__ void kernel_mcl_add_block_sum_new(
    mcl_bn128_g1 data, 
    mcl_bn128_g1 block_sums, 
    const int n,
    Fp_model one, 
    Fp_model p, 
    Fp_model a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
  const int instances = BlockInstances;
  int instance = blockIdx.x * blockDim.x + threadIdx.x;
  if(instances + instance >= n) return;
  using namespace BigInt256;
  Int256 lone, lp, la;
  load(lone, one, 0); 
  load(la, a, 0); 
  load(lp, p, 0); 

  Ect dev_block_sum, dev_a;
  load(dev_block_sum, block_sums, blockIdx.x);
  load(dev_a, data, instance + instances);//offset = instances
  add(dev_a, dev_a, dev_block_sum, lone, lp, specialA_, la, mode_, rp);  
  store(data, dev_a, instance + instances);
}

void mcl_prefix_sum(
    mcl_bn128_g1 data, 
    mcl_bn128_g1 block_sums, 
    mcl_bn128_g1 block_sums2, 
    const int n,//2^16
    Fp_model one, 
    Fp_model p, 
    Fp_model a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp,
    CudaStream stream){
  const int threads = 64;
  int instances = threads;//64
  int prefix_sum_blocks = (n + instances - 1) / instances;//2^10
  int prefix_sum_blocks2 = (prefix_sum_blocks + instances-1) / instances;//2^4

  for(int stride = 1; stride <= 32; stride *= 2){
    int instances = 32 / stride;
    int threads = instances;
    kernel_mcl_prefix_sum_pre_new<64><<<prefix_sum_blocks, threads, 0, stream>>>(data, n, stride, one, p, a, specialA_, mode_, rp);
  }
  for(int stride = 32; stride > 0; stride /= 2){
    int instances = 32 / stride;
    int threads = instances;
    bool save_block_sum = (stride == 1);
    kernel_mcl_prefix_sum_post_new<64><<<prefix_sum_blocks, threads, 0, stream>>>(data, block_sums, n, stride, save_block_sum, one, p, a, specialA_, mode_, rp);
  }

  for(int stride = 1; stride <= 32; stride *= 2){
    int instances = 32 / stride;
    int threads = instances;
    kernel_mcl_prefix_sum_pre_new<64><<<prefix_sum_blocks2, threads, 0, stream>>>(block_sums, prefix_sum_blocks, stride, one, p, a, specialA_, mode_, rp);
  }
  for(int stride = 32; stride > 0; stride /= 2){
    int instances = 32 / stride;
    int threads = instances;
    bool save_block_sum = (stride == 1);
    kernel_mcl_prefix_sum_post_new<64><<<prefix_sum_blocks2, threads, 0, stream>>>(block_sums, block_sums2, prefix_sum_blocks, stride, save_block_sum, one, p, a, specialA_, mode_, rp);
  }
  
  kernel_mcl_prefix_sum_new<16, 8, false><<<1, 8, 0, stream>>>(block_sums2, block_sums2, prefix_sum_blocks2, one, p, a, specialA_, mode_, rp);
  kernel_mcl_add_block_sum_new<64><<<prefix_sum_blocks2-1, threads, 0, stream>>>(block_sums, block_sums2, prefix_sum_blocks, one, p, a, specialA_, mode_, rp);
  kernel_mcl_add_block_sum_new<64><<<prefix_sum_blocks-1, threads, 0, stream>>>(data, block_sums, n, one, p, a, specialA_, mode_, rp);
  //CUDA_CHECK(cudaDeviceSynchronize());
}

}// namespace gpu

#endif
