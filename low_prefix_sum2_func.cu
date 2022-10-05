#ifndef LOW_PREFIX_SUM2_FUNC_CUH
#define LOW_PREFIX_SUM2_FUNC_CUH

#include <stdint.h>
#include "cgbn_ect.cuh"
#include "low_func_gpu.h"
#include "bigint_256.cuh"

namespace gpu{

const int ReduceDepthPerBlock = 64;
//template<int ReduceDepthPerBlock>
__global__ void kernel_mcl_prefix_sum_pre_g2_new(
    mcl_bn128_g2 data, 
    const int n,
    int stride,
    Fp_model one, 
    Fp_model p, 
    Fp_model2 a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
  const int local_instance = threadIdx.x;

  int offset = blockIdx.x * ReduceDepthPerBlock;
  int index = (local_instance + 1) * stride * 2 - 1;
  using namespace BigInt256;
  BigInt256::Point la;
  memcpy(la.c0, a.c0.mont_repr_data, sizeof(Int256));
  memcpy(la.c1, a.c1.mont_repr_data, sizeof(Int256));
  Int256 local_one, local_p;
  memcpy(local_one, one.mont_repr_data, sizeof(Int256));
  memcpy(local_p, p.mont_repr_data, sizeof(Int256));

  if(index < ReduceDepthPerBlock && offset + index < n){
      BigInt256::Ect2 dev_a, dev_b;
      load(dev_a, data, offset + index);
      load(dev_b, data, offset + index-stride);

      BigInt256::add_g2(dev_a, dev_a, dev_b, local_one, local_p, specialA_, la, mode_, rp);  

      store(data, dev_a, offset+index);
    }
}

//template<int ReduceDepthPerBlock>
__global__ void kernel_mcl_prefix_sum_post_g2_new(
    mcl_bn128_g2 data, 
    mcl_bn128_g2 block_sums, 
    const int n,
    int stride, bool save_block_sum,
    Fp_model one, 
    Fp_model p, 
    Fp_model2 a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
  int local_instance = threadIdx.x;

  using namespace BigInt256;
  BigInt256::Point la;
  memcpy(la.c0, a.c0.mont_repr_data, sizeof(Int256));
  memcpy(la.c1, a.c1.mont_repr_data, sizeof(Int256));
  Int256 local_one, local_p;
  memcpy(local_one, one.mont_repr_data, sizeof(BigInt256::Int256));
  memcpy(local_p, p.mont_repr_data, sizeof(BigInt256::Int256));

  int offset = blockIdx.x * ReduceDepthPerBlock;
  int index = (local_instance + 1) * stride * 2 - 1;
  if(index + stride < ReduceDepthPerBlock && offset + index + stride < n){
      BigInt256::Ect2 dev_a, dev_b;
      load(dev_a, data, offset+index+stride);
      load(dev_b, data, offset+index);

      BigInt256::add_g2(dev_a, dev_a, dev_b, local_one, local_p, specialA_, la, mode_, rp);  

      store(data, dev_a, offset + index + stride);
  }
  if(save_block_sum && local_instance == 0){
    Ect2 a; 
    load(a, data, blockIdx.x * ReduceDepthPerBlock + ReduceDepthPerBlock-1);
    store(block_sums, a, blockIdx.x);
  }
}

template<int Instances, int RealInstances, bool SaveBlockSum>
__global__ void kernel_mcl_prefix_sum_g2_new(
    mcl_bn128_g2 data, 
    mcl_bn128_g2 block_sums, 
    const int n,
    Fp_model one, 
    Fp_model p, 
    Fp_model2 a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
  int instance = threadIdx.x + blockIdx.x * blockDim.x;
  int local_instance = threadIdx.x;
  if(instance >= n) return;

  using namespace BigInt256;
  BigInt256::Point la;
  memcpy(la.c0, a.c0.mont_repr_data, sizeof(Int256));
  memcpy(la.c1, a.c1.mont_repr_data, sizeof(Int256));
  Int256 local_one, local_p;
  memcpy(local_one, one.mont_repr_data, sizeof(BigInt256::Int256));
  memcpy(local_p, p.mont_repr_data, sizeof(BigInt256::Int256));

  int offset = blockIdx.x * Instances;
  for(int stride = 1; stride <= RealInstances; stride *= 2){
      __syncthreads();
      int index = (local_instance+1)*stride*2 - 1; 
      if(index < Instances && offset + index < n){
          BigInt256::Ect2 dev_a, dev_b;
          load(dev_a, data, offset + index);
          load(dev_b, data, offset + index-stride);

          BigInt256::add_g2(dev_a, dev_a, dev_b, local_one, local_p, specialA_, la, mode_, rp);  

          store(data, dev_a, offset + index);

      }
      __syncthreads();
  }
  for (unsigned int stride = (Instances >> 1); stride > 0 ; stride>>=1) {
      __syncthreads();
      int index = (local_instance+1)*stride*2 - 1;
      if(index + stride < Instances && offset + index + stride < n){
          BigInt256::Ect2 dev_a, dev_b;
          load(dev_a, data, offset + index + stride);
          load(dev_b, data, offset + index);

          BigInt256::add_g2(dev_a, dev_a, dev_b, local_one, local_p, specialA_, la, mode_, rp);  

          store(data, dev_a, offset + index + stride);
      }
  }
  __syncthreads();
  if(SaveBlockSum && local_instance == 0){
      Ect2 a;
      load(a, data, blockIdx.x * Instances + Instances -1);
      store(block_sums, a, blockIdx.x);
  }
}

template<int BlockInstances>
__global__ void kernel_mcl_add_block_sum_g2_new(
    mcl_bn128_g2 data, 
    mcl_bn128_g2 block_sums, 
    const int n,
    Fp_model one, 
    Fp_model p, 
    Fp_model2 a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
    const int instances = BlockInstances;
    int instance = blockIdx.x * blockDim.x + threadIdx.x;
    if(instances + instance >= n) return;

    using namespace BigInt256;
    BigInt256::Point la;
    memcpy(la.c0, a.c0.mont_repr_data, sizeof(Int256));
    memcpy(la.c1, a.c1.mont_repr_data, sizeof(Int256));

    Int256 local_one, local_p;
    memcpy(local_one, one.mont_repr_data, sizeof(BigInt256::Int256));
    memcpy(local_p, p.mont_repr_data, sizeof(BigInt256::Int256));

    BigInt256::Ect2 dev_a, dev_block_sum;
    load(dev_a, data, instance + instances);
    load(dev_block_sum, block_sums, blockIdx.x);
    BigInt256::add_g2(dev_a, dev_a, dev_block_sum, local_one, local_p, specialA_, la, mode_, rp);  
    store(data, dev_a, instance + instances);
}

void mcl_prefix_sum_g2(
    mcl_bn128_g2 data, 
    mcl_bn128_g2 block_sums, 
    mcl_bn128_g2 block_sums2, 
    const int n,//2^16
    Fp_model one, 
    Fp_model p, 
    Fp_model2 a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp,
    CudaStream stream){

  if(true){
      int threads = 64;
      int instances = threads;//64
      int prefix_sum_blocks = (n + instances - 1) / instances;//2^10
      int prefix_sum_blocks2 = (prefix_sum_blocks + instances-1) / instances;//2^4
      for(int stride = 1; stride <= 32; stride *= 2){
          int instances = 32 / stride;
          int threads = instances;
          kernel_mcl_prefix_sum_pre_g2_new<<<prefix_sum_blocks, threads, 0, stream>>>(data, n, stride, one, p, a, specialA_, mode_, rp);
      }

      for(int stride = 32; stride > 0; stride /= 2){
          int instances = 32 / stride;
          int threads = instances;
          bool save_block_sum = (stride == 1);
          kernel_mcl_prefix_sum_post_g2_new<<<prefix_sum_blocks, threads, 0, stream>>>(data, block_sums, n, stride, save_block_sum, one, p, a, specialA_, mode_, rp);
      }

      for(int stride = 1; stride <= 32; stride *= 2){
          int instances = 32 / stride;
          int threads = instances;
          kernel_mcl_prefix_sum_pre_g2_new<<<prefix_sum_blocks2, threads, 0, stream>>>(block_sums, prefix_sum_blocks, stride, one, p, a, specialA_, mode_, rp);
      }


      for(int stride = 32; stride > 0; stride /= 2){
          int instances = 32 / stride;
          int threads = instances;
          bool save_block_sum = (stride == 1);
          kernel_mcl_prefix_sum_post_g2_new<<<prefix_sum_blocks2, threads, 0, stream>>>(block_sums, block_sums2, prefix_sum_blocks, stride, save_block_sum, one, p, a, specialA_, mode_, rp);
      }

      kernel_mcl_prefix_sum_g2_new<16, 8, false><<<1, 16/2, 0, stream>>>(block_sums2, block_sums2, prefix_sum_blocks2, one, p, a, specialA_, mode_, rp);

      kernel_mcl_add_block_sum_g2_new<64><<<prefix_sum_blocks2-1, threads, 0, stream>>>(block_sums, block_sums2, prefix_sum_blocks, one, p, a, specialA_, mode_, rp);

      kernel_mcl_add_block_sum_g2_new<64><<<prefix_sum_blocks-1, threads, 0, stream>>>(data, block_sums, n, one, p, a, specialA_, mode_, rp);
  }
}

}// namespace gpu

#endif
