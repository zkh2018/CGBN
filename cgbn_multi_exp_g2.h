#ifndef CGBN_MULTI_EXP_G2_H
#define CGBN_MULTI_EXP_G2_H

#include "cgbn_alt_bn128_g2.h"

namespace gpu{

  void prefix_sum_g2(const alt_bn128_g2 in, alt_bn128_g2 out, const int n, CudaStream stream = 0);

  void split_to_bucket_g2(
      alt_bn128_g2 data, 
      alt_bn128_g2 out, 
      const char* density,
      const cgbn_mem_t<BITS>* bn_exponents,
      const int c, const int k,
      const int data_length,
      int *indexs, CudaStream stream = 0);

  void bucket_reduce_sum_g2(
      alt_bn128_g2 data,
      int* starts, int* ends, int *ids,
      int *d_instance_bucket_ids,
      alt_bn128_g2 buckets,
      const int bucket_num,
      const int data_size,
      cgbn_mem_t<BITS>* max_value,
      alt_bn128_g2 t_zero,
      cgbn_mem_t<BITS>* modulus, const uint64_t inv,
      Fp_model non_residue,
      CudaStream stream = 0);

  void prefix_sum_g2(
      alt_bn128_g2 in, 
      alt_bn128_g2 out, 
      alt_bn128_g2 block_sums, 
      const int n,
      const int offset,
      cgbn_mem_t<BITS>* max_value,
      cgbn_mem_t<BITS>* modulus, const uint64_t inv, 
      Fp_model non_residue, 
      CudaStream stream = 0);

  void reverse_g2(alt_bn128_g2 in, alt_bn128_g2 out, const int n, const int offset, CudaStream stream = 0);

  void prefix_sum_g2(
      alt_bn128_g2 data, 
      alt_bn128_g2 block_sums, 
      alt_bn128_g2 block_sums2, 
      const int n,
      cgbn_mem_t<BITS>* max_value,
      cgbn_mem_t<BITS>* modulus, const uint64_t inv, 
      Fp_model non_residue, 
      CudaStream stream = 0);


}// namespace gpu
#endif

