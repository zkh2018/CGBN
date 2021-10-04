#ifndef CGBN_MULTI_EXP_H
#define CGBN_MULTI_EXP_H

namespace gpu{
  void bucket_counter(
      const char* density,
      const cgbn_mem_t<BITS>* bn_exponents,
      const int c, const int k,
      const int data_length,
      const int bucket_nums,
      int* bucket_counters);

  void prefix_sum(const int *in, int *out, const int n);
  void prefix_sum(const alt_bn128_g1 in, alt_bn128_g1 out, const int n);

  void split_to_bucket(
      alt_bn128_g1 data, 
      alt_bn128_g1 out, 
      const char* density,
      const cgbn_mem_t<BITS>* bn_exponents,
      const int c, const int k,
      const int data_length,
      int *indexs);

  void bucket_reduce_sum(
      alt_bn128_g1 data,
      int* starts, int* ends,
      alt_bn128_g1 buckets,
      const int bucket_num,
      cgbn_mem_t<BITS>* max_value,
      alt_bn128_g1 t_zero,
      cgbn_mem_t<BITS>* modulus, const uint64_t inv);

  void prefix_sum(
      alt_bn128_g1 in, 
      alt_bn128_g1 out, 
      alt_bn128_g1 block_sums, 
      const int n,
      const int offset,
      cgbn_mem_t<BITS>* max_value,
      cgbn_mem_t<BITS>* modulus, const uint64_t inv);

  void reverse(alt_bn128_g1 in, alt_bn128_g1 out, const int n, const int offset);

  void prefix_sum(
      alt_bn128_g1 data, 
      alt_bn128_g1 block_sums, 
      alt_bn128_g1 block_sums2, 
      const int n,
      cgbn_mem_t<BITS>* max_value,
      cgbn_mem_t<BITS>* modulus, const uint64_t inv);


}// namespace gpu
#endif

