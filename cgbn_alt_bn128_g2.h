#ifndef CGBN_ALT_BN128_G2_H
#define CGBN_ALT_BN128_G2_H

#include "cgbn_fp2.h"

namespace gpu{

const int REDUCE_BLOCKS_PER_RANGE = 512;
const int INSTANCES_PER_BLOCK = 32;

struct alt_bn128_g2{
  Fp_model2 x, y, z;
  alt_bn128_g2(){}
  alt_bn128_g2(const int count);
  void init(const int count);
  void init_host(const int count);
  void release();
  void release_host();
  void copy_from_cpu(const alt_bn128_g2& host);
  void copy_from_gpu(const alt_bn128_g2& gpu);
  void copy_to_cpu(alt_bn128_g2& host);
  void clear(CudaStream stream = 0);
};

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
    const int max_reduce_depth);

void alt_bn128_g2_reduce_sum2(
    alt_bn128_g2 data, 
    alt_bn128_g2 out, 
    const uint32_t n,
    cgbn_mem_t<BITS>* max_value,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    Fp_model non_residue, 
    CudaStream stream = 0);

}//namespace gpu

#endif
