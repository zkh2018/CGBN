#ifndef CGBN_ALT_BN128_G1_H
#define CGBN_ALT_BN128_G1_H

#include "cgbn_fp.h"

namespace gpu{

struct alt_bn128_g1{
  Fp_model x, y, z;
  alt_bn128_g1(){}
  alt_bn128_g1(const int count);
  void init(const int count);
  void init_host(const int count);
  void release();
  void release_host();
  void copy_from_cpu(const alt_bn128_g1& g1);
  void copy_to_cpu(alt_bn128_g1& g1);
};

int alt_bn128_g1_add(alt_bn128_g1 a, alt_bn128_g1 b, alt_bn128_g1 c, const uint32_t count, uint32_t *tmp_res, cgbn_mem_t<BITS>* max_value, cgbn_mem_t<BITS>* modulus, const uint64_t inv);

int alt_bn128_g1_reduce_sum(
    alt_bn128_g1 values, 
    Fp_model scalars, 
    const size_t *index_it,
    alt_bn128_g1 partial, 
    uint32_t *counters,
    const uint32_t ranges_size,
    const uint32_t *firsts,
    const uint32_t *seconds,
    uint32_t *tmp_res, 
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    alt_bn128_g1 t_one,
    Fp_model field_zero,
    Fp_model field_one,
    char *density,
    cgbn_mem_t<BITS>* bn_exponents,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv
    );

int alt_bn128_g1_reduce_sum_one_range(
    alt_bn128_g1 values, 
    Fp_model scalars, 
    const size_t *index_it,
    alt_bn128_g1 partial, 
    uint32_t *counters,
    const uint32_t ranges_size,
    const uint32_t *firsts,
    const uint32_t *seconds,
    uint32_t *tmp_res, 
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    alt_bn128_g1 t_one,
    Fp_model field_zero,
    Fp_model field_one,
    char *density,
    cgbn_mem_t<BITS>* bn_exponents,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    const int max_reduce_depth);

void alt_bn128_g1_reduce_sum(
    alt_bn128_g1 partial_in, 
    const uint32_t *counters_in,
    alt_bn128_g1 partial_out, 
    uint32_t *counters_out,
    const uint32_t ranges_size,
    uint32_t *tmp_res, 
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    const int max_reduce_depth);

void init_error_report();

} //gpu

#endif
