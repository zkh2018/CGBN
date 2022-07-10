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
  void resize(const int count);
  void resize_host(const int count);
  void release();
  void release_host();
  void copy_from_cpu(const alt_bn128_g1& g1, CudaStream stream = 0);
  void copy_from_gpu(const alt_bn128_g1& g1, CudaStream stream = 0);
  void copy_to_cpu(alt_bn128_g1& g1, CudaStream stream = 0);
  void clear(CudaStream stream = 0);
};

int alt_bn128_g1_add(alt_bn128_g1 a, alt_bn128_g1 b, alt_bn128_g1 c, const uint32_t count, cgbn_mem_t<BITS>* max_value, cgbn_mem_t<BITS>* modulus, const uint64_t inv);

int alt_bn128_g1_reduce_sum(
    alt_bn128_g1 values, 
    Fp_model scalars, 
    const size_t *index_it,
    alt_bn128_g1 partial, 
    uint32_t *counters,
    const uint32_t ranges_size,
    const uint32_t *firsts,
    const uint32_t *seconds,
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
    char* flags,
    const uint32_t ranges_size,
    const uint32_t *firsts,
    const uint32_t *seconds,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    Fp_model field_zero,
    Fp_model field_one,
    char *density,
    cgbn_mem_t<BITS>* bn_exponents,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    cgbn_mem_t<BITS>* field_modulus, const uint64_t field_inv,
    const int max_reduce_depth);

int alt_bn128_g1_reduce_sum_one_range5(
    alt_bn128_g1 values, 
    Fp_model scalars, 
    const size_t *index_it,
    alt_bn128_g1 partial, 
    uint32_t *counters,
    char* flags,
    const uint32_t ranges_size,
    const uint32_t *firsts,
    uint32_t *seconds,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    Fp_model field_zero,
    Fp_model field_one,
    char *density,
    cgbn_mem_t<BITS>* bn_exponents,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    cgbn_mem_t<BITS>* field_modulus, const uint64_t field_inv,
    const int max_reduce_depth,
    cudaStream_t stream);

void alt_bn128_g1_reduce_sum(
    alt_bn128_g1 partial_in, 
    const uint32_t *counters_in,
    alt_bn128_g1 partial_out, 
    uint32_t *counters_out,
    const uint32_t ranges_size,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    const int max_reduce_depth);

void alt_bn128_g1_reduce_sum2(
    alt_bn128_g1 data, 
    alt_bn128_g1 out, 
    const uint32_t n,
    cgbn_mem_t<BITS>* max_value,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    CudaStream stream = 0);

void alt_bn128_g1_reduce_sum_one_instance(
    alt_bn128_g1 partial_in, 
    const uint32_t *counters_in,
    alt_bn128_g1 partial_out, 
    uint32_t *counters_out,
    cgbn_mem_t<BITS>* max_value,
    alt_bn128_g1 t_zero,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv,
    const int max_reduce_depth);

void alt_bn128_g1_elementwise_mul_scalar(
    Fp_model datas, 
    Fp_model sconst, 
    const uint32_t n,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv);

void fft_internal(
    Fp_model in,
    const int n,
    Fp_model twiddles,
    const int twiddles_len,
    const int twiddle_offset,
    const int *in_offsets,
    const int *out_offsets,
    const int *stage_lengths,
    const int *radixs,
    const int *strides,
    cgbn_mem_t<BITS>* max_value, 
    cgbn_mem_t<BITS>* modulus, 
    const uint64_t inv,
    Fp_model out);

void fft_copy(
    Fp_model in,
    Fp_model out,
    const int *in_offsets,
    const int *out_offsets,
    const int *strides,
    const int n,
    const int radix);

void butterfly_2(
        Fp_model out,
        Fp_model twiddles, 
        const int twiddle_offset,
        const int *strides, 
        const uint32_t stage_length, 
        const int* out_offsets, 
        const int n,
        cgbn_mem_t<BITS>* max_value, 
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv);

void butterfly_4(
        Fp_model out,
        Fp_model twiddles, 
        const int twiddles_len,
        const int twiddle_offset,
        const int* strides, 
        const uint32_t stage_length, 
        const int* out_offsets, 
        const int n,
        cgbn_mem_t<BITS>* max_value, 
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv);

void multiply_by_coset_and_constant(
        Fp_model inputs,
        const int n,
        Fp_model g,
        Fp_model c, 
        Fp_model one,
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv,
        const int gmp_bits);

void calc_xor(
        Fp_model xor_results,
        const int n,
        const int offset,
        Fp_model g,
        Fp_model one,
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv,
        const int gmp_num_bits);

void multiply(
        Fp_model inputs,
        Fp_model xor_results,
        const int n,
        const int offset,
        Fp_model c, 
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv);

void calc_H(
        Fp_model A,
        Fp_model B,
        Fp_model C,
        Fp_model out,
        Fp_model Z_inverse_at_coset,
        const int n,
        cgbn_mem_t<BITS>* max_value, 
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv);

void init_error_report();
void warm_up();

} //gpu

#endif
