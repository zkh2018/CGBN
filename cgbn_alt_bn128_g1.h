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

void alt_bn128_g1_elementwise_mul_scalar(
    Fp_model datas, 
    Fp_model sconst, 
    const uint32_t n,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv);

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

void warm_up();

} //gpu

#endif
