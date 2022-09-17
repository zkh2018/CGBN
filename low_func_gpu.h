#ifndef LOW_FUNC_GPU_H
#define LOW_FUNC_GPU_H

#include "cgbn_fp.h"
#include "cgbn_ect.h"

namespace gpu{

void gpu_mcl_add(uint32_t* z, uint32_t *x, uint32_t *y, uint32_t *p);
void gpu_mcl_sub(uint32_t* z, uint32_t *x, uint32_t *y, uint32_t *p);
void gpu_mcl_mul(uint32_t* z, uint32_t *x, uint32_t *y, uint32_t *p, const uint64_t rp, const bool print);
void gpu_mcl_ect_add(mcl_bn128_g1 R, mcl_bn128_g1 P, mcl_bn128_g1 Q, Fp_model one, Fp_model p, Fp_model a, const int specialA_, const int model_, const uint64_t rp);
void gpu_mcl_ect_add_new(mcl_bn128_g1 R, mcl_bn128_g1 P, mcl_bn128_g1 Q, Fp_model one, Fp_model p, Fp_model a, const int specialA_, const int model_, const uint64_t rp);

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
    cgbn_mem_t<BITS>* field_modulus, const uint64_t field_inv,
    Fp_model one, Fp_model p, Fp_model a, const int specialA_, const int mode_, const uint64_t rp,
    const int max_reduce_depth, cudaStream_t stream);

void mcl_bucket_counter(
    const bool with_density,
    const char* density,
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    const int bucket_nums,
    int* bucket_counters,
    CudaStream stream);
    
void mcl_prefix_sum(const int *in, int *out, const int n, CudaStream stream);

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
    CudaStream stream);

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
    CudaStream stream);

void mcl_bucket_reduce_sum_one_bucket(
    mcl_bn128_g1 data,
    const int bucket_id,
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
    CudaStream stream);

void mcl_reverse(mcl_bn128_g1 in, mcl_bn128_g1 out, const int n, const int offset, CudaStream stream);

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
    CudaStream stream);

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
    CudaStream stream);

void gpu_mcl_sub_g2(uint32_t* z, uint32_t *x, uint32_t *y, uint32_t *p);
void gpu_mcl_add_g2(uint32_t* z, uint32_t *x, uint32_t *y, uint32_t *p);
void gpu_mont_red(uint32_t* z, uint32_t *xy, uint32_t *p, const uint64_t rp);
void gpu_mul_wide(uint32_t*z, uint32_t*x, uint32_t*y);
void gpu_sub_wide(uint32_t*z, uint32_t*x, uint32_t*y);
void gpu_fp2Dbl_mulPreW(uint32_t*z, uint32_t*x, uint32_t*y, uint32_t*p);
void gpu_sqr_g2(uint32_t*y, uint32_t*x, uint32_t*p, const uint64_t rp);
void gpu_mcl_mul_g2(uint32_t* z, uint32_t*x, uint32_t*y, uint32_t*p, const uint64_t rp);
void gpu_mcl_ect_add_g2(mcl_bn128_g2 R, mcl_bn128_g2 P, mcl_bn128_g2 Q, Fp_model one, Fp_model p, Fp_model2 a, const int specialA_, const int model_, const uint64_t rp);
void gpu_mcl_ect_add_g2_new(mcl_bn128_g2 R, mcl_bn128_g2 P, mcl_bn128_g2 Q, Fp_model one, Fp_model p, Fp_model2 a, const int specialA_, const int model_, const uint64_t rp);


///mcl_g2
int mcl_bn128_g2_reduce_sum(
    mcl_bn128_g2 values, 
    Fp_model scalars, 
    const size_t *index_it,
    mcl_bn128_g2 partial, 
    uint32_t *counters,
    char* flags,
    const uint32_t ranges_size,
    const uint32_t *firsts,
    uint32_t *seconds,
    mcl_bn128_g2 t_zero,
    Fp_model field_zero,
    Fp_model field_one,
    char *density,
    cgbn_mem_t<BITS>* bn_exponents,
    cgbn_mem_t<BITS>* field_modulus, const uint64_t field_inv,
    Fp_model one, Fp_model p, Fp_model2 a, const int specialA_, const int mode_, const uint64_t rp,
    const int max_reduce_depth, cudaStream_t stream);

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
    mcl_bn128_g2 t_zero,
    Fp_model field_zero,
    Fp_model field_one,
    char *density,
    cgbn_mem_t<BITS>* bn_exponents,
    cgbn_mem_t<BITS>* field_modulus, const uint64_t field_inv,
    Fp_model one, Fp_model p, Fp_model2 a, const int specialA_, const int mode_, const uint64_t rp,
    const int max_reduce_depth, const int values_size, cudaStream_t stream);


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
    CudaStream stream);

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
    CudaStream stream);

void mcl_reverse_g2(mcl_bn128_g2 in, mcl_bn128_g2 out, const int n, const int offset, CudaStream stream);

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
    CudaStream stream);

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
    CudaStream stream);


} // namespace gpu

#endif
