
#include <gmp.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "cgbn/cgbn.h"
#include "low_func_gpu.h"
#include "low_func.cuh"
#include "gpu_support.h"

namespace gpu{

void gpu_mcl_add(uint32_t* z, uint32_t *x, uint32_t *y, uint32_t *p){
  cgbn_error_report_t *report = nullptr;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_mcl_add<<<1, 8>>>(report, z, x, y, p);
}

void gpu_mcl_sub(uint32_t* z, uint32_t *x, uint32_t *y, uint32_t *p){
  cgbn_error_report_t *report = nullptr;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_mcl_sub<<<1, 8>>>(report, z, x, y, p);
}

void gpu_mcl_mul(uint32_t* z, uint32_t *x, uint32_t *y, uint32_t *p, const uint64_t rp){
  cgbn_error_report_t *report = nullptr;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_mcl_mul<<<1, 8>>>(report, z, x, y, p, rp);
}

void gpu_mcl_ect_add(mcl_bn128_g1 R, mcl_bn128_g1 P, mcl_bn128_g1 Q, Fp_model one, Fp_model p, Fp_model a, const int specialA_, const int model_, const uint64_t rp){
  cgbn_error_report_t *report = nullptr;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_ect_add<<<1, 8>>>(report, R, P, Q, one, p, a, specialA_, model_, rp);
}

void gpu_mcl_sub_g2(uint32_t* z, uint32_t *x, uint32_t *y, uint32_t *p){
  cgbn_error_report_t *report = nullptr;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_mcl_sub_g2<<<1, 8>>>(report, z, x, y, p);
}
void gpu_mcl_add_g2(uint32_t* z, uint32_t *x, uint32_t *y, uint32_t *p){
  cgbn_error_report_t *report = nullptr;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_mcl_add_g2<<<1, 8>>>(report, z, x, y, p);
}

void gpu_mont_red(uint32_t* z, uint32_t *xy, uint32_t *p, const uint64_t rp){
  cgbn_error_report_t *report = nullptr;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_mont_red<<<1, 8>>>(report, z, xy, p, rp);
}

void gpu_mul_wide(uint32_t*z, uint32_t*x, uint32_t*y){
  cgbn_error_report_t *report = nullptr;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_mul_wide<<<1, 8>>>(report, z, x, y);
}

void gpu_sub_wide(uint32_t*z, uint32_t*x, uint32_t*y){
  cgbn_error_report_t *report = nullptr;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_sub_wide<<<1, 8>>>(report, z, x, y);
}
void gpu_fp2Dbl_mulPreW(uint32_t*z, uint32_t*x, uint32_t*y, uint32_t*p){
  cgbn_error_report_t *report = nullptr;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_fp2Dbl_mulPreW<<<1, 8>>>(report, z, x, y, p);
}
void gpu_sqr_g2(uint32_t*y, uint32_t*x, uint32_t*p, const uint64_t rp){
  cgbn_error_report_t *report = nullptr;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_sqr_g2<<<1, 8>>>(report, y, x, p, rp);
}
void gpu_mcl_mul_g2(uint32_t* z, uint32_t*x, uint32_t*y, uint32_t*p, const uint64_t rp){
  cgbn_error_report_t *report = nullptr;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_mcl_mul_g2<<<1, 8>>>(report, z, x, y, p, rp);
}

void gpu_mcl_ect_add_g2(mcl_bn128_g2 R, mcl_bn128_g2 P, mcl_bn128_g2 Q, Fp_model one, Fp_model p, Fp_model2 a, const int specialA_, const int model_, const uint64_t rp){
  cgbn_error_report_t *report = nullptr;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_ect_add_g2<<<1, 8>>>(report, R, P, Q, one, p, a, specialA_, model_, rp);
}

} // namespace gpu

