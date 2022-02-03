
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

} // namespace gpu

