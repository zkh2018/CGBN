#ifndef LOW_FUNC_GPU_H
#define LOW_FUNC_GPU_H

#include "cgbn_fp.h"
#include "cgbn_ect.h"

namespace gpu{

void gpu_mcl_add(uint32_t* z, uint32_t *x, uint32_t *y, uint32_t *p);
void gpu_mcl_sub(uint32_t* z, uint32_t *x, uint32_t *y, uint32_t *p);
void gpu_mcl_mul(uint32_t* z, uint32_t *x, uint32_t *y, uint32_t *p, const uint64_t rp);
void gpu_mcl_ect_add(mcl_bn128_g1 R, mcl_bn128_g1 P, mcl_bn128_g1 Q, Fp_model one, Fp_model p, Fp_model a, const int specialA_, const int model_, const uint64_t rp);

} // namespace gpu

#endif
