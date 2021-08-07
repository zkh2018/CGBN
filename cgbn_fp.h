#ifndef CGBN_FP_H
#define CGBN_FP_H

#include "cgbn_math.h"

namespace gpu{

int fp_add(cgbn_mem_t<BITS>* in1, cgbn_mem_t<BITS>* in2, cgbn_mem_t<BITS>* module_data, cgbn_mem_t<BITS>* max_value, const uint32_t count);

int fp_sub(cgbn_mem_t<BITS>* in1, cgbn_mem_t<BITS>* in2, cgbn_mem_t<BITS>* module_data, cgbn_mem_t<BITS>* max_value, const uint32_t count);

/**
 *
 */
int fp_mul_reduce(cgbn_mem_t<BITS>* in1, cgbn_mem_t<BITS>* in2, uint64_t inv, cgbn_mem_t<BITS>* module_data, cgbn_mem_t<BITS>* tmp_buffer, uint32_t *res, const uint32_t count);

}//gpu

#endif
