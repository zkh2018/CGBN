#ifndef CGBN_FP_H
#define CGBN_FP_H

#include "cgbn_math.h"

namespace gpu{

int fp_sub(cgbn_mem_t<BITS>* in1, cgbn_mem_t<BITS>* in2, cgbn_mem_t<BITS>* module_data, const uint32_t count);

}//gpu

#endif
