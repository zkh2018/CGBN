#ifndef CGBN_MULTI_EXP_H
#define CGBN_MULTI_EXP_H

#include "cgbn_math.h"

namespace gpu{

  const int BUCKET_INSTANCES = 64;
  const int BUCKET_INSTANCES_G2 = 64;

  void bucket_counter(
      const bool with_density,
      const char* density,
      const cgbn_mem_t<BITS>* bn_exponents,
      const int c, const int k,
      const int data_length,
      const int bucket_nums,
      int* bucket_counters, CudaStream stream = 0);

  void prefix_sum(const int *in, int *out, const int n, CudaStream stream = 0);


}// namespace gpu
#endif

