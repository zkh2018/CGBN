#ifndef CGBN_FP2_H
#define CGBN_FP2_H

#include "cgbn_math.h"
#include "cgbn_fp.h"

namespace gpu{

struct Fp_model2 {
  Fp_model c0, c1;

  Fp_model2(){}
  Fp_model2(const int count);
  void init(const int count);
  void init_host(const int count);
  void release();
  void release_host();
  void copy_from_cpu(const Fp_model2& fp);
  void copy_to_cpu(Fp_model2& fp);
  void clear(CudaStream stream = 0);
};

}//namespace gpu

#endif
