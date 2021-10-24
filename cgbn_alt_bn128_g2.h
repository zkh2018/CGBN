#ifndef CGBN_ALT_BN128_G2_H
#define CGBN_ALT_BN128_G2_H

#include "cgbn_fp2.h"

namespace gpu{

struct alt_bn128_g2{
  Fp_model2 x, y, z;
  alt_bn128_g2(){}
  alt_bn128_g2(const int count);
  void init(const int count);
  void init_host(const int count);
  void release();
  void release_host();
  void copy_from_cpu(const alt_bn128_g2& g1);
  void copy_to_cpu(alt_bn128_g2& g1);
  void clear(CudaStream stream = 0);
};

}//namespace gpu

#endif
