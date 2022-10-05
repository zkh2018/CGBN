#ifndef CGBN_ALT_BN128_G2_H
#define CGBN_ALT_BN128_G2_H

#include "cgbn_fp2.h"

namespace gpu{

const int REDUCE_BLOCKS_PER_RANGE = 512;
const int INSTANCES_PER_BLOCK = 32;

struct alt_bn128_g2{
  Fp_model2 x, y, z;
  alt_bn128_g2(){}
  alt_bn128_g2(const int count);
  void init(const int count);
  void init_host(const int count);
  void resize(const int count);
  void resize_host(const int count);
  void release();
  void release_host();
  void copy_from_cpu(const alt_bn128_g2& host, CudaStream stream = 0);
  void copy_from_gpu(const alt_bn128_g2& gpu, CudaStream stream = 0);
  void copy_to_cpu(alt_bn128_g2& host, CudaStream stream = 0);
  void clear(CudaStream stream = 0);
};


}//namespace gpu

#endif
