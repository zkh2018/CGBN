#ifndef CGBN_ALT_BN128_G1_H
#define CGBN_ALT_BN128_G1_H

#include "cgbn_fp.h"

namespace gpu{

struct alt_bn128_g1{
  Fp_model x, y, z;
  alt_bn128_g1(){}
  alt_bn128_g1(const int count);
  void init(const int count);
  void init_host(const int count);
  void release();
  void release_host();
  void copy_from_cpu(const alt_bn128_g1& g1);
  void copy_to_cpu(alt_bn128_g1& g1);
};

int alt_bn128_g1_add(alt_bn128_g1 a, alt_bn128_g1 b, alt_bn128_g1 c, const uint32_t count, uint32_t *tmp_res, cgbn_mem_t<BITS>* tmp_buffer, cgbn_mem_t<BITS>* max_value, bool debug);

}//gpu

#endif
