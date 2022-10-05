#ifndef CGBN_FP_H
#define CGBN_FP_H

#include "cgbn_math.h"

namespace gpu{

struct Fp_model {
  cgbn_mem_t<BITS>* mont_repr_data;
  //cgbn_mem_t<BITS>* modulus_data;
  //uint64_t inv;
  int _count = 0;

  Fp_model(){
    _count = 0;
    mont_repr_data = nullptr;
    //modulus_data = nullptr;
  }
  Fp_model(const int count);
  void init(const int count);
  void init_host(const int count);
  void resize(const int count);
  void resize_host(const int count);
  void release();
  void release_host();
  void copy_from_cpu(const Fp_model& fp, CudaStream stream=0);
  void copy_from_gpu(const Fp_model& fp, CudaStream stream=0);
  void copy_to_cpu(Fp_model& fp, CudaStream stream=0);
  void clear(CudaStream stream = 0);
};

}//gpu

#endif
