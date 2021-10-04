#ifndef CGBN_FP_H
#define CGBN_FP_H

#include "cgbn_math.h"

namespace gpu{

struct Fp_model {
  cgbn_mem_t<BITS>* mont_repr_data;
  //cgbn_mem_t<BITS>* modulus_data;
  //uint64_t inv;
  int _count;

  Fp_model(){
    _count = 0;
    mont_repr_data = nullptr;
    //modulus_data = nullptr;
  }
  Fp_model(const int count);
  void init(const int count);
  void init_host(const int count);
  void release();
  void release_host();
  void copy_from_cpu(const Fp_model& fp);
  void copy_to_cpu(Fp_model& fp);
  void clear();
};

int fp_add(cgbn_mem_t<BITS>* in1, cgbn_mem_t<BITS>* in2, cgbn_mem_t<BITS>* module_data, cgbn_mem_t<BITS>* max_value, const uint32_t count);

int fp_sub(cgbn_mem_t<BITS>* in1, cgbn_mem_t<BITS>* in2, cgbn_mem_t<BITS>* module_data, cgbn_mem_t<BITS>* max_value, const uint32_t count);

int fp_mul_reduce(cgbn_mem_t<BITS>* in1, cgbn_mem_t<BITS>* in2, uint64_t inv, cgbn_mem_t<BITS>* module_data, cgbn_mem_t<BITS>* tmp_buffer, uint32_t *res, const uint32_t count);

}//gpu

#endif
