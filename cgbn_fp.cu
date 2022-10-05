#include "cgbn_fp.h"
#include "cgbn_math.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>

#include "gpu_support.h"

namespace gpu{

Fp_model::Fp_model(const int count){
  init(count);
}

void Fp_model::init(const int count){
  _count = count;
  gpu_malloc((void**)&mont_repr_data, count * sizeof(cgbn_mem_t<BITS>));
  //gpu_malloc((void**)&modulus_data, count * sizeof(cgbn_mem_t<BITS>));
}
void Fp_model::init_host(const int count){
  _count = count;
  mont_repr_data = (cgbn_mem_t<BITS>*)malloc(count * sizeof(cgbn_mem_t<BITS>));
  //modulus_data = (cgbn_mem_t<BITS>*)malloc(count * sizeof(cgbn_mem_t<BITS>));
}
void Fp_model::resize(const int count){
  if(_count < count){
    if(_count > 0){
      gpu_free(mont_repr_data);
    }
    _count = count;
    gpu_malloc((void**)&mont_repr_data, count * sizeof(cgbn_mem_t<BITS>));
  }
  //gpu_malloc((void**)&modulus_data, count * sizeof(cgbn_mem_t<BITS>));
}
void Fp_model::resize_host(const int count){
  if(_count < count){
    if(_count > 0){
      free(mont_repr_data);
    }
    _count = count;
    mont_repr_data = (cgbn_mem_t<BITS>*)malloc(count * sizeof(cgbn_mem_t<BITS>));
  }
  //modulus_data = (cgbn_mem_t<BITS>*)malloc(count * sizeof(cgbn_mem_t<BITS>));
}
void Fp_model::release(){
  if(mont_repr_data != nullptr)
      gpu_free(mont_repr_data);
  //gpu_free(modulus_data);
}
void Fp_model::release_host(){
  if(mont_repr_data != nullptr)
      free(mont_repr_data);
  //free(modulus_data);
}

void Fp_model::copy_from_cpu(const Fp_model& fp, CudaStream stream){
  copy_cpu_to_gpu(mont_repr_data, fp.mont_repr_data, sizeof(cgbn_mem_t<BITS>) * _count, stream);
  //copy_cpu_to_gpu(modulus_data, fp.modulus_data, sizeof(cgbn_mem_t<BITS>) * _count);
}
void Fp_model::copy_from_gpu(const Fp_model& fp, CudaStream stream){
  copy_gpu_to_gpu(mont_repr_data, fp.mont_repr_data, sizeof(cgbn_mem_t<BITS>) * _count, stream);
}
void Fp_model::copy_to_cpu(Fp_model& fp, CudaStream stream){
  copy_gpu_to_cpu(fp.mont_repr_data, mont_repr_data, sizeof(cgbn_mem_t<BITS>) * _count, stream);
  //copy_gpu_to_cpu(fp.modulus_data, modulus_data, sizeof(cgbn_mem_t<BITS>) * _count);
}

void Fp_model::clear(CudaStream stream){
  gpu_set_zero(this->mont_repr_data, _count * sizeof(cgbn_mem_t<BITS>), stream);
}

}//gpu
