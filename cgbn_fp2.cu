#include "cgbn_fp2.h"
#include "cgbn_fp.cuh"

#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>

#include "cgbn/cgbn.h"
#include "utility/cpu_support.h"
#include "utility/cpu_simple_bn_math.h"
#include "utility/gpu_support.h"

namespace gpu{

Fp_model2::Fp_model2(const int count){
  init(count);
}

void Fp_model2::init(const int count){
  c0.init(count);
  c1.init(count);
}
void Fp_model2::init_host(const int count){
  c0.init_host(count);
  c1.init_host(count);
}

void Fp_model2::release(){
  c0.release();
  c1.release();
}
void Fp_model2::release_host(){
  c0.release_host();
  c1.release_host();
}

void Fp_model2::copy_from_cpu(const Fp_model2& fp){
  c0.copy_from_cpu(fp.c0);
  c1.copy_from_cpu(fp.c1);
}
void Fp_model2::copy_to_cpu(Fp_model2& fp){
  c0.copy_to_cpu(fp.c0);
  c1.copy_to_cpu(fp.c1);
}

void Fp_model2::clear(CudaStream stream){
  c0.clear(stream);
  c1.clear(stream);
}

}//namespace gpu
