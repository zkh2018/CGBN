
#include "cgbn_fp.h"

#include <cuda_runtime.h>
#include <cuda.h>

#include "cgbn/cgbn.h"
#include "utility/cpu_support.h"
#include "utility/cpu_simple_bn_math.h"
#include "utility/gpu_support.h"

#define TPI 16
typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

namespace gpu{

__global__ void kernel_fp_sub(cgbn_error_report_t* report, cgbn_mem_t<BITS>* const in1, cgbn_mem_t<BITS>* const in2, cgbn_mem_t<BITS>* module_data, const uint32_t count){
  int instance = (blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance >= count) return;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());
  env_t::cgbn_t tin1, tin2, tmodule_data, tscratch;
  cgbn_load(bn_env, tin1, in1);
  cgbn_load(bn_env, tin2, in2);
  cgbn_load(bn_env, tmodule_data, module_data);

  if(cgbn_compare(bn_env, tin1, tin2) < 0){
    uint32_t carry = cgbn_add(bn_env, tscratch, tin1, tmodule_data);
    if(carry > 0){
      cgbn_mem_t<BITS> max_value;
      for(int i = 0; i < BITS/32; i++){
        max_value._limbs[i] = 0xffffffff;
      }

      env_t::cgbn_t tmax_value;
      cgbn_load(bn_env, tmax_value, &max_value);
      cgbn_sub(bn_env, tin1, tmax_value, tin2);
      cgbn_add(bn_env, tmax_value, tin1, tscratch);
      cgbn_add_ui32(bn_env, tin1, tmax_value, 1);
    }else{
      cgbn_sub(bn_env, tin1, tscratch, tin2);
    }
    cgbn_store(bn_env, in1, tin1);
  }else{
      cgbn_sub(bn_env, tscratch, tin1, tin2);
      cgbn_store(bn_env, in1, tscratch);
  }
}

int fp_sub(cgbn_mem_t<BITS>* in1, cgbn_mem_t<BITS>* in2, cgbn_mem_t<BITS>* module_data, const uint32_t count){
  cgbn_error_report_t *report;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_fp_sub<<<1, TPI>>>(report, in1, in2, module_data, count);
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  return 0;
}

}//gpu
