#ifndef CGBN_FP_CUH
#define CGBN_FP_CUH

#include "cgbn_math.h"
#include <cuda_runtime.h>
#include <cuda.h>

#include "cgbn/cgbn.h"
#include "utility/cpu_support.h"
#include "utility/cpu_simple_bn_math.h"
#include "utility/gpu_support.h"

namespace gpu{

#define TPI 8
typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;
#define max_threads_per_block  (512/TPI)

inline __device__ void device_fp_add(env_t& bn_env, cgbn_mem_t<BITS>* const in1, cgbn_mem_t<BITS>* const in2, cgbn_mem_t<BITS>* module_data, cgbn_mem_t<BITS>* max_value){
  env_t::cgbn_t tin1, tin2, tmodule_data, tscratch;

  cgbn_load(bn_env, tin1, in1);
  cgbn_load(bn_env, tin2, in2);
  cgbn_load(bn_env, tmodule_data, module_data);

  uint32_t carry = cgbn_add(bn_env, tscratch, tin1, tin2);
  if(carry || cgbn_compare(bn_env, tscratch, tmodule_data) >= 0){
    if(carry){
      env_t::cgbn_t tmax_value;
      cgbn_load(bn_env, tmax_value, max_value);
      cgbn_sub(bn_env, tin1, tmax_value, tmodule_data);
      cgbn_add(bn_env, tin2, tin1, tscratch);
      cgbn_add_ui32(bn_env, tin1, tin2, 1);
    }else{
      cgbn_sub(bn_env, tin1, tscratch, tmodule_data);
    }
    cgbn_store(bn_env, in1, tin1);
  }else{
    cgbn_store(bn_env, in1, tscratch);
  }
}
inline __device__ void device_fp_add(env_t& bn_env, cgbn_mem_t<BITS>* const in1, cgbn_mem_t<BITS>* const in2, cgbn_mem_t<BITS>* module_data, const env_t::cgbn_t& tmax_value){
  env_t::cgbn_t tin1, tin2, tmodule_data, tscratch;

  cgbn_load(bn_env, tin1, in1);
  cgbn_load(bn_env, tin2, in2);
  cgbn_load(bn_env, tmodule_data, module_data);

  uint32_t carry = cgbn_add(bn_env, tscratch, tin1, tin2);
  if(carry || cgbn_compare(bn_env, tscratch, tmodule_data) >= 0){
    if(carry){
      cgbn_sub(bn_env, tin1, tmax_value, tmodule_data);
      cgbn_add(bn_env, tin2, tin1, tscratch);
      cgbn_add_ui32(bn_env, tin1, tin2, 1);
    }else{
      cgbn_sub(bn_env, tin1, tscratch, tmodule_data);
    }
    cgbn_store(bn_env, in1, tin1);
  }else{
    cgbn_store(bn_env, in1, tscratch);
  }
}

inline __device__ void device_fp_add(env_t& bn_env, env_t::cgbn_t& tout, const env_t::cgbn_t& tin1, const env_t::cgbn_t& tin2, cgbn_mem_t<BITS>* module_data, const env_t::cgbn_t& tmax_value){
  env_t::cgbn_t tmodule_data, tscratch;

  cgbn_load(bn_env, tmodule_data, module_data);

  uint32_t carry = cgbn_add(bn_env, tscratch, tin1, tin2);
  if(carry || cgbn_compare(bn_env, tscratch, tmodule_data) >= 0){
    if(carry){
      env_t::cgbn_t tmp_sub, tmp_add;
      cgbn_sub(bn_env, tmp_sub, tmax_value, tmodule_data);
      cgbn_add(bn_env, tmp_add, tmp_sub, tscratch);
      cgbn_add_ui32(bn_env, tout, tmp_add, 1);
    }else{
      cgbn_sub(bn_env, tout, tscratch, tmodule_data);
    }
  }else{
    //cgbn_store(bn_env, in1, tscratch);
    cgbn_set(bn_env, tout, tscratch);
  }
}

inline __device__ void device_fp_add(env_t& bn_env, env_t::cgbn_t& tout, cgbn_mem_t<BITS>* const in1, const env_t::cgbn_t& tin2, cgbn_mem_t<BITS>* module_data, const env_t::cgbn_t& tmax_value){
  env_t::cgbn_t tin1, tmodule_data, tscratch, tmp;

  cgbn_load(bn_env, tin1, in1);
  cgbn_load(bn_env, tmodule_data, module_data);

  uint32_t carry = cgbn_add(bn_env, tscratch, tin1, tin2);
  if(carry || cgbn_compare(bn_env, tscratch, tmodule_data) >= 0){
    if(carry){
      cgbn_sub(bn_env, tin1, tmax_value, tmodule_data);
      cgbn_add(bn_env, tmp, tin1, tscratch);
      cgbn_add_ui32(bn_env, tout, tmp, 1);
    }else{
      cgbn_sub(bn_env, tout, tscratch, tmodule_data);
    }
  }else{
    cgbn_set(bn_env, tout, tscratch);
  }
}

inline __device__ void device_fp_add(env_t& bn_env, env_t::cgbn_t& tout, const env_t::cgbn_t& tin1, const env_t::cgbn_t& tin2, const env_t::cgbn_t& tmodule_data, const env_t::cgbn_t& tmax_value){
  env_t::cgbn_t tscratch;

  uint32_t carry = cgbn_add(bn_env, tscratch, tin1, tin2);
  if(carry || cgbn_compare(bn_env, tscratch, tmodule_data) >= 0){
    if(carry){
      env_t::cgbn_t sub_res, add_res;
      cgbn_sub(bn_env, sub_res, tmax_value, tmodule_data);
      cgbn_add(bn_env, add_res, sub_res, tscratch);
      cgbn_add_ui32(bn_env, tout, add_res, 1);
    }else{
      cgbn_sub(bn_env, tout, tscratch, tmodule_data);
    }
  }else{
    cgbn_set(bn_env, tout, tscratch);
  }
}

inline __device__ void device_fp_sub(env_t& bn_env, cgbn_mem_t<BITS>* const in1, cgbn_mem_t<BITS>* const in2, cgbn_mem_t<BITS>* module_data, cgbn_mem_t<BITS>* max_value){
  env_t::cgbn_t tin1, tin2, tmodule_data, tscratch;
  cgbn_load(bn_env, tin1, in1);
  cgbn_load(bn_env, tin2, in2);
  cgbn_load(bn_env, tmodule_data, module_data);

  if(cgbn_compare(bn_env, tin1, tin2) < 0){
    uint32_t carry = cgbn_add(bn_env, tscratch, tin1, tmodule_data);
    if(carry > 0){
      env_t::cgbn_t tmax_value;
      cgbn_load(bn_env, tmax_value, max_value);
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

inline __device__ void device_fp_sub(env_t& bn_env, env_t::cgbn_t& tout, const env_t::cgbn_t& tin1, const env_t::cgbn_t& tin2, cgbn_mem_t<BITS>* module_data, const env_t::cgbn_t& tmax_value){
  env_t::cgbn_t tmodule_data, tscratch;
  cgbn_load(bn_env, tmodule_data, module_data);

  if(cgbn_compare(bn_env, tin1, tin2) < 0){
    uint32_t carry = cgbn_add(bn_env, tscratch, tin1, tmodule_data);
    if(carry > 0){
      env_t::cgbn_t tmp_sub, tmp_add;
      cgbn_sub(bn_env, tmp_sub, tmax_value, tin2);
      cgbn_add(bn_env, tmp_add, tmp_sub, tscratch);
      cgbn_add_ui32(bn_env, tout, tmp_add, 1);
    }else{
      cgbn_sub(bn_env, tout, tscratch, tin2);
    }
  }else{
      cgbn_sub(bn_env, tout, tin1, tin2);
  }
}

inline __device__ void device_fp_sub(env_t& bn_env, env_t::cgbn_t& tout, const env_t::cgbn_t& tin1, const env_t::cgbn_t& tin2, const env_t::cgbn_t& tmodule_data, const env_t::cgbn_t& tmax_value){
  env_t::cgbn_t tscratch;

  if(cgbn_compare(bn_env, tin1, tin2) < 0){
    uint32_t carry = cgbn_add(bn_env, tscratch, tin1, tmodule_data);
    if(carry > 0){
      env_t::cgbn_t tmp_sub, tmp_add;
      cgbn_sub(bn_env, tmp_sub, tmax_value, tin2);
      cgbn_add(bn_env, tmp_add, tmp_sub, tscratch);
      cgbn_add_ui32(bn_env, tout, tmp_add, 1);
    }else{
      cgbn_sub(bn_env, tout, tscratch, tin2);
    }
  }else{
      cgbn_sub(bn_env, tout, tin1, tin2);
  }
}
inline __device__ void device_mul_reduce(const env_t& bn_env, uint32_t* res,cgbn_mem_t<BITS>* const in1, cgbn_mem_t<BITS>* const in2, cgbn_mem_t<BITS>* module_data, cgbn_mem_t<BITS>* tmp_buffer, const uint64_t inv){
  const int group_thread = threadIdx.x & (TPI-1);
  env_t::cgbn_t  tin1, tin2, tmodule_data, tb, tres,tres2, add_res;                                             
  cgbn_load(bn_env, tin1, in1);  
  cgbn_load(bn_env, tin2, in2);   
  cgbn_load(bn_env, tmodule_data, module_data);     

  const int n = BITS/32;
  env_t::cgbn_wide_t tc;
  cgbn_mul_wide(bn_env, tc, tin1, tin2);
  cgbn_store(bn_env, res, tc._low);
  cgbn_store(bn_env, res + n, tc._high);

  for(int i = 0; i < n; i+=2){
    cgbn_load(bn_env, tres, res+i);
    cgbn_load(bn_env, tres2, res+n+i);

    if(group_thread == 0){
      uint64_t *p64 = (uint64_t*)(res+i);
      uint64_t k = inv * p64[0];
      uint32_t *p32 = (uint32_t*)&k;
      tmp_buffer->_limbs[0] = p32[0];
      tmp_buffer->_limbs[1] = p32[1];
      for(int j = 2; j < BITS/32; j++){
        tmp_buffer->_limbs[j] = 0;
      }
    }

    cgbn_load(bn_env, tb, tmp_buffer);      

    env_t::cgbn_wide_t mul_res;
    cgbn_mul_wide(bn_env, mul_res, tmodule_data, tb);

    uint32_t carryout = cgbn_add(bn_env, add_res, mul_res._low, tres);
    cgbn_store(bn_env, res+i, add_res);   
    
    cgbn_store(bn_env, tmp_buffer, mul_res._high);
    if(group_thread == 0){
      uint64_t tmp_carry = ((uint64_t*)tmp_buffer->_limbs)[0];
      tmp_carry += carryout;
      uint32_t *p = (uint32_t*)&tmp_carry;
      tmp_buffer->_limbs[0] = p[0];
      tmp_buffer->_limbs[1] = p[1];
    }

    cgbn_load(bn_env, tb, tmp_buffer);      
    cgbn_add(bn_env, add_res, tres2, tb);
    cgbn_store(bn_env, res+n+i, add_res);   
  }
  cgbn_load(bn_env, tres, res+n);
  if(cgbn_compare(bn_env, tres, tmodule_data) >= 0){
    cgbn_sub(bn_env, tres2, tres, tmodule_data);
    cgbn_store(bn_env, res+n, tres2);
  }
}

inline __device__ void device_mul_reduce(const env_t& bn_env, uint32_t* res,cgbn_mem_t<BITS>* const in1, const env_t::cgbn_t tin2, cgbn_mem_t<BITS>* module_data, cgbn_mem_t<BITS>* tmp_buffer, const uint64_t inv){
  const int group_thread = threadIdx.x & (TPI-1);
  env_t::cgbn_t  tin1, tmodule_data, tb, tres,tres2, add_res;                                             
  cgbn_load(bn_env, tin1, in1);  
  cgbn_load(bn_env, tmodule_data, module_data);     

  const int n = BITS/32;
  env_t::cgbn_wide_t tc;
  cgbn_mul_wide(bn_env, tc, tin1, tin2);
  cgbn_store(bn_env, res, tc._low);
  cgbn_store(bn_env, res + n, tc._high);

  for(int i = 0; i < n; i+=2){
    cgbn_load(bn_env, tres, res+i);
    cgbn_load(bn_env, tres2, res+n+i);

    if(group_thread == 0){
      uint64_t *p64 = (uint64_t*)(res+i);
      uint64_t k = inv * p64[0];
      uint32_t *p32 = (uint32_t*)&k;
      tmp_buffer->_limbs[0] = p32[0];
      tmp_buffer->_limbs[1] = p32[1];
      for(int j = 2; j < BITS/32; j++){
        tmp_buffer->_limbs[j] = 0;
      }
    }

    cgbn_load(bn_env, tb, tmp_buffer);      

    env_t::cgbn_wide_t mul_res;
    cgbn_mul_wide(bn_env, mul_res, tmodule_data, tb);

    uint32_t carryout = cgbn_add(bn_env, add_res, mul_res._low, tres);
    cgbn_store(bn_env, res+i, add_res);   
    
    cgbn_store(bn_env, tmp_buffer, mul_res._high);
    if(group_thread == 0){
      uint64_t tmp_carry = ((uint64_t*)tmp_buffer->_limbs)[0];
      tmp_carry += carryout;
      uint32_t *p = (uint32_t*)&tmp_carry;
      tmp_buffer->_limbs[0] = p[0];
      tmp_buffer->_limbs[1] = p[1];
    }

    cgbn_load(bn_env, tb, tmp_buffer);      
    cgbn_add(bn_env, add_res, tres2, tb);
    cgbn_store(bn_env, res+n+i, add_res);   
  }
  cgbn_load(bn_env, tres, res+n);
  if(cgbn_compare(bn_env, tres, tmodule_data) >= 0){
    cgbn_sub(bn_env, tres2, tres, tmodule_data);
    cgbn_store(bn_env, res+n, tres2);
  }
}

inline __device__ void device_mul_reduce(const env_t& bn_env, uint32_t* res, const env_t::cgbn_t& tin1, const env_t::cgbn_t tin2, cgbn_mem_t<BITS>* module_data, cgbn_mem_t<BITS>* tmp_buffer, const uint64_t inv){
  const int group_thread = threadIdx.x & (TPI-1);
  env_t::cgbn_t tmodule_data, tb, tres,tres2, add_res;                                             
  cgbn_load(bn_env, tmodule_data, module_data);     

  const int n = BITS/32;
  env_t::cgbn_wide_t tc;
  cgbn_mul_wide(bn_env, tc, tin1, tin2);
  cgbn_store(bn_env, res, tc._low);
  cgbn_store(bn_env, res + n, tc._high);

  for(int i = 0; i < n; i+=2){
    cgbn_load(bn_env, tres, res+i);
    cgbn_load(bn_env, tres2, res+n+i);

    if(group_thread == 0){
      uint64_t *p64 = (uint64_t*)(res+i);
      uint64_t k = inv * p64[0];
      uint32_t *p32 = (uint32_t*)&k;
      tmp_buffer->_limbs[0] = p32[0];
      tmp_buffer->_limbs[1] = p32[1];
      for(int j = 2; j < BITS/32; j++){
        tmp_buffer->_limbs[j] = 0;
      }
    }

    cgbn_load(bn_env, tb, tmp_buffer);      

    env_t::cgbn_wide_t mul_res;
    cgbn_mul_wide(bn_env, mul_res, tmodule_data, tb);

    uint32_t carryout = cgbn_add(bn_env, add_res, mul_res._low, tres);
    cgbn_store(bn_env, res+i, add_res);   
    
    cgbn_store(bn_env, tmp_buffer, mul_res._high);
    if(group_thread == 0){
      uint64_t tmp_carry = ((uint64_t*)tmp_buffer->_limbs)[0];
      tmp_carry += carryout;
      uint32_t *p = (uint32_t*)&tmp_carry;
      tmp_buffer->_limbs[0] = p[0];
      tmp_buffer->_limbs[1] = p[1];
    }

    cgbn_load(bn_env, tb, tmp_buffer);      
    cgbn_add(bn_env, add_res, tres2, tb);
    cgbn_store(bn_env, res+n+i, add_res);   
  }
  cgbn_load(bn_env, tres, res+n);
  if(cgbn_compare(bn_env, tres, tmodule_data) >= 0){
    cgbn_sub(bn_env, tres2, tres, tmodule_data);
    cgbn_store(bn_env, res+n, tres2);
  }
}

inline __device__ void device_mul_reduce(const env_t& bn_env, uint32_t* res, const env_t::cgbn_t& tin1, const env_t::cgbn_t& tin2, const env_t::cgbn_t& tmodule_data, cgbn_mem_t<BITS>* tmp_buffer, const uint64_t inv){
  const int group_thread = threadIdx.x & (TPI-1);
  env_t::cgbn_t tb, tres,tres2, add_res;                                             

  const int n = BITS/32;
  env_t::cgbn_wide_t tc;
  cgbn_mul_wide(bn_env, tc, tin1, tin2);
  cgbn_store(bn_env, res, tc._low);
  cgbn_store(bn_env, res + n, tc._high);

  for(int i = 0; i < n; i+=2){
    cgbn_load(bn_env, tres, res+i);
    cgbn_load(bn_env, tres2, res+n+i);

    if(group_thread == 0){
      uint64_t *p64 = (uint64_t*)(res+i);
      uint64_t k = inv * p64[0];
      uint32_t *p32 = (uint32_t*)&k;
      tmp_buffer->_limbs[0] = p32[0];
      tmp_buffer->_limbs[1] = p32[1];
      for(int j = 2; j < BITS/32; j++){
        tmp_buffer->_limbs[j] = 0;
      }
    }

    cgbn_load(bn_env, tb, tmp_buffer);      

    env_t::cgbn_wide_t mul_res;
    cgbn_mul_wide(bn_env, mul_res, tmodule_data, tb);

    uint32_t carryout = cgbn_add(bn_env, add_res, mul_res._low, tres);
    cgbn_store(bn_env, res+i, add_res);   
    
    cgbn_store(bn_env, tmp_buffer, mul_res._high);
    if(group_thread == 0){
      uint64_t tmp_carry = ((uint64_t*)tmp_buffer->_limbs)[0];
      tmp_carry += carryout;
      uint32_t *p = (uint32_t*)&tmp_carry;
      tmp_buffer->_limbs[0] = p[0];
      tmp_buffer->_limbs[1] = p[1];
    }

    cgbn_load(bn_env, tb, tmp_buffer);      
    cgbn_add(bn_env, add_res, tres2, tb);
    cgbn_store(bn_env, res+n+i, add_res);   
  }
  cgbn_load(bn_env, tres, res+n);
  if(cgbn_compare(bn_env, tres, tmodule_data) >= 0){
    cgbn_sub(bn_env, tres2, tres, tmodule_data);
    cgbn_store(bn_env, res+n, tres2);
  }
}

inline __device__ void device_squared(const env_t& bn_env, const Fp_model& x, uint32_t *res, cgbn_mem_t<BITS>* tmp_buffer, const int offset){
  device_mul_reduce(bn_env, res, 
      x.mont_repr_data + offset, 
      x.mont_repr_data + offset,
      x.modulus_data + offset,
      tmp_buffer + offset, 
      x.inv);
}


} //namespace gpu

#endif
