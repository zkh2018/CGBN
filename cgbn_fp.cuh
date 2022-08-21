#ifndef CGBN_FP_CUH
#define CGBN_FP_CUH

#include "cgbn_math.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace gpu{

inline __device__ uint32_t sync_mask(){
    uint32_t group_thread=threadIdx.x & TPI-1, warp_thread=threadIdx.x & 31;
    const uint32_t TPI_ONES=(1ull<<TPI)-1;

    return TPI_ONES<<(group_thread ^ warp_thread);
}

inline __device__ void device_fp_add(const env_t& bn_env, cgbn_mem_t<BITS>* const in1, cgbn_mem_t<BITS>* const in2, cgbn_mem_t<BITS>* module_data, cgbn_mem_t<BITS>* max_value){
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
inline __device__ void device_fp_add(const env_t& bn_env, cgbn_mem_t<BITS>* const in1, cgbn_mem_t<BITS>* const in2, cgbn_mem_t<BITS>* module_data, const env_t::cgbn_t& tmax_value){
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

inline __device__ void device_fp_add(const env_t& bn_env, env_t::cgbn_t& tout, const env_t::cgbn_t& tin1, const env_t::cgbn_t& tin2, cgbn_mem_t<BITS>* module_data, const env_t::cgbn_t& tmax_value){
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

inline __device__ void device_fp_add(const env_t& bn_env, env_t::cgbn_t& tout, cgbn_mem_t<BITS>* const in1, const env_t::cgbn_t& tin2, cgbn_mem_t<BITS>* module_data, const env_t::cgbn_t& tmax_value){
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

inline __device__ void device_fp_add(const env_t& bn_env, env_t::cgbn_t& tout, const env_t::cgbn_t& tin1, const env_t::cgbn_t& tin2, const env_t::cgbn_t& tmodule_data, const env_t::cgbn_t& tmax_value){
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

inline __device__ void device_fp_sub(const env_t& bn_env, cgbn_mem_t<BITS>* const in1, cgbn_mem_t<BITS>* const in2, cgbn_mem_t<BITS>* module_data, cgbn_mem_t<BITS>* max_value){
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

inline __device__ void device_fp_sub(const env_t& bn_env, env_t::cgbn_t& tout, const env_t::cgbn_t& tin1, const env_t::cgbn_t& tin2, cgbn_mem_t<BITS>* module_data, const env_t::cgbn_t& tmax_value){
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

inline __device__ void device_fp_sub(const env_t& bn_env, env_t::cgbn_t& tout, const env_t::cgbn_t& tin1, const env_t::cgbn_t& tin2, const env_t::cgbn_t& tmodule_data, const env_t::cgbn_t& tmax_value){
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

  const int n = NUM;
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
      for(int j = 2; j < NUM; j++){
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

  const int n = NUM;
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
      for(int j = 2; j < NUM; j++){
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

  const int n = NUM;
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
      for(int j = 2; j < NUM; j++){
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

inline __device__ void device_mont_mul(uint64_t *wide_r, const uint64_t *modulus, const uint64_t inv){
    uint64_t k = wide_r[0] * inv;
    uint64_t carry = 0;

    asm(
      "{\n\t"
      ".reg .u64 c;\n\t"
      ".reg .u64 t;\n\t"
      ".reg .u64 nc;\n\t"

        //c = k * p[0] + r[0]
      "mad.lo.cc.u64 c, %10, %6, %0;\n\t"
      "madc.hi.cc.u64 c, %10, %6, 0;\n\t"

      // t = r[1] + c
      "addc.cc.u64 t, %1, c;\n\t"
      // nc = carry
      "addc.u64 nc, 0, 0;\n\t"
      // (r[1],c) = k * p[1] + (t, nc)
      "mad.lo.cc.u64 %1, %10, %7, t;\n\t"
      "madc.hi.cc.u64 c, %10, %7, nc;\n\t"

        // t = r[2] + c
      "addc.cc.u64 t, %2, c;\n\t"
      // nc = 0 + carry
      "addc.u64 nc, 0, 0;\n\t"
      // (r[2],c) = k * p[2] + (t, nc)
      "mad.lo.cc.u64 %2, %10, %8, t;\n\t"
      "madc.hi.cc.u64 c, %10, %8, nc;\n\t"

        // t = r[3] + c
      "addc.cc.u64 t, %3, c;\n\t"
      // nc = carry
      "addc.u64 nc, 0, 0;\n\t"
      //(r[3], c) = k * p[3] + (t, nc)
      "mad.lo.cc.u64 %3, %10, %9, t;\n\t"
      "madc.hi.cc.u64 c, %10, %9, nc;\n\t"

      "addc.cc.u64 %4, %4, c;\n\t"
      "addc.u64 %5, 0, 0;\n\t"
      "}"
      : "+l"(wide_r[0]),
      "+l"(wide_r[1]),
      "+l"(wide_r[2]),
      "+l"(wide_r[3]),
      "+l"(wide_r[4]),
      "=l"(carry)
      : "l"(modulus[0]),
      "l"(modulus[1]),
      "l"(modulus[2]),
      "l"(modulus[3]),
      "l"(k)
    );

    k = wide_r[1] * inv;

    asm(
      "{\n\t"
      ".reg .u64 c;\n\t"
      ".reg .u64 t;\n\t"
      ".reg .u64 nc;\n\t"

      "mad.lo.cc.u64 c, %10, %6, %0;\n\t"
      "madc.hi.cc.u64 c, %10, %6, 0;\n\t"
      
      "addc.cc.u64 t, %1, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %1, %10, %7, t;\n\t"
      "madc.hi.cc.u64 c, %10, %7, nc;\n\t"

      "addc.cc.u64 t, %2, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %2, %10, %8, t;\n\t"
      "madc.hi.cc.u64 c, %10, %8, nc;\n\t"

      "addc.cc.u64 t, %3, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %3, %10, %9, t;\n\t"
      "madc.hi.cc.u64 c, %10, %9, nc;\n\t"

      "addc.cc.u64 c, c, %5;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "addc.cc.u64 %4, %4, c;\n\t"
      "addc.u64 %5, nc, 0;\n\t"
      "}"
      : "+l"(wide_r[1]),
      "+l"(wide_r[2]),
      "+l"(wide_r[3]),
      "+l"(wide_r[4]),
      "+l"(wide_r[5]),
      "+l"(carry)
      : "l"(modulus[0]),
      "l"(modulus[1]),
      "l"(modulus[2]),
      "l"(modulus[3]),
      "l"(k)
    );

    k = wide_r[2] * inv;

    asm(
      "{\n\t"
      ".reg .u64 c;\n\t"
      ".reg .u64 t;\n\t"
      ".reg .u64 nc;\n\t"

      "mad.lo.cc.u64 c, %10, %6, %0;\n\t"
      "madc.hi.cc.u64 c, %10, %6, 0;\n\t"
      
      "addc.cc.u64 t, %1, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %1, %10, %7, t;\n\t"
      "madc.hi.cc.u64 c, %10, %7, nc;\n\t"

      "addc.cc.u64 t, %2, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %2, %10, %8, t;\n\t"
      "madc.hi.cc.u64 c, %10, %8, nc;\n\t"

      "addc.cc.u64 t, %3, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %3, %10, %9, t;\n\t"
      "madc.hi.cc.u64 c, %10, %9, nc;\n\t"

      "addc.cc.u64 c, c, %5;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "addc.cc.u64 %4, %4, c;\n\t"
      "addc.u64 %5, nc, 0;\n\t"
      "}"
      : "+l"(wide_r[2]),
      "+l"(wide_r[3]),
      "+l"(wide_r[4]),
      "+l"(wide_r[5]),
      "+l"(wide_r[6]),
      "+l"(carry)
      : "l"(modulus[0]),
      "l"(modulus[1]),
      "l"(modulus[2]),
      "l"(modulus[3]),
      "l"(k)
    );
    k = wide_r[3] * inv;

    asm(
      "{\n\t"
      ".reg .u64 c;\n\t"
      ".reg .u64 t;\n\t"
      ".reg .u64 nc;\n\t"

      "mad.lo.cc.u64 c, %10, %6, %0;\n\t"
      "madc.hi.cc.u64 c, %10, %6, 0;\n\t"
      
      "addc.cc.u64 t, %1, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %1, %10, %7, t;\n\t"
      "madc.hi.cc.u64 c, %10, %7, nc;\n\t"

      "addc.cc.u64 t, %2, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %2, %10, %8, t;\n\t"
      "madc.hi.cc.u64 c, %10, %8, nc;\n\t"

      "addc.cc.u64 t, %3, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %3, %10, %9, t;\n\t"
      "madc.hi.cc.u64 c, %10, %9, nc;\n\t"

      "addc.cc.u64 c, c, %5;\n\t"
      //"addc.u64 nc, 0, 0;\n\t"
      "addc.cc.u64 %4, %4, c;\n\t"
      //"addc.u64 %5, nc, 0;\n\t"
      "}"
      : "+l"(wide_r[3]),
      "+l"(wide_r[4]),
      "+l"(wide_r[5]),
      "+l"(wide_r[6]),
      "+l"(wide_r[7]),
      "+l"(carry)
      : "l"(modulus[0]),
      "l"(modulus[1]),
      "l"(modulus[2]),
      "l"(modulus[3]),
      "l"(k)
    );

    //memcpy(ret, wide_r + 4, sizeof(uint64_t) * 4);
}

inline __device__ void device_mul_wide(const uint64_t *a, const uint64_t *b, uint64_t *c){
    uint64_t r[12] = {0};

    asm(
      "{\n\t"
      ".reg .u64 c;\n\t"
      ".reg .u64 nc;\n\t"
      ".reg .u64 t;\n\t"
      //r[0], c = a[0] * b[0] 
      "mad.lo.cc.u64 %0, %8, %12, 0;\n\t"
      "madc.hi.cc.u64 c, %8, %12, 0;\n\t"
      
      //r[1], c = a[0] * b[1] + c
      "madc.lo.cc.u64 %1, %8, %13, c;\n\t"
      "madc.hi.cc.u64 c, %8, %13, 0;\n\t"
    
      //r[2], c = a[0] * b[2] + c
      "madc.lo.cc.u64 %2, %8, %14, c;\n\t"
      "madc.hi.cc.u64 c, %8, %14, 0;\n\t"

      //r[3], c = a[0] * b[3] + c
      "madc.lo.cc.u64 %3, %8, %15, c;\n\t"
      "madc.hi.cc.u64 %4, %8, %15, 0;\n\t"

      //r[1], c = a[1] * b[0] + c
      "mad.lo.cc.u64 %1, %9, %12, %1;\n\t"
      "madc.hi.cc.u64 c, %9, %12, 0;\n\t"
      
      //t = r[2] + c
      "addc.cc.u64 t, %2, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      //r[2], c = a[1] * b[1] + c
      "mad.lo.cc.u64 %2, %9, %13, t;\n\t"
      "madc.hi.cc.u64 c, %9, %13, nc;\n\t"

      "addc.cc.u64 t, %3, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      //r[3], c = a[1] * b[2] + c
      "mad.lo.cc.u64 %3, %9, %14, t;\n\t"
      "madc.hi.cc.u64 c, %9, %14, nc;\n\t"

      "addc.cc.u64 t, %4, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      //r[4], c = a[1] * b[3] + c
      "mad.lo.cc.u64 %4, %9, %15, t;\n\t"
      "madc.hi.cc.u64 %5, %9, %15, nc;\n\t"

      //r[2], c = a[2] * b[0] + c
      "mad.lo.cc.u64 %2, %10, %12, %2;\n\t"
      "madc.hi.cc.u64 c, %10, %12, 0;\n\t"
      
      "addc.cc.u64 t, %3, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %3, %10, %13, t;\n\t"
      "madc.hi.cc.u64 c, %10, %13, nc;\n\t"
      
      "addc.cc.u64 t, %4, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %4, %10, %14, t;\n\t"
      "madc.hi.cc.u64 c, %10, %14, nc;\n\t"

      "addc.cc.u64 t, %5, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %5, %10, %15, t;\n\t"
      "madc.hi.cc.u64 %6, %10, %15, nc;\n\t"

      "mad.lo.cc.u64 %3, %11, %12, %3;\n\t"
      "madc.hi.cc.u64 c, %11, %12, 0;\n\t"
      
      "addc.cc.u64 t, %4, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %4, %11, %13, t;\n\t"
      "madc.hi.cc.u64 c, %11, %13, nc;\n\t"
      
      "addc.cc.u64 t, %5, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %5, %11, %14, t;\n\t"
      "madc.hi.cc.u64 c, %11, %14, nc;\n\t"
      
      "addc.cc.u64 t, %6, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %6, %11, %15, t;\n\t"
      "madc.hi.cc.u64 %7, %11, %15, nc;\n\t"
      "}"
      : "+l"(r[0]),
      "+l"(r[1]),
      "+l"(r[2]),
      "+l"(r[3]),
      "+l"(r[4]),
      "+l"(r[5]),
      "+l"(r[6]),
      "+l"(r[7])
      : "l"(a[0]),
      "l"(a[1]),
      "l"(a[2]),
      "l"(a[3]),
      "l"(b[0]),
      "l"(b[1]),
      "l"(b[2]),
      "l"(b[3])
    );

    #pragma unroll
    for(int i = 0; i < 8; i++){
        c[i] = r[i];
    }
}

inline __device__ int dev_is_ge(const uint64_t *a, const uint64_t *b){
    for (int i = 3; i >= 0; --i) {
        if (a[i] < b[i]) {
            return 0;
        } else if (a[i] > b[i]) {
            return 1;
        }
    }
    return 1;
}

inline __device__ void dev_sub_mod(const uint64_t *a, const uint64_t *b, uint64_t *c){
   asm(
      "sub.cc.u64 %0, %4, %8;\n\t"
      "subc.cc.u64 %1, %5, %9;\n\t"
      "subc.cc.u64 %2, %6, %10;\n\t"
      "subc.u64 %3, %7, %11;"
      : "=l"(c[0]),
      "=l"(c[1]),
      "=l"(c[2]),
      "=l"(c[3])
      : "l"(a[0]),
      "l"(a[1]),
      "l"(a[2]),
      "l"(a[3]),
      "l"(b[0]),
      "l"(b[1]),
      "l"(b[2]),
      "l"(b[3])
    );
}

inline __device__ void dev_mont_mul(const uint64_t *a, const uint64_t *b, const uint64_t *modulus, const uint64_t inv, uint64_t *c){
    uint64_t wide_r[8];
    device_mul_wide(a, b, wide_r);  
    device_mont_mul(wide_r, modulus, inv);
    #pragma unroll
    for(int i = 0; i < 4; i++){
        c[i] = wide_r[i + 4];
    }
    //reduce
    if(dev_is_ge(c, modulus)){
        uint64_t sub[4];
        dev_sub_mod(c, modulus, sub);
        memcpy(c, sub, 4 * sizeof(uint64_t));
    }
}

inline __device__ void device_mul_reduce(const env_t& bn_env, uint32_t* res, const env_t::cgbn_t& tin1, const env_t::cgbn_t& tin2, const env_t::cgbn_t& tmodule_data, uint32_t* tmp_buffer, const uint64_t inv){
  //const int group_thread = threadIdx.x & (TPI-1);
  cg::thread_group tg = cg::tiled_partition(cg::this_thread_block(), TPI);
  const int n = NUM;
  env_t::cgbn_wide_t tc;
  cgbn_mul_wide(bn_env, tc, tin1, tin2);
  cgbn_store(bn_env, res, tc._low);
  cgbn_store(bn_env, res + n, tc._high);

#if false
  env_t::cgbn_t tb, tres, add_res;                                             
  for(int i = 0; i < n; i+=2){
    cgbn_load(bn_env, tres, res+i);
    uint64_t *p64 = (uint64_t*)(res+i);
    uint64_t k = inv * p64[0];
    //if(group_thread == 0){
    //    printf("k = %lu\n", k);
    //}
    //carry1 = mpn_mul_1(tmp, modulus, n, k)
    cgbn_mul_ui64(bn_env, tc, tmodule_data, k); 
    uint32_t th[2];
    cgbn_get_ui64(bn_env, tc._high, th, 0);
    cgbn_store(bn_env, tmp_buffer, tc._low);
    //if(group_thread == 0){
    //    uint64_t *p64_buf = (uint64_t*)tmp_buffer;
    //    printf("mul_1 %lu:", *(uint64_t*)th);
    //    for(int j = 0; j < n/2; j++){
    //        printf("%lu ", p64_buf[j]);
    //    }
    //    printf("\n");
    //}

    //carry2 = mpn_add_n(res+i, res+i, tmp, n);
    uint32_t carryout = cgbn_add(bn_env, add_res, tc._low, tres);
    cgbn_store(bn_env, res+i, add_res);   
    //if(group_thread == 0){
    //    printf("add %d:", carryout);
    //    for(int j = 0; j < n/2; j++){
    //        printf("%lu ", p64[j]);
    //    }
    //    printf("\n");
    //}

    //mpn_add_1(res+n+i, res+n+i, n-i, carry1+carry2);
    cgbn_load(bn_env, tres, res+n+i);
    uint64_t tmp_carry = *(uint64_t*)th;
    tmp_carry += carryout;
    //if(group_thread == 0){
    //    printf("carry=%lu\n", tmp_carry);
    //}
    uint32_t* p32 = (uint32_t*)&tmp_carry;
    //if(group_thread > 1) tmp_buffer[group_thread] = 0;
    //tmp_buffer[0] = p32[0];
    //tmp_buffer[1] = p32[1];
    //cgbn_load(bn_env, tb, tmp_buffer);      
    cgbn_set_ui32(bn_env, tb, p32[0], p32[1]);
    cgbn_add(bn_env, add_res, tres, tb);
    cgbn_store(bn_env, res+n+i, add_res);   
    //if(group_thread == 0){
    //    for(int j = 0; j < n/2-i/2; j++){
    //        printf("%lu ", p64[6+j]);
    //    }
    //    printf("\n");
    //}

  }

  cgbn_load(bn_env, tres, res+n);
  if(cgbn_compare(bn_env, tres, tmodule_data) >= 0){
    cgbn_sub(bn_env, add_res, tres, tmodule_data);
    cgbn_store(bn_env, res+n, add_res);
  }
#else
  cgbn_store(bn_env, tmp_buffer, tmodule_data);
  tg.sync();
  if((threadIdx.x & TPI-1) == 0){
    uint64_t *p64_res = (uint64_t*)(res);
    uint64_t *p64_buf = (uint64_t*)(tmp_buffer);
    device_mont_mul(p64_res, p64_buf, inv);
    if(dev_is_ge(p64_res + 4, p64_buf)){
        uint64_t sub[4];
        dev_sub_mod(p64_res + 4, p64_buf, sub);
        memcpy(p64_res+4, sub, 4 * sizeof(uint64_t));
    }
  }
  tg.sync();
#endif

}

inline __device__ void device_squared(const env_t& bn_env, const Fp_model& x, uint32_t *res, cgbn_mem_t<BITS>* tmp_buffer, const int offset, cgbn_mem_t<BITS>* modulus, const uint64_t inv){
  device_mul_reduce(bn_env, res, 
      x.mont_repr_data + offset, 
      x.mont_repr_data + offset,
      //x.modulus_data + offset,
      modulus,
      tmp_buffer + offset, 
      inv);
}

static inline cgbn_error_report_t* get_error_report(){
  static cgbn_error_report_t* report = nullptr;
  if(report == nullptr){
    CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  }
  return report;
}


} //namespace gpu

#endif
