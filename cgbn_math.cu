#include "cgbn_math.h"
#include <cuda_runtime.h>
#include <cuda.h>

#include "cgbn/cgbn.h"
#include "utility/cpu_support.h"
#include "utility/cpu_simple_bn_math.h"
#include "utility/gpu_support.h"

#define TPI 8
typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

namespace gpu{

void gpu_malloc(void** ptr, size_t size){
  CUDA_CHECK(cudaMalloc(ptr, size));
  CUDA_CHECK(cudaMemset(*ptr, 0, size));
}
void gpu_free(void* ptr){
  CUDA_CHECK(cudaFree(ptr));
}
void copy_cpu_to_gpu(void* dst, const void* src, size_t size){
  CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}
void copy_gpu_to_cpu(void* dst, const void* src, size_t size){
  CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}
void copy_gpu_to_gpu(void* dst, const void* src, size_t size){
  CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}


void gpu_buffer::resize(int new_n){
  if(total_n == 0){
    total_n = new_n;
    n = new_n;
    gpu_malloc((void**)&ptr, n * sizeof(cgbn_mem_t<BITS>));
  }else if(total_n < new_n){
    gpu_free((void*)ptr);
    total_n = new_n;
    n = new_n;
    gpu_malloc((void**)&ptr, n * sizeof(cgbn_mem_t<BITS>));
  }else{
    n = new_n;
  }
}
void gpu_buffer::resize_host(int new_n){
  if(total_n == 0){
    total_n = new_n;
    n = new_n;
    ptr = (cgbn_mem_t<BITS>*)malloc(n * sizeof(cgbn_mem_t<BITS>)); 
  }else if(total_n < n){
    free(ptr);
    total_n = n;
    n = new_n;
    ptr = (cgbn_mem_t<BITS>*)malloc(n * sizeof(cgbn_mem_t<BITS>));
  }else{
    n = new_n;
  }
  memset(ptr, 0, sizeof(cgbn_mem_t<BITS>) * new_n);
}

void gpu_buffer::release(){
  CUDA_CHECK(cudaFree(ptr));
}
void gpu_buffer::release_host(){
  free(ptr);
}

void gpu_buffer::copy_from_host(gpu_buffer& buf){
  copy_cpu_to_gpu(ptr, buf.ptr, n * sizeof(cgbn_mem_t<BITS>));
}
void gpu_buffer::copy_to_host(gpu_buffer& buf){
  copy_gpu_to_cpu(buf.ptr, ptr, n * sizeof(cgbn_mem_t<BITS>));
}
void gpu_buffer::copy_from_host(const uint32_t* data, const uint32_t n){
  copy_cpu_to_gpu((void*)ptr, (void*)data, n * sizeof(uint32_t));
}
void gpu_buffer::copy_to_host(uint32_t* data, const uint32_t n){
  copy_cpu_to_gpu((void*)data, (void*)ptr, n * sizeof(uint32_t));
}

__global__ void kernel_add(cgbn_error_report_t* report, cgbn_mem_t<BITS>* c, cgbn_mem_t<BITS>* const a, cgbn_mem_t<BITS>* const b, uint32_t *carry, const uint32_t count){
  int instance = (blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance >= count) return;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
  env_t::cgbn_t  ta, tb, tc;                                             // define a, b, r as 1024-bit bignums
  cgbn_load(bn_env, ta, a);      // load my instance's a value
  cgbn_load(bn_env, tb, b);      // load my instance's b value
  *carry = cgbn_add(bn_env, tc, ta, tb);                           // r=a+b
  cgbn_store(bn_env, c, tc);   // store r into sum
}
__global__ void kernel_add(cgbn_error_report_t* report, uint32_t* c, cgbn_mem_t<BITS>* const a, cgbn_mem_t<BITS>* const b, uint32_t *carry, const uint32_t count){
  int instance = (blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance >= count) return;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
  env_t::cgbn_t  ta, tb, tc;                                             // define a, b, r as 1024-bit bignums
  cgbn_load(bn_env, ta, a);      // load my instance's a value
  cgbn_load(bn_env, tb, b);      // load my instance's b value
  *carry = cgbn_add(bn_env, tc, ta, tb);                           // r=a+b
  cgbn_store(bn_env, c, tc);   // store r into sum
}
__global__ void kernel_add_1(cgbn_error_report_t* report, cgbn_mem_t<BITS>* c, cgbn_mem_t<BITS>* const a, const uint32_t b, uint32_t* carry, const uint32_t count){
  int instance = (blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance >= count) return;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
  env_t::cgbn_t  ta, tr;                                             // define a, b, r as 1024-bit bignums
  cgbn_load(bn_env, ta, a);      // load my instance's a value
  //cgbn_load(bn_env, tb, b);      // load my instance's b value
  *carry = cgbn_add_ui32(bn_env, tr, ta, b);                           // r=a+b
  cgbn_store(bn_env, c, tr);   // store r into sum
}

__global__ void kernel_sub(cgbn_error_report_t* report, cgbn_mem_t<BITS>* c, cgbn_mem_t<BITS>* const a, cgbn_mem_t<BITS>* const b, uint32_t* carry, const uint32_t count){
  int instance = (blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance >= count) return;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
  env_t::cgbn_t  ta, tb, tr;                                             // define a, b, r as 1024-bit bignums
  cgbn_load(bn_env, ta, a);      // load my instance's a value
  cgbn_load(bn_env, tb, b);      // load my instance's b value
  *carry = cgbn_sub(bn_env, tr, ta, tb);                           // r=a+b
  cgbn_store(bn_env, c, tr);   // store r into sum
}

__global__ void kernel_sub_1(cgbn_error_report_t* report, cgbn_mem_t<BITS>* c, cgbn_mem_t<BITS>* const a, const uint32_t b, uint32_t* carry, const uint32_t count){
  int instance = (blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance >= count) return;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
  env_t::cgbn_t  ta, tr;                                             // define a, b, r as 1024-bit bignums
  cgbn_load(bn_env, ta, a);      // load my instance's a value
  //cgbn_load(bn_env, tb, b);      // load my instance's b value
  *carry = cgbn_sub_ui32(bn_env, tr, ta, b);                           // r=a+b
  cgbn_store(bn_env, c, tr);   // store r into sum
}
__global__ void kernel_mul_1(cgbn_error_report_t* report, cgbn_mem_t<BITS>* c, cgbn_mem_t<BITS>* const a, const uint32_t b, uint32_t* carry, const uint32_t count){
  int instance = (blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance >= count) return;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
  env_t::cgbn_t  ta, tr;                                             // define a, b, r as 1024-bit bignums
  cgbn_load(bn_env, ta, a);      // load my instance's a value
  *carry = cgbn_mul_ui32(bn_env, tr, ta, b);                           // r=a+b
  cgbn_store(bn_env, c, tr);   // store r into sum
}

__global__ void kernel_mul(cgbn_error_report_t* report, 
    cgbn_mem_t<BITS>* c_low, 
    cgbn_mem_t<BITS>* c_high, 
    cgbn_mem_t<BITS>* const a, cgbn_mem_t<BITS>* const b, 
    const uint32_t count){
  int instance = (blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance >= count) return;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
  env_t::cgbn_t  ta, tb;                                             // define a, b, r as 1024-bit bignums
  env_t::cgbn_wide_t tc;
  cgbn_load(bn_env, ta, a);      // load my instance's a value
  cgbn_load(bn_env, tb, b);      // load my instance's b value
  //cgbn_mul(bn_env, tc, ta, tb);                           // r=a+b
  cgbn_mul_wide(bn_env, tc, ta, tb);
  cgbn_store(bn_env, c_low, tc._low);   // store r into sum
  cgbn_store(bn_env, c_high, tc._high);   // store r into sum
}
__global__ void kernel_mul(cgbn_error_report_t* report, 
    uint32_t* c, 
    cgbn_mem_t<BITS>* const a, cgbn_mem_t<BITS>* const b, 
    const uint32_t count){
  int instance = (blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance >= count) return;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
  env_t::cgbn_t  ta, tb;                                             // define a, b, r as 1024-bit bignums
  env_t::cgbn_wide_t tc;
  cgbn_load(bn_env, ta, a);      // load my instance's a value
  cgbn_load(bn_env, tb, b);      // load my instance's b value
  //cgbn_mul(bn_env, tc, ta, tb);                           // r=a+b
  cgbn_mul_wide(bn_env, tc, ta, tb);
  cgbn_store(bn_env, c, tc._low);   // store r into sum
  cgbn_store(bn_env, c + BITS/32, tc._high);   // store r into sum
}

__global__ void kernel_addmul_1(cgbn_error_report_t* report, uint32_t* res, uint32_t* res2, cgbn_mem_t<BITS>* const a, const uint32_t b, uint32_t* carry, const uint32_t count){
  int instance = (blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance >= count) return;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
  env_t::cgbn_t  ta, tres, mul_res, tres2, add_res;                                             // define a, b, r as 1024-bit bignums
  cgbn_load(bn_env, ta, a);      // load my instance's a value
  cgbn_load(bn_env, tres, res);      // load my instance's a value
  cgbn_load(bn_env, tres2, res2);      // load my instance's a value
  uint32_t tmp_carry = cgbn_mul_ui32(bn_env, mul_res, ta, b);                           
  uint32_t tmp_carry2 = cgbn_add(bn_env, add_res, mul_res, tres);
  uint32_t carryout = cgbn_add_ui32(bn_env, mul_res, add_res, tmp_carry + tmp_carry2);
  cgbn_store(bn_env, res, mul_res);   // store r into sum
  *carry = cgbn_add_ui32(bn_env, add_res, tres2, carryout);
  cgbn_store(bn_env, res2, add_res);   // store r into sum
}

__global__ void kernel_addmul_1(cgbn_error_report_t* report, uint32_t* res, uint32_t n, uint32_t i, cgbn_mem_t<BITS>* const a, uint64_t inv, const uint32_t count){
  int instance = (blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance >= count) return;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
  env_t::cgbn_t  ta, tb, tres, mul_res, tres2, add_res;                                             // define a, b, r as 1024-bit bignums
  cgbn_load(bn_env, ta, a);      // load my instance's a value
  for(int i = 0; i < n; i+=2){
    cgbn_load(bn_env, tres, res+i);      // load my instance's a value
    cgbn_load(bn_env, tres2, res+n+i);      // load my instance's a value

    uint64_t* p64 = (uint64_t*)res;
    uint64_t k = inv * p64[i/2]; 
    uint32_t* p32 = (uint32_t*)&k;
    cgbn_mem_t<BITS> b;
    b._limbs[0] = p32[0]; 
    b._limbs[1] = p32[1]; 
    cgbn_load(bn_env, tb, &b);      // load my instance's a value

    cgbn_mul(bn_env, mul_res, ta, tb);                           
    uint32_t carryout = cgbn_add(bn_env, add_res, mul_res, tres);
    cgbn_store(bn_env, res+i, add_res);   // store r into sum
    cgbn_add_ui32(bn_env, add_res, tres2, carryout);
    cgbn_store(bn_env, res+n+i, add_res);   // store r into sum
  }
}


int add_two_num(cgbn_mem_t<BITS>* c, cgbn_mem_t<BITS>* const a, cgbn_mem_t<BITS>* const b, uint32_t* carry, const uint32_t count){
  cgbn_error_report_t *report;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_add<<<1, TPI>>>(report, c, a, b, carry, count);
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  return 0;
}

int add_1(cgbn_mem_t<BITS>* c, cgbn_mem_t<BITS>* const a, const uint32_t b, uint32_t* carry, const uint32_t count){
  cgbn_error_report_t *report;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_add_1<<<1, TPI>>>(report, c, a, b, carry, count);
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  return 0;
}


int sub_two_num(cgbn_mem_t<BITS>* c, cgbn_mem_t<BITS>* const a, cgbn_mem_t<BITS>* const b, uint32_t* carry, const uint32_t count){
  cgbn_error_report_t *report;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_sub<<<1, TPI>>>(report, c, a, b, carry, count);
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  return 0;
}

int sub_1(cgbn_mem_t<BITS>* c, cgbn_mem_t<BITS>* const a, const uint32_t b, uint32_t* carry, const uint32_t count){
  cgbn_error_report_t *report;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_sub_1<<<1, TPI>>>(report, c, a, b, carry, count);
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  return 0;
}
int mul_1(cgbn_mem_t<BITS>* c, cgbn_mem_t<BITS>* const a, const uint32_t b, uint32_t* carry, const uint32_t count){
  cgbn_error_report_t *report;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_mul_1<<<1, TPI>>>(report, c, a, b, carry, count);
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  return 0;
}

int addmul_1(uint32_t *res, uint32_t *res2, cgbn_mem_t<BITS>* const a, const uint32_t b, uint32_t* carry, const uint32_t count){
  cgbn_error_report_t *report;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_addmul_1<<<1, TPI>>>(report, res, res2, a, b, carry, count);
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  return 0;
}

int addmul_1(uint32_t *res, uint32_t n, uint32_t i, cgbn_mem_t<BITS>* const a, uint64_t inv, const uint32_t count){
  cgbn_error_report_t *report;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_addmul_1<<<1, TPI>>>(report, res, n, i, a, inv, count);
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  return 0;
}

int mul_two_num(
    cgbn_mem_t<BITS>* c_low, 
    cgbn_mem_t<BITS>* c_high, 
    cgbn_mem_t<BITS>* const a, cgbn_mem_t<BITS>* const b, const uint32_t count){
  cgbn_error_report_t *report;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_mul<<<1, TPI>>>(report, c_low, c_high, a, b, count);
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  return 0;
}
int mul_two_num(
    uint32_t *c, 
    cgbn_mem_t<BITS>* const a, cgbn_mem_t<BITS>* const b, const uint32_t count){
  cgbn_error_report_t *report;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  kernel_mul<<<1, TPI>>>(report, c, a, b, count);
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  return 0;
}


__global__ void kernel_add(cgbn_error_report_t* report, 
    cgbn_mem_t<BITS>* x1, cgbn_mem_t<BITS>* y1, cgbn_mem_t<BITS>* z1, 
    cgbn_mem_t<BITS>* x2, cgbn_mem_t<BITS>* y2, cgbn_mem_t<BITS>* z2, 
    cgbn_mem_t<BITS>* x_out, cgbn_mem_t<BITS>* y_out, cgbn_mem_t<BITS>* z_out){
  int instance = instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  //if(instance >= count) return;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
  env_t::cgbn_t  tx1, ty1, tz1, ty2, tx2, tz2;                                             // define a, b, r as 1024-bit bignums
  cgbn_load(bn_env, tx1, x1);      // load my instance's a value
  cgbn_load(bn_env, ty1, y1);      // load my instance's b value
  cgbn_load(bn_env, tz1, z1);      // load my instance's b value
  cgbn_load(bn_env, tx2, x2);      // load my instance's a value
  cgbn_load(bn_env, ty2, y2);      // load my instance's b value
  cgbn_load(bn_env, tz2, z2);      // load my instance's b value

  //cgbn_store(bn_env, c, tc);   // store r into sum
  //z1z1=squared z1
  env_t::cgbn_t z1z1, z2z2, u1, u2, z1_cubed, z2_cubed, s1, s2;
  cgbn_mul(bn_env, z1z1, tz1, tz1);                           // r=a+b
  //z2z2=squared z2
  cgbn_mul(bn_env, z2z2, tz2, tz2);                           // r=a+b
  //u1=x1*z2z2
  cgbn_mul(bn_env, u1, tx1, z2z2);                           // r=a+b
  //u2=x2*z1z1
  cgbn_mul(bn_env, u2, tx2, z1z1);                           // r=a+b
  //z1_cubed=z1*z1z1
  cgbn_mul(bn_env, z1_cubed, tz1, z1z1);                           // r=a+b
  //z2_cubed=z2*z2z2
  cgbn_mul(bn_env, z2_cubed, tz2, z2z2);                           // r=a+b
  //s1=y1*z2_cubed
  cgbn_mul(bn_env, s1, ty1, z2_cubed);                           // r=a+b
  //s2=y2*z1_cubed
  cgbn_mul(bn_env, s2, ty2, z1_cubed);                           // r=a+b
  //if(u1==u2 && s1==s2)
  {
  }
  //h=u2-u1
  //alt_bn128_Fq S2_minus_S1 = S2-S1;
  //alt_bn128_Fq I = (H+H).squared();                    // I = (2 * H)^2
  //alt_bn128_Fq J = H * I;                              // J = H * I
  //alt_bn128_Fq r = S2_minus_S1 + S2_minus_S1;          // r = 2 * (S2-S1)
  //alt_bn128_Fq V = U1 * I;                             // V = U1 * I
  //alt_bn128_Fq X3 = r.squared() - J - (V+V);           // X3 = r^2 - J - 2 * V
  //alt_bn128_Fq S1_J = S1 * J;
  //alt_bn128_Fq Y3 = r * (V-X3) - (S1_J+S1_J);          // Y3 = r * (V-X3)-2 S1 J
  //alt_bn128_Fq Z3 = ((this->Z+other.Z).squared()-Z1Z1-Z2Z2) * H; // Z3 = ((Z1+Z2)^2-Z1Z1-Z2Z2) * H

  //return alt_bn128_G1(X3, Y3, Z3);)
}
int add(
    cgbn_mem_t<BITS>* x1, cgbn_mem_t<BITS>* y1, cgbn_mem_t<BITS>* z1, 
    cgbn_mem_t<BITS>* x2, cgbn_mem_t<BITS>* y2, cgbn_mem_t<BITS>* z2, 
    cgbn_mem_t<BITS>* x_out, cgbn_mem_t<BITS>* y_out, cgbn_mem_t<BITS>* z_out){
  return 0;
}

//int main(){
//  cgbn_mem_t<BITS> a, b, c;
//  add_two_num(&c, &a, &b);
//  return 0;
//}
}
