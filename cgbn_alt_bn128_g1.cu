#include "cgbn_alt_bn128_g1.h"
#include "cgbn_fp.cuh"

#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>

#include "cgbn/cgbn.h"
#include "utility/cpu_support.h"
#include "utility/cpu_simple_bn_math.h"
#include "utility/gpu_support.h"

namespace gpu{

alt_bn128_g1::alt_bn128_g1(const int count){
  init(count);
}
void alt_bn128_g1::init(const int count){
  x.init(count);
  y.init(count);
  z.init(count);
}
void alt_bn128_g1::init_host(const int count){
  x.init_host(count);
  y.init_host(count);
  z.init_host(count);
}
void alt_bn128_g1::copy_from_cpu(const alt_bn128_g1& g1){
  x.copy_from_cpu(g1.x);
  y.copy_from_cpu(g1.y);
  z.copy_from_cpu(g1.z);
}
void alt_bn128_g1::copy_to_cpu(alt_bn128_g1& g1){
  g1.x.copy_from_cpu(x);
  g1.y.copy_from_cpu(y);
  g1.z.copy_from_cpu(z);
}

struct DevFp{
  env_t::cgbn_t mont, modulus;
  uint64_t inv;
  inline __device__ bool is_zero(env_t& bn_env){
    return cgbn_equals_ui32(bn_env, mont, 0);
  }
  inline __device__ bool isequal(env_t& bn_env, const DevFp& other){
    return cgbn_equals(bn_env, mont, other.mont);
  }

  inline __device__ DevFp squared(env_t& bn_env, uint32_t *res, cgbn_mem_t<BITS>* tmp_buffer){
    device_mul_reduce(bn_env, res, mont, mont, modulus, tmp_buffer, inv);
    DevFp ret;
    ret.inv = inv;
    cgbn_set(bn_env, ret.modulus, modulus);
    cgbn_load(bn_env, ret.mont, res + BITS/32);
    return ret;
  }
  inline __device__ DevFp mul(env_t& bn_env, const DevFp& other, uint32_t *res, cgbn_mem_t<BITS>* tmp_buffer){
    device_mul_reduce(bn_env, res, mont, other.mont, modulus, tmp_buffer, inv);
    DevFp ret;
    ret.inv = inv;
    cgbn_set(bn_env, ret.modulus, modulus);
    cgbn_load(bn_env, ret.mont, res + BITS/32);
    return ret;

  }
  inline __device__ DevFp sub(env_t& bn_env, const DevFp& other, const env_t::cgbn_t& max_value){
    DevFp ret;
    device_fp_sub(bn_env, ret.mont, mont, other.mont, modulus, max_value);
    cgbn_set(bn_env, ret.modulus, modulus);
    ret.inv = inv;
    return ret;
  }
  inline __device__ DevFp add(env_t& bn_env, const DevFp& other, const env_t::cgbn_t& max_value){
    DevFp ret;
    device_fp_add(bn_env, ret.mont, mont, other.mont, modulus, max_value);
    cgbn_set(bn_env, ret.modulus, modulus);
    ret.inv = inv;
    return ret;
  }
};


struct DevAltBn128G1{
  DevFp x, y, z;

  __device__ void load(env_t& bn_env, alt_bn128_g1& a, const int offset){
    cgbn_load(bn_env, x.mont, a.x.mont_repr_data + offset);
    cgbn_load(bn_env, x.modulus, a.x.modulus_data + offset);

    cgbn_load(bn_env, y.mont, a.y.mont_repr_data + offset);
    cgbn_load(bn_env, y.modulus, a.y.modulus_data + offset);

    cgbn_load(bn_env, z.mont, a.z.mont_repr_data + offset);
    cgbn_load(bn_env, z.modulus, a.z.modulus_data + offset);
  }
  __device__ void store(env_t& bn_env, alt_bn128_g1& a, const int offset){
    cgbn_store(bn_env, a.x.mont_repr_data + offset, x.mont);
    cgbn_store(bn_env, a.x.modulus_data + offset, x.modulus);

    cgbn_store(bn_env, a.y.mont_repr_data + offset, y.mont);
    cgbn_store(bn_env, a.y.modulus_data + offset, y.modulus);

    cgbn_store(bn_env, a.z.mont_repr_data + offset, z.mont);
    cgbn_store(bn_env, a.z.modulus_data + offset, z.modulus);
  }
  __device__ void store(env_t& bn_env, DevFp& x_, DevFp& y_, DevFp& z_, alt_bn128_g1& a, const int offset){
    cgbn_store(bn_env, a.x.mont_repr_data + offset, x_.mont);
    cgbn_store(bn_env, a.x.modulus_data + offset, x_.modulus);

    cgbn_store(bn_env, a.y.mont_repr_data + offset, y_.mont);
    cgbn_store(bn_env, a.y.modulus_data + offset, y_.modulus);

    cgbn_store(bn_env, a.z.mont_repr_data + offset, z_.mont);
    cgbn_store(bn_env, a.z.modulus_data + offset, z_.modulus);
  }

  inline __device__ bool is_zero(env_t& bn_env){
    return z.is_zero(bn_env);
  }
  inline __device__ void dbl(env_t& bn_env, alt_bn128_g1& c, uint32_t* tmp_res, cgbn_mem_t<BITS>* tmp_buffer, env_t::cgbn_t& max_value, const int instance){
    if(is_zero(bn_env)){
      store(bn_env, c, instance);
      return;
    }

    const int n = BITS/32;
    uint32_t *res = tmp_res + instance * 3 * n;
    cgbn_mem_t<BITS>* buffer = tmp_buffer + instance;
    //A = squared(a.x)
    DevFp A = x.squared(bn_env, res, buffer);
    //B = squared(a.y)
    DevFp B = y.squared(bn_env, res, buffer);
    //C = squared(B)
    DevFp C = B.squared(bn_env, res, buffer);
    //D = squared(a.x + B) - A - C
    DevFp xb = x.add(bn_env, B, max_value);
    DevFp xb2 = xb.squared(bn_env, res, buffer);
    xb = xb2.sub(bn_env, A, max_value);
    DevFp tmp_D = xb.sub(bn_env, C, max_value);
    //D = D+D
    DevFp D = tmp_D.add(bn_env, tmp_D, max_value);
    //E = A + A + A
    DevFp A2 = A.add(bn_env, A, max_value);
    DevFp E = A2.add(bn_env, A, max_value);
    //F = squared(E)
    DevFp F = E.squared(bn_env, res, buffer);
    //X3 = F - (D+D)
    DevFp X3 = F.sub(bn_env, D.add(bn_env, D, max_value), max_value);
    //eightC = C+C
    DevFp eightC1 = C.add(bn_env, C, max_value);
    //eightC = eightC + eightC
    DevFp eightC2 = eightC1.add(bn_env, eightC1, max_value);
    //eightC = eightC + eightC
    DevFp eightC = eightC2.add(bn_env, eightC2, max_value);
    //Y3 = E * (D - X3) - eightC
    DevFp dx3 = D.sub(bn_env, X3, max_value);
    DevFp edx3 = E.mul(bn_env, dx3, res, buffer);
    DevFp Y3 = edx3.sub(bn_env, eightC, max_value);
    //Y1Z1 = (a.y * a.z)
    DevFp Y1Z1 = y.mul(bn_env, z, res, buffer);
    //Z3 = Y1Z1 + Y1Z1
    DevFp Z3 = Y1Z1.add(bn_env, Y1Z1, max_value);
    //c.x = X3, c.y = Y3, c.z = Z3
    store(bn_env, X3, Y3, Z3, c, instance);
  }
};

__global__ void kernel_alt_bn128_g1_add(cgbn_error_report_t* report, alt_bn128_g1 a, alt_bn128_g1 b, alt_bn128_g1 c, const uint32_t count, uint32_t *tmp_res, cgbn_mem_t<BITS>* tmp_buffer, cgbn_mem_t<BITS>* max_value){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int instance = tid / TPI;
  if(instance >= count) return;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  

  DevAltBn128G1 dev_a, dev_b;
  dev_a.load(bn_env, a, instance);
  dev_b.load(bn_env, b, instance);

  if(dev_a.is_zero(bn_env)){
    dev_b.store(bn_env, c, instance);
    return;
  }
  if(dev_b.is_zero(bn_env)){
    dev_a.store(bn_env, c, instance);
    return;
  }

  const int n = BITS / 32;
  uint32_t *res = tmp_res + instance * 3 * n;
  cgbn_mem_t<BITS>* buffer = tmp_buffer + instance;
  env_t::cgbn_t tmax_value;
  cgbn_load(bn_env, tmax_value, max_value);

  //z1=squared(a.z)
  DevFp Z1 = dev_a.z.squared(bn_env, res, buffer);
  //z2=squared(b.z)
  DevFp Z2 = dev_b.z.squared(bn_env, res, buffer);
  //u1=a.x * z2
  DevFp U1 = dev_a.x.mul(bn_env, Z2, res, buffer);
  //u2=b.x * z1
  DevFp U2 = dev_b.x.mul(bn_env, Z1, res, buffer);
  //z1_cubed = a.z * z1
  DevFp Z1_cubed = dev_a.z.mul(bn_env, Z1, res, buffer);
  //z2_cubed = b.z * z2
  DevFp Z2_cubed = dev_b.z.mul(bn_env, Z2, res, buffer);
  //s1 = a.y * z2_cubed
  DevFp S1 = dev_a.y.mul(bn_env, Z2_cubed, res, buffer);
  //s2 = b.y * z1_cubed
  DevFp S2 = dev_b.y.mul(bn_env, Z1_cubed, res, buffer);
  //if(u1 == u2) reutrn a.db1()
  if(U1.isequal(bn_env, U2)){
    dev_a.dbl(bn_env, c, tmp_res, tmp_buffer, tmax_value, instance);
    return;
  }

  //h = u2-u1
  DevFp H = U2.sub(bn_env, U1, tmax_value);
  //s2_minus_s1 = s2-s1
  DevFp S2_minus_S1 = S2.sub(bn_env, S1, tmax_value);
  //i = squared(h+h)
  DevFp h2 = H.add(bn_env, H, tmax_value);
  DevFp I = h2.squared(bn_env, res, buffer);
  //j = h * i
  DevFp J = H.mul(bn_env, I, res, buffer);
  //r = s2_minus_s1 + s2_minus_s1
  DevFp R = S2_minus_S1.add(bn_env, S2_minus_S1, tmax_value);
  //v = u1 * i
  DevFp V = U1.mul(bn_env, I, res, buffer);
  //x3 = square(r) - j - (v+v)
  DevFp r2 = R.squared(bn_env, res, buffer);
  DevFp v2 = V.add(bn_env, V, tmax_value);
  DevFp rj = r2.sub(bn_env, J, tmax_value);
  DevFp X3 = rj.sub(bn_env, v2, tmax_value);
  //s1_j = s1 * j
  DevFp S1_J = S1.mul(bn_env, J, res, buffer);
  //y3 = r * (v - x3) - (s1_j+s1_j)
  DevFp vx = V.sub(bn_env, X3, tmax_value);
  DevFp s1_j2 = S1_J.add(bn_env, S1_J, tmax_value);
  DevFp rvx = R.mul(bn_env, vx, res, buffer);
  DevFp Y3 = rvx.sub(bn_env, s1_j2, tmax_value);
  //z3 = (square(a.z + b.z) - z1 - z2) * h 
  DevFp abz = dev_a.z.add(bn_env, dev_b.z, tmax_value);
  DevFp abz2 = abz.squared(bn_env, res, buffer);
  DevFp abz2_z1 = abz2.sub(bn_env, Z1, tmax_value);
  DevFp abz2_z1_z2 = abz2_z1.sub(bn_env, Z2, tmax_value);
  DevFp Z3 = abz2_z1_z2.mul(bn_env, H, res, buffer);
  //c.x = x3 c.y = y3 c.z = z3
  dev_a.store(bn_env, X3, Y3, Z3, c, instance);
}

int alt_bn128_g1_add(alt_bn128_g1 a, alt_bn128_g1 b, alt_bn128_g1 c, const uint32_t count, uint32_t *tmp_res, cgbn_mem_t<BITS>* tmp_buffer, cgbn_mem_t<BITS>* max_value){
  cgbn_error_report_t *report;
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 

  uint32_t instances = std::min(count, (uint32_t)max_threads_per_block);
  uint32_t threads = instances * TPI;
  uint32_t blocks = (count + instances - 1) / instances;

  kernel_alt_bn128_g1_add<<<blocks, threads>>>(report, a, b, c, count, tmp_res, tmp_buffer, max_value);

  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  CUDA_CHECK(cgbn_error_report_free(report));
  return 0;
}

}
