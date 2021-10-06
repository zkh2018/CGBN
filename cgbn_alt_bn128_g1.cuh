#ifndef CGBN_ALT_BN128_G1_CUH
#define CGBN_ALT_BN128_G1_CUH

#include "cgbn_fp.cuh"

namespace gpu{
struct DevFp{
  env_t::cgbn_t mont;//, modulus;
  //uint64_t inv;
  __device__ DevFp zero(env_t& bn_env){
    DevFp zero;
    cgbn_set_ui32(bn_env, zero.mont, 0);
    return zero;
  }

  __device__ void set_zero(env_t& bn_env){
    cgbn_set_ui32(bn_env, mont, 0);
  }
  __device__ void set_one(env_t& bn_env){
    cgbn_set_ui32(bn_env, mont, 1);
  }
  __device__ void set_ui32(env_t& bn_env, const uint32_t value){
    cgbn_set_ui32(bn_env, mont,value);
  }

  __device__ void copy_from(env_t& bn_env, const DevFp& other){
    //inv = other.inv;
    cgbn_set(bn_env, mont, other.mont);
    //cgbn_set(bn_env, modulus, other.modulus);
  }
  inline __device__ bool is_zero(env_t& bn_env) const {
    return cgbn_equals_ui32(bn_env, mont, 0);
  }
  inline __device__ bool isequal(env_t& bn_env, const DevFp& other) const {
    return cgbn_equals(bn_env, mont, other.mont);
  }

  inline __device__ DevFp squared(env_t& bn_env, uint32_t *res, uint32_t* tmp_buffer, const env_t::cgbn_t& modulus, const uint64_t inv) const {
    device_mul_reduce(bn_env, res, mont, mont, modulus, tmp_buffer, inv);
    DevFp ret;
    //ret.inv = inv;
    //cgbn_set(bn_env, ret.modulus, modulus);
    cgbn_load(bn_env, ret.mont, res + 8);
    return ret;
  }
  inline __device__ DevFp mul(env_t& bn_env, const DevFp& other, uint32_t *res, uint32_t* tmp_buffer, const env_t::cgbn_t& modulus, const uint64_t inv) const {
    device_mul_reduce(bn_env, res, mont, other.mont, modulus, tmp_buffer, inv);
    DevFp ret;
    //ret.inv = inv;
    //cgbn_set(bn_env, ret.modulus, modulus);
    cgbn_load(bn_env, ret.mont, res + 8);
    return ret;

  }
  inline __device__ DevFp sub(env_t& bn_env, const DevFp& other, const env_t::cgbn_t& max_value, const env_t::cgbn_t& modulus) const {
    DevFp ret;
    device_fp_sub(bn_env, ret.mont, mont, other.mont, modulus, max_value);
    //cgbn_set(bn_env, ret.modulus, modulus);
    //ret.inv = inv;
    return ret;
  }
  inline __device__ DevFp add(env_t& bn_env, const DevFp& other, const env_t::cgbn_t& max_value, const env_t::cgbn_t& modulus) const {
    DevFp ret;
    device_fp_add(bn_env, ret.mont, mont, other.mont, modulus, max_value);
    //cgbn_set(bn_env, ret.modulus, modulus);
    //ret.inv = inv;
    return ret;
  }

  inline __device__ void load(env_t& bn_env, const Fp_model& a, const int offset){
    cgbn_load(bn_env, mont, a.mont_repr_data + offset);
    //cgbn_load(bn_env, modulus, a.modulus_data + offset);
    //inv = a.inv;
  }
  inline __device__ void store(env_t& bn_env, cgbn_mem_t<BITS>* a, const int offset){
    cgbn_store(bn_env, a + offset, mont);
  }
  inline __device__ DevFp as_bigint(env_t& bn_env, uint32_t *res, uint32_t* buffer, const env_t::cgbn_t& modulus, const uint64_t inv){
    DevFp one;
    one.set_zero(bn_env);
    one.set_one(bn_env);
    return this->mul(bn_env, one, res, buffer, modulus, inv);
  }

  inline __device__ void print_array(env_t& bn_env, env_t::cgbn_t& data, uint32_t* buffer){
    cgbn_store(bn_env, buffer, data);
    int group_tid = threadIdx.x % TPI;
    if(group_tid == 0){
      for(int i = 0; i < BITS/32; i++){
        printf("%u ", buffer[i]);
      }
      printf("\n");
    }
  }
  inline __device__ void print_array_64(env_t& bn_env, env_t::cgbn_t& data, uint32_t* buffer){
    cgbn_store(bn_env, buffer, data);
    int group_tid = threadIdx.x % TPI;
    if(group_tid == 0){
      uint64_t *p = (uint64_t*)buffer;
      for(int i = 0; i < BITS/64; i++){
        printf("%lu ", p[i]);
      }
      printf("\n");
    }
  }
  inline __device__ void print(env_t& bn_env, uint32_t* buffer){
    int group_tid = threadIdx.x % TPI;
    if(group_tid == 0)
    print_array(bn_env, mont, buffer);
  }
  inline __device__ void print_64(env_t& bn_env, uint32_t* buffer){
    int group_tid = threadIdx.x % TPI;
    if(group_tid == 0)
    print_array_64(bn_env, mont, buffer);
  }
};


struct DevAltBn128G1{
  DevFp x, y, z;

  __device__ void load(env_t& bn_env, alt_bn128_g1& a, const int offset){
    cgbn_load(bn_env, x.mont, a.x.mont_repr_data + offset);
    //cgbn_load(bn_env, x.modulus, a.x.modulus_data + offset);
    //x.inv = a.x.inv;

    cgbn_load(bn_env, y.mont, a.y.mont_repr_data + offset);
    //cgbn_load(bn_env, y.modulus, a.y.modulus_data + offset);
    //y.inv = a.y.inv;

    cgbn_load(bn_env, z.mont, a.z.mont_repr_data + offset);
    //cgbn_load(bn_env, z.modulus, a.z.modulus_data + offset);
    //z.inv = a.z.inv;
  }
  __device__ void store(env_t& bn_env, alt_bn128_g1& a, const int offset){
    cgbn_store(bn_env, a.x.mont_repr_data + offset, x.mont);
    //cgbn_store(bn_env, a.x.modulus_data + offset, x.modulus);
    //a.x.inv = x.inv;

    cgbn_store(bn_env, a.y.mont_repr_data + offset, y.mont);
    //cgbn_store(bn_env, a.y.modulus_data + offset, y.modulus);
    //a.y.inv = y.inv;

    cgbn_store(bn_env, a.z.mont_repr_data + offset, z.mont);
    //cgbn_store(bn_env, a.z.modulus_data + offset, z.modulus);
    //a.z.inv = z.inv;
  }
  __device__ void store(env_t& bn_env, DevFp& x_, DevFp& y_, DevFp& z_, alt_bn128_g1& a, const int offset){
    cgbn_store(bn_env, a.x.mont_repr_data + offset, x_.mont);
    //cgbn_store(bn_env, a.x.modulus_data + offset, x_.modulus);
    //a.x.inv = x_.inv;

    cgbn_store(bn_env, a.y.mont_repr_data + offset, y_.mont);
    //cgbn_store(bn_env, a.y.modulus_data + offset, y_.modulus);
    //a.y.inv = y_.inv;

    cgbn_store(bn_env, a.z.mont_repr_data + offset, z_.mont);
    //cgbn_store(bn_env, a.z.modulus_data + offset, z_.modulus);
    //a.z.inv = z_.inv;
  }

  inline __device__ bool is_zero(env_t& bn_env) const {
    return z.is_zero(bn_env);
  }
  inline __device__ bool is_equal(env_t& bn_env, DevAltBn128G1& other, uint32_t* res, uint32_t* buffer, const env_t::cgbn_t& modulus, const uint64_t inv ) const {
    if(this->is_zero(bn_env)){
      return other.is_zero(bn_env);
    }
    if(other.is_zero(bn_env)){
      return false;
    }

    DevFp Z1 = this->z.squared(bn_env, res, buffer, modulus, inv);
    DevFp Z2 = other.z.squared(bn_env, res, buffer, modulus, inv);
    DevFp XZ2 = x.mul(bn_env, Z2, res, buffer, modulus, inv);
    DevFp XZ1 = other.x.mul(bn_env, Z1, res, buffer, modulus, inv);
    if(!XZ2.isequal(bn_env, XZ1)){
      return false;
    }
    DevFp Z1_cubed = this->z.mul(bn_env, Z1, res, buffer, modulus, inv);
    DevFp Z2_cubed = other.z.mul(bn_env, Z2, res, buffer, modulus, inv);
    DevFp YZ2 = this->y.mul(bn_env, Z2_cubed, res, buffer, modulus, inv);
    DevFp YZ1 = other.y.mul(bn_env, Z1_cubed, res, buffer, modulus, inv);
    if(!YZ2.isequal(bn_env, YZ1)){
      return false;
    }
    return true;
  }

  __device__ void set(env_t& bn_env, const DevFp& x_, const DevFp& y_, const DevFp& z_){
    x.copy_from(bn_env, x_);
    y.copy_from(bn_env, y_);
    z.copy_from(bn_env, z_);
  }
  __device__ void copy_from(env_t& bn_env, const DevAltBn128G1& other){
    x.copy_from(bn_env, other.x);
    y.copy_from(bn_env, other.y);
    z.copy_from(bn_env, other.z);
  }
  __device__ void set_zero(env_t& bn_env){
    x.set_zero(bn_env);
    y.set_one(bn_env);
    z.set_zero(bn_env);
  }

  inline __device__ void dbl(env_t& bn_env, DevAltBn128G1* dev_c, uint32_t* res, uint32_t* buffer, env_t::cgbn_t& max_value, const env_t::cgbn_t& modulus, const uint64_t inv) const {
    if(is_zero(bn_env)){
      //store(bn_env, c, instance);
      dev_c->copy_from(bn_env, *this);
      return;
    }

    //A = squared(a.x)
    DevFp A = x.squared(bn_env, res, buffer, modulus, inv);
    //B = squared(a.y)
    DevFp B = y.squared(bn_env, res, buffer, modulus, inv);
    //C = squared(B)
    DevFp C = B.squared(bn_env, res, buffer, modulus, inv);
    //D = squared(a.x + B) - A - C
    DevFp xb = x.add(bn_env, B, max_value, modulus);
    DevFp xb2 = xb.squared(bn_env, res, buffer, modulus, inv);
    xb = xb2.sub(bn_env, A, max_value, modulus);
    DevFp tmp_D = xb.sub(bn_env, C, max_value, modulus);
    //D = D+D
    DevFp D = tmp_D.add(bn_env, tmp_D, max_value, modulus);
    //E = A + A + A
    DevFp A2 = A.add(bn_env, A, max_value, modulus);
    DevFp E = A2.add(bn_env, A, max_value, modulus);
    //F = squared(E)
    DevFp F = E.squared(bn_env, res, buffer, modulus, inv);
    //X3 = F - (D+D)
    DevFp X3 = F.sub(bn_env, D.add(bn_env, D, max_value, modulus), max_value, modulus);
    //eightC = C+C
    DevFp eightC1 = C.add(bn_env, C, max_value, modulus);
    //eightC = eightC + eightC
    DevFp eightC2 = eightC1.add(bn_env, eightC1, max_value, modulus);
    //eightC = eightC + eightC
    DevFp eightC = eightC2.add(bn_env, eightC2, max_value, modulus);
    //Y3 = E * (D - X3) - eightC
    DevFp dx3 = D.sub(bn_env, X3, max_value, modulus);
    DevFp edx3 = E.mul(bn_env, dx3, res, buffer, modulus, inv);
    DevFp Y3 = edx3.sub(bn_env, eightC, max_value, modulus);
    //Y1Z1 = (a.y * a.z)
    DevFp Y1Z1 = y.mul(bn_env, z, res, buffer, modulus, inv);
    //Z3 = Y1Z1 + Y1Z1
    DevFp Z3 = Y1Z1.add(bn_env, Y1Z1, max_value, modulus);
    //c.x = X3, c.y = Y3, c.z = Z3
    dev_c->set(bn_env, X3, Y3, Z3);
    //store(bn_env, X3, Y3, Z3, c, instance);
  }
};

inline __device__ void dev_alt_bn128_g1_add(env_t& bn_env, const DevAltBn128G1& dev_a, const DevAltBn128G1& dev_b, DevAltBn128G1* dev_c, uint32_t* res, uint32_t* buffer, env_t::cgbn_t& tmax_value, const env_t::cgbn_t& modulus, const uint64_t inv){

  if(dev_a.is_zero(bn_env)){
   // dev_b.store(bn_env, c, instance);
    dev_c->copy_from(bn_env, dev_b);
    return;
  }
  if(dev_b.is_zero(bn_env)){
    //dev_a.store(bn_env, c, instance);
    dev_c->copy_from(bn_env, dev_a);
    return;
  }

  //z1=squared(a.z)
  DevFp Z1 = dev_a.z.squared(bn_env, res, buffer, modulus, inv);
  //z2=squared(b.z)
  DevFp Z2 = dev_b.z.squared(bn_env, res, buffer, modulus, inv);
  //u1=a.x * z2
  DevFp U1 = dev_a.x.mul(bn_env, Z2, res, buffer, modulus, inv);
  //u2=b.x * z1
  DevFp U2 = dev_b.x.mul(bn_env, Z1, res, buffer, modulus, inv);
  //z1_cubed = a.z * z1
  DevFp Z1_cubed = dev_a.z.mul(bn_env, Z1, res, buffer, modulus, inv);
  //z2_cubed = b.z * z2
  DevFp Z2_cubed = dev_b.z.mul(bn_env, Z2, res, buffer, modulus, inv);
  //s1 = a.y * z2_cubed
  DevFp S1 = dev_a.y.mul(bn_env, Z2_cubed, res, buffer, modulus, inv);
  //s2 = b.y * z1_cubed
  DevFp S2 = dev_b.y.mul(bn_env, Z1_cubed, res, buffer, modulus, inv);
  //if(u1 == u2) reutrn a.db1()
  if(U1.isequal(bn_env, U2) && S1.isequal(bn_env, S2)){
    dev_a.dbl(bn_env, dev_c, res, buffer, tmax_value, modulus, inv);
    return;
  }

  //h = u2-u1
  DevFp H = U2.sub(bn_env, U1, tmax_value, modulus);
  //s2_minus_s1 = s2-s1
  DevFp S2_minus_S1 = S2.sub(bn_env, S1, tmax_value, modulus);
  //i = squared(h+h)
  DevFp h2 = H.add(bn_env, H, tmax_value, modulus);
  DevFp I = h2.squared(bn_env, res, buffer, modulus, inv);
  //j = h * i
  DevFp J = H.mul(bn_env, I, res, buffer, modulus, inv);
  //r = s2_minus_s1 + s2_minus_s1
  DevFp R = S2_minus_S1.add(bn_env, S2_minus_S1, tmax_value, modulus);
  //v = u1 * i
  DevFp V = U1.mul(bn_env, I, res, buffer, modulus, inv);
  //x3 = square(r) - j - (v+v)
  DevFp r2 = R.squared(bn_env, res, buffer, modulus, inv);
  DevFp v2 = V.add(bn_env, V, tmax_value, modulus);
  DevFp rj = r2.sub(bn_env, J, tmax_value, modulus);
  DevFp X3 = rj.sub(bn_env, v2, tmax_value, modulus);
  //s1_j = s1 * j
  DevFp S1_J = S1.mul(bn_env, J, res, buffer, modulus, inv);
  //y3 = r * (v - x3) - (s1_j+s1_j)
  DevFp vx = V.sub(bn_env, X3, tmax_value, modulus);
  DevFp s1_j2 = S1_J.add(bn_env, S1_J, tmax_value, modulus);
  DevFp rvx = R.mul(bn_env, vx, res, buffer, modulus, inv);
  DevFp Y3 = rvx.sub(bn_env, s1_j2, tmax_value, modulus);
  //z3 = (square(a.z + b.z) - z1 - z2) * h 
  DevFp abz = dev_a.z.add(bn_env, dev_b.z, tmax_value, modulus);
  DevFp abz2 = abz.squared(bn_env, res, buffer, modulus, inv);
  DevFp abz2_z1 = abz2.sub(bn_env, Z1, tmax_value, modulus);
  DevFp abz2_z1_z2 = abz2_z1.sub(bn_env, Z2, tmax_value, modulus);
  DevFp Z3 = abz2_z1_z2.mul(bn_env, H, res, buffer, modulus, inv);
  //c.x = x3 c.y = y3 c.z = z3
  dev_c->set(bn_env, X3, Y3, Z3);
  //dev_a.store(bn_env, X3, Y3, Z3, c, instance);
}

}// namespace gpu

#endif