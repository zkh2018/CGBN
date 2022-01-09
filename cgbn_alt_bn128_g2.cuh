#ifndef CGBN_ALT_BN128_G2_CUH
#define CGBN_ALT_BN128_G2_CUH

#include "cgbn_alt_bn128_g1.cuh"

namespace gpu{
struct DevFp2{
  DevFp c0, c1;
  //squared_complex
  inline __device__ DevFp2 squared(env_t& bn_env, uint32_t *res, uint32_t* tmp_buffer, const env_t::cgbn_t& modulus, const env_t::cgbn_t& max_value, const uint64_t inv, const DevFp& non_residue) const {
    DevFp ab = c0.mul(bn_env, c1, res, tmp_buffer, modulus, inv);
    DevFp a_add_b = c0.add(bn_env, c1, max_value, modulus);
    DevFp nrb = non_residue.mul(bn_env, c1, res, tmp_buffer, modulus, inv);
    DevFp nrab = non_residue.mul(bn_env, ab, res, tmp_buffer, modulus, inv);
    DevFp a_add_nrb = c0.add(bn_env, nrb, max_value, modulus);
    DevFp first = a_add_b.mul(bn_env, a_add_nrb, res, tmp_buffer, modulus, inv);
    DevFp first_ab = first.sub(bn_env, ab, max_value, modulus);
    
    DevFp2 ret;
    ret.c0 = first_ab.sub(bn_env, nrab, max_value, modulus);
    ret.c1 = ab.add(bn_env, ab, max_value, modulus);
    return ret;
  }

  inline __device__ DevFp2 mul(env_t& bn_env, const DevFp2& other, uint32_t *res, uint32_t* buffer, const env_t::cgbn_t& modulus, const env_t::cgbn_t& max_value, const uint64_t inv, const DevFp& non_residue) const {
    DevFp aA = this->c0.mul(bn_env, other.c0, res, buffer, modulus, inv); 
    DevFp bB = this->c1.mul(bn_env, other.c1, res, buffer, modulus, inv);
    DevFp nrb = non_residue.mul(bn_env, bB, res, buffer, modulus, inv);
    DevFp a_add_b = this->c0.add(bn_env, this->c1, max_value, modulus);
    DevFp A_add_B = other.c0.add(bn_env, other.c1, max_value, modulus);
    DevFp first = a_add_b.mul(bn_env, A_add_B, res, buffer, modulus, inv); 
    DevFp first_sub_aA = first.sub(bn_env, aA, max_value, modulus);

    DevFp2 ret;
    ret.c0 = aA.add(bn_env, nrb, max_value, modulus);
    ret.c1 = first_sub_aA.sub(bn_env, bB, max_value, modulus);
    return ret;
  }

  inline __device__ DevFp2 negative(env_t& bn_env, const env_t::cgbn_t& max_value, const env_t::cgbn_t& modulus) const {
    DevFp2 ret;
    ret.c0 = c0.negative(bn_env, max_value, modulus);
    ret.c1 = c1.negative(bn_env, max_value, modulus);
    return ret;
  }

  inline __device__ DevFp2 sub(env_t& bn_env, const DevFp2& other, const env_t::cgbn_t& max_value, const env_t::cgbn_t& modulus) const {
    DevFp2 ret;
    ret.c0 = this->c0.sub(bn_env, other.c0, max_value, modulus);
    ret.c1 = this->c1.sub(bn_env, other.c1, max_value, modulus);
    return ret;
  }
  inline __device__ DevFp2 add(env_t& bn_env, const DevFp2& other, const env_t::cgbn_t& max_value, const env_t::cgbn_t& modulus) const {
    DevFp2 ret;
    ret.c0 = this->c0.add(bn_env, other.c0, max_value, modulus);
    ret.c1 = this->c1.add(bn_env, other.c1, max_value, modulus);
    return ret;
  }

  inline __device__ bool is_zero(env_t& bn_env) const {
    return (c0.is_zero(bn_env) && c1.is_zero(bn_env));
  }

  inline __device__ bool isequal(env_t& bn_env, const DevFp2& other) const {
    return (c0.isequal(bn_env, other.c0) && c1.isequal(bn_env, other.c1));
  }

  inline __device__ void copy_from(env_t& bn_env, const DevFp2& other){
    c0.copy_from(bn_env, other.c0);
    c1.copy_from(bn_env, other.c1);
  }

  inline __device__ void set_zero(env_t& bn_env){
    c0.set_zero(bn_env);
    c1.set_zero(bn_env);
  }
  inline __device__ void set_one(env_t& bn_env){
    c0.set_one(bn_env);
    c1.set_one(bn_env);
  }

  inline __device__ void load(env_t& bn_env, const Fp_model2& a, const int offset){
    c0.load(bn_env, a.c0, offset);
    c1.load(bn_env, a.c1, offset);
  }
  //inline __device__ void store(env_t& bn_env, cgbn_mem_t<BITS>* a, const int offset){
  //  cgbn_store(bn_env, a + offset, mont);
  //}
};

struct DevAltBn128G2{
  DevFp2 x, y, z;
  inline __device__ bool is_zero(env_t& bn_env) const {
    return z.is_zero(bn_env);
  }

  inline __device__ bool is_equal(env_t& bn_env, DevAltBn128G2& other, uint32_t* res, uint32_t* buffer, const env_t::cgbn_t& modulus, const env_t::cgbn_t& max_value, const uint64_t inv, const DevFp& non_residue) const {
    if(this->is_zero(bn_env)){
      return other.is_zero(bn_env);
    }
    if(other.is_zero(bn_env)){
      return false;
    }

    DevFp2 Z1 = this->z.squared(bn_env, res, buffer, modulus, max_value, inv, non_residue);
    DevFp2 Z2 = other.z.squared(bn_env, res, buffer, modulus, max_value, inv, non_residue);
    DevFp2 XZ2 = x.mul(bn_env, Z2, res, buffer, modulus, max_value, inv, non_residue);
    DevFp2 XZ1 = other.x.mul(bn_env, Z1, res, buffer, modulus, max_value, inv, non_residue);
    if(!XZ2.isequal(bn_env, XZ1)){
      return false;
    }
    DevFp2 Z1_cubed = this->z.mul(bn_env, Z1, res, buffer, modulus, max_value, inv, non_residue);
    DevFp2 Z2_cubed = other.z.mul(bn_env, Z2, res, buffer, modulus, max_value, inv, non_residue);
    DevFp2 YZ2 = this->y.mul(bn_env, Z2_cubed, res, buffer, modulus, max_value, inv, non_residue);
    DevFp2 YZ1 = other.y.mul(bn_env, Z1_cubed, res, buffer, modulus, max_value, inv, non_residue);
    if(!YZ2.isequal(bn_env, YZ1)){
      return false;
    }
    return true;
  }
  inline __device__ void copy_from(env_t& bn_env, const DevAltBn128G2& other){
    x.copy_from(bn_env, other.x);
    y.copy_from(bn_env, other.y);
    z.copy_from(bn_env, other.z);
  }
  
  inline __device__ void set(env_t& bn_env, const DevFp2& x_, const DevFp2& y_, const DevFp2& z_){
    x.copy_from(bn_env, x_);
    y.copy_from(bn_env, y_);
    z.copy_from(bn_env, z_);
  }
  inline __device__ void load(env_t& bn_env, alt_bn128_g2& a, const int offset){
    cgbn_load(bn_env, x.c0.mont, a.x.c0.mont_repr_data + offset);
    cgbn_load(bn_env, x.c1.mont, a.x.c1.mont_repr_data + offset);
    cgbn_load(bn_env, y.c0.mont, a.y.c0.mont_repr_data + offset);
    cgbn_load(bn_env, y.c1.mont, a.y.c1.mont_repr_data + offset);
    cgbn_load(bn_env, z.c0.mont, a.z.c0.mont_repr_data + offset);
    cgbn_load(bn_env, z.c1.mont, a.z.c1.mont_repr_data + offset);
  }
  inline __device__ void store(env_t& bn_env, alt_bn128_g2& a, const int offset){
    cgbn_store(bn_env, a.x.c0.mont_repr_data + offset, x.c0.mont);
    cgbn_store(bn_env, a.x.c1.mont_repr_data + offset, x.c1.mont);
    cgbn_store(bn_env, a.y.c0.mont_repr_data + offset, y.c0.mont);
    cgbn_store(bn_env, a.y.c1.mont_repr_data + offset, y.c1.mont);
    cgbn_store(bn_env, a.z.c0.mont_repr_data + offset, z.c0.mont);
    cgbn_store(bn_env, a.z.c1.mont_repr_data + offset, z.c1.mont);
  }

  inline __device__ void load(env_t& bn_env, uint32_t* data, const int offset){
    cgbn_load(bn_env, x.c0.mont, data + offset);
    cgbn_load(bn_env, x.c1.mont, data + offset + 8);
    cgbn_load(bn_env, y.c0.mont, data + offset + 16);
    cgbn_load(bn_env, y.c1.mont, data + offset + 24);
    cgbn_load(bn_env, z.c0.mont, data + offset + 32);
    cgbn_load(bn_env, z.c1.mont, data + offset + 40);
    //const int group_id = threadIdx.x & (TPI-1);
    //x.c0.mont._limbs[0] = data[offset + group_id];
    //x.c1.mont._limbs[0] = data[offset + 8 + group_id];
    //y.c0.mont._limbs[0] = data[offset + 16 + group_id];
    //y.c1.mont._limbs[0] = data[offset + 24 + group_id];
    //z.c0.mont._limbs[0] = data[offset + 32 + group_id];
    //z.c1.mont._limbs[0] = data[offset + 40 + group_id];
  }
  inline __device__ void store(env_t& bn_env, uint32_t* data, const int offset){
    cgbn_store(bn_env, data + offset    , x.c0.mont);   
    cgbn_store(bn_env, data + offset + 8 , x.c1.mont);
    cgbn_store(bn_env, data + offset + 16, y.c0.mont);
    cgbn_store(bn_env, data + offset + 24, y.c1.mont);
    cgbn_store(bn_env, data + offset + 32, z.c0.mont);
    cgbn_store(bn_env, data + offset + 40, z.c1.mont);
    //const int group_id = threadIdx.x & (TPI-1);
    //data[offset + group_id] = x.c0.mont._limbs[0];
    //data[offset + 8 + group_id] = x.c1.mont._limbs[0];
    //data[offset + 16 + group_id] = y.c0.mont._limbs[0];
    //data[offset + 24 + group_id] = y.c1.mont._limbs[0];
    //data[offset + 32 + group_id] = z.c0.mont._limbs[0];
    //data[offset + 40 + group_id] = z.c1.mont._limbs[0];
  }

  inline __device__ void dbl(env_t& bn_env, DevAltBn128G2* dev_c, uint32_t* res, uint32_t* buffer, const env_t::cgbn_t& max_value, const env_t::cgbn_t& modulus, const uint64_t inv, const DevFp& non_residue) const {
    if(is_zero(bn_env)){
      //store(bn_env, c, instance);
      dev_c->copy_from(bn_env, *this);
      return;
    }

    //A = squared(a.x)
    DevFp2 A = x.squared(bn_env, res, buffer, modulus, max_value, inv, non_residue);
    //B =2 squared(a.y)
    DevFp2 B = y.squared(bn_env, res, buffer, modulus, max_value, inv, non_residue);
    //C = squared(B)
    DevFp2 C = B.squared(bn_env, res, buffer, modulus, max_value, inv, non_residue);
    //D = squared(a.x + B) - A - C
    DevFp2 xb = x.add(bn_env, B, max_value, modulus);
    DevFp2 xb2 = xb.squared(bn_env, res, buffer, modulus, max_value, inv, non_residue);
    xb = xb2.sub(bn_env, A, max_value, modulus);
    DevFp2 tmp_D = xb.sub(bn_env, C, max_value, modulus);
    //D = D+D
    DevFp2 D = tmp_D.add(bn_env, tmp_D, max_value, modulus);
    //E = A + A + A
    DevFp2 A2 = A.add(bn_env, A, max_value, modulus);
    DevFp2 E = A2.add(bn_env, A, max_value, modulus);
    //F = squared(E)
    DevFp2 F = E.squared(bn_env, res, buffer, modulus, max_value, inv, non_residue);
    //X3 = F - (D+D)
    DevFp2 X3 = F.sub(bn_env, D.add(bn_env, D, max_value, modulus), max_value, modulus);
    //eightC = C+C
    DevFp2 eightC1 = C.add(bn_env, C, max_value, modulus);
    //eightC = eightC + eightC
    DevFp2 eightC2 = eightC1.add(bn_env, eightC1, max_value, modulus);
    //eightC = eightC + eightC
    DevFp2 eightC = eightC2.add(bn_env, eightC2, max_value, modulus);
    //Y3 = E * (D - X3) - eightC
    DevFp2 dx3 = D.sub(bn_env, X3, max_value, modulus);
    DevFp2 edx3 = E.mul(bn_env, dx3, res, buffer, modulus, max_value, inv, non_residue);
    DevFp2 Y3 = edx3.sub(bn_env, eightC, max_value, modulus);
    //Y1Z1 = (a.y * a.z)
    DevFp2 Y1Z1 = y.mul(bn_env, z, res, buffer, modulus, max_value, inv, non_residue);
    //Z3 = Y1Z1 + Y1Z1
    DevFp2 Z3 = Y1Z1.add(bn_env, Y1Z1, max_value, modulus);
    //c.x = X3, c.y = Y3, c.z = Z3
    dev_c->set(bn_env, X3, Y3, Z3);
    //store(bn_env, X3, Y3, Z3, c, instance);
  }
};

inline __device__ void dev_alt_bn128_g2_add(env_t& bn_env, const DevAltBn128G2& dev_a, const DevAltBn128G2& dev_b, DevAltBn128G2* dev_c, uint32_t* res, uint32_t* buffer, env_t::cgbn_t& tmax_value, const env_t::cgbn_t& modulus, const uint64_t inv, const DevFp& non_residue){
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
  DevFp2 Z1 = dev_a.z.squared(bn_env, res, buffer, modulus, tmax_value, inv, non_residue);
  //z2=squared(b.z)
  DevFp2 Z2 = dev_b.z.squared(bn_env, res, buffer, modulus, tmax_value, inv, non_residue);
  //u1=a.x * z2
  DevFp2 U1 = dev_a.x.mul(bn_env, Z2, res, buffer, modulus, tmax_value, inv, non_residue);
  //u2=b.x * z1
  DevFp2 U2 = dev_b.x.mul(bn_env, Z1, res, buffer, modulus, tmax_value, inv, non_residue);
  //z1_cubed = a.z * z1
  DevFp2 Z1_cubed = dev_a.z.mul(bn_env, Z1, res, buffer, modulus, tmax_value, inv, non_residue);
  //z2_cubed = b.z * z2
  DevFp2 Z2_cubed = dev_b.z.mul(bn_env, Z2, res, buffer, modulus, tmax_value, inv, non_residue);
  //s1 = a.y * z2_cubed
  DevFp2 S1 = dev_a.y.mul(bn_env, Z2_cubed, res, buffer, modulus, tmax_value, inv, non_residue);
  //s2 = b.y * z1_cubed
  DevFp2 S2 = dev_b.y.mul(bn_env, Z1_cubed, res, buffer, modulus, tmax_value, inv, non_residue);
  //if(u1 == u2) reutrn a.db1()
  if(U1.isequal(bn_env, U2) && S1.isequal(bn_env, S2)){
    dev_a.dbl(bn_env, dev_c, res, buffer, tmax_value, modulus, inv, non_residue);
    return;
  }

  //h = u2-u1
  DevFp2 H = U2.sub(bn_env, U1, tmax_value, modulus);
  //s2_minus_s1 = s2-s1
  DevFp2 S2_minus_S1 = S2.sub(bn_env, S1, tmax_value, modulus);
  //i = squared(h+h)
  DevFp2 h2 = H.add(bn_env, H, tmax_value, modulus);
  DevFp2 I = h2.squared(bn_env, res, buffer, modulus, tmax_value, inv, non_residue);
  //j = h * i
  DevFp2 J = H.mul(bn_env, I, res, buffer, modulus, tmax_value, inv, non_residue);
  //r = s2_minus_s1 + s2_minus_s1
  DevFp2 R = S2_minus_S1.add(bn_env, S2_minus_S1, tmax_value, modulus);
  //v = u1 * i
  DevFp2 V = U1.mul(bn_env, I, res, buffer, modulus, tmax_value, inv, non_residue);
  //x3 = square(r) - j - (v+v)
  DevFp2 r2 = R.squared(bn_env, res, buffer, modulus, tmax_value, inv, non_residue);
  DevFp2 v2 = V.add(bn_env, V, tmax_value, modulus);
  DevFp2 rj = r2.sub(bn_env, J, tmax_value, modulus);
  DevFp2 X3 = rj.sub(bn_env, v2, tmax_value, modulus);
  //s1_j = s1 * j
  DevFp2 S1_J = S1.mul(bn_env, J, res, buffer, modulus, tmax_value, inv, non_residue);
  //y3 = r * (v - x3) - (s1_j+s1_j)
  DevFp2 vx = V.sub(bn_env, X3, tmax_value, modulus);
  DevFp2 s1_j2 = S1_J.add(bn_env, S1_J, tmax_value, modulus);
  DevFp2 rvx = R.mul(bn_env, vx, res, buffer, modulus, tmax_value, inv, non_residue);
  DevFp2 Y3 = rvx.sub(bn_env, s1_j2, tmax_value, modulus);
  //z3 = (square(a.z + b.z) - z1 - z2) * h 
  DevFp2 abz = dev_a.z.add(bn_env, dev_b.z, tmax_value, modulus);
  DevFp2 abz2 = abz.squared(bn_env, res, buffer, modulus, tmax_value, inv, non_residue);
  DevFp2 abz2_z1 = abz2.sub(bn_env, Z1, tmax_value, modulus);
  DevFp2 abz2_z1_z2 = abz2_z1.sub(bn_env, Z2, tmax_value, modulus);
  DevFp2 Z3 = abz2_z1_z2.mul(bn_env, H, res, buffer, modulus, tmax_value, inv, non_residue);
  //c.x = x3 c.y = y3 c.z = z3
  dev_c->set(bn_env, X3, Y3, Z3);
  //dev_a.store(bn_env, X3, Y3, Z3, c, instance);
}
} //namespace gpu

#endif
