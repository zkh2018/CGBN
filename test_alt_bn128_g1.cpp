#include "cgbn_math.h"
#include "cgbn_fp.h"
#include "cgbn_alt_bn128_g1.h"
#include <string.h>
#include "assert.h"
#include <time.h>

using namespace gpu;

const int N = BITS/64;

struct Fp{
  mp_limb_t mont[N], modulus[N];
  uint64_t inv;
  Fp mul(const Fp& other){
    mp_limb_t res[2*N];
    mpn_mul_n(res, mont, other.mont, N);

    for(int i = 0; i < N; i++){
      mp_limb_t k =  inv * res[i];
      mp_limb_t carry = mpn_addmul_1(res + i, modulus, N, k);
      mpn_add_1(res + N + i, res + N + i, N - i, carry);
    }
    if(mpn_cmp(res + N, modulus, N) >= 0){
      mp_limb_t borrow = mpn_sub(res + N, res + N, N, modulus, N);
    }
    Fp ret;
    mpn_copyi(ret.mont, res + N, N);
    mpn_copyi(ret.modulus, modulus, N);
    ret.inv = inv;
    return ret;
  }
  Fp add(const Fp& other){
    mp_limb_t scratch[N+1];
    mp_limb_t carry = mpn_add_n(scratch, mont, other.mont, N);
    scratch[N] = carry;
    if(carry || mpn_cmp(scratch, modulus, N) >= 0){
      const mp_limb_t borrow = mpn_sub(scratch, scratch, N+1, modulus, N);
    }
    Fp ret;
    mpn_copyi(ret.mont, scratch, N);
    mpn_copyi(ret.modulus, modulus, N);
    ret.inv = inv;
    return ret;
  }
  Fp sub(const Fp& other){
    mp_limb_t scratch[N+1];
    if(mpn_cmp(mont, other.mont, N) < 0){
      const mp_limb_t carry = mpn_add_n(scratch, mont, modulus, N);
      scratch[N] = carry;
    }else{
      mpn_copyi(scratch, mont, N);
    }
    const mp_limb_t borrow = mpn_sub(scratch , scratch, N+1, other.mont, N);
    Fp ret;
    mpn_copyi(ret.mont, scratch, N);
    mpn_copyi(ret.modulus, modulus, N);
    ret.inv = inv;
    return ret;
  }
  Fp squared(){
    return mul(*this); 
  }

  bool is_zero(){
    for(int i = 0; i < N; i++){
      if(mont[i]) return false;
    }
    return 0;
  }
  bool isequal(Fp& other){
    return (mpn_cmp(mont, other.mont, N) == 0);
  }
  void copy(Fp& other){
    mpn_copyi(mont, other.mont, N);
    mpn_copyi(modulus, other.modulus, N);
    inv = other.inv;
  }

  void rand_init(){
    for(int i = 0; i < N; i++){
      mont[i] = rand();
      modulus[i] = rand();
    }
    inv = rand();
  }
  void print_ui32(mp_limb_t *array){
    uint32_t *p = (uint32_t*)array;
    for(int i = 0; i < N*2; i++){
      printf("%u ", p[i]);
    }
    printf("\n");
  }
  void print(){
    printf("mont: \n");
    print_ui32(mont);
    printf("modulus: \n");
    print_ui32(modulus);
    printf("inv: %lu\n\n", inv);
  }
};

struct AltBn128G1{
  Fp x, y, z;
  bool is_zero(){
    return z.is_zero();
  }

  AltBn128G1 dbl(){
    Fp A = x.squared();
    Fp B = y.squared();
    Fp C = B.squared();
    Fp xb = x.add(B);
    Fp xb2 = xb.squared();
    xb = xb2.sub(A);
    Fp tmp_D = xb.sub(C);
    Fp D = tmp_D.add(tmp_D);
    Fp A2 = A.add(A);
    Fp E = A2.add(A);
    Fp F = E.squared();
    Fp X3 = F.sub(D.add(D));
    Fp eightC1 = C.add(C);
    Fp eightC2 = eightC1.add(eightC1);
    Fp eightC = eightC2.add(eightC2);
    Fp dx3 = D.sub(X3);
    Fp edx3 = E.mul(dx3);
    Fp Y3 = edx3.sub(eightC);
    Fp Y1Z1 = y.mul(z);
    Fp Z3 = Y1Z1.add(Y1Z1);
    AltBn128G1 ret;
    ret.x.copy(X3);
    ret.y.copy(Y3);
    ret.z.copy(Z3);
    return ret;
  }
  AltBn128G1 add(AltBn128G1& other){
    if(is_zero()){
      return other;
    }
    if(other.is_zero()){
      return *this;
    }

    Fp Z1 = z.squared();
    Fp Z2 = other.z.squared();
    Fp U1 = x.mul(Z2);
    Fp U2 = other.x.mul(Z1);
    Fp Z1_cubed = z.mul(Z1);
    Fp Z2_cubed = other.z.mul(Z2);
    Fp S1 = y.mul(Z2_cubed);
    Fp S2 = other.y.mul(Z1_cubed);
    if(U1.isequal(U2)){
      return dbl();
    }
    Fp H = U2.sub(U1);
    Fp S2_minus_S1 = S2.sub(S1);
    Fp h2 = H.add(H);
    Fp I = h2.squared();
    Fp J = H.mul(I);
    Fp R = S2_minus_S1.add(S2_minus_S1);
    Fp V = U1.mul(I);
    Fp r2 = R.squared();
    Fp v2 = V.add(V);
    Fp rj = r2.sub(J);
    Fp X3 = rj.sub(v2);
    Fp S1_J = S1.mul(J);
    Fp vx = V.sub(X3);
    Fp s1_j2 = S1_J.add(S1_J);
    Fp rvx = R.mul(vx);
    Fp Y3 = rvx.sub(s1_j2);
    Fp abz = z.add(other.z);
    Fp abz2 = abz.squared();
    Fp abz2_z1 = abz2.sub(Z1);
    Fp abz2_z1_z2 = abz2_z1.sub(Z2);
    Fp Z3 = abz2_z1_z2.mul(H);

    AltBn128G1 ret;
    ret.x.copy(X3);
    ret.y.copy(Y3);
    ret.z.copy(Z3);

    return ret;
  }

  void rand_init(){
    x.rand_init();
    y.rand_init();
    z.rand_init();
  }
  void print(){
    printf("x:\n");
    x.print();
    printf("y:\n");
    y.print();
    printf("z:\n");
    z.print();
  }
};

void copy(AltBn128G1& a, alt_bn128_g1& da){
  copy_cpu_to_gpu(da.x.mont_repr_data, a.x.mont, sizeof(a.x.mont));
  copy_cpu_to_gpu(da.x.modulus_data, a.x.modulus, sizeof(a.x.modulus));
  da.x.inv = a.x.inv;

  copy_cpu_to_gpu(da.y.mont_repr_data, a.y.mont, sizeof(a.y.mont));
  copy_cpu_to_gpu(da.y.modulus_data, a.y.modulus, sizeof(a.y.modulus));
  da.y.inv = a.y.inv;

  copy_cpu_to_gpu(da.z.mont_repr_data, a.z.mont, sizeof(a.z.mont));
  copy_cpu_to_gpu(da.z.modulus_data, a.z.modulus, sizeof(a.z.modulus));
  da.z.inv = a.z.inv;
}

void copy_back(AltBn128G1& a, alt_bn128_g1& da){
  copy_gpu_to_cpu(a.x.mont, da.x.mont_repr_data, sizeof(a.x.mont));
  copy_gpu_to_cpu(a.x.modulus, da.x.modulus_data, sizeof(a.x.modulus));
  a.x.inv = da.x.inv;

  copy_gpu_to_cpu(a.y.mont, da.y.mont_repr_data, sizeof(a.y.mont));
  copy_gpu_to_cpu(a.y.modulus, da.y.modulus_data, sizeof(a.y.modulus));
  a.x.inv = da.x.inv;

  copy_gpu_to_cpu(a.z.mont, da.z.mont_repr_data, sizeof(a.z.mont));
  copy_gpu_to_cpu(a.z.modulus, da.z.modulus_data, sizeof(a.z.modulus));
  a.z.inv = da.z.inv;
}

void run_gpu(AltBn128G1& a, AltBn128G1& b, AltBn128G1& c){
  alt_bn128_g1 da, db, dc;
  const int data_num = 1;
  da.init(data_num);
  db.init(data_num);
  dc.init(data_num);

  copy(a, da);
  copy(b, db);

  uint32_t *gpu_res;
  gpu_malloc((void**)&gpu_res, data_num * BITS/32*3 * sizeof(uint32_t));
  gpu_buffer tmp_buffer, max_value, dmax_value;
  tmp_buffer.resize(data_num);
  max_value.resize_host(1);
  dmax_value.resize(1);
  for(int i = 0; i < BITS/32; i++){
    max_value.ptr->_limbs[i] = 0xffffffff;
  }
  dmax_value.copy_from_host(max_value);
  alt_bn128_g1_add(da, db, dc, data_num, gpu_res, tmp_buffer.ptr, dmax_value.ptr);
  copy_back(c, dc);

}

void test(){
  AltBn128G1 a, b, c, gpu_c;
  srand((unsigned)time(0));
  a.rand_init();
  b.rand_init();
  c = a.add(b);
 // printf("a:\n");
 // a.print();
 // printf("b:\n");
 // b.print();
 // printf("c:\n");
 // c.print();

  run_gpu(a, b, gpu_c);
 // printf("gpu c:\n");
 // gpu_c.print();
 int cmp_ret = memcmp(c.x.mont, gpu_c.x.mont, sizeof(c.x.mont));
 printf("compare result = %d\n", cmp_ret);
}

int main(){
  test();
  return 0;
}

