#include "cgbn_math.h"
#include "cgbn_fp.h"
#include "cgbn_alt_bn128_g1.h"
#include <string.h>
#include "assert.h"
#include <time.h>

using namespace gpu;

const int N = BITS/64;
const int const_modulue=10;
const int const_inv = 1000;

struct Fp{
  mp_limb_t mont[N], modulus[N];
  uint64_t inv;
  void set_zero(){
    memset(mont, 0, sizeof(mp_limb_t) * N);
    //memset(modulus, 0, sizeof(mp_limb_t) * N);
    for(int i = 0; i < N; i++){
      modulus[i] = const_modulue;
    }
    inv = const_inv;
  }
  void set_one(){
    set_zero();
    mont[0] = 1;
    for(int i = 0; i < N; i++){
      modulus[i] = const_modulue;
    }
    inv = const_inv;
  }

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
    return true;
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
      modulus[i] = const_modulue;//rand();
    }
    //inv = rand();
    inv = const_inv;
  }
  void print_ui32(mp_limb_t *array){
    uint32_t *p = (uint32_t*)array;
    for(int i = 0; i < N*2; i++){
      printf("%u ", p[i]);
    }
    printf("\n");
  }
  void print_ui64(mp_limb_t *array){
    for(int i = 0; i < N; i++){
      printf("%lu ", array[i]);
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
  void print_ui64(){
    printf("mont: \n");
    print_ui64(mont);
    printf("modulus: \n");
    print_ui64(modulus);
    printf("inv: %lu\n\n", inv);
  }
};

struct AltBn128G1{
  Fp x, y, z;
  void set_zero(){
    x.set_zero();
    y.set_one();
    z.set_zero();
  }
  void set_one(){
    x.set_one();
    y.set_one();
    z.set_one();
  }
  bool is_zero(){
    return z.is_zero();
  }
  bool is_equal(AltBn128G1& other){
    if(is_zero()){
      return other.is_zero();
    }
    if(other.is_zero()){
      return false;
    }

    Fp Z1 = z.squared();
    Fp Z2 = other.z.squared();
    Fp XZ2 = x.mul(Z2);
    Fp XZ1 = other.x.mul(Z1);
    if(!XZ2.isequal(XZ1)){
      return false;
    }
    Fp Z1_cubed = z.mul(Z1);
    Fp Z2_cubed = other.z.mul(Z2);
    Fp YZ2 = y.mul(Z2_cubed);
    Fp YZ1 = other.y.mul(Z1_cubed);
    if(!YZ2.isequal(YZ1)){
      return false;
    }
    return true;
  }

  AltBn128G1 dbl(){
    if(is_zero()){
      return *this;
    }

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

    if(U1.isequal(U2) && S1.isequal(S2)){
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
  void init_a(){
    mp_limb_t x_mont[BITS/64] = {10635073041846494731u, 18385197196700739644u, 8847835663514441480u, 1691939150287291782u};
    mp_limb_t x_modulus[BITS/64] = {4332616871279656263u, 10917124144477883021u, 13281191951274694749u, 3486998266802970665u};
    
    mp_limb_t y_mont[BITS/64] = {15231017389912311329u, 5016431550731635077u, 8941756931650430670u, 1423108634455529116u};
    mp_limb_t y_modulus[BITS/64] = {4332616871279656263u, 10917124144477883021u, 13281191951274694749u, 3486998266802970665u};

    mp_limb_t z_mont[BITS/64] = {15230403791020821917u, 754611498739239741u, 7381016538464732716u, 1011752739694698287u};
    mp_limb_t z_modulus[BITS/64] = {4332616871279656263u, 10917124144477883021u, 13281191951274694749u, 3486998266802970665u};

    const size_t size = BITS/64 * sizeof(mp_limb_t);
    memcpy(x.mont, x_mont, size);
    memcpy(x.modulus, x_modulus, size);
    x.inv = 0;

    memcpy(y.mont, y_mont, size);
    memcpy(y.modulus, y_modulus, size);
    y.inv = 0;
    
    memcpy(z.mont, z_mont, size);
    memcpy(z.modulus, z_modulus, size);
    z.inv = 0;
  }
  void init_b(){
    mp_limb_t x_mont[BITS/64] = {9870344786918826698u, 16706809717572462584u, 8162712831543794517u, 354779194311042116u};
    mp_limb_t x_modulus[BITS/64] = {4332616871279656263u, 10917124144477883021u, 13281191951274694749u, 3486998266802970665u};

    mp_limb_t y_mont[BITS/64] = {12006604663029098484u, 7700291543686466839u, 1525725619814363843u, 2854626384676222931u};
    mp_limb_t y_modulus[BITS/64] = {4332616871279656263u, 10917124144477883021u, 13281191951274694749u, 3486998266802970665u};

    mp_limb_t z_mont[BITS/64] = {15230403791020821917u, 754611498739239741u, 7381016538464732716u, 1011752739694698287u};
    mp_limb_t z_modulus[BITS/64] = {4332616871279656263u, 10917124144477883021u, 13281191951274694749u, 3486998266802970665u};

    const size_t size = BITS/64 * sizeof(mp_limb_t);
    memcpy(x.mont, x_mont, size);
    memcpy(x.modulus, x_modulus, size);
    x.inv = 0;

    memcpy(y.mont, y_mont, size);
    memcpy(y.modulus, y_modulus, size);
    y.inv = 0;
    
    memcpy(z.mont, z_mont, size);
    memcpy(z.modulus, z_modulus, size);
    z.inv = 0;
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

void copy(AltBn128G1& a, alt_bn128_g1& da, const int offset = 0){
  copy_cpu_to_gpu(da.x.mont_repr_data + offset, a.x.mont, sizeof(a.x.mont));
  copy_cpu_to_gpu(da.x.modulus_data + offset, a.x.modulus, sizeof(a.x.modulus));
  da.x.inv = a.x.inv;

  copy_cpu_to_gpu(da.y.mont_repr_data + offset, a.y.mont, sizeof(a.y.mont));
  copy_cpu_to_gpu(da.y.modulus_data + offset, a.y.modulus, sizeof(a.y.modulus));
  da.y.inv = a.y.inv;

  copy_cpu_to_gpu(da.z.mont_repr_data + offset, a.z.mont, sizeof(a.z.mont));
  copy_cpu_to_gpu(da.z.modulus_data + offset, a.z.modulus, sizeof(a.z.modulus));
  da.z.inv = a.z.inv;
}
void copy_fp(const Fp& a, Fp_model& da, const int offset = 0){
  copy_cpu_to_gpu(da.mont_repr_data + offset, a.mont, sizeof(a.mont));
  copy_cpu_to_gpu(da.modulus_data + offset, a.modulus, sizeof(a.modulus));
  da.inv = a.inv;
}

void copy_back(AltBn128G1& a, alt_bn128_g1& da, const int offset = 0){
  copy_gpu_to_cpu(a.x.mont, da.x.mont_repr_data + offset, sizeof(a.x.mont));
  copy_gpu_to_cpu(a.x.modulus, da.x.modulus_data + offset, sizeof(a.x.modulus));
  //a.x.inv = da.x.inv;

  copy_gpu_to_cpu(a.y.mont, da.y.mont_repr_data + offset, sizeof(a.y.mont));
  copy_gpu_to_cpu(a.y.modulus, da.y.modulus_data + offset, sizeof(a.y.modulus));
  //a.x.inv = da.x.inv;

  copy_gpu_to_cpu(a.z.mont, da.z.mont_repr_data + offset, sizeof(a.z.mont));
  copy_gpu_to_cpu(a.z.modulus, da.z.modulus_data + offset, sizeof(a.z.modulus));
  //a.z.inv = da.z.inv;
}

void reduce_sum(AltBn128G1* values, Fp* scalars, const size_t *index_it, AltBn128G1* partial, uint32_t *counters, const uint32_t ranges_size, const int* firsts, const int* seconds, AltBn128G1& zero, AltBn128G1& one, Fp& fp_zero, Fp& fp_one, char *density, mp_limb_t* bn_exponents){
  for(int j = 0; j < ranges_size; j++){
    AltBn128G1 result ;
    result.set_zero();
    int count = 0;
    for(int i = firsts[j]; i < seconds[j]; i++){
      auto& scalar = scalars[index_it[i]];
      if(scalar.isequal(fp_zero)){
      }else if(scalar.isequal(fp_one)){
        result = result.add(values[i]); 
      }else{
        density[i] = 1;
        //bn_exponents[i] = scalar.as_bigint();
        count += 1;
      }
    }
    partial[j] = result;
    counters[j] = count;
  }
}

void reduce_sum_other_order(AltBn128G1* values, AltBn128G1* scalar_start, const size_t *index_it, AltBn128G1* partial, uint32_t *counters, const uint32_t ranges_size, const int* firsts, const int* seconds, AltBn128G1& zero, AltBn128G1& one){
  for(int j = 0; j < ranges_size; j++){
    AltBn128G1 result ;
    result.set_zero();
    int count = 0;
    for(int i = seconds[j]-1; i >= firsts[j]; i--){
      auto& scale = scalar_start[index_it[i]];
      if(scale.is_equal(zero)){
      }else if(scale.is_equal(one)){
        result = result.add(values[i]); 
      }else{
        count += 1;
      }
    }
    partial[j] = result;
    counters[j] = count;
  }
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
  a.init_a();
  b.init_b();
  c = a.add(b);

  run_gpu(a, b, gpu_c);
 int cmp_ret = memcmp(c.x.mont, gpu_c.x.mont, sizeof(c.x.mont));
 printf("compare result = %d\n", cmp_ret);
}

void run_gpu_reduce_sum(AltBn128G1* values, Fp* scalar_start, const size_t*index_it, AltBn128G1* partial, uint32_t *counters, const uint32_t ranges_size, const int* firsts, const int* seconds, AltBn128G1& zero, AltBn128G1& one, const int step, Fp fp_zero, Fp fp_one, char *density, mp_limb_t* bn_exponents){
  alt_bn128_g1 d_values, d_partial, d_zero, d_one;
  Fp_model d_scalars, d_fp_zero, d_fp_one;
  const int data_num = ranges_size;
  d_values.init(step * ranges_size);
  d_scalars.init(step * ranges_size);
  d_partial.init(data_num);
  d_zero.init(1);
  d_one.init(1);
  d_fp_zero.init(1);
  d_fp_one.init(1);

  for(int i = 0; i < step * ranges_size; i++){
    copy_fp(scalar_start[i], d_scalars, i);
    copy(values[i], d_values, i);
  }
  copy(zero, d_zero);
  copy(one, d_one);
  copy_fp(fp_zero, d_fp_zero);
  copy_fp(fp_one, d_fp_one);

  uint32_t *gpu_res, *d_counters;
  size_t * d_index_it;
  uint32_t *d_firsts, *d_seconds;
  char *d_density;
  gpu_malloc((void**)&gpu_res, data_num * BITS/32*3 * sizeof(uint32_t));
  gpu_malloc((void**)&d_counters, ranges_size  * sizeof(uint32_t));
  gpu_malloc((void**)&d_index_it, step * ranges_size  * sizeof(size_t));
  gpu_malloc((void**)&d_firsts, ranges_size  * sizeof(uint32_t));
  gpu_malloc((void**)&d_seconds, ranges_size  * sizeof(uint32_t));
  gpu_malloc((void**)&d_density, step * ranges_size  * sizeof(char));
  gpu_buffer tmp_buffer, max_value, dmax_value, d_bn_exponents, h_bn_exponents;
  tmp_buffer.resize(data_num);
  d_bn_exponents.resize(step * ranges_size);
  h_bn_exponents.resize_host(step * ranges_size);
  max_value.resize_host(1);
  dmax_value.resize(1);
  for(int i = 0; i < BITS/32; i++){
    max_value.ptr->_limbs[i] = 0xffffffff;
  }
  dmax_value.copy_from_host(max_value);
  copy_cpu_to_gpu(d_index_it, index_it, sizeof(size_t) * step * ranges_size);
  copy_cpu_to_gpu(d_firsts, firsts, sizeof(int) * ranges_size);
  copy_cpu_to_gpu(d_seconds, seconds, sizeof(int) * ranges_size);
  clock_t start = clock();
  alt_bn128_g1_reduce_sum(
      d_values, 
      d_scalars, 
      d_index_it, 
      d_partial, 
      d_counters, 
      ranges_size, 
      d_firsts, d_seconds, 
      gpu_res, tmp_buffer.ptr, dmax_value.ptr, d_zero, d_one, d_fp_zero, d_fp_one, d_density, d_bn_exponents.ptr);
  clock_t end = clock();
  printf("gpu kernel times: %fms\n", (double)(end-start)*1000.0 / CLOCKS_PER_SEC);
  for(int i = 0; i < ranges_size; i++){
    copy_back(partial[i], d_partial, i);
  }
  copy_gpu_to_cpu(counters, d_counters, ranges_size * sizeof(int));
  copy_gpu_to_cpu(density, d_density, step * ranges_size * sizeof(char));
  d_bn_exponents.copy_to_host(h_bn_exponents);
  for(int i = 0; i < step*ranges_size; i++){
    memcpy(bn_exponents + i * 4, h_bn_exponents.ptr[i]._limbs, 32);
  }
}

void test_reduce_sum(){
  const int ranges_size = 16;
  const int step = 30130;
  printf("ranges_size = %d, reduce depth=%d\n", ranges_size, step);
  AltBn128G1 *values, *partial, *partial2, *partial3, zero, one;
  Fp* scalars, fp_zero, fp_one;
  srand((unsigned)time(0));
  size_t * index_it;
  uint32_t *counters = new uint32_t[ranges_size];
  uint32_t *counters2 = new uint32_t[ranges_size];
  uint32_t *counters3 = new uint32_t[ranges_size];
  char*density = new char[step * ranges_size];
  char*density2 = new char[step * ranges_size];
  mp_limb_t* bn_exponents = new mp_limb_t[step*ranges_size*4];
  mp_limb_t* bn_exponents2 = new mp_limb_t[step*ranges_size*4];
  int* firsts = new int[ranges_size];
  int *seconds = new int[ranges_size];
  //values = new AltBn128G1[ranges_size];
  partial = new AltBn128G1[ranges_size];
  partial2 = new AltBn128G1[ranges_size];
  partial3 = new AltBn128G1[ranges_size];
  for(int i = 0; i < ranges_size; i++){
    firsts[i] = step * i;
    seconds[i] = step * (i+1);
  }
  index_it = new size_t[step * ranges_size];
  values = new AltBn128G1[step * ranges_size];
  scalars = new Fp[step * ranges_size];
  for(int i = 0; i < step * ranges_size; i++){
    index_it[i] = i;
    values[i].rand_init();
    scalars[i].rand_init();
    if(i % 2 == 0){
      scalars[i].set_one();
    }
  }
  zero.set_zero();
  one.set_one();
  fp_zero.set_zero();
  fp_one.set_one();

  clock_t start = clock(); 
  reduce_sum(values, scalars, index_it, partial, counters, ranges_size, firsts, seconds, zero, one, fp_zero, fp_one, density, bn_exponents);
  clock_t end = clock(); 
  printf("cpu times: %fms\n", (double)(end-start)*1000.0 / CLOCKS_PER_SEC);
  
  run_gpu_reduce_sum(values, scalars, index_it, partial2, counters2, ranges_size, firsts, seconds, zero, one, step, fp_zero, fp_one, density2, bn_exponents2);

  int cmp_counter = memcmp(counters, counters2, ranges_size * sizeof(int));
  printf("compare counters = %d\n", cmp_counter);
  for(int i = 0; i < ranges_size; i++){
    if(memcmp(partial[i].x.mont, partial2[i].x.mont, 32)){
      printf("the first different partial is %d\n ", i);
      return;
    }
    //if(memcmp(partial[i].x.mont, partial3[i].x.mont, 32)){
    //  printf("other order: the first different partial is %d\n ", i);
    //  return;
    //}
  }
  printf("compare partial success\n");
  for(int i = 0; i < step * ranges_size; i++){
    if(density[i] != density2[i]){
      printf("density no equal...\n");
      break;
    }
  }
  printf("compare density success\n");
}

int main(){
  //test();
  test_reduce_sum();
  return 0;
}

