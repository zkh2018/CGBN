#include "cgbn_math.h"
#include <string.h>
#include "assert.h"

using namespace gpu;

void to_mpz(mpz_t r, uint32_t *x, uint32_t count) {
  mpz_import(r, count, -1, sizeof(uint32_t), 0, 0, x);
}

void from_mpz(mpz_t s, uint32_t *x, uint32_t count) {
  size_t words;

  if(mpz_sizeinbase(s, 2)>count*32) {
    fprintf(stderr, "from_mpz failed -- result does not fit\n");
    exit(1);
  }

  mpz_export(x, &words, -1, sizeof(uint32_t), 0, 0, s);
  while(words<count)
    x[words++]=0;
}

void test_add(){
  gpu_buffer a, b, c;
  a.resize_host(1);
  b.resize_host(1);
  c.resize_host(1);
  for(int i = 0; i < BITS/32; i++){
    a.ptr->_limbs[i] = i;
    b.ptr->_limbs[i] = i;
    if(i == 0) {
      a.ptr->_limbs[i] = 0xffffffff;
      b.ptr->_limbs[i] = 0xffffffff;
    }
  }

  gpu_buffer da, db, dc;
  da.resize(1);
  db.resize(1);
  dc.resize(1);
  da.copy_from_host(a);
  db.copy_from_host(b);

  uint32_t carry, *d_carry;
  gpu_malloc((void**)&d_carry, sizeof(uint32_t));

  add_two_num(dc.ptr, da.ptr, db.ptr, d_carry, 1);
  dc.copy_to_host(c);
  copy_gpu_to_cpu(&carry, d_carry, sizeof(uint32_t));

  mpz_t ra;
  mpz_init(ra);
  mpz_t rb;
  mpz_init(rb);
  to_mpz(ra, a.ptr->_limbs, BITS/32);
  to_mpz(rb, b.ptr->_limbs, BITS/32);
  mpz_t rc;
  mpz_init(rc);
  mpz_add(rc, ra, rb);
  uint32_t correct2[BITS/32+1];
  from_mpz(rc, correct2, BITS/32+1);

  //printf("carry = %u %u\n", carry, correct2[BITS/32]);
  //for(int i = 0; i < BITS/32; i++){
  //  printf("(%u, %u)\n", correct2[i], c.ptr->_limbs[i]);
  //}
  int cmp_ret = memcmp(correct2, c.ptr->_limbs, BITS/32 * sizeof(int32_t));
  printf("compare add result = %d\n", cmp_ret);

}
void test_add_ui32(){
  gpu_buffer a, c;
  uint32_t b = 10000;
  int offset = 0;
  a.resize_host(1);
  c.resize_host(1);
  for(int i = 0; i < BITS/32; i++){
    a.ptr->_limbs[i] = i;
    if(i == 0) {
      //a.ptr->_limbs[i] = 0xffffffff;
    }
    if(i < offset*2) a.ptr->_limbs[i] = 0;
  }

  gpu_buffer da, dc;
  da.resize(1);
  dc.resize(1);
  da.copy_from_host(a);

  uint32_t carry, *d_carry;
  gpu_malloc((void**)&d_carry, sizeof(uint32_t));

  add_1(dc.ptr, da.ptr, b, d_carry, 1);
  dc.copy_to_host(c);
  copy_gpu_to_cpu(&carry, d_carry, sizeof(uint32_t));

  mp_limb_t ra[BITS/64] =  {0}, rc[BITS/64]={0}, rb = b;
  uint64_t* ptr = (uint64_t*)a.ptr->_limbs;
  for(int i = 0; i < BITS/64; i++){
    //ra[i] = a.ptr->_limbs[i];
    ra[i] = ptr[i];
  }
  mp_limb_t carryout = mpn_add_1(rc + offset, ra + offset, BITS/64 - offset, rb);
  assert(carry == carryout);

  uint32_t *ptr32 = (uint32_t*)rc;
  //for(int i = 0; i < BITS/32; i++){
  //  printf("%u %u, ", ptr32[i], c.ptr->_limbs[i]);
  //  ///if(ptr32[i] != c.ptr->_limbs[i]){
  //  ///  printf("compare add_ui32 result falied: %d %u %u\n", i, ptr32[i], c.ptr->_limbs[i]);
  //  ///  return;
  //  ///}
  //}
  //printf("\n");
  int cmp_ret = memcmp(ptr32, c.ptr->_limbs, BITS/32 * sizeof(uint32_t));
  printf("compare add_ui32 result = %d\n", cmp_ret);

}

void test_mul_reduce(){
  gpu_buffer in1, in2, module_data, out, din1, din2, dmodule_data, dout;
  uint32_t inv = 100;
  in1.resize_host(1);
  in2.resize_host(1);
  module_data.resize_host(1);
  out.resize_host(1);

  din1.resize(1);
  din2.resize(1);
  dmodule_data.resize(1);
  dout.resize(1);

  for(int i = 0; i < BITS/32; i++){
    in1.ptr->_limbs[i] = i*60;
    in2.ptr->_limbs[i] = i*100 + 1;
    module_data.ptr->_limbs[i] = i*50 + 2;
  }

  din1.copy_from_host(in1);
  din2.copy_from_host(in2);
  dmodule_data.copy_from_host(module_data);

  uint32_t *gpu_res;
  gpu_malloc((void**)&gpu_res, BITS/32*3 * sizeof(uint32_t));
  uint32_t *host_res = new uint32_t[BITS/32*3];

  mul_reduce(din1.ptr, din2.ptr, inv, dmodule_data.ptr, gpu_res, 1);
  copy_gpu_to_cpu(host_res, gpu_res, BITS/32*3*sizeof(uint32_t));

  mp_limb_t min1[BITS/64], min2[BITS/64], mmodule_data[BITS/64], mout[BITS/64];
  uint64_t *ptr1 = (uint64_t*)in1.ptr->_limbs;
  uint64_t *ptr2 = (uint64_t*)in2.ptr->_limbs;
  uint64_t *ptr3 = (uint64_t*)module_data.ptr->_limbs;
  for(int i = 0; i < BITS/64; i++){
    min1[i] = ptr1[i];
    min2[i] = ptr2[i];
    mmodule_data[i] = ptr3[i];
	}


  const int n = BITS/64;
  mp_limb_t res[2*n], res2[2*n];
  mpn_mul_n(res, min1, min2, n);
  //uint32_t *p = (uint32_t*)res;
  //for(int i = 0; i < BITS/32*2; i++){
  //  printf("%u ", p[i]);
  //}
  //printf("\n");
  //mp_limb_t minv = *((uint64_t*)inv);
  memcpy(res2, res, 2*n * sizeof(mp_limb_t));

  for (size_t i = 0; i < n; ++i)
  {
    mp_limb_t k = inv * res[i];
    /* calculate res = res + k * mod * b^i */
    mp_limb_t carryout = mpn_addmul_1(res+i, mmodule_data, n, k);

    mp_limb_t tmpk[n] = {0};
    tmpk[0] = k;
    mp_limb_t tmp_res[2*n];
    mpn_mul_n(tmp_res, mmodule_data, tmpk, n);
    mp_limb_t tmp_carry_out = mpn_add_n(res2+i, res2+i, tmp_res, n);
    //if(memcmp(res, res2, 2*n*sizeof(mp_limb_t))!=0){
    //  printf("failed : %d %lu %lu\n", i, carryout, tmp_carry_out);
    //  for(int j = 0; j<2*n; j++){
    //    printf("%lu %lu, ", res[j], res2[j]);
    //  }
    //  printf("\n");
    //  return;
    //}

    assert(tmp_carry_out == 0);
    carryout = mpn_add_1(res+n+i, res+n+i, n-i, carryout);
    tmp_carry_out = mpn_add_1(res2+n+i, res2+n+i, n-i, tmp_carry_out + tmp_res[n]);
   // mp_limb_t maxvalue = 0xffffffff;
   // assert(tmp_carry_out + tmp_res[n] <= maxvalue);
    assert(carryout == 0);
    assert(tmp_carry_out == 0);
  }
  if(mpn_cmp(res+n, mmodule_data, n) >= 0){
    printf("sub...\n");
    mp_limb_t borrow = mpn_sub(res+n, res+n, n, mmodule_data, n);
    assert(borrow == 0);
    borrow = mpn_sub(res2+n, res2+n, n, mmodule_data, n);
    assert(borrow == 0);
  }
  
  uint32_t *ptr32 = (uint32_t*)res;
  uint32_t *tmp_ptr32 = (uint32_t*)res2;
  //for(int i = 0; i < BITS/32; i++){
  //  printf("%u %u %u, ", ptr32[i], tmp_ptr32[i], host_res[i]);
  //}
 // printf("\n");
  int cmp_ret_low = memcmp(ptr32, host_res, BITS/32*sizeof(uint32_t));
  int cmp_ret_high = memcmp(ptr32+BITS/32, host_res+BITS/32, BITS/32*sizeof(uint32_t));
  int cmp_ret2 = memcmp(ptr32, tmp_ptr32, BITS/32*2*sizeof(uint32_t));
  printf("compare mul_reduce result =%d %d %d\n", cmp_ret_low, cmp_ret_high, cmp_ret2);

}
void test_mul(){
  gpu_buffer a, b, c_mul_low, c_mul_high;
  a.resize_host(1);
  b.resize_host(1);
  c_mul_low.resize_host(1);
  c_mul_high.resize_host(1);
  for(int i = 0; i < BITS/32; i++){
    a.ptr->_limbs[i] = i;
    b.ptr->_limbs[i] = i;
    if(i == 0) {
      a.ptr->_limbs[i] = 0xffffffff;
      b.ptr->_limbs[i] = 0xffffffff;
    }
  }

  gpu_buffer da, db, dc_mul_low, dc_mul_high;
  da.resize(1);
  db.resize(1);
  dc_mul_low.resize(1);
  dc_mul_high.resize(1);
  da.copy_from_host(a);
  db.copy_from_host(b);

  uint32_t carry, *d_carry;
  gpu_malloc((void**)&d_carry, sizeof(uint32_t));

  //mpz_t ra;
  //mpz_init(ra);
  //mpz_t rb;
  //mpz_init(rb);
  //to_mpz(ra, a.ptr->_limbs, BITS/32);
  //to_mpz(rb, b.ptr->_limbs, BITS/32);

  //mpz_t rc_mul;
  //mpz_init(rc_mul);
  //mpz_mul(rc_mul, ra, rb);
  const int n = BITS/64;
  mp_limb_t ra[n], rb[n], rc[n*2];
  uint64_t *p64_a = (uint64_t*)a.ptr->_limbs;
  uint64_t *p64_b = (uint64_t*)b.ptr->_limbs;
  for(int i = 0; i < n; i++){
    ra[i] = p64_a[i];
    rb[i] = p64_b[i];
  }
  mpn_mul_n(rc, ra, rb, n);

  mul_two_num(dc_mul_low.ptr, dc_mul_high.ptr, da.ptr, db.ptr, 1);
  dc_mul_low.copy_to_host(c_mul_low);
  dc_mul_high.copy_to_host(c_mul_high);

  //uint32_t correct2_mul[BITS/32 * 2] = {0};
  //from_mpz(rc_mul, correct2_mul, BITS/32 * 2);

  //printf("mul carry = %u %u\n", carry, correct2_mul[BITS/32]);
  //for(int i = 0; i < BITS/32; i++){
  //  printf("(%u, %u)\n", correct2_mul[i], c_mul_low.ptr->_limbs[i]);
  //}
  //for(int i = 0; i < BITS/32; i++){
  //  printf("(%u, %u)\n", correct2_mul[i + BITS/32], c_mul_high.ptr->_limbs[i]);
  //}
  //printf("\n");

  //int cmp_ret = memcmp(correct2_mul, c_mul_low.ptr->_limbs, BITS/32 * sizeof(int32_t));
  uint32_t* p32 = (uint32_t*)rc; 
  int cmp_ret = memcmp(p32, c_mul_low.ptr->_limbs, BITS/32*sizeof(uint32_t));
  printf("compare mul low result = %d\n", cmp_ret);
  //cmp_ret = memcmp(correct2_mul + BITS/32, c_mul_high.ptr->_limbs, BITS/32 * sizeof(int32_t));
  cmp_ret = memcmp(p32 + n*2, c_mul_high.ptr->_limbs, BITS/32*sizeof(uint32_t));
  printf("compare mul high result = %d\n", cmp_ret);
}

void test_mul_ui32(){
  gpu_buffer a, b, c_mul_low, c_mul_high;
  a.resize_host(1);
  b.resize_host(1);
  c_mul_low.resize_host(1);
  c_mul_high.resize_host(1);
  mp_limb_t k = 100;
  
  for(int i = 0; i < BITS/32; i++){
    a.ptr->_limbs[i] = i;
    b.ptr->_limbs[i] = 0;
  }
  uint32_t* p = (uint32_t*)&k;
  b.ptr->_limbs[0] = p[0];
  b.ptr->_limbs[1] = p[1];

  gpu_buffer da, db, dc_mul_low, dc_mul_high;
  da.resize(1);
  db.resize(1);
  dc_mul_low.resize(1);
  dc_mul_high.resize(1);
  da.copy_from_host(a);
  db.copy_from_host(b);

  uint32_t carry, *d_carry;
  gpu_malloc((void**)&d_carry, sizeof(uint32_t));

  //mpz_t ra;
  //mpz_init(ra);
  //mpz_t rb;
  //mpz_init(rb);
  //to_mpz(ra, a.ptr->_limbs, BITS/32);
  //to_mpz(rb, b.ptr->_limbs, BITS/32);

  //mpz_t rc_mul;
  //mpz_init(rc_mul);
  //mpz_mul(rc_mul, ra, rb);
  const int n = BITS/64;
  mp_limb_t ra[n], rb[n], rc[n];
  uint64_t *p64_a = (uint64_t*)a.ptr->_limbs;
  uint64_t *p64_b = (uint64_t*)b.ptr->_limbs;
  for(int i = 0; i < n; i++){
    ra[i] = p64_a[i];
    rb[i] = p64_b[i];
  }
  //mpn_mul_n(rc, ra, rb, n);
  mpn_mul_1(rc, ra, n, k);

  mul_two_num(dc_mul_low.ptr, dc_mul_high.ptr, da.ptr, db.ptr, 1);
  dc_mul_low.copy_to_host(c_mul_low);
  dc_mul_high.copy_to_host(c_mul_high);

  //uint32_t correct2_mul[BITS/32 * 2] = {0};
  //from_mpz(rc_mul, correct2_mul, BITS/32 * 2);

  //printf("mul carry = %u %u\n", carry, correct2_mul[BITS/32]);
  //for(int i = 0; i < BITS/32; i++){
  //  printf("(%u, %u)\n", correct2_mul[i], c_mul_low.ptr->_limbs[i]);
  //}
  //for(int i = 0; i < BITS/32; i++){
  //  printf("(%u, %u)\n", correct2_mul[i + BITS/32], c_mul_high.ptr->_limbs[i]);
  //}
  //printf("\n");

  //int cmp_ret = memcmp(correct2_mul, c_mul_low.ptr->_limbs, BITS/32 * sizeof(int32_t));
  uint32_t* p32 = (uint32_t*)rc; 
  int cmp_ret = memcmp(p32, c_mul_low.ptr->_limbs, BITS/32*sizeof(uint32_t));
  printf("compare mul_ui32 result = %d\n", cmp_ret);
  //cmp_ret = memcmp(correct2_mul + BITS/32, c_mul_high.ptr->_limbs, BITS/32 * sizeof(int32_t));
}
int main(){
  printf("gmp_num_bits=%u\n", GMP_NUMB_BITS);
  test_add();
  test_mul();
  test_add_ui32();
  test_mul_reduce();
  test_mul_ui32();
  return 0;
}
