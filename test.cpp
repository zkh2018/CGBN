#include "cgbn_math.h"
#include "cgbn_fp.h"
#include "cgbn_alt_bn128_g1.h"
#include <string.h>
#include "assert.h"
#include <time.h>

using namespace gpu;

const int data_num = 1024;

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

  clock_t start = clock();
  add_two_num(dc.ptr, da.ptr, db.ptr, d_carry, 1);
  clock_t end = clock();
  printf("gpu add : %fms\n", (double)(end-start)*1000.0/CLOCKS_PER_SEC);
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
  clock_t cpu_start = clock();
  mpz_add(rc, ra, rb);
  clock_t cpu_end = clock();
  printf("cpu add: %fms\n", (double)(cpu_end-cpu_start)*1000.0/CLOCKS_PER_SEC);
  uint32_t correct2[BITS/32+1];
  from_mpz(rc, correct2, BITS/32+1);

  //printf("carry = %u %u\n", carry, correct2[BITS/32]);
  //for(int i = 0; i < BITS/32; i++){
  //  printf("(%u, %u)\n", correct2[i], c.ptr->_limbs[i]);
  //}
  int cmp_ret = memcmp(correct2, c.ptr->_limbs, BITS/32 * sizeof(int32_t));
  printf("compare add result = %d\n\n", cmp_ret);

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
  //15230403791020821917 754611498739239741 7381016538464732716 1011752739694698287                                                                                                              
  //4332616871279656263 10917124144477883021 13281191951274694749 
  //3486998266802970665                                                                                                            
  //9786893198990664585                        
  //uint64_t test_in1[4] = {15230403791020821917, 754611498739239741, 7381016538464732716, 1011752739694698287}; 
  //uint64_t test_in2[4] = {4332616871279656263, 10917124144477883021, 13281191951274694749, 3486998266802970665};
  uint64_t test_in1[4] = {10423178207724922205u, 5683435405506315886u, 15722591489810629541u, 3357719272609950010u};
  uint64_t test_in2[4] = {4332616871279656263u, 10917124144477883021u, 13281191951274694749u, 3486998266802970665u};
  uint64_t test_inv = 9786893198990664585u;
  gpu_buffer in1, in2, module_data, out, din1, din2, dmodule_data, dtmp_buffer, dout;
  //uint32_t inv = 100;
  uint64_t inv = test_inv;
  in1.resize_host(data_num);
  in2.resize_host(data_num);
  module_data.resize_host(data_num);
  out.resize_host(data_num);

  din1.resize(data_num);
  din2.resize(data_num);
  dmodule_data.resize(data_num);
  dtmp_buffer.resize(data_num);
  dout.resize(data_num);

  uint32_t *test_p32_1 = (uint32_t*)test_in1;
  uint32_t *test_p32_2 = (uint32_t*)test_in2;

  for(int it = 0; it < data_num; it++){
    for(int i = 0; i < BITS/32; i++){
      in1.ptr[it]._limbs[i] = test_p32_1[i];
      in2.ptr[it]._limbs[i] = test_p32_1[i];
      module_data.ptr[it]._limbs[i] = test_p32_2[i];
    }
  }

  din1.copy_from_host(in1);
  din2.copy_from_host(in2);
  dmodule_data.copy_from_host(module_data);

  uint32_t *gpu_res;
  gpu_malloc((void**)&gpu_res, data_num * BITS/32*3 * sizeof(uint32_t));
  uint32_t *host_res = new uint32_t[data_num * BITS/32*3];
  
  clock_t start = clock();
  fp_mul_reduce(din1.ptr, din2.ptr, inv, dmodule_data.ptr, dtmp_buffer.ptr, gpu_res, data_num);
  clock_t end = clock();
  printf("gpu mul_reduce calc times: %fms\n", (double)(end-start) * 1000.0/CLOCKS_PER_SEC);
  copy_gpu_to_cpu(host_res, gpu_res, data_num * BITS/32*3*sizeof(uint32_t));

  const int n = BITS/64;
  mp_limb_t *min1 = new mp_limb_t[data_num * n];
  mp_limb_t *min2 = new mp_limb_t[data_num * n];
  mp_limb_t *mmodule_data = new mp_limb_t[data_num * n];
  mp_limb_t *mout = new mp_limb_t[data_num * n];
  for(int it = 0; it < data_num; it++){
    uint64_t *ptr1 = (uint64_t*)in1.ptr[it]._limbs;
    uint64_t *ptr2 = (uint64_t*)in2.ptr[it]._limbs;
    uint64_t *ptr3 = (uint64_t*)module_data.ptr[it]._limbs;
    for(int i = 0; i < n; i++){
      min1[it * n + i] = ptr1[i];
      min2[it * n + i] = ptr2[i];
      mmodule_data[it * n + i] = ptr3[i];
    }
  }


  mp_limb_t *res = new mp_limb_t[data_num * 2*n];
  mp_limb_t *res2 = new mp_limb_t[data_num * 2*n];
  for(int i = 0; i < data_num; i++)
    mpn_mul_n(res + i * 2 * n, min1 + i * n, min2 + i * n, n);

  memcpy(res2, res, data_num * 2*n * sizeof(mp_limb_t));

  //printf("cpu: \n");
  clock_t cpu_start = clock();
  for(int it = 0; it < data_num; it++){
    for (size_t i = 0; i < n; ++i)
    {
      mp_limb_t k = inv * res[it * n*2 + i];
      /* calculate res = res + k * mod * b^i */
      mp_limb_t carryout = mpn_addmul_1(res+it*2*n +i, mmodule_data + it*n, n, k);

      //mp_limb_t tmpk[n] = {0};
      //tmpk[0] = k;
      //mp_limb_t tmp_res[2*n] = {0};
      //mpn_mul_n(tmp_res, mmodule_data, tmpk, n);

      //mp_limb_t tmp_carry_out = mpn_add_n(res2+i, res2+i, tmp_res, n);

      carryout = mpn_add_1(res+it*2*n +n+i, res+it*2*n +n+i, n-i, carryout);
      //tmp_carry_out = mpn_add_1(res2+n+i, res2+n+i, n-i, tmp_carry_out + tmp_res[n]);
    }
    //printf("\n");
    if(mpn_cmp(res+it*2*n +n, mmodule_data+it*n, n) >= 0){
      mp_limb_t borrow = mpn_sub(res+it*2*n +n, res+it*2*n +n, n, mmodule_data+it*n, n);
      assert(borrow == 0);
      //borrow = mpn_sub(res2+n, res2+n, n, mmodule_data, n);
      //assert(borrow == 0);
    }
  }
  clock_t cpu_end = clock();
  printf("cpu mul_reuce calc times: %f\n", (double)(cpu_end-cpu_start)*1000.0/CLOCKS_PER_SEC);
  
  uint32_t *ptr32 = (uint32_t*)res;
  //uint32_t *tmp_ptr32 = (uint32_t*)res2;
  //for(int i = 0; i < BITS/32; i++){
  //  printf("%u %u %u, ", ptr32[i], tmp_ptr32[i], host_res[i]);
  //}
 // printf("\n");
  int cmp_ret_low = 0;
  int cmp_ret_high = 0;
  for(int i = 0; i < data_num; i++){
    auto *ptr = host_res + i*3*BITS/32;
    auto *ptr2 = ptr32 + i*2*BITS/32;
    cmp_ret_low |= memcmp(ptr2, ptr, BITS/32*sizeof(uint32_t));
    cmp_ret_high |= memcmp(ptr2 +BITS/32, ptr+BITS/32, BITS/32*sizeof(uint32_t));
  }
  //int cmp_ret2 = memcmp(ptr32, tmp_ptr32, BITS/32*2*sizeof(uint32_t));
  //printf("compare mul_reduce result =%d %d %d\n", cmp_ret_low, cmp_ret_high, cmp_ret2);
  printf("compare mul_reduce result =%d %d\n\n", cmp_ret_low, cmp_ret_high);

}
void test_mul(){
  gpu_buffer a, b, c_mul_low, c_mul_high;
  a.resize_host(1);
  b.resize_host(1);
  c_mul_low.resize_host(1);
  c_mul_high.resize_host(1);
  uint64_t test_in1[4] = {10423178207724922205u, 5683435405506315886u, 15722591489810629541u, 3357719272609950010u};
  uint64_t test_in2[4] = {4332616871279656263u, 10917124144477883021u, 13281191951274694749u, 3486998266802970665u};
  uint32_t *test_p32_1 = (uint32_t*)test_in1;
  uint32_t *test_p32_2 = (uint32_t*)test_in2;
  for(int i = 0; i < BITS/32; i++){
    a.ptr->_limbs[i] = test_p32_1[i];
    b.ptr->_limbs[i] = test_p32_2[i];
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
  clock_t cpu_start = clock();
  mpn_mul_n(rc, ra, rb, n);
  clock_t cpu_end = clock();

  clock_t start = clock();
  mul_two_num(dc_mul_low.ptr, dc_mul_high.ptr, da.ptr, db.ptr, 1);
  clock_t end = clock();
  printf("gpu mul: %fms\n", (double)(end-start)*1000.0/CLOCKS_PER_SEC);
  printf("cpu mul: %fms\n", (double)(cpu_end-cpu_start)*1000.0/CLOCKS_PER_SEC);
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
  printf("compare mul high result = %d\n\n", cmp_ret);
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
  printf("compare mul_ui32 result = %d\n\n", cmp_ret);
  //cmp_ret = memcmp(correct2_mul + BITS/32, c_mul_high.ptr->_limbs, BITS/32 * sizeof(int32_t));
}

void test_sub(){
  gpu_buffer a, b, c, max_value;
  a.resize_host(1);
  b.resize_host(1);
  c.resize_host(1);
  max_value.resize_host(1);
  for(int i = 0; i < BITS/32; i++){
    a.ptr->_limbs[i] = i+2;
    b.ptr->_limbs[i] = i+10;
    max_value.ptr->_limbs[i] = 0xffffffff;
  }

  gpu_buffer da, db, dc, dc_tmp, dmax_value;
  da.resize(1);
  db.resize(1);
  dc.resize(1);
  dc_tmp.resize(1);
  dmax_value.resize(1);
  da.copy_from_host(a);
  db.copy_from_host(b);
  dmax_value.copy_from_host(max_value);

  uint32_t carry, *d_carry;
  gpu_malloc((void**)&d_carry, sizeof(uint32_t));

  clock_t start = clock();
  //sub_two_num(dc.ptr, da.ptr, db.ptr, d_carry, 1);
  sub_two_num(dc_tmp.ptr, dmax_value.ptr, db.ptr, d_carry, 1);
  add_two_num(dc.ptr, dc_tmp.ptr, da.ptr, d_carry, 1);
  add_1(dc.ptr, dc.ptr, 1, d_carry, 1);
  clock_t end = clock();
  printf("gpu sub: %fms\n", (double)(end-start)*1000.0/CLOCKS_PER_SEC);
  dc.copy_to_host(c);
  copy_gpu_to_cpu(&carry, d_carry, sizeof(uint32_t));

  const int n = BITS/64;
  mp_limb_t ma[n+1], mb[n], mc[n], mmax_value[n];
  memcpy(ma, a.ptr->_limbs, n * sizeof(mp_limb_t));
  memcpy(mb, b.ptr->_limbs, n * sizeof(mp_limb_t));
  memcpy(mmax_value, max_value.ptr->_limbs, n * sizeof(mp_limb_t));
  clock_t cpu_start = clock();
  ma[n] = 1;
  mp_limb_t borrow = mpn_sub(mc, ma, n+1, mb, n);
  clock_t cpu_end = clock();
  printf("cpu sub: %fms\n", (double)(cpu_end-cpu_start)*1000.0/CLOCKS_PER_SEC);

  uint32_t *p_c = (uint32_t*)mc;
  int cmp_ret = memcmp(p_c, c.ptr->_limbs, BITS/32 * sizeof(int32_t));
  printf("compare sub result = %d, %lu %d\n\n", cmp_ret, borrow, (int32_t)carry);
}

void test_fp_sub(){
  uint64_t test_in1[4] = {10423178207724922205u, 5683435405506315886u, 15722591489810629541u, 3357719272609950010u};
  uint64_t test_in2[4] = {4332616871279656263u, 10917124144477883021u, 13281191951274694749u, 3486998266802970665u};
  gpu_buffer in1, in2, module_data, out, max_value, din1, din2, dmodule_data, dmax_value, dout;
  in1.resize_host(data_num);
  in2.resize_host(data_num);
  module_data.resize_host(data_num);
  out.resize_host(data_num);
  max_value.resize_host(data_num);

  din1.resize(data_num);
  din2.resize(data_num);
  dmodule_data.resize(data_num);
  dout.resize(data_num);
  dmax_value.resize(data_num);

  uint32_t *test_p32_1 = (uint32_t*)test_in1;
  uint32_t *test_p32_2 = (uint32_t*)test_in2;

  for(int it = 0; it < data_num; it++){
    for(int i = 0; i < BITS/32; i++){
      in1.ptr[it]._limbs[i] = test_p32_1[i];
      in2.ptr[it]._limbs[i] = test_p32_1[i];
      module_data.ptr[it]._limbs[i] = test_p32_2[i];
      max_value.ptr[it]._limbs[i] = 0xffffffff;
    }
  }

  din1.copy_from_host(in1);
  din2.copy_from_host(in2);
  dmodule_data.copy_from_host(module_data);
  dmax_value.copy_from_host(max_value);

  const int n = BITS/64;
  mp_limb_t *min1 = new mp_limb_t[data_num * n];
  mp_limb_t *min2 = new mp_limb_t[data_num * n];
  mp_limb_t *mmodule_data = new mp_limb_t[data_num * n];
  mp_limb_t *mout = new mp_limb_t[data_num * n];
  for(int it = 0; it < data_num; it++){
    uint64_t *ptr1 = (uint64_t*)in1.ptr[it]._limbs;
    uint64_t *ptr2 = (uint64_t*)in2.ptr[it]._limbs;
    uint64_t *ptr3 = (uint64_t*)module_data.ptr[it]._limbs;
    for(int i = 0; i < BITS/64; i++){
      min1[it * n + i] = ptr1[i];
      min2[it * n + i] = ptr2[i];
      mmodule_data[it * n + i] = ptr3[i];
    }
  }

  clock_t start = clock();
  fp_sub(din1.ptr, din2.ptr, dmodule_data.ptr, dmax_value.ptr, data_num);
  clock_t end = clock();
  printf("gpu fp_sub calc times: %fms\n", (double)(end-start) * 1000.0/CLOCKS_PER_SEC);
  din1.copy_to_host(in1);


  mp_limb_t *scratch = new mp_limb_t[data_num * (n+1)];
  clock_t cpu_start = clock();
  for(int it = 0; it < data_num; it++){
    if(mpn_cmp(min1 + it*n, min2 + it*n, n) < 0){
      printf("in1 less than in2\n");
      const mp_limb_t carry = mpn_add_n(scratch + it*(n+1), min1 + it*n, mmodule_data + it*n, n);
      scratch[it*(n+1) + n] = carry;
    }else{
      mpn_copyi(scratch + it*(n+1), min1 + it*n, n);
    }
    const mp_limb_t borrow = mpn_sub(scratch + it*(n+1), scratch + it*(n+1), n+1, min2 + it*n, n);
  }
  clock_t cpu_end = clock();
  printf("cpu fp_sub calc times: %fms\n", (double)(cpu_end-cpu_start) * 1000.0/CLOCKS_PER_SEC);

  int cmp_ret = 0;
  for(int it = 0; it < data_num; it++){
    cmp_ret |= memcmp(in1.ptr[it]._limbs, scratch + it*(n+1), n * sizeof(mp_limb_t));
  }
  printf("compare fb_sub result = %d\n\n", cmp_ret);
}

void test_fp_add(){
  uint64_t test_in1[4] = {10423178207724922205u, 5683435405506315886u, 15722591489810629541u, 3357719272609950010u};
  uint64_t test_in2[4] = {4332616871279656263u, 10917124144477883021u, 13281191951274694749u, 3486998266802970665u};
  gpu_buffer in1, in2, module_data, max_value, out, din1, din2, dmodule_data, dmax_value, dout;
  in1.resize_host(data_num);
  in2.resize_host(data_num);
  module_data.resize_host(data_num);
  max_value.resize_host(data_num);
  out.resize_host(data_num);

  din1.resize(data_num);
  din2.resize(data_num);
  dmodule_data.resize(data_num);
  dmax_value.resize(data_num);
  dout.resize(data_num);

  uint32_t *test_p32_1 = (uint32_t*)test_in1;
  uint32_t *test_p32_2 = (uint32_t*)test_in2;

  for(int it = 0; it < data_num; it++){
    for(int i = 0; i < BITS/32; i++){
      in1.ptr[it]._limbs[i] = test_p32_1[i];
      in2.ptr[it]._limbs[i] = test_p32_1[i];
      module_data.ptr[it]._limbs[i] = test_p32_2[i];
      max_value.ptr[it]._limbs[i] = 0xffffffff;
    }
  }

  din1.copy_from_host(in1);
  din2.copy_from_host(in2);
  dmodule_data.copy_from_host(module_data);
  dmax_value.copy_from_host(max_value);

  const int n = BITS/64;
  mp_limb_t *min1 = new mp_limb_t[data_num * n];
  mp_limb_t *min2 = new mp_limb_t[data_num * n];
  mp_limb_t *mmodule_data = new mp_limb_t[data_num * n];
  mp_limb_t *mout = new mp_limb_t[data_num * n];
  for(int it = 0; it < data_num; it++){
    uint64_t *ptr1 = (uint64_t*)in1.ptr[it]._limbs;
    uint64_t *ptr2 = (uint64_t*)in2.ptr[it]._limbs;
    uint64_t *ptr3 = (uint64_t*)module_data.ptr[it]._limbs;
    for(int i = 0; i < BITS/64; i++){
      min1[it*n + i] = ptr1[i];
      min2[it*n + i] = ptr2[i];
      mmodule_data[it*n + i] = ptr3[i];
    }
  }
  clock_t start = clock();
  fp_add(din1.ptr, din2.ptr, dmodule_data.ptr, dmax_value.ptr, data_num);
  clock_t end = clock();
  printf("gpu fp_add calc times: %fms\n", (double)(end-start) * 1000.0/CLOCKS_PER_SEC);
  din1.copy_to_host(in1);


  mp_limb_t *scratch = new mp_limb_t[data_num * (n+1)];
  clock_t cpu_start = clock();

  for(int it = 0; it < data_num; it++){
    mp_limb_t carry = mpn_add_n(&scratch[it*(n+1)], &min1[it*n], &min2[it*n], n);
    scratch[it*(n+1) + n] = carry;
    if(carry || mpn_cmp(&scratch[it*(n+1)], &mmodule_data[it*n], n) >= 0){
      const mp_limb_t borrow = mpn_sub(scratch + it*(n+1), scratch + it*(n+1), n+1, mmodule_data + it*n, n);
    }
    mpn_copyi(min1 + it*n, scratch+it*(n+1), n);
  }
  clock_t cpu_end = clock();
  printf("cpu fp_add calc times: %fms\n", (double)(cpu_end-cpu_start) * 1000.0/CLOCKS_PER_SEC);

  int cmp_ret = 0;
  for(int it = 0; it < data_num; it++){
    cmp_ret |= memcmp(in1.ptr[it]._limbs, scratch + it*(n+1), n * sizeof(mp_limb_t));
  }
  printf("compare fb_add result = %d\n\n", cmp_ret);
}

void test_alt_bn128_g1_add(){
  alt_bn128_g1 a, b, c, da, db, dc;
  a.init_host(data_num);
  b.init_host(data_num);
  c.init_host(data_num);
  da.init(data_num);
  db.init(data_num);
  dc.init(data_num);

  for(int i = 0; i < data_num; i++){
    for(int j = 0; j < BITS/32; j++){
      a.x.mont_repr_data[i]._limbs[j] = i+j;
      a.x.modulus_data[i]._limbs[j] = i+j;
      a.y.mont_repr_data[i]._limbs[j] = i+j;
      a.y.modulus_data[i]._limbs[j] = i+j;
      a.z.mont_repr_data[i]._limbs[j] = i+j;
      a.z.modulus_data[i]._limbs[j] = i+j;

      b.x.mont_repr_data[i]._limbs[j] = i+j;
      b.x.modulus_data[i]._limbs[j] = i+j;
      b.y.mont_repr_data[i]._limbs[j] = i+j;
      b.y.modulus_data[i]._limbs[j] = i+j;
      b.z.mont_repr_data[i]._limbs[j] = i+j;
      b.z.modulus_data[i]._limbs[j] = i+j;
    }
  }
  da.copy_from_cpu(a);
  db.copy_from_cpu(b);
  dc.copy_from_cpu(c);

  uint32_t *gpu_res;
  gpu_malloc((void**)&gpu_res, data_num * BITS/32*3 * sizeof(uint32_t));
  gpu_buffer tmp_buffer, max_value, dmax_value;
  tmp_buffer.resize(data_num);
  max_value.resize_host(1);
  dmax_value.resize(1);
  clock_t gpu_start = clock();
  alt_bn128_g1_add(da, db, dc, data_num, gpu_res, tmp_buffer.ptr, dmax_value.ptr);
  clock_t gpu_end = clock();
  printf("gpu alt_bn128_g1_add calc times: %f\n", (double)(gpu_end - gpu_start) * 1000.0 / CLOCKS_PER_SEC);


}

int main(){
  //printf("gmp_num_bits=%u\n", GMP_NUMB_BIS);
  printf("data number = %d\n\n", data_num);
  //test_add();
  //test_add();
  //test_mul();
  //test_add_ui32();
  //test_mul_reduce();
  //test_mul_ui32();
  //test_sub();
  //test_fp_sub();
  //test_fp_add();

  test_alt_bn128_g1_add();
  return 0;
}
