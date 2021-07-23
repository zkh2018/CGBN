#include "cgbn_math.h"
#include <string.h>

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

int main(){
  gpu_buffer a, b, c, c_mul_low, c_mul_high;
  a.resize_host(1);
  b.resize_host(1);
  c.resize_host(1);
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

  gpu_buffer da, db, dc, dc_mul_low, dc_mul_high;
  da.resize(1);
  db.resize(1);
  dc.resize(1);
  dc_mul_low.resize(1);
  dc_mul_high.resize(1);
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

  //mul
  mpz_t rc_mul;
  mpz_init(rc_mul);
  mpz_mul(rc_mul, ra, rb);
  mul_two_num(dc_mul_low.ptr, dc_mul_high.ptr, da.ptr, db.ptr, 1);
  dc_mul_low.copy_to_host(c_mul_low);
  dc_mul_high.copy_to_host(c_mul_high);

  uint32_t correct2_mul[BITS/32 * 2] = {0};
  from_mpz(rc_mul, correct2_mul, BITS/32 * 2);

  //printf("mul carry = %u %u\n", carry, correct2_mul[BITS/32]);
  //for(int i = 0; i < BITS/32; i++){
  //  printf("(%u, %u)\n", correct2_mul[i], c_mul_low.ptr->_limbs[i]);
  //}
  //for(int i = 0; i < BITS/32; i++){
  //  printf("(%u, %u)\n", correct2_mul[i + BITS/32], c_mul_high.ptr->_limbs[i]);
  //}
  //printf("\n");

  cmp_ret = memcmp(correct2_mul, c_mul_low.ptr->_limbs, BITS/32 * sizeof(int32_t));
  printf("compare mul low result = %d\n", cmp_ret);
  cmp_ret = memcmp(correct2_mul + BITS/32, c_mul_high.ptr->_limbs, BITS/32 * sizeof(int32_t));
  printf("compare mul high result = %d\n", cmp_ret);
  
  return 0;
}
