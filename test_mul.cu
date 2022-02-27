
#include <gmp.h>
#include "cgbn/cgbn.h"
#include <stdint.h>

#define TPI 8
#define BITS 256
typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

__global__ void kernel(
    cgbn_error_report_t* report, 
    uint32_t *a, uint32_t *b, uint32_t *c){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int instance = tid / TPI;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  env_t::cgbn_t la, lb;
  env_t::cgbn_wide_t lc;
  cgbn_load(bn_env, la, a);
  //cgbn_load(bn_env, lb, b);
  cgbn_set_ui32(bn_env, lb, b[0], b[1]);
  cgbn_mul_wide(bn_env, lc, la, lb);
  cgbn_store(bn_env, c, lc._low);
  cgbn_store(bn_env, c+8, lc._high);
}
__global__ void kernel2(
    cgbn_error_report_t* report, 
    uint32_t *a, uint32_t *b, uint32_t *c){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int instance = tid / TPI;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  env_t::cgbn_t la, lb;
  env_t::cgbn_wide_t lc;
  cgbn_load(bn_env, la, a);
  cgbn_load(bn_env, lb, b);
  __shared__ uint32_t buf[16];
  uint32_t tmp1 = cgbn_mul_ui32(bn_env, lc._low, la, b[1]); 
  uint32_t tmp2 = cgbn_mul_ui32(bn_env, lc._high, la, b[0]); 
  cgbn_store(bn_env, buf, lc._low);
  cgbn_store(bn_env, buf+8, lc._high);
  uint32_t carry = cgbn_get_ui32(bn_env, lc._high, 0);
  cgbn_shift_right(bn_env, lc._high, lc._high, 32);
  uint32_t tmp3 = cgbn_add(bn_env, lc._low, lc._low, lc._high);
  uint32_t right = cgbn_get_ui32(bn_env, lc._low, 7);
  uint64_t tmp4 = (uint64_t)right + tmp2;
  uint32_t tmp5 = (uint32_t)tmp4;
  uint32_t tmp6 = 0;
  if(tmp5 != tmp4){
        tmp6 = 1;
  }
  if(threadIdx.x == 0){
    for(int i = 0; i < 16; i++){
        printf("%u ", buf[i]);
        if(i == 7) printf("\n");
    }
    printf("\n%u %u %u %lu, %u\n", tmp1, tmp2, tmp3, (uint64_t)tmp2 + right, tmp5);
  }
  cgbn_shift_left(bn_env, lc._low, lc._low, 32); 
  cgbn_set_ui32(bn_env, lc._high, tmp5, tmp1 + tmp6);
  cgbn_add_ui32(bn_env, lc._low, lc._low, carry);
  cgbn_store(bn_env, c, lc._low);
  cgbn_store(bn_env, c+8, lc._high);
}
int main(){
    //uint32_t a[8], b[8], c[16], c2[16];
    //for(int i = 0; i < 8; i++){
    //    a[i] = 0xffff;
    //    b[i] = 0;
    //}
    //a[0] = 0xfffff;
    //b[0] = 0xfffff;
    //b[1] = 0xffff;
    uint32_t a[8] = {3632069959, 1008765974, 1752287885, 2541841041, 2172737629 ,3092268470 , 3778125865, 811880050};
    uint64_t q = 15798470839202889540;///10173942680800069712;
    uint32_t b[8] = {0};
    b[0] = ((uint32_t*)&q)[0];
    b[1] = ((uint32_t*)&q)[1];
    uint32_t c[16], c2[16];

    uint32_t* da, *db, *dc, *dc2;
    cudaMalloc((void**)&da, 8 * sizeof(int));
    cudaMalloc((void**)&db, 8 * sizeof(int));
    cudaMalloc((void**)&dc, 16 * sizeof(int));
    cudaMalloc((void**)&dc2, 16 * sizeof(int));
    cudaMemcpy(da, a, 8 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, 8 * sizeof(int), cudaMemcpyHostToDevice);
    cgbn_error_report_t* report = nullptr;
    cgbn_error_report_alloc(&report); 
    for(int i = 0; i < 1; i++)
    kernel<<<1, 8>>>(report, da, db, dc);
    for(int i = 0; i < 1; i++)
    kernel2<<<1, 8>>>(report, da, db, dc2);
    cudaMemcpy(c, dc, 16 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(c2, dc2, 16 * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 16; i++){
        printf("%u ", c[i]);
    }
    printf("\n");
    for(int i = 0; i < 16; i++){
        printf("%u ", c2[i]);
    }
    printf("\n");
    return;
}
