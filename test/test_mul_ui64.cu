#include <iostream>
#include <vector>

#include <gmp.h>
#include "cgbn/cgbn.h"

#define TPI 8
#define BITS 256
typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;


__global__ void kernel(
    cgbn_error_report_t* report, 
    uint32_t*a, uint64_t b){
    context_t bn_context(cgbn_report_monitor, report, 0);
    env_t          bn_env(bn_context.env<env_t>());  
    env_t::cgbn_t la;
    env_t::cgbn_wide_t lwt;
    cgbn_load(bn_env, la, a);
    cgbn_mul_ui64(bn_env, lwt, la, b);  
    uint32_t th[2];
    cgbn_get_ui64(bn_env, lwt._high, th, 0);
    if(threadIdx.x == 0){
        printf("%lu\n", *((uint64_t*)th));
    }
}

int main(){
    uint64_t a[4] = {7277650350209642826, 7090795642739345965, 14870515224833849630, 3332466137311909057};
    uint64_t b = 678154506081227331;

    uint64_t c[4];
    uint64_t carry = mpn_mul_1((mp_limb_t*)c, (const mp_limb_t*)a, 4, b);
    printf("c\n");
    uint32_t* p = (uint32_t*)&carry;
    printf("carry = %lu %u %u\n", carry, p[0], p[1]);

    uint32_t *da;
    cudaMalloc((void**)&da, 32);
    cudaMemcpy(da, a, 32, cudaMemcpyHostToDevice);
    cgbn_error_report_t *report = nullptr;
    cgbn_error_report_alloc(&report); 
    kernel<<<1, 8>>>(report, da, b);
    cudaDeviceSynchronize();

    return;
}
