#include "msm/asm_cuda.h"

#include <gmp.h>
#include "cgbn/cgbn.h"
#include "cgbn_fp.h"
#include "cgbn_alt_bn128_g1.cuh"
#include <stdint.h>

#define BITS 384
typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;
using namespace gpu;

__global__ void kernel_msm_mul(
    uint64_t *a, uint64_t *b, uint64_t *c){
    limb_t r[12];

    asm(
      "{\n\t"
      ".reg .u64 c;\n\t"
      ".reg .u64 nc;\n\t"
      ".reg .u64 t;\n\t"
      
      "mad.lo.cc.u64 %0, %12, %18, 0;\n\t"
      "madc.hi.cc.u64 c, %12, %18, 0;\n\t"
      
      "madc.lo.cc.u64 %1, %12, %19, c;\n\t"
      "madc.hi.cc.u64 c, %12, %19, 0;\n\t"

      "madc.lo.cc.u64 %2, %12, %20, c;\n\t"
      "madc.hi.cc.u64 c, %12, %20, 0;\n\t"

      "madc.lo.cc.u64 %3, %12, %21, c;\n\t"
      "madc.hi.cc.u64 c, %12, %21, 0;\n\t"

      "madc.lo.cc.u64 %4, %12, %22, c;\n\t"
      "madc.hi.cc.u64 c, %12, %22, 0;\n\t"

      "madc.lo.cc.u64 %5, %12, %23, c;\n\t"
      "madc.hi.u64 %6, %12, %23, 0;\n\t"


      "mad.lo.cc.u64 %1, %13, %18, %1;\n\t"
      "madc.hi.cc.u64 c, %13, %18, 0;\n\t"
      
      "addc.cc.u64 t, %2, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %2, %13, %19, t;\n\t"
      "madc.hi.cc.u64 c, %13, %19, nc;\n\t"

      "addc.cc.u64 t, %3, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %3, %13, %20, t;\n\t"
      "madc.hi.cc.u64 c, %13, %20, nc;\n\t"

      "addc.cc.u64 t, %4, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %4, %13, %21, t;\n\t"
      "madc.hi.cc.u64 c, %13, %21, nc;\n\t"

      "addc.cc.u64 t, %5, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %5, %13, %22, t;\n\t"
      "madc.hi.cc.u64 c, %13, %22, nc;\n\t"

      "addc.cc.u64 t, %6, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %6, %13, %23, t;\n\t"
      "madc.hi.u64 %7, %13, %23, nc;\n\t"


      "mad.lo.cc.u64 %2, %14, %18, %2;\n\t"
      "madc.hi.cc.u64 c, %14, %18, 0;\n\t"
      
      "addc.cc.u64 t, %3, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %3, %14, %19, t;\n\t"
      "madc.hi.cc.u64 c, %14, %19, nc;\n\t"
      
      "addc.cc.u64 t, %4, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %4, %14, %20, t;\n\t"
      "madc.hi.cc.u64 c, %14, %20, nc;\n\t"

      "addc.cc.u64 t, %5, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %5, %14, %21, t;\n\t"
      "madc.hi.cc.u64 c, %14, %21, nc;\n\t"

      "addc.cc.u64 t, %6, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %6, %14, %22, t;\n\t"
      "madc.hi.cc.u64 c, %14, %22, nc;\n\t"

      "addc.cc.u64 t, %7, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %7, %14, %23, t;\n\t"
      "madc.hi.u64 %8, %14, %23, nc;\n\t"



      "mad.lo.cc.u64 %3, %15, %18, %3;\n\t"
      "madc.hi.cc.u64 c, %15, %18, 0;\n\t"
      
      "addc.cc.u64 t, %4, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %4, %15, %19, t;\n\t"
      "madc.hi.cc.u64 c, %15, %19, nc;\n\t"
      
      "addc.cc.u64 t, %5, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %5, %15, %20, t;\n\t"
      "madc.hi.cc.u64 c, %15, %20, nc;\n\t"
      
      "addc.cc.u64 t, %6, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %6, %15, %21, t;\n\t"
      "madc.hi.cc.u64 c, %15, %21, nc;\n\t"
      
      "addc.cc.u64 t, %7, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %7, %15, %22, t;\n\t"
      "madc.hi.cc.u64 c, %15, %22, nc;\n\t"
      
      "addc.cc.u64 t, %8, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %8, %15, %23, t;\n\t"
      "madc.hi.u64 %9, %15, %23, nc;\n\t"
      



      "mad.lo.cc.u64 %4, %16, %18, %4;\n\t"
      "madc.hi.cc.u64 c, %16, %18, 0;\n\t"
      
      "addc.cc.u64 t, %5, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %5, %16, %19, t;\n\t"
      "madc.hi.cc.u64 c, %16, %19, nc;\n\t"
      
      "addc.cc.u64 t, %6, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %6, %16, %20, t;\n\t"
      "madc.hi.cc.u64 c, %16, %20, nc;\n\t"
      
      "addc.cc.u64 t, %7, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %7, %16, %21, t;\n\t"
      "madc.hi.cc.u64 c, %16, %21, nc;\n\t"
      
      "addc.cc.u64 t, %8, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %8, %16, %22, t;\n\t"
      "madc.hi.cc.u64 c, %16, %22, nc;\n\t"
      
      "addc.cc.u64 t, %9, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %9, %16, %23, t;\n\t"
      "madc.hi.u64 %10, %16, %23, nc;\n\t"
      


      "mad.lo.cc.u64 %5, %17, %18, %5;\n\t"
      "madc.hi.cc.u64 c, %17, %18, 0;\n\t"
      
      "addc.cc.u64 t, %6, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %6, %17, %19, t;\n\t"
      "madc.hi.cc.u64 c, %17, %19, nc;\n\t"
      
      "addc.cc.u64 t, %7, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %7, %17, %20, t;\n\t"
      "madc.hi.cc.u64 c, %17, %20, nc;\n\t"
      
      "addc.cc.u64 t, %8, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %8, %17, %21, t;\n\t"
      "madc.hi.cc.u64 c, %17, %21, nc;\n\t"
      
      "addc.cc.u64 t, %9, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %9, %17, %22, t;\n\t"
      "madc.hi.cc.u64 c, %17, %22, nc;\n\t"
      
      "addc.cc.u64 t, %10, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %10, %17, %23, t;\n\t"
      "madc.hi.u64 %11, %17, %23, nc;\n\t"

      "}"
      : "+l"(r[0]),
      "+l"(r[1]),
      "+l"(r[2]),
      "+l"(r[3]),
      "+l"(r[4]),
      "+l"(r[5]),
      "+l"(r[6]),
      "+l"(r[7]),
      "+l"(r[8]),
      "+l"(r[9]),
      "+l"(r[10]),
      "+l"(r[11])
      : "l"(a[0]),
      "l"(a[1]),
      "l"(a[2]),
      "l"(a[3]),
      "l"(a[4]),
      "l"(a[5]),
      "l"(b[0]),
      "l"(b[1]),
      "l"(b[2]),
      "l"(b[3]),
      "l"(b[4]),
      "l"(b[5])
    );

    #pragma unroll
    for(int i = 0; i < 12; i++){
        c[i] = r[i];
    }
}


__global__ void kernel_msm_mont_mul(
    uint64_t *a, uint64_t *b, uint64_t *c, uint64_t *p, uint64_t inv){
    mul_mont_384((limb_t*)c, (limb_t*)a, (limb_t*)b, (limb_t*)p, (limb_t)inv); 
}

__global__ void kernel_cgbn_mul(
    cgbn_error_report_t* report, 
    uint32_t *a, uint32_t *b, uint32_t *c){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int instance = tid / TPI;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  

  env_t::cgbn_t dev_a, dev_b;
  env_t::cgbn_wide_t dev_c;
  cgbn_load(bn_env, dev_a, a);
  cgbn_load(bn_env, dev_b, b);

  cgbn_mul_wide(bn_env, dev_c, dev_a, dev_b);
  cgbn_store(bn_env, c, dev_c._low);
  cgbn_store(bn_env, c + 12, dev_c._high);
}

__global__ void kernel_cgbn_mont_mul(
    cgbn_error_report_t* report, 
    uint32_t *a, uint32_t *b, uint32_t *c, uint32_t *p, uint64_t inv){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int instance = tid / TPI;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  


  __shared__ uint32_t res[NUM * 3], buffer[NUM];
  
  DevFp dev_a, dev_b, dev_c, dev_p;
  cgbn_load(bn_env, dev_a.mont, a);
  cgbn_load(bn_env, dev_b.mont, b);
  cgbn_load(bn_env, dev_p.mont, p);
  dev_c = dev_a.mul(bn_env, dev_b, res, buffer, dev_p.mont, inv);
  //uint32_t np0=cgbn_bn2mont(bn_env, dev_a.mont, dev_a.mont, dev_p.mont);
  //cgbn_bn2mont(bn_env, dev_b.mont, dev_b.mont, dev_p.mont);
  //cgbn_mont_mul(bn_env, dev_c.mont, dev_a.mont, dev_b.mont, dev_p.mont, np0);
  //cgbn_mont2bn(bn_env, dev_c.mont, dev_c.mont, dev_p.mont, np0);
  cgbn_store(bn_env, c, dev_c.mont);
}

int main(){
    const int N = 6;
    uint64_t a[N], b[N], c1[N*2], c2[N*2];
    uint64_t p[6] = {
      TO_LIMB_T(0x8508c00000000001), TO_LIMB_T(0x170b5d4430000000),
        TO_LIMB_T(0x1ef3622fba094800), TO_LIMB_T(0x1a22d9f300f5138f),
          TO_LIMB_T(0xc63b05c06ca1493b), TO_LIMB_T(0x1ae3a4617c510ea)
    };
    for(int i = 0; i < N; i++){
        a[i] = i;
        b[i] = i*10 + i;
    }
    uint64_t *dev_a, *dev_b, *dev_c1, *dev_c2, *dev_p;
    uint64_t inv = 0x8508bfffffffffff;

    cudaMalloc((void**)&dev_a, sizeof(int64_t) * N);
    cudaMalloc((void**)&dev_b, sizeof(int64_t) * N);
    cudaMalloc((void**)&dev_c1, sizeof(int64_t) * N*2);
    cudaMalloc((void**)&dev_c2, sizeof(int64_t) * N*2);
    cudaMalloc((void**)&dev_p, sizeof(int64_t) * N);

    cudaMemcpy(dev_a, a, sizeof(int64_t) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(int64_t) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_p, p, sizeof(int64_t) * N, cudaMemcpyHostToDevice);

    const int iters = 100;
    cgbn_error_report_t* report = nullptr;
    cgbn_error_report_alloc(&report); 

    for(int i = 0; i < iters; i++){
        kernel_msm_mul<<<1, 1>>>(dev_a, dev_b, dev_c1);
        kernel_cgbn_mul<<<1, TPI>>>(report, (uint32_t*)dev_a, (uint32_t*)dev_b, (uint32_t*)dev_c2);
    }
    cudaMemcpy(c1, dev_c1, 2*N * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(c2, dev_c2, 2*N * sizeof(int64_t), cudaMemcpyDeviceToHost);

    int cmp = memcmp(c1, c2, 2*N*sizeof(int64_t));
    if(cmp != 0){
        for(int i = 0; i < 2*N; i++){
            printf("(%lu,%lu) ", c1[i], c2[i]);
        }
        printf("\n");
    }else{
        printf("compare success\n");
    }

    for(int i = 0; i < iters; i++)
        kernel_msm_mont_mul<<<1, 1>>>(dev_a, dev_b, dev_c1, dev_p, inv);
    for(int i = 0; i < iters; i++)
        kernel_cgbn_mont_mul<<<1, TPI>>>(report, (uint32_t*)dev_a, (uint32_t*)dev_b, (uint32_t*)dev_c2, (uint32_t*)dev_p, inv);

    cudaMemcpy(c1, dev_c1, N * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(c2, dev_c2, N * sizeof(int64_t), cudaMemcpyDeviceToHost);

    cmp = memcmp(c1, c2, N*sizeof(int64_t));
    if(cmp != 0){
        for(int i = 0; i < N; i++){
            printf("(%lu,%lu) ", c1[i], c2[i]);
        }
        printf("\n");
    }else{
        printf("compare success\n");
    }
    return 0;
}
