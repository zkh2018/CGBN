#include <gmp.h>
#include "cgbn/cgbn.h"
#include <stdint.h>

#define TPI 8
#define BITS 256
typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

__global__ void kernel_sub_wide(
    cgbn_error_report_t* report, 
    uint32_t* z, uint32_t* x, uint32_t* y){
  context_t bn_context(cgbn_report_monitor, report, 0);
  env_t          bn_env(bn_context.env<env_t>());  
  env_t::cgbn_wide_t lz, lx, ly;
  cgbn_load(bn_env, lx._low, x);
  cgbn_load(bn_env, lx._high, x + 8);
  cgbn_load(bn_env, ly._low, y);
  cgbn_load(bn_env, ly._high, y + 8);
  if(cgbn_compare(bn_env, lx._low, ly._low) >= 0){
    if(threadIdx.x == 0)
    printf("first...\n");
    cgbn_sub(bn_env, lz._low, lx._low, ly._low);
    cgbn_sub(bn_env, lz._high, lx._high, ly._high);
  }else{
    if(threadIdx.x == 0)
    printf("second...\n");
    env_t::cgbn_t max;
    cgbn_set(bn_env, max, 0xffffffff);
    cgbn_sub(bn_env, lz._low, max, ly._low);
    cgbn_add(bn_env, lz._low, lz._low, lx._low);
    cgbn_add_ui32(bn_env, lz._low, lz._low, 1);
    cgbn_sub_ui32(bn_env, lz._high, lx._high, 1);
    cgbn_sub(bn_env, lz._high, lz._high, ly._high);
  }
  cgbn_store(bn_env, z, lz._low);
  cgbn_store(bn_env, z + 8, lz._high);
}

void cpu_sub_wide(uint64_t *z, uint64_t *x, uint64_t *y){
    mpn_sub_n(z, x, y, 8);
}


void test_sub_wide(){
    uint64_t z1[8], z2[8];
    //uint64_t x[8] = {16619935976129290828, 9458866309631348105, 13690923725331915767, 8204588786242678379, 11956923625381298830, 13012433824332031229, 12731089116711111744, 503377215233786018};
    uint64_t x[8] = {15706470583754492422, 4586667022969251858, 6349980725307984340, 3318513001389615183, 8606768448218914400, 4222581582277747575, 3422314914783991510, 596779857884086507};
    //uint64_t y[8] = {9002772277281726706, 9674413318396093378, 5997904044282333898, 2575931123308814000, 11251353354874625306, 15107836024188185635, 18150878448637691612, 303291201399149411};
    uint64_t y[8] = {13865163003810320274, 14636758252655260683, 18388200920246596554, 13373283754468364518, 9673096196086582325, 5224440756211953446, 11325029591588937384, 61276170598357236};
    cpu_sub_wide(z1, x, y);

    uint32_t* d_z, *d_x, *d_y;
    cudaMalloc((void**)&d_z, 64);
    cudaMalloc((void**)&d_x, 64);
    cudaMalloc((void**)&d_y, 64);
    cudaMemcpy(d_x, x, 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, 64, cudaMemcpyHostToDevice);
    cgbn_error_report_t* report = nullptr;
    cgbn_error_report_alloc(&report); 
    kernel_sub_wide<<<1, 8>>>(report, d_z, d_x, d_y);
    cudaMemcpy(z2, d_z, 64, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 8; i++){
        printf("%lu ", z1[i]);
    }
    printf("\n");
    for(int i = 0; i < 8; i++){
        printf("%lu ", z2[i]);
    }
    printf("\n");
}


struct Fp2Dbl{
    env_t::cgbn_wide_t a,b;
};
struct DevFp{
    env_t::cgbn_t mont;
};
struct MclFp2{
    DevFp c0, c1;
};

inline __device__ uint32_t dev_sub_wide(env_t& bn_env, env_t::cgbn_wide_t& lz, env_t::cgbn_wide_t& lx, env_t::cgbn_wide_t& ly){
  if(cgbn_compare(bn_env, lx._low, ly._low) >= 0){
    if(threadIdx.x == 0){
        printf("first..\n");
    }
    cgbn_sub(bn_env, lz._low, lx._low, ly._low);
    return cgbn_sub(bn_env, lz._high, lx._high, ly._high);
  }else{
    if(threadIdx.x == 0){
        printf("second..\n");
    }
    env_t::cgbn_t max;
    env_t::cgbn_wide_t tmpz;
    cgbn_set(bn_env, max, 0xffffffff);
    cgbn_sub(bn_env, tmpz._low, max, ly._low);
    cgbn_add(bn_env, tmpz._low, tmpz._low, lx._low);
    cgbn_add_ui32(bn_env, lz._low, tmpz._low, 1);
    cgbn_sub_ui32(bn_env, tmpz._high, lx._high, 1);
    uint32_t ret = cgbn_sub(bn_env, lz._high, tmpz._high, ly._high);
    return ret;
  }
}

inline __device__ void dev_fp2Dbl_mulPreW(
    env_t& bn_env,
    Fp2Dbl& z,
    MclFp2& x, MclFp2& y,
    env_t::cgbn_t& lp){

    // const Fp& a = x.a;
    // const Fp& b = x.b;
    // const Fp& c = y.a;
    // const Fp& d = y.b;
    //FpDbl& d1 = z.b;
    DevFp s, t;
    // Fp::addPre(s, a, b);
    cgbn_add(bn_env, s.mont, x.c0.mont, x.c1.mont);  
    //Fp::addPre(t, c, d);
    cgbn_add(bn_env, t.mont, y.c0.mont, y.c1.mont);  
    //FpDbl::mulPre(d1, s, t);
    cgbn_mul_wide(bn_env, z.b, s.mont, t.mont); 

    __shared__ uint32_t tmp[64];
    cgbn_store(bn_env, tmp, z.b._low);
    cgbn_store(bn_env, tmp + 8, z.b._high);
    if(threadIdx.x == 0){
        printf("gpu:\n");
        printf("d1:");
        for(int i = 0; i < 8; i++){
            printf("%lu ", ((uint64_t*)tmp)[i]);
        }
        printf("\n");
    }

    //FpDbl::mulPre(d0, a, c);
    cgbn_mul_wide(bn_env, z.a, x.c0.mont, y.c0.mont); 
    //FpDbl::mulPre(d2, b, d);
    env_t::cgbn_wide_t d2;
    cgbn_mul_wide(bn_env, d2, x.c1.mont, y.c1.mont); 

    cgbn_store(bn_env, tmp, z.b._low);
    cgbn_store(bn_env, tmp + 8, z.b._high);
    if(threadIdx.x == 0){
        printf("d1:");
        for(int i = 0; i < 8; i++){
            printf("%lu ", ((uint64_t*)tmp)[i]);
        }
        printf("\n");
    }
    cgbn_store(bn_env, tmp, z.a._low);
    cgbn_store(bn_env, tmp + 8, z.a._high);
    if(threadIdx.x == 0){
        printf("d0:");
        for(int i = 0; i < 8; i++){
            printf("%lu ", ((uint64_t*)tmp)[i]);
        }
        printf("\n");
    }
    cgbn_store(bn_env, tmp, d2._low);
    cgbn_store(bn_env, tmp + 8, d2._high);
    if(threadIdx.x == 0){
        printf("d2:");
        for(int i = 0; i < 8; i++){
            printf("%lu ", ((uint64_t*)tmp)[i]);
        }
        printf("\n");
    }
    
    
    // FpDbl::subPre(d1, d1, d0);
    dev_sub_wide(bn_env, z.b, z.b, z.a);

    cgbn_store(bn_env, tmp, z.b._low);
    cgbn_store(bn_env, tmp + 8, z.b._high);
    if(threadIdx.x == 0){
        printf("d1:");
        for(int i = 0; i < 8; i++){
            printf("%lu ", ((uint64_t*)tmp)[i]);
        }
        printf("\n");
    }
    //FpDbl::subPre(d1, d1, d2);
    dev_sub_wide(bn_env, z.b, z.b, d2);

    cgbn_store(bn_env, tmp, z.b._low);
    cgbn_store(bn_env, tmp + 8, z.b._high);
    if(threadIdx.x == 0){
        printf("d1:");
        for(int i = 0; i < 8; i++){
            printf("%lu ", ((uint64_t*)tmp)[i]);
        }
        printf("\n");
    }
    //FpDbl::sub(d0, d0, d2);
    if(dev_sub_wide(bn_env, z.a, z.a, d2)){
        cgbn_add(bn_env, z.a._high, z.a._high, lp);
    }
}

__global__ void kernel_fp2Dbl_mulPreW(
    cgbn_error_report_t* report, 
    uint32_t* z, uint32_t* x, uint32_t* y,
    uint32_t *p
){
  context_t bn_context(cgbn_report_monitor, report, 0);
  env_t          bn_env(bn_context.env<env_t>());  
  Fp2Dbl lz;
  MclFp2 lx, ly;
  cgbn_load(bn_env, lx.c0.mont, x);
  cgbn_load(bn_env, lx.c1.mont, x + 8);
  cgbn_load(bn_env, ly.c0.mont, y);
  cgbn_load(bn_env, ly.c1.mont, y + 8);
  env_t::cgbn_t lp;
  cgbn_load(bn_env, lp, p);
  dev_fp2Dbl_mulPreW(bn_env, lz, lx, ly, lp);
  cgbn_store(bn_env, z, lz.a._low);
  cgbn_store(bn_env, z + 8, lz.a._high);
  cgbn_store(bn_env, z + 16, lz.b._low);
  cgbn_store(bn_env, z + 24, lz.b._high);
}

void test_mulPreW(){
    uint64_t x[8] = {17504212154399292175, 13406480092263516530, 11052506301539585977, 370724817334823578, 2738505452492009767, 11348385648817470847, 5987438616802946354, 2267050511450853472};
    uint64_t y[8] = {9346595941626535758, 6891782256290078197, 8914842573943073617, 3049015830585680570, 854176251880143851, 909606410567745456, 6999250755325914748, 1124442457008671669};
    uint64_t p[4] = {4332616871279656263, 10917124144477883021, 13281191951274694749, 3486998266802970665};
    uint64_t z[16];
    uint32_t *d_x, *d_y, *d_z, *d_p;
    cudaMalloc((void**)&d_x, 64);
    cudaMalloc((void**)&d_y, 64);
    cudaMalloc((void**)&d_z, 128);
    cudaMalloc((void**)&d_p, 32);
    cudaMemcpy(d_x, x, 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, p, 32, cudaMemcpyHostToDevice);
    cgbn_error_report_t* report = nullptr;
    cgbn_error_report_alloc(&report); 
    kernel_fp2Dbl_mulPreW<<<1, 8>>>(report, d_z, d_x, d_y, d_p);
    cudaMemcpy(z, d_z, 128, cudaMemcpyDeviceToHost);
    for(int i = 0; i < 16; i++){
        printf("%lu ", z[i]);
    }
    printf("\n");
}

int main(){
    //test_sub_wide();
    test_mulPreW();
    return 0;
}
