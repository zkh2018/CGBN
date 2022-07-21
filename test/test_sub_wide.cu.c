#include <gmp.h>
#include "cgbn/cgbn.h"
#include <stdint.h>
#include <vector>
#include <iostream>
using namespace std;

#define TPI 8
#define BITS 256
typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;
const int N_32 = 8;
const int N_64 = 4;

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
    cgbn_sub(bn_env, lz._low, lx._low, ly._low);
    cgbn_sub(bn_env, lz._high, lx._high, ly._high);
  }else{
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
    cgbn_sub(bn_env, lz._low, lx._low, ly._low);
    return cgbn_sub(bn_env, lz._high, lx._high, ly._high);
  }else{
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
    //FpDbl::mulPre(d0, a, c);
    cgbn_mul_wide(bn_env, z.a, x.c0.mont, y.c0.mont); 
    //FpDbl::mulPre(d2, b, d);
    env_t::cgbn_wide_t d2;
    cgbn_mul_wide(bn_env, d2, x.c1.mont, y.c1.mont); 

    // FpDbl::subPre(d1, d1, d0);
    dev_sub_wide(bn_env, z.b, z.b, z.a);

    //FpDbl::subPre(d1, d1, d2);
    dev_sub_wide(bn_env, z.b, z.b, d2);

    //FpDbl::sub(d0, d0, d2);
    if(dev_sub_wide(bn_env, z.a, z.a, d2)){
        cgbn_add(bn_env, z.a._high, z.a._high, lp);
    }
}

__global__ void kernel_fp2Dbl_mulPreW(
    cgbn_error_report_t* report, 
    uint32_t* z, uint32_t* x, uint32_t* y,
    uint32_t *p,
    const int n
){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int instance = tid / 8;
    if(instance >= n) return;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  Fp2Dbl lz;
  MclFp2 lx, ly;
  cgbn_load(bn_env, lx.c0.mont, x + instance * 16);
  cgbn_load(bn_env, lx.c1.mont, x + instance * 16 + 8);
  cgbn_load(bn_env, ly.c0.mont, y + instance * 16);
  cgbn_load(bn_env, ly.c1.mont, y + instance * 16 + 8);
  env_t::cgbn_t lp;
  cgbn_load(bn_env, lp, p);
  dev_fp2Dbl_mulPreW(bn_env, lz, lx, ly, lp);
  cgbn_store(bn_env, z + instance * 32, lz.a._low);
  cgbn_store(bn_env, z + instance * 32 + 8, lz.a._high);
  cgbn_store(bn_env, z + instance * 32 + 16, lz.b._low);
  cgbn_store(bn_env, z + instance * 32 + 24, lz.b._high);
}

void test_mulPreW(){
    uint64_t x[8] = {17504212154399292175, 13406480092263516530, 11052506301539585977, 370724817334823578, 2738505452492009767, 11348385648817470847, 5987438616802946354, 2267050511450853472};
    uint64_t y[8] = {9346595941626535758, 6891782256290078197, 8914842573943073617, 3049015830585680570, 854176251880143851, 909606410567745456, 6999250755325914748, 1124442457008671669};
    uint64_t p[4] = {4332616871279656263, 10917124144477883021, 13281191951274694749, 3486998266802970665};
    const int n = 16000;
    std::vector<uint64_t> all_x(n * 8), all_y(n * 8), all_z(n * 16);
    for(int i = 0; i < n; i++){
        memcpy(&all_x[i * 8], x, 8 * sizeof(int64_t)); 
        memcpy(&all_y[i * 8], y, 8 * sizeof(int64_t)); 
    }

    {
        
        clock_t start = clock();
        for(int i = 0; i < n; i++){
            uint64_t d0[8], d1[8], d2[8];
            uint64_t s[4], t[4];
            uint64_t *x = &all_x[i * 8];
            uint64_t *y = &all_y[i * 8];
            mpn_add_n(s, x, &x[4], 4); 
            mpn_add_n(t, y, &y[4], 4); 
            mpn_mul_n(d1, s, t, 4);
            mpn_mul_n(d0, x, y, 4);
            mpn_mul_n(d2, &x[4], &y[4], 4); 
            mpn_sub_n(d1, d1, d0, 8);
            mpn_sub_n(d1, d1, d2, 8);
            if(mpn_sub_n(d0, d0, d2, 8)){
                mpn_add_n(&d0[4], &d0[4], p, 4);
            }
            memcpy(&all_z[i * 16], d0, 8 * sizeof(int64_t));
            memcpy(&all_z[i * 16 + 8], d1, 8 * sizeof(int64_t));
        }
        clock_t end = clock();
        printf("cpu time: %fus\n", 1000*1000*(double)(end-start)/CLOCKS_PER_SEC);

        ///for(int i = 0; i < 8; i++){
        ///    printf("%lu ", d0[i]);
        ///}
        ///for(int i = 0; i < 8; i++){
        ///    printf("%lu ", d1[i]);
        ///}
        ///printf("\n");
    }
    ///uint64_t z[16];
    std::vector<uint64_t> z(n * 16);
    uint32_t *d_x, *d_y, *d_z, *d_p;
    cudaMalloc((void**)&d_x, n * 64);
    cudaMalloc((void**)&d_y, n * 64);
    cudaMalloc((void**)&d_z, n * 128);
    cudaMalloc((void**)&d_p, 32);
    cudaMemcpy(d_x, all_x.data(), n * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, all_y.data(), n * 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, p, 32, cudaMemcpyHostToDevice);
    cgbn_error_report_t* report = nullptr;
    cgbn_error_report_alloc(&report); 
    int local_instances = 64;
    int threads = local_instances * 8;
    int blocks = (n + threads - 1) / threads;
    printf("threads = %d, blocks=%d\n", threads, blocks);
    kernel_fp2Dbl_mulPreW<<<blocks, threads>>>(report, d_z, d_x, d_y, d_p, n);
    cudaMemcpy(z.data(), d_z, n * 128, cudaMemcpyDeviceToHost);
    //for(int i = 0; i < 16; i++){
    //    printf("%lu ", z[i]);
    //}
    //printf("\n");
    for(int i = 0; i < n; i++){
        int cmp = memcmp(&z[i * 16], &all_z[i * 16], 16 * sizeof(uint64_t)); 
        if(cmp != 0){
            printf("first diff = %d \n", i);
            break;
        }
    }
    int cmp = memcmp(z.data(), all_z.data(), n * 128);
    printf("cmp = %d\n", cmp);
}

inline __device__ void dev_mont_red(env_t& bn_env, 
        env_t::cgbn_t& lz, env_t::cgbn_wide_t& lxy, env_t::cgbn_t& lp, 
        uint32_t* p, uint32_t *buf, uint32_t *pq, const uint64_t rp){
        cgbn_store(bn_env, buf, lxy._low);
        cgbn_store(bn_env, buf + 8, lxy._high);
        const int group_id = threadIdx.x & (TPI-1);
        uint64_t* p64_pq = (uint64_t*)pq;
        uint64_t *p64_buf = (uint64_t*)buf;
        if(group_id == 0){
            p64_buf[N_64*2] = 0;
        }
        uint32_t th[2];
        env_t::cgbn_wide_t lwt;
        cgbn_get_ui64(bn_env, lxy._low, th, 0);
        //q = xy[0] * rp;
        uint64_t q = *(uint64_t*)th * rp;
        //pq = p * q
        cgbn_mul_ui64(bn_env, lwt, lp, q);
        cgbn_store(bn_env, pq, lwt._low);
        cgbn_get_ui64(bn_env, lwt._high, th, 0);
        if(group_id == 0){
            p64_pq[N_64] = *(uint64_t*)th;
        }
        env_t::cgbn_t lc;
        //buf = pq + xy
        uint32_t carry = cgbn_add(bn_env, lc, lxy._low, lwt._low);
        cgbn_store(bn_env, buf, lc);
        cgbn_get_ui64(bn_env, lxy._high, th, 0);

        uint64_t buf4 = carry + *(uint64_t*)th + p64_pq[N_64];
        if(group_id == 0){
            p64_buf[N_64] = buf4;
        }

        if(buf4 < *(uint64_t*)th || buf4 < carry || buf4 < p64_pq[N_64]){
            env_t::cgbn_t la;
            cgbn_load(bn_env, la, buf + N_32 + 2);     
            cgbn_add_ui32(bn_env, la, la, 1); 
            cgbn_store(bn_env, buf + N_32 + 2, la);
        }

        //if(group_id == 0){
        //    printf("buf4 = %lu, %lu\n", p64_buf[N_64], p64_buf[N_64*2]);
        //}
        uint64_t *c = p64_buf + 1;
        for(int i = 1; i < 4; i++){
            q = c[0] * rp;
            //if(group_id == 0){
            //    for(int j = 0; j < 4; j++)
            //        printf("%lu ", c[j]);
            //    printf("%lu \n", p64_buf[N_64*2]);
            //}
            //pq = p*q
            cgbn_mul_ui64(bn_env, lwt, lp, q);
            cgbn_store(bn_env, pq, lwt._low);
            cgbn_get_ui64(bn_env, lwt._high, th, 0);
            if(group_id == 0){
                p64_pq[N_64] = *(uint64_t*)th;
            }
            cgbn_load(bn_env, lc, (uint32_t*)c);
            //c = c + pq
            carry = cgbn_add(bn_env, lc, lc, lwt._low);
            cgbn_store(bn_env, (uint32_t*)c, lc);

            //c[N] += pq[N]
            buf4 = carry + c[N_64] + *(uint64_t*)th;
            if(group_id == 0){
                //printf("%d, %lu, %lu\n", i, buf4, p64_pq[N_64]);
                c[N_64] = buf4;
            }
            if(buf4 < *(uint64_t*)th || buf4 < carry || buf4 < p64_pq[N_64]){
                env_t::cgbn_t la;
                cgbn_load(bn_env, la, (uint32_t*)(c + N_64 + 1));     
                cgbn_add_ui32(bn_env, la, la, 1); 
                //if(group_id == 0){
                //    printf("%lu %lu\n", c[N_64], c[N_64+1]);
                //}
                if(group_id < N_32 - i * 2)
                    cgbn_store(bn_env, (uint32_t*)(c + N_64 + 1), la);
                //if(group_id == 0){
                //    printf("%lu %lu\n", c[N_64], c[N_64+1]);
                //}
            }
            c++;
        }

        //if(group_id == 0){
        //    for(int i = 0; i<5; i++){
        //        printf("%lu ", c[i]);
        //    }
        //    printf("\n");
        //}
        cgbn_load(bn_env, lc, (uint32_t*)c);
        carry = cgbn_sub(bn_env, lz, lc, lp);
        if(c[N_64] == 0 && carry){
            cgbn_set(bn_env, lz, lc);
        }
}
template<int Instances>
__global__ void kernel_mont_red(
    cgbn_error_report_t* report, 
    uint32_t* z, uint32_t*xy, uint32_t* p, const uint64_t rp, const int n){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int instance = tid / 8;
    if(instance >= n) return;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  __shared__ uint32_t cache_buf[Instances * (N_32*2+2)], cache_pq[Instances * (N_32 + 2)];
  env_t::cgbn_t lz, lp;
  env_t::cgbn_wide_t lxy;
  cgbn_load(bn_env, lp, p);
  cgbn_load(bn_env, lxy._low, xy + instance * 16);
  cgbn_load(bn_env, lxy._high, xy + instance * 16 + 8);
  dev_mont_red(bn_env, lz, lxy, lp, p, cache_buf, cache_pq, rp);
  cgbn_store(bn_env, z + instance * 8, lz);
}

void test_mont_red(){
    ///uint64_t xy[8] = {350024899820998845, 10701496500823572861, 10487677729325716844, 10683094384378792728, 17973394756262313675, 15075780239999670660, 15781612571399597230, 47843774714984954};
    //uint64_t xy[8] = {6381586897798729768, 10016599911609461703, 42610866849821252, 16332603064443471215, 9551413591996050823, 11594754781359390022, 8169577449298419839, 3190887580392153910};
    //uint64_t xy[8] = {12901158869322336760, 1760982898811485273, 3821559897752039931, 5484803367439840043, 2832195709154589428, 3586402354694468814, 10483059522430710091, 443597267001008308};
    //uint64_t xy[8] = {14552259481042486510, 3231802661448234502, 1207150056786510271, 12910906534128644860, 4150184641963046917, 991433677009212840, 6910353580225735697, 3101614545678270856};
    uint64_t xy[8] = {10429895110800583235, 5272122913542525819, 12740881035615201776, 14771886802373817387, 6537186679166961715, 10405311241031360291, 1234404491868396534, 222089625327330384};
    uint64_t p[4] = {4332616871279656263, 10917124144477883021, 13281191951274694749, 3486998266802970665};
    uint64_t rp = 9786893198990664585;
    const int n =1600;
    std::vector<uint64_t> all_xy(n*8);
    for(int i = 0; i < n; i++){
        //memcpy(&all_xy[i*8], xy, 8 * sizeof(uint64_t));
        for(int j = 0; j < 8; j++){
            all_xy[i * 8 + j] = i * 8 + j;
        }
    }

    uint64_t correct_z[4];
    {
        clock_t start = clock();
        for(int it = 0; it < n; it++){
            //host
            const int N = 4;
            uint64_t pq[N+1];//N+1
            uint64_t buf[N*2+1];//N*2+1
            //copyC<N - 1>(buf + N + 1, xy + N + 1);
            //memcpy(buf, xy, 64);
            memcpy(buf + N + 1, xy + N + 1, (N-1) * sizeof(int64_t));
            buf[N * 2] = 0;
            uint64_t q = xy[0] * rp;
            pq[N] = mpn_mul_1(pq, p, N, q);
            uint64_t carry = mpn_add_n(buf, xy, pq, N);
            uint64_t up = 0;
            buf[N] = xy[N] + pq[N];
            if(buf[N] < xy[N] && buf[N] < pq[N]){
                up = 1;
                buf[N * 2] = mpn_add_1(buf + N + 1, buf+N+1, N - 1, 1);
            }
            buf[N] += carry;
            //printf("buf[4]=%lu\n", buf[N]);
            uint64_t *c = buf + 1;
            for (size_t i = 1; i < N; i++) {
                q = c[0] * rp;
                //for(int j = 0; j < 4; j++)
                //    printf("%lu ", c[j]);
                //printf("\n");
                pq[N] = mpn_mul_1(pq, p, N, q);
                //uint64_t up = AddPre<N + 1, Tag>::f(c, c, pq);
                uint64_t up = mpn_add_n(c, c, pq, N+1);
                if(up){
                    //AddUnitPre<Tag>::f(c + N + 1, N - i, 1);
                    mpn_add_1(c+N+1, c+N+1, N-i, 1);
                }
                //carry = mpn_add_n(c, c, pq, N);
                //up = 0;
                //c[N] += pq[N];
                //printf("%d, %lu %lu\n", (int)i, c[N], pq[N]);
                //if(c[N] < pq[N]){
                //    mpn_add_1(c+N+1, c+N+1, N-i, 1);
                //}
                //c[N] += carry;
                c++;
            }
            //uint ret = SubPre<N, Tag>::f(z, c, p);
            //for(int i = 0; i<5; i++){
            //    printf("%lu ", c[i]);
            //}
            //printf("\n");
            //uint64_t ret = mpn_sub_n(correct_z, c, p, N);
            //if(c[N] <= 0){
            //	memcpy(correct_z, c, N * sizeof(uint64_t));
            //}
            if(c[N]){
                mpn_sub_n(correct_z, c, p, N);
            }else{
                if(mpn_sub_n(correct_z, c, p, N)){
                    memcpy(correct_z, c, N * sizeof(uint64_t));
                }
            }
        }
        //printf("end host\n");
        clock_t end = clock();
        printf("cpu time=%fus\n", 1000*1000*(double)(end-start)/CLOCKS_PER_SEC);
    }

    uint32_t *d_xy, *d_z, *d_p;
    cudaMalloc((void**)&d_xy, n*64);
    cudaMalloc((void**)&d_p, 32);
    cudaMalloc((void**)&d_z, n*32);
    cudaMemcpy(d_xy, all_xy.data(), n*64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, p, 32, cudaMemcpyHostToDevice);
    cgbn_error_report_t* report = nullptr;
    cgbn_error_report_alloc(&report); 
    const int local_instances = 64;
    int threads = local_instances * 8;
    int blocks = (n + local_instances - 1) / local_instances;
    //for(int it = 0; it < n; it++){
        kernel_mont_red<local_instances><<<blocks, threads>>>(report, d_z, d_xy, d_p, rp, n);
    //}
    //uint64_t z[4];
    std::vector<uint64_t> z(n * 4);
    cudaMemcpy(z.data(), d_z, n*32, cudaMemcpyDeviceToHost);
    //printf("\n");
    //for(int i = 0; i < 4; i++){
    //    printf("%lu ", correct_z[i]);
    //}
    //printf("\n");
    //for(int i = 0; i < 4; i++){
    //    printf("%lu ", z[i]);
    //}
    //printf("\n");
}

inline __device__ void dev_mcl_mul_g2(env_t& bn_env, MclFp2& z, MclFp2& x, MclFp2& y, env_t::cgbn_t& lp, uint32_t* p, uint32_t* buf, uint32_t *pq , const uint64_t rp){
    Fp2Dbl d;
    dev_fp2Dbl_mulPreW(bn_env, d, x, y, lp);
    __shared__ uint32_t cache[32];
    cgbn_store(bn_env, cache, d.a._low);
    cgbn_store(bn_env, cache + 8, d.a._high);
    if(threadIdx.x == 0){
        printf("xy:");
        for(int i = 0; i < 8; i++){
            printf("%lu, ", ((uint64_t*)cache)[i]);
        }
        printf("\n");
    }
    dev_mont_red(bn_env, z.c0.mont, d.a, lp, p, buf, pq, rp);

    cgbn_store(bn_env, cache, d.b._low);
    cgbn_store(bn_env, cache + 8, d.b._high);
    if(threadIdx.x == 0){
        printf("xy:");
        for(int i = 0; i < 8; i++){
            printf("%lu, ", ((uint64_t*)cache)[i]);
        }
        printf("\n");
    }
    dev_mont_red(bn_env, z.c1.mont, d.b, lp, p, buf, pq, rp);
}

__global__ void kernel_mcl_mul_g2(
    cgbn_error_report_t* report, 
    uint32_t* z,
    uint32_t *x, uint32_t*y,
    uint32_t *p, uint64_t rp){
    context_t bn_context(cgbn_report_monitor, report, 0);
    env_t          bn_env(bn_context.env<env_t>());  
    MclFp2 lz, ly,lx;
    env_t::cgbn_t lp;
    cgbn_load(bn_env, lx.c0.mont, x);
    cgbn_load(bn_env, lx.c1.mont, x + 8);
    cgbn_load(bn_env, ly.c0.mont, y);
    cgbn_load(bn_env, ly.c1.mont, y + 8);
    cgbn_load(bn_env, lp, p);
    __shared__ uint32_t cache_buf[N_32*2+2], cache_pq[N_32 + 2];
    dev_mcl_mul_g2(bn_env, lz, lx, ly, lp, p, cache_buf, cache_pq, rp);
    cgbn_store(bn_env, z, lz.c0.mont);
    cgbn_store(bn_env, z + 8, lz.c1.mont);
}

void test_mcl_mul_g2(){
    uint64_t x[8] = {
    17504212154399292175, 13406480092263516530, 11052506301539585977, 370724817334823578, 5533594257695532713, 1807355348778647184, 18217392116320771811, 1206937569312116885
    };
    uint64_t y[8] = {
    9346595941626535758, 6891782256290078197, 8914842573943073617, 3049015830585680570, 854176251880143851, 909606410567745456, 6999250755325914748, 1124442457008671669
    };
    uint64_t p[4] = {
    4332616871279656263, 10917124144477883021, 13281191951274694749, 3486998266802970665
    };
    uint64_t rp = 9786893198990664585;

    uint32_t *d_x, *d_y, *d_z, *d_p;
    cudaMalloc((void**)&d_x, 64);
    cudaMalloc((void**)&d_y, 64);
    cudaMalloc((void**)&d_p, 32);
    cudaMalloc((void**)&d_z, 64);
    cudaMemcpy(d_x, x, 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, 64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, p, 32, cudaMemcpyHostToDevice);
    cgbn_error_report_t* report = nullptr;
    cgbn_error_report_alloc(&report); 
    kernel_mcl_mul_g2<<<1, 8>>>(report, d_z, d_x, d_y, d_p, rp);
    uint64_t z[8];
    cudaMemcpy(z, d_z, 64, cudaMemcpyDeviceToHost);
    for(int i = 0; i<8; i++){
        printf("%lu ", z[i]);
    }
    printf("\n");
}

int main(){
    //test_sub_wide();
    //test_mulPreW();
    test_mont_red();
    //test_mcl_mul_g2();
    return 0;
}
