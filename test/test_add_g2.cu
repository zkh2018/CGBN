#include <gmp.h>
#include "cgbn/cgbn.h"
#include "cgbn_alt_bn128_g2.h"
#include "low_func.cuh"
#include <stdint.h>
#include <vector>
#include <iostream>
using namespace std;
using namespace gpu;

#define TPI 8
#define BITS 256
typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;
const int N_32 = 8;
const int N_64 = 4;

void test_add_g2(){
    printf("start..\n");
    uint64_t x[8] = {
    17504212154399292175, 13406480092263516530, 11052506301539585977, 370724817334823578, 5533594257695532713, 1807355348778647184, 18217392116320771811, 1206937569312116885
    };
    uint64_t y[8] = {
    9346595941626535758, 6891782256290078197, 8914842573943073617, 3049015830585680570, 854176251880143851, 909606410567745456, 6999250755325914748, 1124442457008671669
    };
    uint64_t p[4] = {
    4332616871279656263, 10917124144477883021, 13281191951274694749, 3486998266802970665
    };
    uint64_t tmp_one[4] = {15230403791020821917, 754611498739239741, 7381016538464732716, 1011752739694698287};
    uint64_t tmp_a[8] = {0};
    uint64_t rp = 9786893198990664585;

    //const int n = 786432;//1024 * 1024;
    const int n = 196608;//1024 * 1024;
    //std::vector<int64_t> tmpx(n*8), tmpy(n*8);
    //for(int i = 0; i < n*8; i++){
    //    tmpx[i] = i;
    //    tmpy[i] = i*2+1;
    //}
    //printf("1\n");
    mcl_bn128_g2 h_partial;
    h_partial.init_host(n);
    FILE*fp = fopen("partial.txt", "r");
    if(fp == NULL){
        printf("open file failed..\n");
    }else{
        printf("opern file success\n");
    }
    for(int i = 0; i < n; i++){
        for(int j = 0; j < 8; j++){
            fscanf(fp, "%u ", &h_partial.x.c0.mont_repr_data[i]._limbs[j]);
        }
        for(int j = 0; j < 8; j++){
            fscanf(fp, "%u ", &h_partial.x.c1.mont_repr_data[i]._limbs[j]);
        }
        for(int j = 0; j < 8; j++){
            fscanf(fp, "%u ", &h_partial.y.c0.mont_repr_data[i]._limbs[j]);
        }
        for(int j = 0; j < 8; j++){
            fscanf(fp, "%u ", &h_partial.y.c1.mont_repr_data[i]._limbs[j]);
        }
        for(int j = 0; j < 8; j++){
            fscanf(fp, "%u ", &h_partial.z.c0.mont_repr_data[i]._limbs[j]);
        }
        for(int j = 0; j < 8; j++){
            fscanf(fp, "%u ", &h_partial.z.c1.mont_repr_data[i]._limbs[j]);
        }
    }
    printf("after read..\n");
    fclose(fp);

    cgbn_error_report_t* report = nullptr;
    cgbn_error_report_alloc(&report); 
    printf("alloc report..\n");

    mcl_bn128_g2 R, P, Q;
    R.init(n);
    P.init(n);
    Q.init(n);
    Fp_model one, lp;
    Fp_model2 a;
    one.init(1);
    lp.init(1);
    a.init(1);
    //cudaMemcpy(P.x.c0.mont_repr_data, &tmpx[0], n * 32, cudaMemcpyHostToDevice);
    //printf("2\n");
    //cudaMemcpy(P.x.c1.mont_repr_data, &tmpx[4*n], n*32, cudaMemcpyHostToDevice);
    //printf("3\n");
    //cudaMemcpy(Q.x.c0.mont_repr_data, &tmpy[0], n*32, cudaMemcpyHostToDevice);
    //printf("4\n");
    //cudaMemcpy(Q.x.c1.mont_repr_data, &tmpy[4*n], n*32, cudaMemcpyHostToDevice);
    //printf("5...\n");
    cudaMemcpy(lp.mont_repr_data, p, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(one.mont_repr_data, tmp_one, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(a.c0.mont_repr_data, tmp_a, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(a.c1.mont_repr_data, tmp_a+4, 32, cudaMemcpyHostToDevice);
    P.copy_from_cpu(h_partial);
    const int threads = 32 * TPI;
    const int half_n = n / 4;
    const int blocks = (half_n/2+31) / 32;
    for(int i = 0;i < 4; i++){
        kernel_ect_add_g2<32><<<blocks, threads>>>(report, R, P, Q, one, lp, a, 0, 0, rp, i * half_n, half_n);
    }
    cudaDeviceSynchronize();
    //mcl_bn128_g2 result;
    //result.init_host(n);
    //R.copy_to_cpu(result);
}
int main(){
    test_add_g2();
    return 0;
}
