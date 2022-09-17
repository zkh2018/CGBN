#include <iostream>
#include <vector>


#include "low_func_gpu.h"

using namespace gpu;
/*
a
9855157384507968064 12866218247869696363 15583024243348139216 2743046514653319853
10563783233724534797 4967502387935820643 6067998445747067568 3441554560241989970
15230403791020821917 754611498739239741 7381016538464732716 1011752739694698287

b
4252204126867602537 9918196088281580021 5151322999762652470 2368356264329432293
14429909543139381222 6582645255282250537 3381912363200108806 2176551161836059481
15230403791020821917 754611498739239741 7381016538464732716 1011752739694698287

one
15230403791020821917 754611498739239741 7381016538464732716 1011752739694698287
d_p
4332616871279656263 10917124144477883021 13281191951274694749 3486998266802970665
d_a
0 0 0 0
specialA_=0
mode_=0
rp=-460954743
17694023113564280320, 6441032102864233943, 3232518032662114548, 2157088571651720264,
6950605467205326756, 5725299854034035246, 258437952663700456, 2168515629986009173,
9399645581057894607, 15170430263790897170, 12319587621102534769, 3131008411267277742,
*/

uint64_t ax[4] = {9855157384507968064, 12866218247869696363, 15583024243348139216, 2743046514653319853};
uint64_t ay[4] = {10563783233724534797, 4967502387935820643, 6067998445747067568, 3441554560241989970};
uint64_t az[4] = {15230403791020821917, 754611498739239741, 7381016538464732716, 1011752739694698287};

uint64_t bx[4] = {4252204126867602537, 9918196088281580021, 5151322999762652470, 2368356264329432293};
uint64_t by[4] = {14429909543139381222, 6582645255282250537, 3381912363200108806, 2176551161836059481};
uint64_t bz[4] = {15230403791020821917, 754611498739239741, 7381016538464732716, 1011752739694698287};

uint64_t cx[4] = {17694023113564280320, 6441032102864233943, 3232518032662114548, 2157088571651720264};
uint64_t cy[4] = {6950605467205326756, 5725299854034035246, 258437952663700456, 2168515629986009173};
uint64_t cz[4] = {9399645581057894607, 15170430263790897170, 12319587621102534769, 3131008411267277742};

uint64_t one[4] = {15230403791020821917, 754611498739239741, 7381016538464732716, 1011752739694698287}; 
uint64_t p[4] = {4332616871279656263, 10917124144477883021, 13281191951274694749, 3486998266802970665};
uint64_t a_[4] = {0, 0, 0, 0};

int specialA_ = 0;
int model_ = 0;
uint64_t rp =  9786893198990664585;

int main(){
    mcl_bn128_g1 dR, dP, dQ;
    Fp_model done, dp, da;
    //dR.init(1);
    //dP.init(1);
    //dQ.init(1);
    //done.init(1);
    //dp.init(1);
    //da.init(1);
    cudaMalloc((void**)&dR.x.mont_repr_data, 32);
    cudaMalloc((void**)&dR.y.mont_repr_data, 32);
    cudaMalloc((void**)&dR.z.mont_repr_data, 32);

    cudaMalloc((void**)&dP.x.mont_repr_data, 32);
    cudaMalloc((void**)&dP.y.mont_repr_data, 32);
    cudaMalloc((void**)&dP.z.mont_repr_data, 32);

    cudaMalloc((void**)&dQ.x.mont_repr_data, 32);
    cudaMalloc((void**)&dQ.y.mont_repr_data, 32);
    cudaMalloc((void**)&dQ.z.mont_repr_data, 32);

    cudaMalloc((void**)&done.mont_repr_data, 32);
    cudaMalloc((void**)&dp.mont_repr_data, 32);
    cudaMalloc((void**)&da.mont_repr_data, 32);
    //FILE *fpa = fopen("a.txt", "r");
    //FILE *fpb = fopen("b.txt", "r");
    //FILE *fpc = fopen("c.txt", "r");
    //auto f = [](FILE*fp, uint64_t *data){
    //    for(int j = 0; j < 4; j++){
    //        fscanf(fp, "%lu ", &data[j]);
    //    }
    //};
    for(int i = 0; i < 1; i++){
        //f(fpa, ax);
        //f(fpa, ay);
        //f(fpa, az);

        //f(fpb, bx);
        //f(fpb, by);
        //f(fpb, bz);

        //f(fpc, cx);
        //f(fpc, cy);
        //f(fpc, cz);

        //copy_cpu_to_gpu(dP.x.mont_repr_data, ax, 32);
        //copy_cpu_to_gpu(dP.y.mont_repr_data, ay, 32);
        //copy_cpu_to_gpu(dP.z.mont_repr_data, az, 32);

        //copy_cpu_to_gpu(dQ.x.mont_repr_data, bx, 32);
        //copy_cpu_to_gpu(dQ.y.mont_repr_data, by, 32);
        //copy_cpu_to_gpu(dQ.z.mont_repr_data, bz, 32);

        //copy_cpu_to_gpu(done.mont_repr_data, one, 32);
        //copy_cpu_to_gpu(dp.mont_repr_data, p, 32);
        //copy_cpu_to_gpu(da.mont_repr_data, a_, 32);
        cudaMemcpy(dP.x.mont_repr_data, ax, 32, cudaMemcpyHostToDevice);
        cudaMemcpy(dP.y.mont_repr_data, ay, 32, cudaMemcpyHostToDevice);
        cudaMemcpy(dP.z.mont_repr_data, az, 32, cudaMemcpyHostToDevice);

        cudaMemcpy(dQ.x.mont_repr_data, ax, 32, cudaMemcpyHostToDevice);
        cudaMemcpy(dQ.y.mont_repr_data, ay, 32, cudaMemcpyHostToDevice);
        cudaMemcpy(dQ.z.mont_repr_data, az, 32, cudaMemcpyHostToDevice);

        cudaMemcpy(done.mont_repr_data, one, 32, cudaMemcpyHostToDevice);
        cudaMemcpy(dp.mont_repr_data, p, 32, cudaMemcpyHostToDevice);
        cudaMemcpy(da.mont_repr_data, a_, 32, cudaMemcpyHostToDevice);

        gpu_mcl_ect_add(dR, dP, dQ, done, dp, da, specialA_, model_, rp);  

        uint64_t rx[4], ry[4], rz[4];
        //copy_gpu_to_cpu(rx, dR.x.mont_repr_data, 32);
        //copy_gpu_to_cpu(ry, dR.y.mont_repr_data, 32);
        //copy_gpu_to_cpu(rz, dR.z.mont_repr_data, 32);
        cudaMemcpy(rx, dR.x.mont_repr_data, 32, cudaMemcpyDeviceToHost);
        cudaMemcpy(ry, dR.y.mont_repr_data, 32, cudaMemcpyDeviceToHost);
        cudaMemcpy(rz, dR.z.mont_repr_data, 32, cudaMemcpyDeviceToHost);
        int cmp1 = memcmp(rx, cx, 32);
        int cmp2 = memcmp(ry, cy, 32);
        int cmp3 = memcmp(rz, cz, 32);
        if(cmp1 != 0 || cmp2 != 0 || cmp3 != 0){
            printf("compare old failed, %d %d %d\n", cmp1, cmp2, cmp3);
            return;
        }else{
            printf("compare old success\n");
        }

        gpu_mcl_ect_add_new(dR, dP, dQ, done, dp, da, specialA_, model_, rp);  

        cudaMemcpy(rx, dR.x.mont_repr_data, 32, cudaMemcpyDeviceToHost);
        cudaMemcpy(ry, dR.y.mont_repr_data, 32, cudaMemcpyDeviceToHost);
        cudaMemcpy(rz, dR.z.mont_repr_data, 32, cudaMemcpyDeviceToHost);
        cmp1 = memcmp(rx, cx, 32);
        cmp2 = memcmp(ry, cy, 32);
        cmp3 = memcmp(rz, cz, 32);
        if(cmp1 != 0 || cmp2 != 0 || cmp3 != 0){
            printf("compare new failed, %d %d %d\n", cmp1, cmp2, cmp3);
            return;
        }else{
            printf("compare new success\n");
        }
    }
    //printf("compare success\n");
    //fclose(fpa);
    //fclose(fpb);
    //fclose(fpc);
    return;
}
