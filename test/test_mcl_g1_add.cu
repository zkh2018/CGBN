#include <iostream>
#include <vector>


#include "low_func_gpu.h"

using namespace gpu;

uint64_t ax[4] = {7093228299899950924, 2590038744256132367, 16520413636585797661, 1192609966037654901};
uint64_t ay[4] = {819477174237376354, 4356376756460091735, 7311316776602624733, 560085712742477326};
uint64_t az[4] = {15230403791020821917, 754611498739239741, 7381016538464732716, 1011752739694698287};

uint64_t bx[4] = {4252204126867602537, 9918196088281580021, 5151322999762652470, 2368356264329432293};
uint64_t by[4] = {14429909543139381222, 6582645255282250537, 3381912363200108806, 2176551161836059481};
uint64_t bz[4] = {15230403791020821917, 754611498739239741, 7381016538464732716, 1011752739694698287};

uint64_t cx[4] = {14082156349980393301, 9569459154142384405, 3920744438930614333, 2083992559355414077};
uint64_t cy[4] = {13043119889934885395, 7069690677259295523, 1218866525624105937, 21693625678536891};
uint64_t cz[4] = {15605719900677203229, 7328157344025447653, 7077653436886406425, 1175746298291777391};

uint64_t one[4] = {15230403791020821917, 754611498739239741, 7381016538464732716, 1011752739694698287}; 
uint64_t p[4] = {4332616871279656263, 10917124144477883021, 13281191951274694749, 3486998266802970665};
uint64_t a_[4] = {0, 0, 0, 0};

int specialA_ = 0;
int model_ = 0;
uint64_t rp =  9786893198990664585;

int main(){
    mcl_bn128_g1 dR, dP, dQ;
    Fp_model done, dp, da;
    dR.init(1);
    dP.init(1);
    dQ.init(1);
    done.init(1);
    dp.init(1);
    da.init(1);

    FILE *fpa = fopen("a.txt", "r");
    FILE *fpb = fopen("b.txt", "r");
    FILE *fpc = fopen("c.txt", "r");
    auto f = [](FILE*fp, uint64_t *data){
        for(int j = 0; j < 4; j++){
            fscanf(fp, "%lu ", &data[j]);
        }
    };
    for(int i = 0; i < 1; i++){
        f(fpa, ax);
        f(fpa, ay);
        f(fpa, az);

        f(fpb, bx);
        f(fpb, by);
        f(fpb, bz);

        f(fpc, cx);
        f(fpc, cy);
        f(fpc, cz);

        copy_cpu_to_gpu(dP.x.mont_repr_data, ax, 32);
        copy_cpu_to_gpu(dP.y.mont_repr_data, ay, 32);
        copy_cpu_to_gpu(dP.z.mont_repr_data, az, 32);

        copy_cpu_to_gpu(dQ.x.mont_repr_data, bx, 32);
        copy_cpu_to_gpu(dQ.y.mont_repr_data, by, 32);
        copy_cpu_to_gpu(dQ.z.mont_repr_data, bz, 32);

        copy_cpu_to_gpu(done.mont_repr_data, one, 32);
        copy_cpu_to_gpu(dp.mont_repr_data, p, 32);
        copy_cpu_to_gpu(da.mont_repr_data, a_, 32);
        gpu_mcl_ect_add(dR, dP, dQ, done, dp, da, specialA_, model_, rp);  

        uint64_t rx[4], ry[4], rz[4];
        copy_gpu_to_cpu(rx, dR.x.mont_repr_data, 32);
        copy_gpu_to_cpu(ry, dR.y.mont_repr_data, 32);
        copy_gpu_to_cpu(rz, dR.z.mont_repr_data, 32);
        int cmp1 = memcmp(rx, cx, 32);
        int cmp2 = memcmp(ry, cy, 32);
        int cmp3 = memcmp(rz, cz, 32);
        if(cmp1 != 0 || cmp2 != 0 || cmp3 != 0){
            printf("compare failed, %d %d %d\n", cmp1, cmp2, cmp3);
            return;
        }
    }
    printf("compare success\n");
    fclose(fpa);
    fclose(fpb);
    fclose(fpc);
    return;
}
