#include <cuda_runtime.h>
#include "bigint_256.cuh"
#include <stdio.h>
#include <gmp.h>

using namespace BigInt256;
typedef Int Unit;

__global__ void MulWideKernel(const Int* a, const Int* b, Int* c){
    __shared__ Int ret[N*2];
    dev_mul_wide(a, b, ret);
    memcpy(c, ret, N*2*sizeof(Int));
}

__global__ void MclAddKernel(const Int* a, const Int* b, const Int* p, Int* c){
    __shared__ Int ret[N];
    dev_mcl_add(a, b, p, ret);
    memcpy(c, ret, N*sizeof(Int));
}

void MulWide(const Int* a, const Int* b, Int* c){
    mpn_mul_n((mp_limb_t*)c, (const mp_limb_t*)a, (const mp_limb_t*)b, N);
}

void MclAdd(const Int* x, const Int* y, const Int* p, Int* z){
    //AddPre<N, Tag>::f(z, x, y);
    mpn_add_n((mp_limb_t*)z, (const mp_limb_t*)x, (const mp_limb_t*)y, N);
    Unit a = z[N - 1];
    Unit b = p[N - 1];
    if (a < b) return;
    if (a > b) {
        //SubPre<N, Tag>::f(z, z, p);
		mpn_sub_n((mp_limb_t*)z, (const mp_limb_t*)z, (const mp_limb_t*)p, N);
        return;
    }
    /* the top of z and p are same */
    //SubIfPossible<N, Tag>::f(z, p);
	Unit tmp[N - 1];
	//if (SubPre<N - 1, Tag>::f(tmp, z, p) == 0) {
	Unit ret = mpn_sub_n((mp_limb_t*)tmp, (const mp_limb_t*)z, (const mp_limb_t*)p, N-1);
	if(ret == 0){
		//copyC<N - 1>(z, tmp);
		memcpy(z, tmp, (N-1) * sizeof(Unit));
		z[N - 1] = 0;
	}
}

void print(const Int* x, const int n, const char* desc){
    printf("%s\n", desc);
    for(int i = 0; i < n; i++){
        printf("%lu ", x[i]);
    }
    printf("\n");
}

void TestMulWide(){
    //Int a[N] = {15230403791020821917, 754611498739239741, 7381016538464732716, 1011752739694698287};
    //Int b[N] = {4332616871279656263, 10917124144477883021, 13281191951274694749, 3486998266802970665};
    Int a[N] = {14068117839353947303, 9889041191422804744, 14467662803655596382, 2851639840953967274};
    Int b[N] = {4330347990600174557, 5331985108397172523, 815369053386666885, 3274728313373436549};
    Int c1[N*2], c2[N*2];
    MulWide(a, b, c1);

    Int* dev_a, *dev_b, *dev_c;
    cudaMalloc((void**)&dev_a, N*sizeof(Int));
    cudaMalloc((void**)&dev_b, N*sizeof(Int));
    cudaMalloc((void**)&dev_c, 2*N*sizeof(Int));
    cudaMemcpy(dev_a, a, sizeof(Int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(Int)*N, cudaMemcpyHostToDevice);
    MulWideKernel<<<1,1>>>(dev_a, dev_b, dev_c);
    cudaMemcpy(c2, dev_c, sizeof(Int)* N*2, cudaMemcpyDeviceToHost);
    int ret = memcmp(c1, c2, N*2*sizeof(Int));
    printf("ret = %d\n", ret);
    if(ret != 0){
       print(c1, N*2, "c1:"); 
       print(c2, N*2, "c2:"); 
    }
}

void TestMclAdd(){
    //Int a[N] = {15230403791020821917, 754611498739239741, 7381016538464732716, 1011752739694698287};
    //Int b[N] = {4332616871279656263, 10917124144477883021, 13281191951274694749, 3486998266802970665};
    Int a[N] = {14068117839353947303, 9889041191422804744, 14467662803655596382, 2851639840953967274};
    Int b[N] = {4330347990600174557, 5331985108397172523, 815369053386666885, 3274728313373436549};
    Int p[N] = {4332616871279656263, 10917124144477883021, 13281191951274694749, 3486998266802970665};
    Int c1[N], c2[N];
    MclAdd(a, b, p, c1);

    Int* dev_a, *dev_b, *dev_p, *dev_c;
    cudaMalloc((void**)&dev_a, N*sizeof(Int));
    cudaMalloc((void**)&dev_b, N*sizeof(Int));
    cudaMalloc((void**)&dev_p, N*sizeof(Int));
    cudaMalloc((void**)&dev_c, N*sizeof(Int));
    cudaMemcpy(dev_a, a, sizeof(Int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(Int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_p, p, sizeof(Int)*N, cudaMemcpyHostToDevice);
    MclAddKernel<<<1,1>>>(dev_a, dev_b, dev_p, dev_c);
    cudaMemcpy(c2, dev_c, sizeof(Int)* N, cudaMemcpyDeviceToHost);
    int ret = memcmp(c1, c2, N*sizeof(Int));
    printf("ret = %d\n", ret);
    if(ret != 0){
       print(c1, N, "c1:"); 
       print(c2, N, "c2:"); 
    }
}

int main(){
    TestMulWide();
    TestMclAdd();
    return 0;
}
