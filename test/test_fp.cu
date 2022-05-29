#include <iostream>
#include <vector>

#include <gmp.h>
#include "cgbn/cgbn.h"
#include "cgbn_fp.h"
#include "cgbn_alt_bn128_g1.cuh"

#define TPI 8
#define BITS 256
typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;
using namespace gpu;

uint64_t out1[4] = {13234290029914954971, 15949499366892279633, 11849505437050900688, 1295465949628936515};
uint64_t tw1[4] = {3605074180500004507, 8631770623530483452, 4430987740696927800, 1344638258457153879};
uint64_t z0[4] = {13549183084534785271, 18015951436638683593, 15372804613078859648, 2500121956885912024};
uint64_t z1[4] = {8272388832304539702, 8626540742494063589, 4178673508047320253, 497531985373161047};
uint64_t z2[4] = {4938796920257629340, 2224294988562428380, 2326078485242717023, 3239146591364966857};
uint64_t z3[4] = {7316948643007030864, 2798420992875277001, 10960839859516136073, 1721314747333763878};
uint64_t t1[4] = {13596519318755815826, 17343332041894265620, 4417691147046881922, 2252270281447908216};
uint64_t t2[4] = {15589337475311570566, 11424961735369340590, 15139513367563456326, 2218846732706924925};
uint64_t t3[4] = {13501846850313754716, 241826757673549950, 7881174005401285759, 2747973632323915833};
uint64_t t4[4] = {5846900875334107623, 8725034132925632941, 6499025599805878929, 2263215504842367834};
uint64_t j[4] = {9184314736506630027, 6863229837856182560, 10210050104002964590, 3114093135882422479};

__global__ void kernel_mul(
    cgbn_error_report_t* report, 
    uint32_t *a, uint32_t *b, uint32_t *c, 
    uint32_t* modulus,
    uint64_t inv){
    context_t bn_context(cgbn_report_monitor, report, 0);
    env_t          bn_env(bn_context.env<env_t>());  
    DevFp da, db, dc;
    cgbn_load(bn_env, da.mont, a);
    cgbn_load(bn_env, db.mont, b);
    env_t::cgbn_t local_modulus;
    cgbn_load(bn_env, local_modulus, modulus);

    __shared__ uint32_t buffer[32], res[32];
    dc = da.mul(bn_env, db, res, buffer, local_modulus, inv);
    cgbn_store(bn_env, c, dc.mont);
}

void mul(uint64_t* a, uint64_t *b, uint64_t *c, uint64_t *modulus, uint64_t inv){
	const int n = 4;
	mp_limb_t res[2*n];
	mpn_mul_n(res, a, b, n);
	/*
	   The Montgomery reduction here is based on Algorithm 14.32 in
	   Handbook of Applied Cryptography
	   <http://cacr.uwaterloo.ca/hac/about/chap14.pdf>.
	 */
	for (size_t i = 0; i < n; ++i)
	{
		mp_limb_t k = inv * res[i];
		printf("k = %lu\n", k);
		/* calculate res = res + k * mod * b^i */
		//mp_limb_t carryout = mpn_addmul_1(res+i, modulus, n, k);
		//carryout = mpn_add_1(res+n+i, res+n+i, n-i, carryout);
		mp_limb_t tmp[n];	
	    mp_limb_t carry = mpn_mul_1(tmp, modulus, n, k);	
        printf("carryout=%lu\n", carry);
        mp_limb_t carryout = mpn_add_n(res+i, res + i, tmp, n);
        //mpn_add_1(res+i, res+i, n, carry);
        printf("carryout=%lu\n\n", carryout);
        for(int j = 0; j < 4; j++){
            printf("%lu ", *(res+n+i+j));
        }
        printf("\n");
        mpn_add_1(res+n+i, res+n+i, n-i, carryout + carry);
        for(int j = 0; j < 4; j++){
            printf("%lu ", *(res+n+i+j));
        }
        printf("\n");
		//assert(carryout == 0);
	}

	if (mpn_cmp(res+n, modulus, n) >= 0)
	{
		const mp_limb_t borrow = mpn_sub(res+n, res+n, n, modulus, n);
        printf("borrow=%lu\n", borrow);
		//assert(borrow == 0);
	}

	mpn_copyi(c, res+n, n);
}

__global__ void kernel_sub(
    cgbn_error_report_t* report, 
    uint32_t *a, uint32_t *b, uint32_t *c, 
    uint32_t* modulus,
    uint32_t* max_value,
    uint64_t inv){
    context_t bn_context(cgbn_report_monitor, report, 0);
    env_t          bn_env(bn_context.env<env_t>());  
    DevFp da, db, dc;
    cgbn_load(bn_env, da.mont, a);
    cgbn_load(bn_env, db.mont, b);
    env_t::cgbn_t local_modulus, local_max_value;
    cgbn_load(bn_env, local_modulus, modulus);
    cgbn_load(bn_env, local_max_value, max_value);

    dc = da.sub(bn_env, db, local_max_value, local_modulus);
    cgbn_store(bn_env, c, dc.mont);
}

void sub(uint64_t* a, uint64_t *b, uint64_t *c, uint64_t *modulus, uint64_t inv){
	const int n = 4;
	mp_limb_t scratch[n+1];
	if (mpn_cmp(a, b, n) < 0)
	{
		const mp_limb_t carry = mpn_add_n(scratch, a, modulus, n);
		scratch[n] = carry;
	}
	else
	{
		mpn_copyi(scratch, a, n);
		scratch[n] = 0;
	}

	const mp_limb_t borrow = mpn_sub(scratch, scratch, n+1, b, n);
	//assert(borrow == 0);
	printf("borrow=%d\n", borrow);

	mpn_copyi(c, scratch, n);
}

void add_ui64(uint64_t *a, uint64_t b){
    mp_limb_t c[4];
    mp_limb_t carry = mpn_add_1(c, a, 4, b);    
    for(int i = 0; i < 4; i++){
        printf("%lu ", c[i]);
    }
    printf("\n");
    printf("carry=%lu\n", carry);
}

uint64_t modulus[4] = {4891460686036598785, 2896914383306846353, 13281191951274694749, 3486998266802970665};
uint64_t inv=14042775128853446655;


__global__ void kernel_add_ui64(
        cgbn_error_report_t* report, 
        uint32_t *a, uint64_t b, uint32_t*c){
    context_t bn_context(cgbn_report_monitor, report, 0);
    env_t          bn_env(bn_context.env<env_t>());  
    env_t::cgbn_t da, db, dc;
    cgbn_load(bn_env, da, a);
    __shared__ uint32_t buf[8];
    buf[threadIdx.x] = 0;
    uint32_t *p = (uint32_t*)&b;
    buf[0] = p[0];
    buf[1] = p[1];
    cgbn_load(bn_env, db, buf);
    cgbn_add(bn_env, dc, da, db);
    cgbn_store(bn_env, c, dc);
}

int main(){
    cgbn_error_report_t *report = nullptr;
    cgbn_error_report_alloc(&report); 
    /*
    uint32_t *da, *db, *dc, *dmodulus, *dmax_value;
    cudaMalloc((void**)&da, 32);
    cudaMalloc((void**)&db, 32);
    cudaMalloc((void**)&dc, 32);
    cudaMalloc((void**)&dmodulus, 32);

	uint64_t max_value[4];
	for(int i = 0; i < 4; i++){
		max_value[i] = 0xffffffff;
	}


    //j*t4
    cudaMemcpy(da, out1, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(db, tw1, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(dmodulus, modulus, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(dmax_value, max_value, 32, cudaMemcpyHostToDevice);

    //kernel_sub<<<1, 8>>>(report, da, db, dc, dmodulus, dmax_value, inv);
    kernel_mul<<<1, 8>>>(report, da, db, dc, dmodulus, inv);
    cudaDeviceSynchronize();

    uint64_t c1[4], c2[4];
    cudaMemcpy(c2, dc, 32, cudaMemcpyDeviceToHost);

    mul(out1, tw1, c1, modulus, inv);

    auto f = [](uint64_t *a){
        for(int i = 0; i < 4; i++){
            printf("%lu ", a[i]);
        }
        printf("\n");
    };
	printf("host:");
    f(c1);
	printf("device:");
    f(c2);
    */
    uint64_t a[4] = {17644445527039459864, 6117862080735121102, 94430381396260237, 140733007764640};
    uint64_t b = 203232170204987391+1;
    uint64_t correct[4] = {17847677697244447256, 6117862080735121102, 94430381396260237, 140733007764640};
    printf("correct:\n");
    for(int i = 0; i < 4; i++){
        printf("%lu ", correct[i]);
    }
    printf("\n");
    add_ui64(a, b);

    uint32_t *da, *db, *dc;
    cudaMalloc((void**)&da, 32);
    cudaMalloc((void**)&db, 32);
    cudaMalloc((void**)&dc, 32);
    cudaMemcpy(da, a, 32, cudaMemcpyHostToDevice);
    kernel_add_ui64<<<1, 8>>>(report, da, b, dc);
    uint64_t c[4];
    cudaMemcpy(c, dc, 32, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 4; i++){
        printf("%lu ", c[i]);
    }
    printf("\n");

    return 0;
}
