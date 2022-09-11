#include <cuda_runtime.h>
#include <gmp.h>

#include <stdio.h>
#include "bigint_256.cuh"
#include "cgbn_alt_bn128_g2.h"
//#include "low_func_gpu.h"
//#include "low_func.cuh"

using BigInt256::N;
using BigInt256::Int;
using BigInt256::Int256;
using namespace gpu;
using namespace BigInt256;

//one:15230403791020821917 754611498739239741 7381016538464732716 1011752739694698287
//p:4332616871279656263 10917124144477883021 13281191951274694749 3486998266802970665
//a_:
//0 0 0 0
//0 0 0 0
//values 0:
//17632392075874326361 686771681694832055 12240117844107911885 259760256799502517
//11378688920152681370 12061601062594514627 5112758659966651911 456042571771077233
//387368582743000074 10050512163553066970 4119814693469794386 1896038484534086291
//3569431071456247879 12368105748002045485 3306831679366552634 3121067581241328303
//15230403791020821917 754611498739239741 7381016538464732716 1011752739694698287
//0 0 0 0
//values 1:
//14182379067695583671 10564135533193299879 1193747285157269860 2579474753088789044
//18329435504127641459 15188361310238567510 10517971578836454787 1180012744745192852
//13768281374437795229 5773016860590742313 5815141406435023500 1349726300420083177
//14916702824358760568 4746496008524954675 15915565169268542370 879307527999266756
//15230403791020821917 754611498739239741 7381016538464732716 1011752739694698287
//0 0 0 0
const int mode_ = 0;
const int specialA_ = 0;
const uint64_t rp=9786893198990664585;
//out
//13712628936793140141 11178027048663624011 13908955125778428489 364828958317829519                                                                       
//16488583920232013686 4406118390151675257 9530784481429205982 282509255794187596
//16307487932227164170 13510122477398691286 8869060505837966149 1503013392412725965
//5143377157453079116 4242295185100531185 4717934346919037966 1383927756181149357
//14996731065530808926 9877363851498467823 7400373514758909591 2319714496289286526
//6950746583974960089 3126760247644052883 5405212918869802876 723970172974115619

//13712628936793140141 11178027048663624011 13908955125778428489 364828958317829519
//16488583920232013686 4406118390151675257 9530784481429205982 282509255794187596
//7642254189667851644 10122618262152476860 753420676998128266 12975760932516336250
//14924887488603318206 854790969854316758 15049038591788751699 12856675296284759642
//14996731065530808926 9877363851498467823 7400373514758909591 2319714496289286526
//6950746583974960089 3126760247644052883 5405212918869802877 723970172974115620

__global__ void kernel2(BigInt256::Ect2 R){
   BigInt256::Ect2 P, Q;
   Int256 one = {15230403791020821917, 754611498739239741, 7381016538464732716, 1011752739694698287};
   Int256 p = {4332616871279656263, 10917124144477883021, 13281191951274694749, 3486998266802970665};
   BigInt256::Point a_;
   Int a_c0[N] = {0}, a_c1[N] = {0};
   a_.c0 = a_c0; a_.c1 = a_c1;

   Int256 p_x_c0 = {13309721960704354645, 9108267262294541978, 11011926952682606932, 3230674604292178113};
   Int256 p_x_c1 = {15173343002083334166, 673699406643579773, 18104480845134585166, 2905684088189023460};
   Int256 p_y_c0 = {4271464053119150746, 9931769675798870999, 5045794387563676626, 150293018333280419};
   Int256 p_y_c1 = {15126966224785968460, 5247032317646106843, 11172505633045305263, 868817351613909621};
   Int256 p_z_c0 = {15230403791020821917, 754611498739239741, 7381016538464732716, 1011752739694698287};
   Int256 p_z_c1 = {0, 0, 0, 0};

   P.x.c0 = p_x_c0;
   P.x.c1 = p_x_c1;

   P.y.c0 = p_y_c0;
   P.y.c1 = p_y_c1;

   P.z.c0 = p_z_c0;
   P.z.c1 = p_z_c1;

   Int256 q_x_c0 = {13309721960704354645, 9108267262294541978, 11011926952682606932, 3230674604292178113};
   Int256 q_x_c1 = {15173343002083334166, 673699406643579773, 18104480845134585166, 2905684088189023460};
   Int256 q_y_c0 = {4271464053119150746, 9931769675798870999, 5045794387563676626, 150293018333280419};
   Int256 q_y_c1 = {15126966224785968460, 5247032317646106843, 11172505633045305263, 868817351613909621};
   Int256 q_z_c0 = {15230403791020821917, 754611498739239741, 7381016538464732716, 1011752739694698287};
   Int256 q_z_c1 = {0, 0, 0, 0};
   Q.x.c0 = q_x_c0;
   Q.x.c1 = q_x_c1;

   Q.y.c0 = q_y_c0;
   Q.y.c1 = q_y_c1;

   Q.z.c0 = q_z_c0;
   Q.z.c1 = q_z_c1;

   //__shared__ Int r_x_c0[N];
   //__shared__ Int r_x_c1[N];
   //__shared__ Int r_y_c0[N];
   //__shared__ Int r_y_c1[N];
   //__shared__ Int r_z_c0[N];
   //__shared__ Int r_z_c1[N];
   //R.x.c0 = r_x_c0;
   //R.x.c1 = r_x_c1;

   //R.y.c0 = r_y_c0;
   //R.y.c1 = r_y_c1;

   //R.z.c0 = r_z_c0;
   //R.z.c1 = r_z_c1;
   printf("%x %x %x %x %x %x\n", R.x.c0, R.x.c1, R.y.c0, R.y.c1, R.z.c0, R.z.c1);


    
   //BigInt256::Ect2 result;
  //Int cache[BigInt256::N * 6];
  //result.x.c0 = cache;
  //result.x.c1 = cache + BigInt256::N;
  //result.y.c0 = cache + 2 * BigInt256::N;
  //result.y.c1 = cache + 3 * BigInt256::N;
  //result.z.c0 = cache + 4 * BigInt256::N;
  //result.z.c1 = cache + 5 * BigInt256::N;
   BigInt256::add_g2(R, P, Q, one, p, specialA_, a_, mode_, rp);
   printf("result:\n\n");
   for(int i = 0; i < 4; i++){
    printf("%lu ", R.x.c0[i]);
   }
   printf("\n");
   for(int i = 0; i < 4; i++){
    printf("%lu ", R.x.c1[i]);
   }
   printf("\n");
   for(int i = 0; i < 4; i++){
    printf("%lu ", R.y.c0[i]);
   }
   printf("\n");
   for(int i = 0; i < 4; i++){
    printf("%lu ", R.y.c1[i]);
   }
   printf("\n");
   for(int i = 0; i < 4; i++){
    printf("%lu ", R.z.c0[i]);
   }
   printf("\n");
   for(int i = 0; i < 4; i++){
    printf("%lu ", R.z.c1[i]);
   }
   printf("\n");
}

typedef uint64_t Unit;
void mcl_add(Unit *z, const Unit *x, Unit *y, const Unit *p){
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

void mcl_sub(Unit *z, const Unit *x, Unit *y, const Unit *p){
	Unit ret = mpn_sub_n((mp_limb_t*)z, (const mp_limb_t*)x, (const mp_limb_t*)y, N);
	//if (SubPre<N, Tag>::f(z, x, y)) {
	if(ret){
	//AddPre<N, Tag>::f(z, z, p);
		mpn_add_n((mp_limb_t*)z, (const mp_limb_t*)z, (const mp_limb_t*)p, N);
	}
}

void mcl_mul(Unit *z, const Unit *x, Unit *y, const Unit *p, const Unit rp){
	Unit carry;
	(void)carry;
	Unit buf[N * 2 + 1];
	Unit *c = buf;
	//MulUnitPre<N, Tag>::f(c, x, y[0]); // x * y[0]
    c[N] = mpn_mul_1((mp_limb_t*)c, (mp_limb_t*)x, N, y[0]);
	Unit q = c[0] * rp;
	Unit t[N + 1];
	//MulUnitPre<N, Tag>::f(t, p, q); // p * q
    t[N] = mpn_mul_1((mp_limb_t*)t, (mp_limb_t*)p, N, q);
	//carry = AddPre<N + 1, Tag>::f(c, c, t);
    carry = mpn_add_n((mp_limb_t*)c, (const mp_limb_t*)c, (const mp_limb_t*)t, N+1);
	//assert(carry == 0);
	c++;
	c[N] = 0;
	for (size_t i = 1; i < N; i++) {
		c[N + 1] = 0;
		//MulUnitPre<N, Tag>::f(t, x, y[i]);
		t[N] = mpn_mul_1((mp_limb_t*)t, (mp_limb_t*)x, N, y[i]);
		//carry = AddPre<N + 1, Tag>::f(c, c, t);
		carry = mpn_add_n((mp_limb_t*)c, (const mp_limb_t*)c, (const mp_limb_t*)t, N+1);
		//assert(carry == 0);
		q = c[0] * rp;
		//MulUnitPre<N, Tag>::f(t, p, q);
		t[N] = mpn_mul_1((mp_limb_t*)t, (mp_limb_t*)p, N, q);
		//carry = AddPre<N + 1, Tag>::f(c, c, t);
		carry = mpn_add_n((mp_limb_t*)c, (const mp_limb_t*)c, (const mp_limb_t*)t, N+1);
		//assert(carry == 0);
		c++;
	}
	//assert(c[N] == 0);
    int borrow = mpn_sub_n((mp_limb_t*)z, (const mp_limb_t*)c, (const mp_limb_t*)p, N);
	//if (SubPre<N, Tag>::f(z, c, p)) {
	if(borrow){
		memcpy(z, c, N * sizeof(Unit));
	}
}

int main(){
    BigInt256::Ect2 R;
    cudaMalloc((void**)&R.x.c0, N * sizeof(Int)); 
    cudaMalloc((void**)&R.x.c1, N * sizeof(Int)); 
    cudaMalloc((void**)&R.y.c0, N * sizeof(Int)); 
    cudaMalloc((void**)&R.y.c1, N * sizeof(Int)); 
    cudaMalloc((void**)&R.z.c0, N * sizeof(Int)); 
    cudaMalloc((void**)&R.z.c1, N * sizeof(Int)); 
    for(int i = 0; i < 1; i++){
        kernel2<<<1,1>>>(R);
    }
    cudaDeviceSynchronize();
    return 0;
}
