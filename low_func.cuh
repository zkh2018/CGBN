#ifndef LOW_FUNC_CUH
#define LOW_FUNC_CUH

#include "cgbn_fp.h"
#include "cgbn_alt_bn128_g1.cuh"
#include "cgbn_multi_exp.h"
#include "cgbn_ect.h"
#include <stdint.h>
#include <thrust/scan.h>

namespace gpu{

#define TPI 8
typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

const int N_32 = 8;
const int N_64 = 4;

inline __device__ bool dev_is_zero(env_t& bn_env, uint32_t* x){
    env_t::cgbn_t a;
    cgbn_load(bn_env, a, x);
    return cgbn_equals_ui32(bn_env, a, 0);
}

inline __device__ bool dev_is_zero(env_t& bn_env, const env_t::cgbn_t& a){
    return cgbn_equals_ui32(bn_env, a, 0);
}

inline __device__ bool dev_is_one(env_t& bn_env, uint32_t* x, uint32_t* one){
    env_t::cgbn_t lx, lone;
    cgbn_load(bn_env, lx, x);
    cgbn_load(bn_env, lone, one);
    return cgbn_equals(bn_env, lx, lone);
}

inline __device__ bool dev_is_one(env_t& bn_env, const env_t::cgbn_t& lx, const env_t::cgbn_t& lone){
    return cgbn_equals(bn_env, lx, lone);
}

inline __device__ bool dev_equal(env_t& bn_env, uint32_t* x, uint32_t *y){
    env_t::cgbn_t a, b;
    cgbn_load(bn_env, a, x);
    cgbn_load(bn_env, b, y);
    return cgbn_equals(bn_env, a, b);
}
inline __device__ bool dev_equal(env_t& bn_env, env_t::cgbn_t& a, env_t::cgbn_t& b){
    return cgbn_equals(bn_env, a, b);
}

inline __device__ void dev_clear(env_t& bn_env, uint32_t *x){
    env_t::cgbn_t zero;
    cgbn_set_ui32(bn_env, zero, 0);
    cgbn_store(bn_env, x, zero);
}

inline __device__ void dev_clear(env_t& bn_env, env_t::cgbn_t& x){
    cgbn_set_ui32(bn_env, x, 0);
}

inline __device__ void dev_mcl_add(env_t& bn_env, env_t::cgbn_t& lz, env_t::cgbn_t& lx, env_t::cgbn_t& ly, env_t::cgbn_t& lp, 
        uint32_t* p, uint32_t* cache){
    cgbn_add(bn_env, lz, lx, ly);
    cgbn_store(bn_env, cache, lz);
    int64_t a_z = ((uint64_t*)cache)[3];
    int64_t b_p = ((uint64_t*)p)[3];
    if(a_z < b_p) return;
    if(a_z > b_p) {
        cgbn_sub(bn_env, lz, lz, lp);
        return;
    }
    
    env_t::cgbn_t a;
    int32_t borrow = cgbn_sub(bn_env, a, lz, lp);
    if(borrow == 0){
        cgbn_set(bn_env, lz, a);
    }
}

inline __device__ void dev_mcl_add(env_t& bn_env, uint32_t* z, uint32_t* x, uint32_t* y, uint32_t* p, uint32_t* cache){
    env_t::cgbn_t lx, ly, lp, lz;
    cgbn_load(bn_env, lx, x);
    cgbn_load(bn_env, ly, y);
    cgbn_load(bn_env, lp, p);
    dev_mcl_add(bn_env, lz, lx, ly, lp, p, cache); 
    cgbn_store(bn_env, z, lz);
    //cgbn_add(bn_env, d, a, b);
    //cgbn_store(bn_env, z, d);
    //int64_t a_z = ((uint64_t*)z)[3];
    //int64_t b_p = ((uint64_t*)p)[3];
    //if(a_z < b_p) return;
    //if(a_z > b_p) {
    //    cgbn_sub(bn_env, a, d, c);
    //    cgbn_store(bn_env, z, a);
    //    return;
    //}
    //
    //int32_t borrow = cgbn_sub(bn_env, a, d, c);
    //if(borrow == 0){
    //    cgbn_store(bn_env, z, a);
    //}
}

__global__ void kernel_mcl_add(
    cgbn_error_report_t* report, 
    uint32_t* z, uint32_t*x, uint32_t*y, uint32_t* p){
  context_t bn_context(cgbn_report_monitor, report, 0);
  env_t          bn_env(bn_context.env<env_t>());  
  __shared__ uint32_t cache[8];
  dev_mcl_add(bn_env, z, x, y, p, cache);
}

inline __device__ void dev_mcl_sub(env_t& bn_env, env_t::cgbn_t& lz, env_t::cgbn_t& lx, env_t::cgbn_t& ly, env_t::cgbn_t& lp){
    int ret = cgbn_sub(bn_env, lz, lx, ly);
    if(ret){
        cgbn_add(bn_env, lz, lz, lp);
        return;
    }
}
inline __device__ void dev_mcl_sub(env_t& bn_env, uint32_t* z, uint32_t* x, uint32_t* y, uint32_t* p){
    env_t::cgbn_t lx, ly, lp, lz;
    cgbn_load(bn_env, lx, x);
    cgbn_load(bn_env, ly, y);
    cgbn_load(bn_env, lp, p);
    dev_mcl_sub(bn_env, lz, lx, ly, lp);
    cgbn_store(bn_env, z, lz);
    //int ret = cgbn_sub(bn_env, d, a, b);
    //if(ret){
    //    cgbn_load(bn_env, c, p);
    //    cgbn_add(bn_env, a, d, c);
    //    cgbn_store(bn_env, z, a);
    //    return;
    //}
    //cgbn_store(bn_env, z, d);
}

__global__ void kernel_mcl_sub(
    cgbn_error_report_t* report, 
    uint32_t* z, uint32_t*x, uint32_t*y, uint32_t* p){
  context_t bn_context(cgbn_report_monitor, report, 0);
  env_t          bn_env(bn_context.env<env_t>());  
  dev_mcl_sub(bn_env, z, x, y, p);
}

inline __device__ void dev_neg(env_t& bn_env, env_t::cgbn_t& ly, env_t::cgbn_t& lx, env_t::cgbn_t& lp){
    if(dev_is_zero(bn_env, lx)){
       if(!dev_equal(bn_env, lx, ly)){
           dev_clear(bn_env, ly);
           return;
       }
    }
    cgbn_sub(bn_env, ly, lp, lx);
}
inline __device__ void dev_neg(env_t& bn_env, uint32_t* y, uint32_t* x, uint32_t* p){
    env_t::cgbn_t ly, lx, lp;
    cgbn_load(bn_env, lx, x);
    cgbn_load(bn_env, ly, y);
    cgbn_load(bn_env, lp, p);
    dev_neg(bn_env, ly, lx, lp);
}

//buf: size = 2 * N_32 + 2
//t : size = N_32 + 2 + N_32
inline __device__ void dev_mcl_mul(env_t& bn_env, 
        env_t::cgbn_t& lz, env_t::cgbn_t& lx, env_t::cgbn_t& ly, env_t::cgbn_t& lp, 
        uint32_t* p, uint32_t *buf, uint32_t *t, const uint64_t rp){
    int group_id = threadIdx.x & (TPI-1);
    cgbn_store(bn_env, t, ly);
    uint64_t* p64_y = (uint64_t*)t;
    const uint64_t* p64_p = (const uint64_t*)p;
    //const uint64_t* p64_t = (const uint64_t*)t;
    uint64_t* c = (uint64_t*)buf;
    c[group_id] = 0;
    //if(group_id == 0){
    //    c[9] = 0;
    //}

    env_t::cgbn_t lc;
    env_t::cgbn_wide_t lwc, lwt;
    env_t::cgbn_t lcache;
    cgbn_set_ui32(bn_env, lcache, t[0], t[1]);
    cgbn_mul_wide(bn_env, lwc, lx, lcache);
    cgbn_store(bn_env, buf, lwc._low);
    uint64_t q = c[0] * rp;
    cgbn_set_ui32(bn_env, lcache, ((uint32_t*)&q)[0], ((uint32_t*)&q)[1]);
    cgbn_mul_wide(bn_env, lwt, lp, lcache);
    //cgbn_store(bn_env, t, lwt._high);
    int32_t carry = cgbn_add(bn_env, lc, lwc._low, lwt._low);
    cgbn_store(bn_env, buf, lc);
    cgbn_store(bn_env, buf + N_32, lwc._high);
    uint32_t th[2];
    cgbn_get_ui64(bn_env, lwt._high, th);
    if(group_id == 0){
        c[N_64] += *((uint64_t*)th) + carry; 
    }
    c++;
    if(group_id == 0){
        c[N_64] = 0;
    }
    for(int i = 1; i < N_64; i++){
        if(group_id == 0){
            c[N_64+1] = 0;
        }
        cgbn_load(bn_env, lc, (uint32_t*)c);
        cgbn_set_ui32(bn_env, lcache, t[i * 2], t[i * 2 + 1]);
        cgbn_mul_wide(bn_env, lwt, lx, lcache);
        cgbn_get_ui64(bn_env, lwt._high, th);
        carry = cgbn_add(bn_env, lc, lc, lwt._low);
        cgbn_store(bn_env, (uint32_t*)c, lc);
        if(group_id == 0){
            c[N_64] += *((uint64_t*)th) + carry; 
        }
        q = c[0] * rp;
        cgbn_set_ui32(bn_env, lcache, ((uint32_t*)&q)[0], ((uint32_t*)&q)[1]);
        cgbn_mul_wide(bn_env, lwt, lp, lcache);
        cgbn_get_ui64(bn_env, lwt._high, th);
        cgbn_load(bn_env, lc, (uint32_t*)c);
        carry = cgbn_add(bn_env, lc, lc, lwt._low);
        cgbn_store(bn_env, (uint32_t*)c, lc);
        if(group_id == 0){
            c[N_64] += *((uint64_t*)th) + carry; 
        }
        c++;
    }
    cgbn_load(bn_env, lc, (uint32_t*)c);
    int sub_ret = cgbn_sub(bn_env, lz, lc, lp);
    if(sub_ret){
        cgbn_set(bn_env, lz, lc);
    }
}

//rp = p[0]p[1], p=&p[2]
inline __device__ void dev_mcl_mul(env_t& bn_env, uint32_t* z, uint32_t* x, uint32_t* y, uint32_t* p, uint32_t *buf, uint32_t *t, const uint64_t rp){
    env_t::cgbn_t lx, ly, lp, lz;
    cgbn_load(bn_env, lx, x);
    cgbn_load(bn_env, ly, y);
    cgbn_load(bn_env, lp, p);

    dev_mcl_mul(bn_env, lz, lx, ly, lp, p, buf, t, rp); 
    cgbn_store(bn_env, z, lz);
}

inline __device__ void dev_mcl_sqr(env_t& bn_env, 
        env_t::cgbn_t& lz, env_t::cgbn_t& lx, env_t::cgbn_t& lp, 
        uint32_t* p, uint32_t *buf, uint32_t *t, const uint64_t rp){
    dev_mcl_mul(bn_env, lz, lx, lx, lp, p, buf, t, rp);
    //cgbn_store(bn_env, z, lz);
}
inline __device__ void dev_mcl_sqr(env_t& bn_env, uint32_t* z, uint32_t* x, uint32_t* p, uint32_t *buf, uint32_t *t, const uint64_t rp){
    env_t::cgbn_t lx, lp, lz;
    cgbn_load(bn_env, lx, x);
    cgbn_load(bn_env, lp, p);

    dev_mcl_mul(bn_env, lz, lx, lx, lp, p, buf, t, rp); 
    cgbn_store(bn_env, z, lz);
}

__global__ void kernel_mcl_mul(
    cgbn_error_report_t* report, 
    uint32_t* z, uint32_t*x, uint32_t*y, uint32_t* p, const uint64_t rp){
  context_t bn_context(cgbn_report_monitor, report, 0);
  env_t          bn_env(bn_context.env<env_t>());  
  __shared__ uint32_t cache_buf[N_32*2+2], cache_t[N_32];
  dev_mcl_mul(bn_env, z, x, y, p, cache_buf, cache_t, rp);
}

struct MclFp: public DevFp {
    uint32_t *ptr;
};

struct DevEct : public DevAltBn128G1 {

inline __device__ bool is_zero(env_t& bn_env){
    return dev_is_zero(bn_env, this->z.mont);
}
    
inline __device__ void dev_dblNoVerifyInfJacobi(env_t& bn_env, DevEct& P, MclFp& one, MclFp& p, const int specialA_, uint32_t* cache_buf, uint32_t *cache_t, MclFp& a_, const uint64_t rp){
    MclFp S, M, t, y2;
    //Fp::sqr(y2, P.y);
    dev_mcl_sqr(bn_env, y2.mont, P.y.mont, p.mont, p.ptr, cache_buf, cache_t, rp); 
    //Fp::mul(S, P.x, y2);
    dev_mcl_mul(bn_env, S.mont, P.x.mont, y2.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
    //const bool isPzOne = P.z.isOne();
    const bool isPzOne = dev_is_one(bn_env, P.z.mont, one.mont);
    //S += S;
    dev_mcl_add(bn_env, S.mont, S.mont, S.mont, p.mont, p.ptr, cache_buf);
    //S += S;
    dev_mcl_add(bn_env, S.mont, S.mont, S.mont, p.mont, p.ptr, cache_buf);
    //Fp::sqr(M, P.x);
    dev_mcl_sqr(bn_env, M.mont, P.x.mont, p.mont, p.ptr, cache_buf, cache_t, rp); 
    switch (specialA_) {
        case 0:
            //Fp::add(t, M, M);
            dev_mcl_add(bn_env, t.mont, M.mont, M.mont, p.mont, p.ptr, cache_buf);
            //M += t;
            dev_mcl_add(bn_env, M.mont, M.mont, t.mont, p.mont, p.ptr, cache_buf);
            break;
        case 1:
            if (isPzOne) {
                //M -= P.z;
                dev_mcl_sub(bn_env, M.mont, M.mont, P.z.mont, p.mont);
            } else {
                //Fp::sqr(t, P.z);
                dev_mcl_sqr(bn_env, t.mont, P.z.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
                //Fp::sqr(t, t);
                dev_mcl_sqr(bn_env, t.mont, t.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
                //M -= t;
                dev_mcl_sub(bn_env, M.mont, M.mont, t.mont, p.mont);
            }
            //Fp::add(t, M, M);
            dev_mcl_add(bn_env, t.mont, M.mont, M.mont, p.mont, p.ptr, cache_buf);
            //M += t;
            dev_mcl_add(bn_env, M.mont, M.mont, t.mont, p.mont, p.ptr, cache_buf);
            break;
        case 2:
        default:
            if (isPzOne) {
                //t = a_;
                cgbn_set(bn_env, t.mont, a_.mont);
            } else {
                //Fp::sqr(t, P.z);
                dev_mcl_sqr(bn_env, t.mont, P.z.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
                //Fp::sqr(t, t);
                dev_mcl_sqr(bn_env, t.mont, t.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
                //t *= a_;
                dev_mcl_mul(bn_env, t.mont, t.mont, a_.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
            }
            //t += M;
            dev_mcl_add(bn_env, t.mont, t.mont, M.mont, p.mont, p.ptr, cache_buf);
            //M += M;
            dev_mcl_add(bn_env, M.mont, M.mont, M.mont, p.mont, p.ptr, cache_buf);
            //M += t;
            dev_mcl_add(bn_env, M.mont, M.mont, t.mont, p.mont, p.ptr, cache_buf);
            break;
    }
	//Fp::sqr(R.x, M);
    dev_mcl_sqr(bn_env, this->x.mont, M.mont, p.mont, p.ptr, cache_buf, cache_t, rp); 
	//R.x -= S;
    dev_mcl_sub(bn_env, this->x.mont, this->x.mont, S.mont, p.mont);
	//R.x -= S;
    dev_mcl_sub(bn_env, this->x.mont, this->x.mont, S.mont, p.mont);
	if (isPzOne) {
		//R.z = P.y;
        cgbn_set(bn_env, this->z.mont, P.y.mont);
	} else {
		//Fp::mul(R.z, P.y, P.z);
        dev_mcl_mul(bn_env, this->z.mont, P.y.mont, P.z.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
	}
	//R.z += R.z;
    dev_mcl_add(bn_env, this->z.mont, this->z.mont, this->z.mont, p.mont, p.ptr, cache_buf);
	//Fp::sqr(y2, y2);
    dev_mcl_sqr(bn_env, y2.mont, y2.mont, p.mont, p.ptr, cache_buf, cache_t, rp); 
	//y2 += y2;
    dev_mcl_add(bn_env, y2.mont, y2.mont, y2.mont, p.mont, p.ptr, cache_buf);
	//y2 += y2;
    dev_mcl_add(bn_env, y2.mont, y2.mont, y2.mont, p.mont, p.ptr, cache_buf);
	//y2 += y2;
    dev_mcl_add(bn_env, y2.mont, y2.mont, y2.mont, p.mont, p.ptr, cache_buf);
	//Fp::sub(R.y, S, R.x);
    dev_mcl_sub(bn_env, this->y.mont, S.mont, this->x.mont, p.mont);
	//R.y *= M;
    dev_mcl_mul(bn_env, this->y.mont, this->y.mont, M.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
	//R.y -= y2;
    dev_mcl_sub(bn_env, this->y.mont, this->y.mont, y2.mont, p.mont);
}

inline __device__ void set(env_t& bn_env, DevEct& other){
    this->x.set(bn_env, other.x);
    this->y.set(bn_env, other.y);
    this->z.set(bn_env, other.z);
}

inline __device__ void clear(env_t& bn_env){
    dev_clear(bn_env, x.mont);
    dev_clear(bn_env, y.mont);
    dev_clear(bn_env, z.mont);
}

};
/*
//inline __device__ void neg(env_t& bn_env, DevEct& R, DevEct& P){
//    if(P.is_zero(bn_env)){
//        R.clear(bn_env);
//        return;
//    }
//    R.x.set(P.x);
//    dev_neg(bn_env, R.y.mont, P.y.mont);
//    R.z.set(P.z);
//}
//
//inline __device__ void sub(env_t& bn_env, DevEct& R, DevEct& P, DevEct& Q){
//    EcT nQ;
//    neg(bn_env, nQ, Q);
//}
*/


inline __device__ void dev_addJacobi(env_t& bn_env, DevEct& R, DevEct& P, DevEct& Q, bool isPzOne, bool isQzOne,
        MclFp& one, MclFp& p, const int specialA_, uint32_t* cache_buf, uint32_t* cache_t, MclFp& a_, const int mode_, const uint64_t rp){
    MclFp r, U1, S1, H, H3;
    if (isPzOne) {
        // r = 1;
    } else {
        //Fp::sqr(r, P.z);
        dev_mcl_sqr(bn_env, r.mont, P.z.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
    }
    if (isQzOne) {
        //U1 = P.x;
        U1.set(bn_env, P.x);
        if (isPzOne) {
            //H = Q.x;
            H.set(bn_env, Q.x);
        } else {
            //Fp::mul(H, Q.x, r);
            dev_mcl_mul(bn_env, H.mont, Q.x.mont, r.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
        }
        //H -= U1;
        dev_mcl_sub(bn_env, H.mont, H.mont, U1.mont, p.mont);
        //S1 = P.y;
        S1.set(bn_env, P.y);
    } else {
        //Fp::sqr(S1, Q.z);
        dev_mcl_sqr(bn_env, S1.mont, Q.z.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
        //Fp::mul(U1, P.x, S1);
        dev_mcl_mul(bn_env, U1.mont, P.x.mont, S1.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
        if (isPzOne) {
            //H = Q.x;
            H.set(bn_env, Q.x);
        } else {
            //Fp::mul(H, Q.x, r);
            dev_mcl_mul(bn_env, H.mont, Q.x.mont, r.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
        }
        //H -= U1;
        dev_mcl_sub(bn_env, H.mont, H.mont, U1.mont, p.mont);
        //S1 *= Q.z;
        dev_mcl_mul(bn_env, S1.mont, S1.mont, Q.z.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
        //S1 *= P.y;
        dev_mcl_mul(bn_env, S1.mont, S1.mont, P.y.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
    }
    if (isPzOne) {
        //r = Q.y;
        r.set(bn_env, Q.y);
    } else {
        //r *= P.z;
        dev_mcl_mul(bn_env, r.mont, r.mont, P.z.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
        //r *= Q.y;
        dev_mcl_mul(bn_env, r.mont, r.mont, Q.y.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
    }
    //r -= S1;
    dev_mcl_sub(bn_env, r.mont, r.mont, S1.mont, p.mont);
    if (H.is_zero(bn_env)) {
        if (r.is_zero(bn_env)) {
            R.dev_dblNoVerifyInfJacobi(bn_env, P, one, p, specialA_, cache_buf, cache_t, a_, rp);
        } else {
            R.clear(bn_env);
        }
        return;
    }
    if (isPzOne) {
        if (isQzOne) {
            //R.z = H;
            R.z.set(bn_env, H);
        } else {
            //Fp::mul(R.z, H, Q.z);
            dev_mcl_mul(bn_env, R.z.mont, H.mont, Q.z.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
        }
    } else {
        if (isQzOne) {
            //Fp::mul(R.z, P.z, H);
            dev_mcl_mul(bn_env, R.z.mont, P.z.mont, H.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
        } else {
            //Fp::mul(R.z, P.z, Q.z);
            dev_mcl_mul(bn_env, R.z.mont, P.z.mont, Q.z.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
            //R.z *= H;
            dev_mcl_mul(bn_env, R.z.mont, R.z.mont, H.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
        }
    }
    //Fp::sqr(H3, H); // H^2
    dev_mcl_sqr(bn_env, H3.mont, H.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
    //Fp::sqr(R.y, r); // r^2
    dev_mcl_sqr(bn_env, R.y.mont, r.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
    ///U1 *= H3; // U1 H^2
    dev_mcl_mul(bn_env, U1.mont, U1.mont, H3.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
    //H3 *= H; // H^3
    dev_mcl_mul(bn_env, H3.mont, H3.mont, H.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
    //R.y -= U1;
    dev_mcl_sub(bn_env, R.y.mont, R.y.mont, U1.mont, p.mont);
    //R.y -= U1;
    dev_mcl_sub(bn_env, R.y.mont, R.y.mont, U1.mont, p.mont);
    //Fp::sub(R.x, R.y, H3);
    dev_mcl_sub(bn_env, R.x.mont, R.y.mont, H3.mont, p.mont);
    //U1 -= R.x;
    dev_mcl_sub(bn_env, U1.mont, U1.mont, R.x.mont, p.mont);
    //U1 *= r;
    dev_mcl_mul(bn_env, U1.mont, U1.mont, r.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
    //H3 *= S1;
    dev_mcl_mul(bn_env, H3.mont, H3.mont, S1.mont, p.mont, p.ptr, cache_buf, cache_t, rp);
    //Fp::sub(R.y, U1, H3);
    dev_mcl_sub(bn_env, R.y.mont, U1.mont, H3.mont, p.mont);
}

inline __device__ void add(env_t& bn_env, DevEct& R, DevEct& P, DevEct& Q,
        MclFp& one, MclFp& p, const int specialA_, uint32_t* cache_buf, uint32_t* cache_t, MclFp& a_, 
        const int mode_, const uint64_t rp){
    if(P.is_zero(bn_env)){
        R.set(bn_env, Q); 
        return;
    }
    if(Q.is_zero(bn_env)){
        R.set(bn_env, P);
        return;
    }
    if(&P == &Q){
        R.dev_dblNoVerifyInfJacobi(bn_env, P, one, p, specialA_, cache_buf, cache_t, a_, rp);
        return;
    }
    bool isPzOne = dev_is_one(bn_env, P.z.mont, one.mont);
    bool isQzOne = dev_is_one(bn_env, Q.z.mont, one.mont);
    //switch (mode_) {
        dev_addJacobi(bn_env, R, P, Q, isPzOne, isQzOne, one, p, specialA_, cache_buf, cache_t, a_, mode_, rp);
        //break;
    //}
}


inline __device__ void load(env_t& bn_env, DevEct& Q, mcl_bn128_g1& P, const int offset){
    cgbn_load(bn_env, Q.x.mont, P.x.mont_repr_data + offset);
    cgbn_load(bn_env, Q.y.mont, P.y.mont_repr_data + offset);
    cgbn_load(bn_env, Q.z.mont, P.z.mont_repr_data + offset);
}

inline __device__ void store(env_t& bn_env, mcl_bn128_g1& Q, DevEct& P, const int offset){
    cgbn_store(bn_env, Q.x.mont_repr_data + offset, P.x.mont);
    cgbn_store(bn_env, Q.y.mont_repr_data + offset, P.y.mont);
    cgbn_store(bn_env, Q.z.mont_repr_data + offset, P.z.mont);
}

inline __device__ void load(env_t& bn_env, MclFp& Q, Fp_model& data, const int offset){
    cgbn_load(bn_env, Q.mont, data.mont_repr_data + offset);
}

__global__ void kernel_ect_add(
    cgbn_error_report_t* report,
    mcl_bn128_g1 R, 
    mcl_bn128_g1 P,
    mcl_bn128_g1 Q,
    Fp_model one, 
    Fp_model p, 
    Fp_model a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int instance = tid / TPI;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  

  __shared__ uint32_t cache_buf[N_32*2+2], cache_t[N_32];
  DevEct lR, lP, lQ;
  MclFp lone, lp, la;
  load(bn_env, lP, P, 0);
  load(bn_env, lQ, Q, 0);

  load(bn_env, lone, one, 0); 
  load(bn_env, la, a, 0); 
  load(bn_env, lp, p, 0); 
  lp.ptr = (uint32_t*)p.mont_repr_data;

  add(bn_env, lP, lP, lQ, lone, lp, specialA_, cache_buf, cache_t, la, mode_, rp);  
  store(bn_env, R, lP, 0);
}

__global__ void kernel_mcl_bn128_g1_reduce_sum_pre(
    cgbn_error_report_t* report, 
    Fp_model scalars,
    const size_t *index_it,
    uint32_t* counters, 
    char* flags,
    const int ranges_size, 
    const uint32_t* firsts,
    const uint32_t* seconds,
    Fp_model field_zero,
    Fp_model field_one,
    char *density,
    cgbn_mem_t<BITS>* bn_exponents,
    cgbn_mem_t<BITS>* field_modulus, const uint64_t field_inv
    ){
  int local_instance = threadIdx.x / TPI;//0~63
  int local_instances = 64;
  int instance = blockIdx.x * local_instances + local_instance;

  int range_offset = blockIdx.y * gridDim.x * local_instances;
  int first = firsts[blockIdx.y];
  int second = seconds[blockIdx.y];
  int reduce_depth = second - first;//30130

  context_t bn_context(cgbn_report_monitor, report, range_offset + instance);
  env_t          bn_env(bn_context.env<env_t>());  

  __shared__ uint32_t cache_res[64 * 24];
  uint32_t *res = &cache_res[local_instance * 24];
  __shared__ uint32_t cache_buffer[512];
  uint32_t *buffer = &cache_buffer[local_instance * 8];
  env_t::cgbn_t local_field_modulus;
  cgbn_load(bn_env, local_field_modulus, field_modulus);

  DevFp dev_field_zero, dev_field_one;
  dev_field_zero.load(bn_env, field_zero, 0);
  dev_field_one.load(bn_env, field_one, 0);
  int count = 0;
  for(int i = first + instance; i < first + reduce_depth; i+= gridDim.x * local_instances){
    const int j = index_it[i];
    DevFp scalar;
    scalar.load(bn_env, scalars, j);
    if(scalar.isequal(bn_env, dev_field_zero)){
    }
    else if(scalar.isequal(bn_env, dev_field_one)){
      flags[j] = 1;
    }
    else{
      const int group_thread = threadIdx.x & (TPI-1);
      if(group_thread == 0){
        density[i] = 1;
      }
      DevFp a = scalar.as_bigint(bn_env, res, buffer, local_field_modulus, field_inv);
      a.store(bn_env, bn_exponents, i);
      count += 1;
    }
  }
  __shared__ int cache_counters[64];
  const int group_thread = threadIdx.x & (TPI-1);
  if(group_thread == 0)
    cache_counters[local_instance] = count;
  __syncthreads();
  if(local_instance == 0){
    for(int i = 1; i < local_instances; i++){
      count += cache_counters[i];
    }
    if(group_thread == 0){
      counters[blockIdx.y * gridDim.x + blockIdx.x] = count;
    }
  }
}

template<int BlockInstances>
__global__ void kernel_mcl_bn128_g1_reduce_sum_one_range5(
    cgbn_error_report_t* report, 
    mcl_bn128_g1 values, 
    Fp_model scalars,
    const size_t *index_it,
    mcl_bn128_g1 partial, 
    const int ranges_size, 
    const int range_id_offset,
    const uint32_t* firsts,
    const uint32_t* seconds,
    char* flags,
    mcl_bn128_g1 t_zero,
    Fp_model one, 
    Fp_model p, 
    Fp_model a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
  const int local_instance = threadIdx.x / TPI;//0~63
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int instance = tid / TPI;

  const int instance_offset = (range_id_offset + blockIdx.y) * gridDim.x * BlockInstances;
  const int first = firsts[range_id_offset + blockIdx.y];
  const int second = seconds[range_id_offset + blockIdx.y];
  const int reduce_depth = second - first;//30130
  if(reduce_depth <= 1) return;
  const int half_depth = (reduce_depth + 1) / 2;

  context_t bn_context(cgbn_report_monitor, report, instance_offset + instance);
  env_t          bn_env(bn_context.env<env_t>());  
  if(instance >= half_depth) return;
  __shared__ uint32_t cache_buf[BlockInstances*(N_32*2+2)], cache_t[BlockInstances * N_32];

  MclFp lone, lp, la;
  load(bn_env, lone, one, 0); 
  load(bn_env, la, a, 0); 
  load(bn_env, lp, p, 0); 
  lp.ptr = (uint32_t*)p.mont_repr_data;

  DevEct result;
  if(flags[index_it[first + instance]] == 1){
	  load(bn_env, result, values, first+instance);
  }else{
	  load(bn_env, result, t_zero, 0);
  }
  for(int i = first + instance+half_depth; i < first + reduce_depth; i+= half_depth){
    const int j = index_it[i];
    if(flags[j] == 1){
      DevEct dev_b;
      load(bn_env, dev_b, values, i);
      add(bn_env, result, result, dev_b, lone, lp, specialA_, cache_buf + local_instance * (N_32 * 2 + 2), cache_t + local_instance * N_32, la, mode_, rp);  
    }
  }
  store(bn_env, partial, result, first + instance);
}

template<int BlockInstances>
__global__ void kernel_mcl_bn128_g1_reduce_sum_one_range7(
    cgbn_error_report_t* report, 
    mcl_bn128_g1 values, 
    Fp_model scalars,
    const size_t *index_it,
    mcl_bn128_g1 partial, 
    const int ranges_size, 
    const int range_id_offset,
    const uint32_t* firsts,
    const uint32_t* seconds,
    char* flags,
    mcl_bn128_g1 t_zero,
    Fp_model one, 
    Fp_model p, 
    Fp_model a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
  const int local_instance = threadIdx.x / TPI;//0~63
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int instance = tid / TPI;

  const int instance_offset = (range_id_offset + blockIdx.y) * gridDim.x * BlockInstances;
  const int first = firsts[range_id_offset + blockIdx.y];
  const int second = seconds[range_id_offset + blockIdx.y];
  const int reduce_depth = second - first;//30130
  if(reduce_depth <= 1) return;
  const int half_depth = (reduce_depth + 1) / 2;

  context_t bn_context(cgbn_report_monitor, report, instance_offset + instance);
  env_t          bn_env(bn_context.env<env_t>());  
  if(instance >= half_depth) return;

  __shared__ uint32_t cache_buf[BlockInstances*(N_32*2+2)], cache_t[BlockInstances * N_32];

  MclFp lone, lp, la;
  load(bn_env, lone, one, 0); 
  load(bn_env, la, a, 0); 
  load(bn_env, lp, p, 0); 
  lp.ptr = (uint32_t*)p.mont_repr_data;

  DevEct result;
  load(bn_env, result, values, first+instance);
  for(int i = first + instance+half_depth; i < first + reduce_depth; i+= half_depth){
      DevEct dev_b;
      load(bn_env, dev_b, values, i);
      add(bn_env, result, result, dev_b, lone, lp, specialA_, cache_buf + local_instance * (N_32*2+2), cache_t + local_instance * N_32, la, mode_, rp);  
      //add(bn_env, result, result, dev_b, lone, lp, specialA_, cache, cache_buf, cache_t, la, mode_, rp);  
  }
  store(bn_env, partial, result, first + instance);
}

__global__ void kernel_mcl_update_seconds(const uint32_t *firsts, uint32_t* seconds, const int range_size){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < range_size){
		int first = firsts[tid];
		int second = seconds[tid];
		seconds[tid] = first + (second - first + 1) / 2;
	}
}

__global__ void kernel_mcl_bn128_g1_reduce_sum_one_range6(
    cgbn_error_report_t* report, 
    mcl_bn128_g1 partial, 
    const int n, 
    const uint32_t* firsts,
    Fp_model one, 
    Fp_model p, 
    Fp_model a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
  int instance = threadIdx.x / TPI;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  

  __shared__ uint32_t cache_buf[N_32*2+2], cache_t[N_32];

  MclFp lone, lp, la;
  load(bn_env, lone, one, 0); 
  load(bn_env, la, a, 0); 
  load(bn_env, lp, p, 0); 
  lp.ptr = (uint32_t*)p.mont_repr_data;

  DevEct result;
  load(bn_env, result, partial, firsts[0]);

  for(int i = 1; i < n; i++){
    DevEct dev_b;
    load(bn_env, dev_b, partial, firsts[i]);
    add(bn_env, result, result, dev_b, lone, lp, specialA_, cache_buf, cache_t, la, mode_, rp);  
  }
  store(bn_env, partial, result, 0);
}

int mcl_bn128_g1_reduce_sum(
    mcl_bn128_g1 values, 
    Fp_model scalars, 
    const size_t *index_it,
    mcl_bn128_g1 partial, 
    uint32_t *counters,
    char* flags,
    const uint32_t ranges_size,
    const uint32_t *firsts,
    uint32_t *seconds,
    mcl_bn128_g1 t_zero,
    Fp_model field_zero,
    Fp_model field_one,
    char *density,
    cgbn_mem_t<BITS>* bn_exponents,
    cgbn_mem_t<BITS>* field_modulus, const uint64_t field_inv,
    Fp_model one, Fp_model p, Fp_model a, const int specialA_, const int mode_, const uint64_t rp,
    const int max_reduce_depth, cudaStream_t stream
    ){
  cgbn_error_report_t *report = get_error_report();

  uint32_t threads = 512;
  const int local_instances = 64 * BlockDepth;
  uint32_t block_x =  (max_reduce_depth + local_instances - 1) / local_instances;
  dim3 blocks(block_x, ranges_size, 1);
  kernel_mcl_bn128_g1_reduce_sum_pre<<<blocks, threads, 0, stream>>>(report, scalars, index_it, counters, flags, ranges_size, firsts, seconds, field_zero, field_one, density, bn_exponents, field_modulus, field_inv);

  int n = max_reduce_depth;
  const int local_instances2 = 64;
  threads = local_instances2 * TPI;
  uint32_t block_x2 =  ((n+1)/2 + local_instances2 - 1) / local_instances2;
  dim3 blocks2(block_x2, ranges_size, 1);
  kernel_mcl_bn128_g1_reduce_sum_one_range5<local_instances2><<<blocks2, dim3(threads, 1, 1), 0, stream>>>(report, values, scalars, index_it, partial, ranges_size, 0, firsts, seconds, flags, t_zero, one, p, a, specialA_, mode_, rp);
  const int update_threads = 64;
  const int update_blocks = (ranges_size + update_threads - 1) / update_threads;
  kernel_mcl_update_seconds<<<update_blocks, update_threads, 0, stream>>>(firsts, seconds, ranges_size);
  //CUDA_CHECK(cudaDeviceSynchronize());
  n = (n+1)/2;
  while(n>=2){
	  uint32_t block_x2 =  ((n+1)/2 + local_instances2 - 1) / local_instances2;
	  dim3 blocks2(block_x2, ranges_size, 1);
	  kernel_mcl_bn128_g1_reduce_sum_one_range7<local_instances2><<<blocks2, dim3(threads, 1, 1), 0, stream>>>(report, partial, scalars, index_it, partial, ranges_size, 0, firsts, seconds, flags, t_zero, one, p, a, specialA_, mode_, rp);
	  //CUDA_CHECK(cudaDeviceSynchronize());
	  kernel_mcl_update_seconds<<<update_blocks, update_threads, 0, stream>>>(firsts, seconds, ranges_size);
	  //CUDA_CHECK(cudaDeviceSynchronize());
	  n = (n+1)/2;
  }
  kernel_mcl_bn128_g1_reduce_sum_one_range6<<<1, TPI, 0, stream>>>(report, partial, ranges_size, firsts, one, p, a, specialA_, mode_, rp);
  //CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

inline __device__ size_t dev_mcl_get_id(const size_t c, const size_t bitno, uint64_t* data){
  const uint64_t one = 1;
  const uint64_t mask = (one << c) - one;
  const size_t limb_num_bits = 64;//sizeof(mp_limb_t) * 8;

  const size_t part = bitno / limb_num_bits;
  const size_t bit = bitno % limb_num_bits;
  size_t id = (data[part] & (mask << bit)) >> bit;
  //const mp_limb_t next_data = (bit + c >= limb_num_bits && part < 3) ? bn_exponents[i].data[part+1] : 0;
  //id |= (next_data & (mask >> (limb_num_bits - bit))) << (limb_num_bits - bit);
  id |= (((bit + c >= limb_num_bits && part < 3) ? data[part+1] : 0) & (mask >> (limb_num_bits - bit))) << (limb_num_bits - bit);

  return id;
}

//with_density = false
__global__ void kernel_mcl_bucket_counter(
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    int* bucket_counters){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for(int i = tid; i < data_length; i+= gridDim.x * blockDim.x){
    size_t id = dev_mcl_get_id(c, k*c, (uint64_t*)bn_exponents[i]._limbs);
    if(id != 0){
      atomicAdd(&bucket_counters[id], 1);
    }
  }
}
//with_density = true
__global__ void kernel_mcl_bucket_counter(
    const char* density,
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    int* bucket_counters){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for(int i = tid; i < data_length; i+= gridDim.x * blockDim.x){
    if(density[i]){
      size_t id = dev_mcl_get_id(c, k*c, (uint64_t*)bn_exponents[i]._limbs);
      if(id != 0){
        atomicAdd(&bucket_counters[id], 1);
      }
    }
  }
}

void mcl_bucket_counter(
    const bool with_density,
    const char* density,
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    const int bucket_nums,
    int* bucket_counters,
    CudaStream stream){
  int threads = 512;
  int blocks = (data_length + threads-1) / threads;
  if(with_density){
    kernel_mcl_bucket_counter<<<blocks, threads, 0, stream>>>(density, bn_exponents, c, k, data_length, bucket_counters);
  }else{
    kernel_mcl_bucket_counter<<<blocks, threads, 0, stream>>>(bn_exponents, c, k, data_length, bucket_counters);
  }
  //CUDA_CHECK(cudaDeviceSynchronize());
}

void mcl_prefix_sum(const int *in, int *out, const int n, CudaStream stream){
  thrust::exclusive_scan(thrust::cuda::par.on(stream), in, in + n, out);
}

__global__ void kernel_mcl_get_bid_and_counter(
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    const int bucket_num,
    int* bucket_counters,
    int* bucket_ids, int* value_ids){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  for(int i = tid; i < data_length; i+= gridDim.x * blockDim.x){
    size_t id = dev_mcl_get_id(c, k*c, (uint64_t*)bn_exponents[i]._limbs);
    if(id >= bucket_num) printf("error bucket_id\n");
    if(id != 0){
      atomicAdd(&bucket_counters[id], 1);
      bucket_ids[i] = id;
      value_ids[i] = i;
    }else{
        bucket_ids[i] = bucket_num+1;
        value_ids[i] = i;
    }
  }
}

__global__ void kernel_mcl_split_to_bucket(
        cgbn_error_report_t* report, 
		mcl_bn128_g1 data,
		mcl_bn128_g1 buckets,
		const int data_length,
		const int bucket_num,
		const int* starts,
		const int* value_ids,
		const int* bucket_ids){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int instance = tid / TPI;
	if(instance >= data_length) return;
    context_t bn_context(cgbn_report_monitor, report, instance);
    env_t          bn_env(bn_context.env<env_t>());  
	int bucket_id = bucket_ids[instance];
	if(bucket_id > 0 && bucket_id < bucket_num){
		int src_i = value_ids[instance];
		int dst_i = instance;
        DevEct a;
        load(bn_env, a, data, src_i);
        store(bn_env, buckets, a, dst_i);
	}
}

void mcl_split_to_bucket(
    mcl_bn128_g1 data, 
    mcl_bn128_g1 out, 
    const bool with_density,
    const char* density,
    const cgbn_mem_t<BITS>* bn_exponents,
    const int c, const int k,
    const int data_length,
    int *starts,
    int *indexs, 
    int* tmp,
    CudaStream stream){
  int threads = 512;
  int blocks = (data_length + threads-1) / threads;
  const int bucket_num = (1<<c);
  int *bucket_ids = tmp, *value_ids = tmp + data_length;
  kernel_mcl_get_bid_and_counter<<<blocks, threads, 0, stream>>>(bn_exponents, c, k, data_length, bucket_num, indexs, bucket_ids, value_ids); 
  thrust::sort_by_key(thrust::cuda::par.on(stream), bucket_ids, bucket_ids + data_length, value_ids); 

  cgbn_error_report_t *report = get_error_report();
  blocks = (data_length + 63) / 64;
  kernel_mcl_split_to_bucket<<<blocks, threads, 0, stream>>>(report, data, out, data_length, bucket_num, starts, value_ids, bucket_ids);
}

__global__ void kernel_mcl_calc_bucket_half_size(const int *starts, const int *ends, int* sizes, const int bucket_num){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < bucket_num){
        sizes[tid] = (ends[tid] - starts[tid])/2;
    }
}

__global__ void kernel_mcl_get_bucket_tids(const int* half_sizes, const int bucket_num, int* bucket_tids, int*bucket_ids){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < bucket_num){
        int start = 0;
        if(tid > 0) start = half_sizes[tid-1];
        for(int i = start; i < half_sizes[tid]; i++){
            bucket_tids[i] = i - start;
            bucket_ids[i] = tid;
        }
    }
}

template<int BlockInstances>
__global__ void kernel_mcl_bucket_reduce_g1(
    cgbn_error_report_t* report, 
    mcl_bn128_g1 data,
    const int *starts, 
    const int *ends,
    const int *bucket_ids,
    const int *bucket_tids,
    const int total_instances,
    mcl_bn128_g1 t_zero,
    Fp_model one, 
    Fp_model p, 
    Fp_model a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int instance = tid / TPI;
  int local_instance = threadIdx.x / TPI;
  if(instance >= total_instances) return;
  int bucket_id = bucket_ids[instance];
  int start = starts[bucket_id];
  int bucket_size = ends[bucket_id] - start;
  if(bucket_size <= 1) return;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  

  __shared__ uint32_t cache_buf[BlockInstances*(N_32*2+2)], cache_t[BlockInstances * N_32];
  int half_bucket_size = bucket_size / 2;
  int bucket_instance = bucket_tids[instance];

  MclFp lone, lp, la;
  load(bn_env, lone, one, 0); 
  load(bn_env, la, a, 0); 
  load(bn_env, lp, p, 0); 
  lp.ptr = (uint32_t*)p.mont_repr_data;

  DevEct result;
  load(bn_env, result, data, start + bucket_instance);
  for(int i = bucket_instance + half_bucket_size; i < bucket_size; i+= half_bucket_size){
      DevEct other;
      load(bn_env, other, data, start + i);
      add(bn_env, result, result, other, lone, lp, specialA_, cache_buf + local_instance * (N_32 * 2 + 2), cache_t + local_instance * N_32, la, mode_, rp);  
  }
  store(bn_env, data, result, start + bucket_instance);
}

__global__ void kernel_mcl_update_ends2(
        const int *starts, 
        int* half_sizes, 
        int* ends, 
        const int bucket_num){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid < bucket_num){
    int start = starts[tid];
    int end = ends[tid];
    int half_bucket_size = (end-start)/2;
    if(end-start > 1) ends[tid] = start + half_bucket_size;
    half_sizes[tid] = half_bucket_size / 2;
  }
}

template<int Offset>
__global__ void kernel_mcl_copy(
    cgbn_error_report_t* report, 
    mcl_bn128_g1 data,
    int* starts, 
    int* ends, 
    mcl_bn128_g1 buckets,
    mcl_bn128_g1 zero,
    const int bucket_num){
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int instance = tid / TPI;
  if(instance >= bucket_num) return;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  int bid = instance;
  int start = starts[bid];
  int end = ends[bid];
  if(end - start == 0){
      DevEct dev_zero;
      load(bn_env, dev_zero, zero, 0);
      store(bn_env, buckets, dev_zero, bid * Offset);
  }else{
      DevEct a;
      load(bn_env, a, data, start);
      store(bn_env, buckets, a, bid * Offset);
  }
}

void mcl_bucket_reduce_sum(
    mcl_bn128_g1 data,
    int* starts, int* ends, int* ids,
    int *d_instance_bucket_ids,
    mcl_bn128_g1 buckets,
    const int bucket_num,
    const int data_size,
    mcl_bn128_g1 t_zero,
    Fp_model one, 
    Fp_model p, 
    Fp_model a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp,
    CudaStream stream){
  cgbn_error_report_t *report = get_error_report();
  int *half_sizes, *bucket_ids;
  int* bucket_tids = d_instance_bucket_ids;
  half_sizes = ids;
  bucket_ids = d_instance_bucket_ids + data_size;
  int threads = 256;
  int blocks = (bucket_num + threads-1) / threads;
  kernel_mcl_calc_bucket_half_size<<<blocks, threads, 0, stream>>>(starts, ends, half_sizes, bucket_num);
  //CUDA_CHECK(cudaDeviceSynchronize());
  while(1){
      thrust::inclusive_scan(thrust::cuda::par.on(stream), half_sizes, half_sizes + bucket_num, half_sizes);
      //CUDA_CHECK(cudaDeviceSynchronize());
      threads = 256;
      blocks = (bucket_num + threads-1) / threads;
      kernel_mcl_get_bucket_tids<<<blocks, threads>>>(half_sizes, bucket_num, bucket_tids, bucket_ids);
      //CUDA_CHECK(cudaDeviceSynchronize());
      int total_instances = 0;
      CUDA_CHECK(cudaMemcpyAsync(&total_instances, half_sizes + bucket_num-1, sizeof(int), cudaMemcpyDeviceToHost, stream)); 
      sync(stream); 
      if(total_instances == 0) break;
      const int local_instances = 64;
      threads = local_instances * TPI;
      blocks = (total_instances + local_instances - 1) / local_instances;
      kernel_mcl_bucket_reduce_g1<local_instances><<<blocks, threads, 0, stream>>>(report, data, starts, ends, bucket_ids, bucket_tids, total_instances, t_zero, one, p, a, specialA_, mode_, rp); 
      //CUDA_CHECK(cudaDeviceSynchronize());
      threads = 256;
      blocks = (bucket_num + threads-1) / threads;
      kernel_mcl_update_ends2<<<blocks, threads, 0, stream>>>(starts, half_sizes, ends, bucket_num);
      //CUDA_CHECK(cudaDeviceSynchronize());
  }
  threads = 512;
  int local_instances = 64;
  blocks = (bucket_num + local_instances-1) / local_instances;
  kernel_mcl_copy<BUCKET_INSTANCES><<<blocks, threads, 0, stream>>>(report, data, starts, ends, buckets, t_zero, bucket_num);
}

__global__ void kernel_mcl_reverse(
      cgbn_error_report_t* report, 
      mcl_bn128_g1 data, mcl_bn128_g1 out, int n, int offset){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int instance = tid / TPI;
  int local_instances = blockDim.x / TPI;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  for(int i = instance; i < n; i += gridDim.x * local_instances){
    int in_i = i * offset;
    int out_i = n - i - 1;
    DevEct a;
    load(bn_env, a, data, in_i);
    store(bn_env, out, a, out_i);
  }
}

void mcl_reverse(mcl_bn128_g1 in, mcl_bn128_g1 out, const int n, const int offset, CudaStream stream){
  const int threads = 512;
  cgbn_error_report_t *report = get_error_report();
  int reverse_blocks = (n + 63) / 64;
  kernel_mcl_reverse<<<reverse_blocks, threads, 0, stream>>>(report, in, out, n, offset);
  //CUDA_CHECK(cudaDeviceSynchronize());
}

template<int BlockInstances, int ReduceDepthPerBlock>
__global__ void kernel_mcl_prefix_sum_pre(
      cgbn_error_report_t* report, 
      mcl_bn128_g1 data, 
      const int n,
      int stride,
      Fp_model one, 
      Fp_model p, 
      Fp_model a, 
      const int specialA_,
      const int mode_,
    const uint64_t rp){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int instance = tid / TPI;
  int local_instance = threadIdx.x / TPI;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  

  __shared__ uint32_t cache_buf[BlockInstances*(N_32*2+2)], cache_t[BlockInstances * N_32];
  MclFp lone, lp, la;
  load(bn_env, lone, one, 0); 
  load(bn_env, la, a, 0); 
  load(bn_env, lp, p, 0); 
  lp.ptr = (uint32_t*)p.mont_repr_data;

  int offset = blockIdx.x * ReduceDepthPerBlock;
  int index = (local_instance + 1) * stride * 2 - 1;
  if(index < ReduceDepthPerBlock && offset + index < n){
    DevEct dev_a, dev_b;
    load(bn_env, dev_a, data, offset + index);
    load(bn_env, dev_b, data, offset + index - stride);
    add(bn_env, dev_a, dev_a, dev_b, lone, lp, specialA_, cache_buf + local_instance * (N_32 * 2 + 2), cache_t + local_instance * N_32, la, mode_, rp);  
    store(bn_env, data, dev_a, offset + index);
  }
}

template<int BlockInstances, int ReduceDepthPerBlock>
__global__ void kernel_mcl_prefix_sum_post(
      cgbn_error_report_t* report, 
      mcl_bn128_g1 data, 
      mcl_bn128_g1 block_sums, 
      const int n,
      int stride, bool save_block_sum,
      Fp_model one, 
      Fp_model p, 
      Fp_model a, 
      const int specialA_,
      const int mode_,
      const uint64_t rp){
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int instance = tid / TPI;
  int local_instance = threadIdx.x / TPI;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  

  __shared__ uint32_t cache_buf[BlockInstances*(N_32*2+2)], cache_t[BlockInstances * N_32];
  MclFp lone, lp, la;
  load(bn_env, lone, one, 0); 
  load(bn_env, la, a, 0); 
  load(bn_env, lp, p, 0); 
  lp.ptr = (uint32_t*)p.mont_repr_data;

  int offset = blockIdx.x * ReduceDepthPerBlock;
  int index = (local_instance + 1) * stride * 2 - 1;
  if(index + stride < ReduceDepthPerBlock && offset + index + stride < n){
    DevEct dev_a, dev_b;
    load(bn_env, dev_a, data, offset + index + stride);
    load(bn_env, dev_b, data, offset + index);
    add(bn_env, dev_a, dev_a, dev_b, lone, lp, specialA_, cache_buf + local_instance * (N_32 * 2 + 2), cache_t + local_instance * N_32, la, mode_, rp);  
    store(bn_env, data, dev_a, offset + index + stride);
  }
  if(save_block_sum && local_instance == 0){
    DevEct dev_a;
    load(bn_env, dev_a, data, blockIdx.x * ReduceDepthPerBlock + ReduceDepthPerBlock - 1);
    store(bn_env, block_sums, dev_a, blockIdx.x);
  }
}

template<int Instances, int RealInstances, bool SaveBlockSum>
__global__ void kernel_mcl_prefix_sum(
    cgbn_error_report_t* report, 
    mcl_bn128_g1 data, 
    mcl_bn128_g1 block_sums, 
    const int n,
    Fp_model one, 
    Fp_model p, 
    Fp_model a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int instance = tid / TPI;
    int local_instance = threadIdx.x / TPI;
    int local_instances = Instances;
    if(instance >= n) return;

    context_t bn_context(cgbn_report_monitor, report, instance);
    env_t          bn_env(bn_context.env<env_t>());  
    __shared__ uint32_t cache_buf[RealInstances*(N_32*2+2)], cache_t[RealInstances * N_32];
    MclFp lone, lp, la;
    load(bn_env, lone, one, 0); 
    load(bn_env, la, a, 0); 
    load(bn_env, lp, p, 0); 
    lp.ptr = (uint32_t*)p.mont_repr_data;

    int offset = blockIdx.x * local_instances;
    for(int stride = 1; stride <= RealInstances; stride *= 2){
        __syncthreads();
        int index = (local_instance+1)*stride*2 - 1; 
        if(index < Instances && offset + index < n){
            DevEct dev_a, dev_b;
            load(bn_env, dev_a, data, offset + index);
            load(bn_env, dev_b, data, offset + index - stride);
            add(bn_env, dev_a, dev_a, dev_b, lone, lp, specialA_, cache_buf + local_instance * (N_32 * 2 + 2), cache_t + local_instance * N_32, la, mode_, rp);  
            store(bn_env, data, dev_a, offset + index);
        }
        __syncthreads();
    }
    for (unsigned int stride = (Instances >> 1); stride > 0 ; stride>>=1) {
        __syncthreads();
        int index = (local_instance+1)*stride*2 - 1;
        if(index + stride < Instances && offset + index + stride < n){
            DevEct dev_a, dev_b;
            load(bn_env, dev_a, data, offset + index + stride);
            load(bn_env, dev_b, data, offset + index);
            add(bn_env, dev_a, dev_a, dev_b, lone, lp, specialA_, cache_buf + local_instance * (N_32 * 2 + 2), cache_t + local_instance * N_32, la, mode_, rp);  
            store(bn_env, data, dev_a, offset + index + stride);
        }
    }
    __syncthreads();
    if(SaveBlockSum && local_instance == 0){
        DevEct dev_a;
        load(bn_env, dev_a, data, blockIdx.x * local_instances + local_instances-1);
        store(bn_env, block_sums, dev_a, blockIdx.x);
    }
}

template<int BlockInstances>
__global__ void kernel_mcl_add_block_sum(
    cgbn_error_report_t* report, 
    mcl_bn128_g1 data, 
    mcl_bn128_g1 block_sums, 
    const int n,
    Fp_model one, 
    Fp_model p, 
    Fp_model a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
  const int instances = BlockInstances;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int instance = i / TPI;
  int local_instance = threadIdx.x / TPI;
  if(instances + instance >= n) return;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  __shared__ uint32_t cache_buf[BlockInstances*(N_32*2+2)], cache_t[BlockInstances * N_32];
  MclFp lone, lp, la;
  load(bn_env, lone, one, 0); 
  load(bn_env, la, a, 0); 
  load(bn_env, lp, p, 0); 
  lp.ptr = (uint32_t*)p.mont_repr_data;

  DevEct dev_block_sum, dev_a;
  load(bn_env, dev_block_sum, block_sums, blockIdx.x);
  load(bn_env, dev_a, data, instance + instances);//offset = instances
  add(bn_env, dev_a, dev_a, dev_block_sum, lone, lp, specialA_, cache_buf + local_instance * (N_32 * 2 + 2), cache_t + local_instance * N_32, la, mode_, rp);  
  store(bn_env, data, dev_a, instance + instances);
}


void mcl_prefix_sum(
    mcl_bn128_g1 data, 
    mcl_bn128_g1 block_sums, 
    mcl_bn128_g1 block_sums2, 
    const int n,//2^16
    Fp_model one, 
    Fp_model p, 
    Fp_model a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp,
    CudaStream stream){
  cgbn_error_report_t *report = get_error_report();
  const int threads = 512;
  int instances = threads / TPI;//64
  int prefix_sum_blocks = (n + instances - 1) / instances;//2^10
  int prefix_sum_blocks2 = (prefix_sum_blocks + instances-1) / instances;//2^4

  for(int stride = 1; stride <= 32; stride *= 2){
    int instances = 32 / stride;
    int threads = instances * TPI;
    kernel_mcl_prefix_sum_pre<32, 64><<<prefix_sum_blocks, threads, 0, stream>>>(report, data, n, stride, one, p, a, specialA_, mode_, rp);
  }
  for(int stride = 32; stride > 0; stride /= 2){
    int instances = 32 / stride;
    int threads = instances * TPI;
    bool save_block_sum = (stride == 1);
    kernel_mcl_prefix_sum_post<32, 64><<<prefix_sum_blocks, threads, 0, stream>>>(report, data, block_sums, n, stride, save_block_sum, one, p, a, specialA_, mode_, rp);
  }

  for(int stride = 1; stride <= 32; stride *= 2){
    int instances = 32 / stride;
    int threads = instances * TPI;
    kernel_mcl_prefix_sum_pre<32, 64><<<prefix_sum_blocks2, threads, 0, stream>>>(report, block_sums, prefix_sum_blocks, stride, one, p, a, specialA_, mode_, rp);
  }
  for(int stride = 32; stride > 0; stride /= 2){
    int instances = 32 / stride;
    int threads = instances * TPI;
    bool save_block_sum = (stride == 1);
    kernel_mcl_prefix_sum_post<32, 64><<<prefix_sum_blocks2, threads, 0, stream>>>(report, block_sums, block_sums2, prefix_sum_blocks, stride, save_block_sum, one, p, a, specialA_, mode_, rp);
  }
  
  kernel_mcl_prefix_sum<16, 8, false><<<1, 128/2, 0, stream>>>(report, block_sums2, block_sums2, prefix_sum_blocks2, one, p, a, specialA_, mode_, rp);
  kernel_mcl_add_block_sum<64><<<prefix_sum_blocks2-1, threads, 0, stream>>>(report, block_sums, block_sums2, prefix_sum_blocks, one, p, a, specialA_, mode_, rp);
  kernel_mcl_add_block_sum<64><<<prefix_sum_blocks-1, threads, 0, stream>>>(report, data, block_sums, n, one, p, a, specialA_, mode_, rp);
  //CUDA_CHECK(cudaDeviceSynchronize());
}

template<int BlockInstances>
__global__ void kernel_mcl_reduce_sum(
    cgbn_error_report_t* report, 
    mcl_bn128_g1 data, 
    mcl_bn128_g1 out, 
    const int half_n,
    const int n,
    Fp_model one, 
    Fp_model p, 
    Fp_model a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int instance = tid / TPI;
  if(instance >= half_n) return;
  int local_instance = threadIdx.x / TPI;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  
  __shared__ uint32_t cache_buf[BlockInstances*(N_32*2+2)], cache_t[BlockInstances * N_32];
  MclFp lone, lp, la;
  load(bn_env, lone, one, 0); 
  load(bn_env, la, a, 0); 
  load(bn_env, lp, p, 0); 
  lp.ptr = (uint32_t*)p.mont_repr_data;

  DevEct dev_a;
  load(bn_env, dev_a, data, instance);
  for(int i = instance + half_n; i < n; i+= half_n){
    DevEct dev_b;
    load(bn_env, dev_b, data, i);
    add(bn_env, dev_a, dev_a, dev_b, lone, lp, specialA_, cache_buf + local_instance * (N_32 * 2 + 2), cache_t + local_instance * N_32, la, mode_, rp);  
  }
  store(bn_env, out, dev_a, instance);
}

void mcl_bn128_g1_reduce_sum2(
    mcl_bn128_g1 data, 
    mcl_bn128_g1 out, 
    const uint32_t n,
    Fp_model one, 
    Fp_model p, 
    Fp_model a, 
    const int specialA_,
    const int mode_,
    const uint64_t rp,
    CudaStream stream){
  cgbn_error_report_t *report = get_error_report();
  int len = n-1;
  const int instances = 64;
  int threads = instances * TPI;
  int half_len = (len + 1) / 2;
  int blocks = (half_len + instances - 1) / instances;
  kernel_mcl_reduce_sum<instances><<<blocks, threads, 0, stream>>>(report, data, out, half_len, len, one, p, a, specialA_, mode_, rp);
  len = half_len;
  while(len > 1){
      int half_len = (len + 1) / 2;
      int blocks = (half_len + instances - 1) / instances;
      kernel_mcl_reduce_sum<instances><<<blocks, threads, 0, stream>>>(report, out, out, half_len, len, one, p, a, specialA_, mode_, rp);
      len = half_len;
  }
}

}// namespace gpu

#endif
