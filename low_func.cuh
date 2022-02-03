#ifndef LOW_FUNC_CUH
#define LOW_FUNC_CUH

#include "cgbn_fp.h"
#include "cgbn_alt_bn128_g1.cuh"
#include "cgbn_ect.h"
#include <stdint.h>

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
        uint32_t* p, uint32_t *buf, uint32_t *t, uint32_t* cache, const uint64_t rp){
    int group_id = threadIdx.x & (TPI-1);
    uint32_t *ty = t + 2*N_32 + 2;
    cgbn_store(bn_env, ty, ly);
    uint64_t* p64_y = (uint64_t*)ty;
    //uint64_t* p64_y = (uint64_t*)y;
    uint64_t* p64_p = (uint64_t*)p;
    uint64_t* p64_t = (uint64_t*)t;
    //uint64_t rp = p64_p[0];
    uint64_t* c = (uint64_t*)buf;
    env_t::cgbn_t lc;
    env_t::cgbn_wide_t lwc, lwt;
    cache[group_id] = 0;
    ((uint64_t*)cache)[0] = p64_y[0];
    env_t::cgbn_t lcache;
    cgbn_load(bn_env, lcache, cache);
    cgbn_mul_wide(bn_env, lwc, lx, lcache);
    cgbn_store(bn_env, buf, lwc._low);
    uint64_t q = c[0] * rp;
    if(group_id == 0){
        ((uint64_t*)cache)[0] = q;
    }
    cgbn_load(bn_env, lcache, cache);
    cgbn_mul_wide(bn_env, lwt, lp, lcache);
    cgbn_store(bn_env, t, lwt._low);
    cgbn_store(bn_env, t + N_32, lwt._high);
    int32_t carry = cgbn_add(bn_env, lc, lwc._low, lwt._low);
    cgbn_store(bn_env, buf, lc);
    cgbn_store(bn_env, buf + N_32, lwc._high);
    if(group_id == 0){
        c[N_64] += p64_t[N_64] + carry; 
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
        if(group_id == 0){
            ((uint64_t*)cache)[0] = p64_y[i];
        }
        cgbn_load(bn_env, lcache, cache);
        cgbn_mul_wide(bn_env, lwt, lx, lcache);
        cgbn_store(bn_env, t, lwt._low);
        cgbn_store(bn_env, t + N_32, lwt._high);
        carry = cgbn_add(bn_env, lc, lc, lwt._low);
        cgbn_store(bn_env, (uint32_t*)c, lc);
        if(group_id == 0){
            c[N_64] += p64_t[N_64] + carry;
        }
        q = c[0] * rp;
        if(group_id == 0){
            ((uint64_t*)cache)[0] = q;
        }
        cgbn_load(bn_env, lcache, cache);
        cgbn_mul_wide(bn_env, lwt, lp, lcache);
        cgbn_store(bn_env, t, lwt._low);
        cgbn_store(bn_env, t + N_32, lwt._high);
        cgbn_load(bn_env, lc, (uint32_t*)c);
        carry = cgbn_add(bn_env, lc, lc, lwt._low);
        cgbn_store(bn_env, (uint32_t*)c, lc);
        if(group_id == 0){
            c[N_64] += p64_t[N_64] + carry;
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
inline __device__ void dev_mcl_mul(env_t& bn_env, uint32_t* z, uint32_t* x, uint32_t* y, uint32_t* p, uint32_t *buf, uint32_t *t, uint32_t* cache, const uint64_t rp){
    env_t::cgbn_t lx, ly, lp, lz;
    cgbn_load(bn_env, lx, x);
    cgbn_load(bn_env, ly, y);
    cgbn_load(bn_env, lp, p);

    dev_mcl_mul(bn_env, lz, lx, ly, lp, p, buf, t, cache, rp); 
    cgbn_store(bn_env, z, lz);
}

inline __device__ void dev_mcl_sqr(env_t& bn_env, 
        env_t::cgbn_t& lz, env_t::cgbn_t& lx, env_t::cgbn_t& lp, 
        uint32_t* p, uint32_t *buf, uint32_t *t, uint32_t* cache, const uint64_t rp){
    dev_mcl_mul(bn_env, lz, lx, lx, lp, p, buf, t, cache, rp);
    //cgbn_store(bn_env, z, lz);
}
inline __device__ void dev_mcl_sqr(env_t& bn_env, uint32_t* z, uint32_t* x, uint32_t* p, uint32_t *buf, uint32_t *t, uint32_t* cache, const uint64_t rp){
    env_t::cgbn_t lx, lp, lz;
    cgbn_load(bn_env, lx, x);
    cgbn_load(bn_env, lp, p);

    dev_mcl_mul(bn_env, lz, lx, lx, lp, p, buf, t, cache, rp); 
    cgbn_store(bn_env, z, lz);
}

__global__ void kernel_mcl_mul(
    cgbn_error_report_t* report, 
    uint32_t* z, uint32_t*x, uint32_t*y, uint32_t* p, const uint64_t rp){
  context_t bn_context(cgbn_report_monitor, report, 0);
  env_t          bn_env(bn_context.env<env_t>());  
  __shared__ uint32_t cache[N_32], cache_buf[N_32*2+2], cache_t[N_32*3+2];
  dev_mcl_mul(bn_env, z, x, y, p, cache_buf, cache_t, cache, rp);
}

struct MclFp: public DevFp {
    uint32_t *ptr;
};

struct DevEct : public DevAltBn128G1 {

inline __device__ bool is_zero(env_t& bn_env){
    return dev_is_zero(bn_env, this->z.mont);
}
    
inline __device__ void dev_dblNoVerifyInfJacobi(env_t& bn_env, DevEct& P, MclFp& one, MclFp& p, const int specialA_, uint32_t *cache, uint32_t* cache_buf, uint32_t *cache_t, MclFp& a_, const uint64_t rp){
    MclFp S, M, t, y2;
    //Fp::sqr(y2, P.y);
    dev_mcl_sqr(bn_env, y2.mont, P.y.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp); 
    //Fp::mul(S, P.x, y2);
    dev_mcl_mul(bn_env, S.mont, P.x.mont, y2.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
    //const bool isPzOne = P.z.isOne();
    const bool isPzOne = dev_is_one(bn_env, P.z.mont, one.mont);
    //S += S;
    dev_mcl_add(bn_env, S.mont, S.mont, S.mont, p.mont, p.ptr, cache);
    //S += S;
    dev_mcl_add(bn_env, S.mont, S.mont, S.mont, p.mont, p.ptr, cache);
    //Fp::sqr(M, P.x);
    dev_mcl_sqr(bn_env, M.mont, P.x.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp); 
    switch (specialA_) {
        case 0:
            //Fp::add(t, M, M);
            dev_mcl_add(bn_env, t.mont, M.mont, M.mont, p.mont, p.ptr, cache);
            //M += t;
            dev_mcl_add(bn_env, M.mont, M.mont, t.mont, p.mont, p.ptr, cache);
            break;
        case 1:
            if (isPzOne) {
                //M -= P.z;
                dev_mcl_sub(bn_env, M.mont, M.mont, P.z.mont, p.mont);
            } else {
                //Fp::sqr(t, P.z);
                dev_mcl_sqr(bn_env, t.mont, P.z.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
                //Fp::sqr(t, t);
                dev_mcl_sqr(bn_env, t.mont, t.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
                //M -= t;
                dev_mcl_sub(bn_env, M.mont, M.mont, t.mont, p.mont);
            }
            //Fp::add(t, M, M);
            dev_mcl_add(bn_env, t.mont, M.mont, M.mont, p.mont, p.ptr, cache);
            //M += t;
            dev_mcl_add(bn_env, M.mont, M.mont, t.mont, p.mont, p.ptr, cache);
            break;
        case 2:
        default:
            if (isPzOne) {
                //t = a_;
                cgbn_set(bn_env, t.mont, a_.mont);
            } else {
                //Fp::sqr(t, P.z);
                dev_mcl_sqr(bn_env, t.mont, P.z.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
                //Fp::sqr(t, t);
                dev_mcl_sqr(bn_env, t.mont, t.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
                //t *= a_;
                dev_mcl_mul(bn_env, t.mont, t.mont, a_.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
            }
            //t += M;
            dev_mcl_add(bn_env, t.mont, t.mont, M.mont, p.mont, p.ptr, cache);
            //M += M;
            dev_mcl_add(bn_env, M.mont, M.mont, M.mont, p.mont, p.ptr, cache);
            //M += t;
            dev_mcl_add(bn_env, M.mont, M.mont, t.mont, p.mont, p.ptr, cache);
            break;
    }
	//Fp::sqr(R.x, M);
    dev_mcl_sqr(bn_env, this->x.mont, M.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp); 
	//R.x -= S;
    dev_mcl_sub(bn_env, this->x.mont, this->x.mont, S.mont, p.mont);
	//R.x -= S;
    dev_mcl_sub(bn_env, this->x.mont, this->x.mont, S.mont, p.mont);
	if (isPzOne) {
		//R.z = P.y;
        cgbn_set(bn_env, this->z.mont, P.y.mont);
	} else {
		//Fp::mul(R.z, P.y, P.z);
        dev_mcl_mul(bn_env, this->z.mont, P.y.mont, P.z.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
	}
	//R.z += R.z;
    dev_mcl_add(bn_env, this->z.mont, this->z.mont, this->z.mont, p.mont, p.ptr, cache);
	//Fp::sqr(y2, y2);
    dev_mcl_sqr(bn_env, y2.mont, y2.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp); 
	//y2 += y2;
    dev_mcl_add(bn_env, y2.mont, y2.mont, y2.mont, p.mont, p.ptr, cache);
	//y2 += y2;
    dev_mcl_add(bn_env, y2.mont, y2.mont, y2.mont, p.mont, p.ptr, cache);
	//y2 += y2;
    dev_mcl_add(bn_env, y2.mont, y2.mont, y2.mont, p.mont, p.ptr, cache);
	//Fp::sub(R.y, S, R.x);
    dev_mcl_sub(bn_env, this->y.mont, S.mont, this->x.mont, p.mont);
	//R.y *= M;
    dev_mcl_mul(bn_env, this->y.mont, this->y.mont, M.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
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
        MclFp& one, MclFp& p, const int specialA_, uint32_t* cache, uint32_t* cache_buf, uint32_t* cache_t, MclFp& a_, const int mode_, const uint64_t rp){
    MclFp r, U1, S1, H, H3;
    if (isPzOne) {
        // r = 1;
    } else {
        //Fp::sqr(r, P.z);
        dev_mcl_sqr(bn_env, r.mont, P.z.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
    }
    if (isQzOne) {
        //U1 = P.x;
        U1.set(bn_env, P.x);
        if (isPzOne) {
            //H = Q.x;
            H.set(bn_env, Q.x);
        } else {
            //Fp::mul(H, Q.x, r);
            dev_mcl_mul(bn_env, H.mont, Q.x.mont, r.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
        }
        //H -= U1;
        dev_mcl_sub(bn_env, H.mont, H.mont, U1.mont, p.mont);
        //S1 = P.y;
        S1.set(bn_env, P.y);
    } else {
        //Fp::sqr(S1, Q.z);
        dev_mcl_sqr(bn_env, S1.mont, Q.z.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
        //Fp::mul(U1, P.x, S1);
        dev_mcl_mul(bn_env, U1.mont, P.x.mont, S1.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
        if (isPzOne) {
            //H = Q.x;
            H.set(bn_env, Q.x);
        } else {
            //Fp::mul(H, Q.x, r);
            dev_mcl_mul(bn_env, H.mont, Q.x.mont, r.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
        }
        //H -= U1;
        dev_mcl_sub(bn_env, H.mont, H.mont, U1.mont, p.mont);
        //S1 *= Q.z;
        dev_mcl_mul(bn_env, S1.mont, S1.mont, Q.z.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
        //S1 *= P.y;
        dev_mcl_mul(bn_env, S1.mont, S1.mont, P.y.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
    }
    if (isPzOne) {
        //r = Q.y;
        r.set(bn_env, Q.y);
    } else {
        //r *= P.z;
        dev_mcl_mul(bn_env, r.mont, r.mont, P.z.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
        //r *= Q.y;
        dev_mcl_mul(bn_env, r.mont, r.mont, Q.y.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
    }
    //r -= S1;
    dev_mcl_sub(bn_env, r.mont, r.mont, S1.mont, p.mont);
    if (H.is_zero(bn_env)) {
        if (r.is_zero(bn_env)) {
            R.dev_dblNoVerifyInfJacobi(bn_env, P, one, p, specialA_, cache, cache_buf, cache_t, a_, rp);
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
            dev_mcl_mul(bn_env, R.z.mont, H.mont, Q.z.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
        }
    } else {
        if (isQzOne) {
            //Fp::mul(R.z, P.z, H);
            dev_mcl_mul(bn_env, R.z.mont, P.z.mont, H.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
        } else {
            //Fp::mul(R.z, P.z, Q.z);
            dev_mcl_mul(bn_env, R.z.mont, P.z.mont, Q.z.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
            //R.z *= H;
            dev_mcl_mul(bn_env, R.z.mont, R.z.mont, H.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
        }
    }
    //Fp::sqr(H3, H); // H^2
    dev_mcl_sqr(bn_env, H3.mont, H.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
    //Fp::sqr(R.y, r); // r^2
    dev_mcl_sqr(bn_env, R.y.mont, r.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
    ///U1 *= H3; // U1 H^2
    dev_mcl_mul(bn_env, U1.mont, U1.mont, H3.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
    //H3 *= H; // H^3
    dev_mcl_mul(bn_env, H3.mont, H3.mont, H.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
    //R.y -= U1;
    dev_mcl_sub(bn_env, R.y.mont, R.y.mont, U1.mont, p.mont);
    //R.y -= U1;
    dev_mcl_sub(bn_env, R.y.mont, R.y.mont, U1.mont, p.mont);
    //Fp::sub(R.x, R.y, H3);
    dev_mcl_sub(bn_env, R.x.mont, R.y.mont, H3.mont, p.mont);
    //U1 -= R.x;
    dev_mcl_sub(bn_env, U1.mont, U1.mont, R.x.mont, p.mont);
    //U1 *= r;
    dev_mcl_mul(bn_env, U1.mont, U1.mont, r.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
    //H3 *= S1;
    dev_mcl_mul(bn_env, H3.mont, H3.mont, S1.mont, p.mont, p.ptr, cache_buf, cache_t, cache, rp);
    //Fp::sub(R.y, U1, H3);
    dev_mcl_sub(bn_env, R.y.mont, U1.mont, H3.mont, p.mont);
}

inline __device__ void add(env_t& bn_env, DevEct& R, DevEct& P, DevEct& Q,
        MclFp& one, MclFp& p, const int specialA_, uint32_t* cache, uint32_t* cache_buf, uint32_t* cache_t, MclFp& a_, 
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
        R.dev_dblNoVerifyInfJacobi(bn_env, P, one, p, specialA_, cache, cache_buf, cache_t, a_, rp);
        return;
    }
    bool isPzOne = dev_is_one(bn_env, P.z.mont, one.mont);
    bool isQzOne = dev_is_one(bn_env, Q.z.mont, one.mont);
    //switch (mode_) {
        dev_addJacobi(bn_env, R, P, Q, isPzOne, isQzOne, one, p, specialA_, cache, cache_buf, cache_t, a_, mode_, rp);
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
    const int model_,
    const uint64_t rp){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int instance = tid / TPI;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());  

  __shared__ uint32_t cache[N_32], cache_buf[N_32*2+2], cache_t[N_32*3+2];
  DevEct lR, lP, lQ;
  MclFp lone, lp, la;
  load(bn_env, lP, P, 0);
  load(bn_env, lQ, Q, 0);

  load(bn_env, lone, one, 0); 
  load(bn_env, la, a, 0); 
  load(bn_env, lp, p, 0); 
  lp.ptr = (uint32_t*)p.mont_repr_data;

  add(bn_env, lR, lP, lQ, lone, lp, specialA_, cache, cache_buf, cache_t, la, model_, rp);  
  store(bn_env, R, lR, 0);
}


}

#endif
