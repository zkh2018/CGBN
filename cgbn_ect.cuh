#ifndef CGBN_ECT_CUH
#define CGBN_ECT_CUH

#include "cgbn_ect.h"
#include "bigint_256.cuh"

namespace gpu{

using namespace BigInt256;

inline __device__ void load(BigInt256::Int* dst, const Fp_model& src, const int offset){
    memcpy(dst, src.mont_repr_data + offset, sizeof(BigInt256::Int256));
}

inline __device__ void load(BigInt256::Ect& dst, const mcl_bn128_g1& src, const int offset){
    memcpy(dst.x, src.x.mont_repr_data + offset, sizeof(BigInt256::Int256));
    memcpy(dst.y, src.y.mont_repr_data + offset, sizeof(BigInt256::Int256));
    memcpy(dst.z, src.z.mont_repr_data + offset, sizeof(BigInt256::Int256));
}

inline __device__ void store(mcl_bn128_g1& dst, const BigInt256::Ect& src, const int offset){
    memcpy(dst.x.mont_repr_data + offset, src.x, sizeof(BigInt256::Int256));
    memcpy(dst.y.mont_repr_data + offset, src.y, sizeof(BigInt256::Int256));
    memcpy(dst.z.mont_repr_data + offset, src.z, sizeof(BigInt256::Int256));
}

inline __device__ void load(BigInt256::Ect2& dst, const mcl_bn128_g2& src, const int offset){
    memcpy(dst.x.c0, src.x.c0.mont_repr_data + offset, sizeof(BigInt256::Int256));
    memcpy(dst.x.c1, src.x.c1.mont_repr_data + offset, sizeof(BigInt256::Int256));
    memcpy(dst.y.c0, src.y.c0.mont_repr_data + offset, sizeof(BigInt256::Int256));
    memcpy(dst.y.c1, src.y.c1.mont_repr_data + offset, sizeof(BigInt256::Int256));
    memcpy(dst.z.c0, src.z.c0.mont_repr_data + offset, sizeof(BigInt256::Int256));
    memcpy(dst.z.c1, src.z.c1.mont_repr_data + offset, sizeof(BigInt256::Int256));
}

inline __device__ void store(mcl_bn128_g2& dst, const BigInt256::Ect2& src, const int offset){
    memcpy(dst.x.c0.mont_repr_data + offset, src.x.c0, sizeof(BigInt256::Int256));
    memcpy(dst.x.c1.mont_repr_data + offset, src.x.c1, sizeof(BigInt256::Int256));
    memcpy(dst.y.c0.mont_repr_data + offset, src.y.c0, sizeof(BigInt256::Int256));
    memcpy(dst.y.c1.mont_repr_data + offset, src.y.c1, sizeof(BigInt256::Int256));
    memcpy(dst.z.c0.mont_repr_data + offset, src.z.c0, sizeof(BigInt256::Int256));
    memcpy(dst.z.c1.mont_repr_data + offset, src.z.c1, sizeof(BigInt256::Int256));
}



} // namespace gpu

#endif
