#ifndef CGBN_ECT_H
#define CGBN_ECT_H

#include "cgbn_alt_bn128_g1.h"
#include "cgbn_alt_bn128_g2.h"
#include "bigint_256.cuh"

namespace gpu{

//const int BUCKET_INSTANCES = 64;

const int N_32 = BITS/32;//8;
const int N_64 = BITS/64;//4


struct mcl_bn128_g1 : public alt_bn128_g1{

};

struct mcl_bn128_g2 : public alt_bn128_g2{

};

} // namespace gpu

#endif
