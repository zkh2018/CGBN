#ifndef CGBN_ECT_H
#define CGBN_ECT_H

#include "cgbn_alt_bn128_g1.h"
#include "cgbn_alt_bn128_g2.h"
namespace gpu{

struct mcl_bn128_g1 : public alt_bn128_g1{

};

struct mcl_bn128_g2 : public alt_bn128_g2{

};

} // namespace gpu

#endif
