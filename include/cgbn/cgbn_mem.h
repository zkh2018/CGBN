#ifndef CGBN_MEM_H
#define CGBN_MEM_H

#include <stdint.h>

template<uint32_t bits>
struct cgbn_mem_t {
  public:
  //uint32_t _limbs[(bits+31)/32];
  uint32_t _limbs[8];
};

#endif
