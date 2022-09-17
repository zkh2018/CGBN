#ifndef BIGINT_256_CUH
#define BIGINT_256_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

namespace BigInt256 {

const int BITS = 256;
const int BASE_BITS = 64;
const int N = BITS / BASE_BITS;
typedef uint64_t Int;
typedef uint64_t Int256[N];

inline __device__ void printInt256(const Int* x, const char* desc){
    printf("%s:\n", desc);
    for(int i = 0; i < 4; i++){
        printf("%lu ", x[i]);
    }
    printf("\n");
}


inline __device__ int dev_is_zero(const Int* x){
    for(int i = 0; i < N; i++){
        if(x[i] != 0){
            return 0;
        }
    }
    return 1;
}

inline __device__ int dev_is_one(const Int* x, const Int* one){
    for(int i = 0; i < N; i++){
        if(x[i] != one[i]){
            return 0;
        }
    }
    return 1;
}

inline __device__ int dev_equal(const Int* a, const Int* b){
    for(int i = 0; i < N; i++){
        if(a[i] != b[i]){
            return 0;
        }
    }
    return 1;
}

inline __device__ void dev_clear(Int* x){
    for(int i = 0; i < N; i++){
        x[i] = 0;
    }
}

inline __device__ void dev_clear_2(Int* x){
    for(int i = 0; i < N*2; i++){
        x[i] = 0;
    }
}

inline __device__ int dev_is_ge(const Int* a, const Int* b){
    for (int i = N-1; i >= 0; --i) {
        if (a[i] < b[i]) {
            return 0;
        } else if (a[i] > b[i]) {
            return 1;
        }
    }
    return 1;
}

inline __device__ Int dev_add(const Int* a, const Int* b, Int* c){
    Int carry_out = 0;
   asm(
      "add.cc.u64 %0, %5, %9;\n\t"
      "addc.cc.u64 %1, %6, %10;\n\t"
      "addc.cc.u64 %2, %7, %11;\n\t"
      "addc.cc.u64 %3, %8, %12;\n\t"
      "addc.u64 %4, 0, 0;\n\t"
      : "=l"(c[0]),
      "=l"(c[1]),
      "=l"(c[2]),
      "=l"(c[3]),
      "=l"(carry_out)
      : "l"(a[0]),
      "l"(a[1]),
      "l"(a[2]),
      "l"(a[3]),
      "l"(b[0]),
      "l"(b[1]),
      "l"(b[2]),
      "l"(b[3])
    );
    return carry_out;
}

inline __device__ Int dev_sub(const Int* a, const Int* b, Int* c){
    Int borrow = 0;
   asm(
      "sub.cc.u64 %0, %5, %9;\n\t"
      "subc.cc.u64 %1, %6, %10;\n\t"
      "subc.cc.u64 %2, %7, %11;\n\t"
      "subc.cc.u64 %3, %8, %12;\n\t"
      "addc.u64 %4, 0, 0;\n\t"
      : "=l"(c[0]),
      "=l"(c[1]),
      "=l"(c[2]),
      "=l"(c[3]),
      "=l"(borrow)
      : "l"(a[0]),
      "l"(a[1]),
      "l"(a[2]),
      "l"(a[3]),
      "l"(b[0]),
      "l"(b[1]),
      "l"(b[2]),
      "l"(b[3])
    );
    return 1-borrow;
}

//c[N*2] = a[N*2] - b[N*2]
inline __device__ Int dev_sub_wide(const Int* a, const Int* b, Int* c){
    Int borrow = 0;
   asm(
      "sub.cc.u64 %0, %9, %17;\n\t"
      "subc.cc.u64 %1, %10, %18;\n\t"
      "subc.cc.u64 %2, %11, %19;\n\t"
      "subc.cc.u64 %3, %12, %20;\n\t"
      "subc.cc.u64 %4, %13, %21;\n\t"
      "subc.cc.u64 %5, %14, %22;\n\t"
      "subc.cc.u64 %6, %15, %23;\n\t"
      "subc.cc.u64 %7, %16, %24;\n\t"
      "addc.u64 %8, 0, 0;\n\t"
      : "=l"(c[0]),
      "=l"(c[1]),
      "=l"(c[2]),
      "=l"(c[3]),
      "=l"(c[4]),
      "=l"(c[5]),
      "=l"(c[6]),
      "=l"(c[7]),
      "=l"(borrow)
      : "l"(a[0]),
      "l"(a[1]),
      "l"(a[2]),
      "l"(a[3]),
      "l"(a[4]),
      "l"(a[5]),
      "l"(a[6]),
      "l"(a[7]),
      "l"(b[0]),
      "l"(b[1]),
      "l"(b[2]),
      "l"(b[3]),
      "l"(b[4]),
      "l"(b[5]),
      "l"(b[6]),
      "l"(b[7])
    );
    return 1-borrow;
}

inline __device__ void dev_mont_mul(uint64_t *wide_r, const uint64_t *modulus, const uint64_t inv){
    uint64_t k = wide_r[0] * inv;
    uint64_t carry = 0;
    asm(
      "{\n\t"
      ".reg .u64 c;\n\t"
      ".reg .u64 t;\n\t"
      ".reg .u64 nc;\n\t"

        //c = k * p[0] + r[0]
      "mad.lo.cc.u64 c, %10, %6, %0;\n\t"
      "madc.hi.cc.u64 c, %10, %6, 0;\n\t"

      // t = r[1] + c
      "addc.cc.u64 t, %1, c;\n\t"
      // nc = carry
      "addc.u64 nc, 0, 0;\n\t"
      // (r[1],c) = k * p[1] + (t, nc)
      "mad.lo.cc.u64 %1, %10, %7, t;\n\t"
      "madc.hi.cc.u64 c, %10, %7, nc;\n\t"

        // t = r[2] + c
      "addc.cc.u64 t, %2, c;\n\t"
      // nc = 0 + carry
      "addc.u64 nc, 0, 0;\n\t"
      // (r[2],c) = k * p[2] + (t, nc)
      "mad.lo.cc.u64 %2, %10, %8, t;\n\t"
      "madc.hi.cc.u64 c, %10, %8, nc;\n\t"

        // t = r[3] + c
      "addc.cc.u64 t, %3, c;\n\t"
      // nc = carry
      "addc.u64 nc, 0, 0;\n\t"
      //(r[3], c) = k * p[3] + (t, nc)
      "mad.lo.cc.u64 %3, %10, %9, t;\n\t"
      "madc.hi.cc.u64 c, %10, %9, nc;\n\t"

      "addc.cc.u64 %4, %4, c;\n\t"
      "addc.u64 %5, 0, 0;\n\t"
      "}"
      : "+l"(wide_r[0]),
      "+l"(wide_r[1]),
      "+l"(wide_r[2]),
      "+l"(wide_r[3]),
      "+l"(wide_r[4]),
      "=l"(carry)
      : "l"(modulus[0]),
      "l"(modulus[1]),
      "l"(modulus[2]),
      "l"(modulus[3]),
      "l"(k)
    );

    k = wide_r[1] * inv;

    asm(
      "{\n\t"
      ".reg .u64 c;\n\t"
      ".reg .u64 t;\n\t"
      ".reg .u64 nc;\n\t"

      "mad.lo.cc.u64 c, %10, %6, %0;\n\t"
      "madc.hi.cc.u64 c, %10, %6, 0;\n\t"
      
      "addc.cc.u64 t, %1, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %1, %10, %7, t;\n\t"
      "madc.hi.cc.u64 c, %10, %7, nc;\n\t"

      "addc.cc.u64 t, %2, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %2, %10, %8, t;\n\t"
      "madc.hi.cc.u64 c, %10, %8, nc;\n\t"

      "addc.cc.u64 t, %3, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %3, %10, %9, t;\n\t"
      "madc.hi.cc.u64 c, %10, %9, nc;\n\t"

      "addc.cc.u64 c, c, %5;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "addc.cc.u64 %4, %4, c;\n\t"
      "addc.u64 %5, nc, 0;\n\t"
      "}"
      : "+l"(wide_r[1]),
      "+l"(wide_r[2]),
      "+l"(wide_r[3]),
      "+l"(wide_r[4]),
      "+l"(wide_r[5]),
      "+l"(carry)
      : "l"(modulus[0]),
      "l"(modulus[1]),
      "l"(modulus[2]),
      "l"(modulus[3]),
      "l"(k)
    );

    k = wide_r[2] * inv;

    asm(
      "{\n\t"
      ".reg .u64 c;\n\t"
      ".reg .u64 t;\n\t"
      ".reg .u64 nc;\n\t"

      "mad.lo.cc.u64 c, %10, %6, %0;\n\t"
      "madc.hi.cc.u64 c, %10, %6, 0;\n\t"
      
      "addc.cc.u64 t, %1, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %1, %10, %7, t;\n\t"
      "madc.hi.cc.u64 c, %10, %7, nc;\n\t"

      "addc.cc.u64 t, %2, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %2, %10, %8, t;\n\t"
      "madc.hi.cc.u64 c, %10, %8, nc;\n\t"

      "addc.cc.u64 t, %3, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %3, %10, %9, t;\n\t"
      "madc.hi.cc.u64 c, %10, %9, nc;\n\t"

      "addc.cc.u64 c, c, %5;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "addc.cc.u64 %4, %4, c;\n\t"
      "addc.u64 %5, nc, 0;\n\t"
      "}"
      : "+l"(wide_r[2]),
      "+l"(wide_r[3]),
      "+l"(wide_r[4]),
      "+l"(wide_r[5]),
      "+l"(wide_r[6]),
      "+l"(carry)
      : "l"(modulus[0]),
      "l"(modulus[1]),
      "l"(modulus[2]),
      "l"(modulus[3]),
      "l"(k)
    );
    k = wide_r[3] * inv;

    asm(
      "{\n\t"
      ".reg .u64 c;\n\t"
      ".reg .u64 t;\n\t"
      ".reg .u64 nc;\n\t"

      "mad.lo.cc.u64 c, %10, %6, %0;\n\t"
      "madc.hi.cc.u64 c, %10, %6, 0;\n\t"
      
      "addc.cc.u64 t, %1, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %1, %10, %7, t;\n\t"
      "madc.hi.cc.u64 c, %10, %7, nc;\n\t"

      "addc.cc.u64 t, %2, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %2, %10, %8, t;\n\t"
      "madc.hi.cc.u64 c, %10, %8, nc;\n\t"

      "addc.cc.u64 t, %3, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %3, %10, %9, t;\n\t"
      "madc.hi.cc.u64 c, %10, %9, nc;\n\t"

      "addc.cc.u64 c, c, %5;\n\t"
      //"addc.u64 nc, 0, 0;\n\t"
      "addc.cc.u64 %4, %4, c;\n\t"
      //"addc.u64 %5, nc, 0;\n\t"
      "}"
      : "+l"(wide_r[3]),
      "+l"(wide_r[4]),
      "+l"(wide_r[5]),
      "+l"(wide_r[6]),
      "+l"(wide_r[7]),
      "+l"(carry)
      : "l"(modulus[0]),
      "l"(modulus[1]),
      "l"(modulus[2]),
      "l"(modulus[3]),
      "l"(k)
    );

    //memcpy(ret, wide_r + 4, sizeof(uint64_t) * 4);
}

inline __device__ void dev_mul_wide(const uint64_t *a, const uint64_t *b, uint64_t *c){
    //uint64_t r[N * 2] = {0};
    asm(
      "{\n\t"
      ".reg .u64 c;\n\t"
      ".reg .u64 nc;\n\t"
      ".reg .u64 t;\n\t"
      //r[0], c = a[0] * b[0] 
      "mad.lo.cc.u64 %0, %8, %12, 0;\n\t"
      "madc.hi.cc.u64 c, %8, %12, 0;\n\t"
      
      //r[1], c = a[0] * b[1] + c
      "madc.lo.cc.u64 %1, %8, %13, c;\n\t"
      "madc.hi.cc.u64 c, %8, %13, 0;\n\t"
    
      //r[2], c = a[0] * b[2] + c
      "madc.lo.cc.u64 %2, %8, %14, c;\n\t"
      "madc.hi.cc.u64 c, %8, %14, 0;\n\t"

      //r[3], c = a[0] * b[3] + c
      "madc.lo.cc.u64 %3, %8, %15, c;\n\t"
      "madc.hi.cc.u64 %4, %8, %15, 0;\n\t"

      //r[1], c = a[1] * b[0] + c
      "mad.lo.cc.u64 %1, %9, %12, %1;\n\t"
      "madc.hi.cc.u64 c, %9, %12, 0;\n\t"
      
      //t = r[2] + c
      "addc.cc.u64 t, %2, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      //r[2], c = a[1] * b[1] + c
      "mad.lo.cc.u64 %2, %9, %13, t;\n\t"
      "madc.hi.cc.u64 c, %9, %13, nc;\n\t"

      "addc.cc.u64 t, %3, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      //r[3], c = a[1] * b[2] + c
      "mad.lo.cc.u64 %3, %9, %14, t;\n\t"
      "madc.hi.cc.u64 c, %9, %14, nc;\n\t"

      "addc.cc.u64 t, %4, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      //r[4], c = a[1] * b[3] + c
      "mad.lo.cc.u64 %4, %9, %15, t;\n\t"
      "madc.hi.cc.u64 %5, %9, %15, nc;\n\t"

      //r[2], c = a[2] * b[0] + c
      "mad.lo.cc.u64 %2, %10, %12, %2;\n\t"
      "madc.hi.cc.u64 c, %10, %12, 0;\n\t"
      
      "addc.cc.u64 t, %3, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %3, %10, %13, t;\n\t"
      "madc.hi.cc.u64 c, %10, %13, nc;\n\t"
      
      "addc.cc.u64 t, %4, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %4, %10, %14, t;\n\t"
      "madc.hi.cc.u64 c, %10, %14, nc;\n\t"

      "addc.cc.u64 t, %5, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %5, %10, %15, t;\n\t"
      "madc.hi.cc.u64 %6, %10, %15, nc;\n\t"

      "mad.lo.cc.u64 %3, %11, %12, %3;\n\t"
      "madc.hi.cc.u64 c, %11, %12, 0;\n\t"
      
      "addc.cc.u64 t, %4, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %4, %11, %13, t;\n\t"
      "madc.hi.cc.u64 c, %11, %13, nc;\n\t"
      
      "addc.cc.u64 t, %5, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %5, %11, %14, t;\n\t"
      "madc.hi.cc.u64 c, %11, %14, nc;\n\t"
      
      "addc.cc.u64 t, %6, c;\n\t"
      "addc.u64 nc, 0, 0;\n\t"
      "mad.lo.cc.u64 %6, %11, %15, t;\n\t"
      "madc.hi.cc.u64 %7, %11, %15, nc;\n\t"
      "}"
      : "+l"(c[0]),
      "+l"(c[1]),
      "+l"(c[2]),
      "+l"(c[3]),
      "+l"(c[4]),
      "+l"(c[5]),
      "+l"(c[6]),
      "+l"(c[7])
      : "l"(a[0]),
      "l"(a[1]),
      "l"(a[2]),
      "l"(a[3]),
      "l"(b[0]),
      "l"(b[1]),
      "l"(b[2]),
      "l"(b[3])
    );

    //#pragma unroll
    //for(int i = 0; i < N*2; i++){
    //    c[i] = r[i];
    //}
}


inline __device__ void dev_mont_mul(const Int* a, const Int* b, const Int* modulus, const uint64_t inv, Int* c){
    uint64_t wide_r[N*2];
    dev_mul_wide(a, b, wide_r);  
    dev_mont_mul(wide_r, modulus, inv);
    #pragma unroll
    for(int i = 0; i < N; i++){
        c[i] = wide_r[i + N];
    }

    //reduce
    if(dev_is_ge(c, modulus)){
        uint64_t sub[N];
        dev_sub(c, modulus, sub);
        memcpy(c, sub, N * sizeof(uint64_t));
    }
}

//c[N+1] = a[N] * b
inline __device__ void dev_mul_1(const Int* a, const Int b, Int* c){
    asm(
        "{\n\t"
        ".reg .u64 c;\n\t"
        ".reg .u64 t;\n\t"
        ".reg .u64 nc;\n\t"

        //r[0] = q * p[0]
        "mad.lo.cc.u64 %0, %9, %5, 0;\n\t"
        "madc.hi.cc.u64 c, %9, %5, 0;\n\t"

        //r[1] = q * p[1]
        "addc.cc.u64 t, 0, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %1, %9, %6, t;\n\t"
        "madc.hi.cc.u64 c, %9, %6, nc;\n\t"

        ////r[2] = q * p[2]
        "addc.cc.u64 t, 0, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %2, %9, %7, t;\n\t"
        "madc.hi.cc.u64 c, %9, %7, nc;\n\t"

        ////r[3] = q * p[3]
        "addc.cc.u64 t, 0, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %3, %9, %8, t;\n\t"
        "madc.hi.cc.u64 %4, %9, %8, nc;\n\t"
      "}"
      : "=l"(c[0]),
      "=l"(c[1]),
      "=l"(c[2]),
      "=l"(c[3]),
      "=l"(c[4])
      : "l"(a[0]),
      "l"(a[1]),
      "l"(a[2]),
      "l"(a[3]),
      "l"(b)
    );
}

//c[N+1] = a[N+1] + b[N+1]
inline __device__ Int dev_add_n_1(const Int* a, const Int* b, Int* c){
    Int carry = 0;
    asm(
        "{\n\t"
            "add.cc.u64 %0, %6, %11;\n\t"
            "addc.cc.u64 %1, %7, %12;\n\t"
            "addc.cc.u64 %2, %8, %13;\n\t"
            "addc.cc.u64 %3, %9, %14;\n\t"
            "addc.cc.u64 %4, %10, %15;\n\t"
            "addc.u64 %5, 0, 0;\n\t"
        "}"
        : "=l"(c[0]),
        "=l"(c[1]),
        "=l"(c[2]),
        "=l"(c[3]),
        "=l"(c[4]),
        "=l"(carry)
        : "l"(a[0]),
        "l"(a[1]),
        "l"(a[2]),
        "l"(a[3]),
        "l"(a[4]),
        "l"(b[0]),
        "l"(b[1]),
        "l"(b[2]),
        "l"(b[3]),
        "l"(b[4])
    );
    return carry;
}

inline __device__ Int dev_add_n(Int* x, const Int y, const int n){
    //asm(
    //    "add.cc.u64 %0, %0, %1;\n\t"
    //    : "=l"(x[0])
    //    : "l"(y)
    //);
    //for(int i = 1; i < n; i++){
    //    asm(
    //        "addc.cc.u64 %0, %0, %1;\n\t"
    //        : "=l"(x[i])
    //        : "l"(y)
    //    );
    //}
    //Int carry = 0;
    //asm(
    //    "addc.cc.u64 %0, 0, 0;\n\t"
    //    : "=l"(carry)
    //);
    //return carry;
    Int t = x[0] + y;
    x[0] = t;
    if(t >= y){
        return 0;
    }
    for(int i = 1; i < n; i++){
        t = x[i] + 1;
        x[i] = t;
        if(t != 0){
            return 0;
        }
    }
    return 1;
}

inline __device__ void dev_mont_red(const Int* xy, const Int* p, const Int rp, Int* z){
    Int pq[N+1];
    Int buf[N*2+1];
    memcpy(buf + N + 1, xy + N + 1, sizeof(Int) * (N-1));
    buf[N*2] = 0;
    Int q = xy[0] * rp;
    dev_mul_1(p, q, pq);

    Int up = dev_add_n_1(xy, pq, buf);
    if(up){
        buf[N*2] = dev_add_n(buf + N + 1, 1, N-1);
    }
    Int* c = buf + 1;
    for(int i = 1; i < N; i++){
        q = c[0] * rp;
        dev_mul_1(p, q, pq);
        up = dev_add_n_1(c, pq, c);
        if(up){
            dev_add_n(c + N + 1, 1, N-i);
        }
        ++c;
    }
    if(c[N]){
        dev_sub(c, p, z);
    }else{
        Int ret = dev_sub(c, p, z);
        if(ret){
            memcpy(z, c, N * sizeof(Int));
        }
    }
}

inline __device__ void dev_as_bigint(const Int* x, const Int* p, const Int rp, Int* res){
    Int one[N] = {0};
    one[0] = 1;
    dev_mont_mul(x, one, p, rp, res);
}

struct Point {
    Int c0[N], c1[N];
    inline __device__ int is_zero() const {
        return (dev_is_zero(c0) && dev_is_zero(c1));
    }
    inline __device__ int is_one(const Int* one) const {
        return (dev_is_one(c0, one) && dev_is_zero(c1));
    }
    inline __device__ void set(const Point& other){
        memcpy(c0, other.c0, N*sizeof(Int));
        memcpy(c1, other.c1, N*sizeof(Int));
    }

    inline __device__ void clear(){
        dev_clear(c0);
        dev_clear(c1);
    }
};


/******mcl******/
inline __device__ void dev_mcl_add(const Int* a, const Int* b, const Int* p, Int* c){
    dev_add(a, b, c);
    if(c[N-1] < p[N-1]){
        return;
    }
    if(c[N-1] > p[N-1]){
        dev_sub(c, p, c);
        return;
    }

    /* the top of z and p are same */
    //tmp[0..N-1] = c[0..N-1] - p;
    Int tmp[N-1];
    Int borrow = 0;
   asm(
      "sub.cc.u64 %0, %4, %7;\n\t"
      "subc.cc.u64 %1, %5, %8;\n\t"
      "subc.cc.u64 %2, %6, %9;\n\t"
      "addc.cc.u64 %3, 0, 0;\n\t"
      : "=l"(tmp[0]),
      "=l"(tmp[1]),
      "=l"(tmp[2]),
      "=l"(borrow)
      : "l"(c[0]),
      "l"(c[1]),
      "l"(c[2]),
      "l"(p[0]),
      "l"(p[1]),
      "l"(p[2])
    );
    borrow = 1-borrow;
    //if(dev_sub(c, p, tmp) == 0){
    if(borrow == 0){
        memcpy(c, tmp, sizeof(Int) * (N-1));
        c[N-1] = 0;
    }
}

inline __device__ void dev_mcl_sub(const Int* a, const Int* b, const Int* p, Int* c){
    int borrow = dev_sub(a, b, c);
    if(borrow){
        dev_add(c, p, c);
    }
}

inline __device__ void dev_mcl_mul(const Int* a, const Int* b, const Int* p, const Int rp, Int* c){
    Int buf[N*2+1];
    Int* ptr = buf;
    dev_mul_1(a, b[0], ptr);
    Int q = ptr[0] * rp;
    Int t[N+1];
    dev_mul_1(p, q, t);
    Int carry = dev_add_n_1(ptr, t, ptr);
    ++ptr;
    ptr[N] = 0;
    for(int i = 1; i < N; i++){
        ptr[N+1] = 0;
        dev_mul_1(a, b[i], t);
        carry = dev_add_n_1(ptr, t, ptr);
        q = ptr[0] * rp;
        dev_mul_1(p, q, t);
        carry = dev_add_n_1(ptr, t, ptr);
        ++ptr;
    }
    if(dev_sub(ptr, p, c)){
        memcpy(c, ptr, sizeof(Int256));
    }
}

inline __device__ void dev_mcl_sqr(const Int* a, const Int* p, const Int rp, Int* c){
    dev_mcl_mul(a, a, p, rp, c);
}

struct Ect{
    Int256 x, y, z;
    inline __device__ int is_zero() const {
        return dev_is_zero(z);
    }

    inline __device__ void dev_dblNoVerifyInfJacobi(const Ect& P, const Int* one, const Int* p, const int specialA_, const Int* a_, const uint64_t rp) {
        Int256 S, M, t, y2;
        dev_clear(S);
        dev_clear(M);
        dev_clear(t);
        dev_clear(y2);
        //Fp::sqr(y2, P.y);
        dev_mcl_sqr(P.y, p, rp, y2); 
        //Fp::mul(S, P.x, y2);
        dev_mcl_mul(P.x, y2, p, rp, S);
        //const bool isPzOne = P.z.isOne();
        const int isPzOne = dev_is_one(P.z, one);
        //S += S;
        dev_mcl_add(S, S, p, S);
        //S += S;
        dev_mcl_add(S, S, p, S);
        //Fp::sqr(M, P.x);
        dev_mcl_sqr(P.x, p, rp, M); 
        switch (specialA_) {
            case 0:
                //Fp::add(t, M, M);
                dev_mcl_add(M, M, p, t);
                //M += t;
                dev_mcl_add(M, t, p, M);
                break;
            case 1:
                if (isPzOne) {
                    //M -= P.z;
                    dev_mcl_sub(M, P.z, p, M);
                } else {
                    //Fp::sqr(t, P.z);
                    dev_mcl_sqr(P.z, p, rp, t);
                    //Fp::sqr(t, t);
                    dev_mcl_sqr(t, p, rp, t);
                    //M -= t;
                    dev_mcl_sub(M, t, p, M);
                }
                //Fp::add(t, M, M);
                dev_mcl_add(M, M, p, t);
                //M += t;
                dev_mcl_add(M, t, p, M);
                break;
            case 2:
            default:
                if (isPzOne) {
                    //t = a_;
                    //t.set(a_);
                    memcpy(t, a_, sizeof(Int256));
                } else {
                    //Fp::sqr(t, P.z);
                    dev_mcl_sqr(P.z, p, rp, t);
                    //Fp::sqr(t, t);
                    dev_mcl_sqr(t, p, rp, t);
                    //t *= a_;
                    dev_mcl_mul(t, a_, p, rp, t);
                }
                //t += M;
                dev_mcl_add(t, M, p, t);
                //M += M;
                dev_mcl_add(M, M, p, M);
                //M += t;
                dev_mcl_add(M, t, p, M);
                break;
        }
        //Fp::sqr(R.x, M);
        dev_mcl_sqr(M, p, rp, this->x); 
        //R.x -= S;
        dev_mcl_sub(this->x, S, p, this->x);
        //R.x -= S;
        dev_mcl_sub(this->x, S, p, this->x);
        if (isPzOne) {
            //R.z = P.y;
            //this->z.set(P.y);
            memcpy(this->z, P.y, sizeof(Int256));
        } else {
            //Fp::mul(R.z, P.y, P.z);
            dev_mcl_mul(P.y, P.z, p, rp, this->z);
        }
        //R.z += R.z;
        dev_mcl_add(this->z, this->z, p, this->z);
        //Fp::sqr(y2, y2);
        dev_mcl_sqr(y2, p, rp, y2); 
        //y2 += y2;
        dev_mcl_add(y2, y2, p, y2);
        //y2 += y2;
        dev_mcl_add(y2, y2, p, y2);
        //y2 += y2;
        dev_mcl_add(y2, y2, p, y2);
        //Fp::sub(R.y, S, R.x);
        dev_mcl_sub(S, this->x, p, this->y);
        //R.y *= M;
        dev_mcl_mul(this->y, M, p, rp, this->y);
        //R.y -= y2;
        dev_mcl_sub(this->y, y2, p, this->y);
    }

    inline __device__ void set(const Ect& other){
        memcpy(this->x, other.x, sizeof(Int256));
        memcpy(this->y, other.y, sizeof(Int256));
        memcpy(this->z, other.z, sizeof(Int256));
    }

    inline __device__ void clear(){
        dev_clear(this->x);
        dev_clear(this->y);
        dev_clear(this->z);
    }
};

inline __device__ void dev_addJacobi(Ect& R, const Ect& P, const Ect& Q, const int isPzOne, const int isQzOne,
        const Int* one, const Int* p, const int specialA_, const Int* a_, const int mode_, const uint64_t rp){
    Int256 r, U1, S1, H, H3;
    dev_clear(r);
    dev_clear(U1);
    dev_clear(S1);
    dev_clear(H);
    dev_clear(H3);
    if (isPzOne) {
        // r = 1;
    } else {
        //Fp::sqr(r, P.z);
        dev_mcl_sqr(P.z, p, rp, r);
    }
    if (isQzOne) {
        //U1 = P.x;
        //U1.set(P.x);
        memcpy(U1, P.x, sizeof(Int256));
        if (isPzOne) {
            //H = Q.x;
            //H.set(Q.x);
            memcpy(H, Q.x, sizeof(Int256));
        } else {
            //Fp::mul(H, Q.x, r);
            dev_mcl_mul(Q.x, r, p, rp, H);
        }
        //H -= U1;
        dev_mcl_sub(H, U1, p, H);
        //printInt256(H, "H");
        //S1 = P.y;
        //S1.set(P.y);
        memcpy(S1, P.y, sizeof(Int256));
    } else {
        //Fp::sqr(S1, Q.z);
        dev_mcl_sqr(Q.z, p, rp, S1);
        //printInt256(S1, "S1");
        //Fp::mul(U1, P.x, S1);
        dev_mcl_mul(P.x, S1, p, rp, U1);
        if (isPzOne) {
            //H = Q.x;
            //H.set(Q.x);
            memcpy(H, Q.x, sizeof(Int256));
        } else {
            //Fp::mul(H, Q.x, r);
            dev_mcl_mul(Q.x, r, p, rp, H);
            //printInt256(H, "H");
        }
        //H -= U1;
        dev_mcl_sub(H, U1, p, H);
        //S1 *= Q.z;
        dev_mcl_mul(S1, Q.z, p, rp, S1);
        //S1 *= P.y;
        dev_mcl_mul(S1, P.y, p, rp, S1);
        //printInt256(S1, "S1");
    }
    if (isPzOne) {
        //r = Q.y;
        //r.set(Q.y);
        memcpy(r, Q.y, sizeof(Int256));
    } else {
        //r *= P.z;
        dev_mcl_mul(r, P.z, p, rp, r);
        //printInt256(r, "r");
        //r *= Q.y;
        dev_mcl_mul(r, Q.y, p, rp, r);
        //printInt256(r, "r");
    }
    //r -= S1;
    dev_mcl_sub(r, S1, p, r);
    //if (H.is_zero()) {
    if(dev_is_zero(H)){
        ///printf("H is zero\n");
        //if (r.is_zero()) {
        if(dev_is_zero(r)){
            //printf("r is zero\n");
            R.dev_dblNoVerifyInfJacobi(P, one, p, specialA_, a_, rp);
        } else {
            R.clear();
        }
        return;
    }
    if (isPzOne) {
        if (isQzOne) {
            //R.z = H;
            //R.z.set(H);
            memcpy(R.z, H, sizeof(Int256));
            //printInt256(R.z, "R.z");
        } else {
            //Fp::mul(R.z, H, Q.z);
            dev_mcl_mul(H, Q.z, p, rp, R.z);
        }
    } else {
        if (isQzOne) {
            //Fp::mul(R.z, P.z, H);
            dev_mcl_mul(P.z, H, p, rp, R.z);
        } else {
            //Fp::mul(R.z, P.z, Q.z);
            dev_mcl_mul(P.z, Q.z, p, rp, R.z);
            //R.z *= H;
            dev_mcl_mul(R.z, H, p, rp, R.z);
        }
    }
    //Fp::sqr(H3, H); // H^2
    dev_mcl_sqr(H, p, rp, H3);
    //printInt256(H, "H");
    //printInt256(H3, "H3");
    //Fp::sqr(R.y, r); // r^2
    dev_mcl_sqr(r, p, rp, R.y);
    //printInt256(r, "r");
    //printInt256(R.y, "R.y");

    ///U1 *= H3; // U1 H^2
    dev_mcl_mul(U1, H3, p, rp, U1);
    //H3 *= H; // H^3
    dev_mcl_mul(H3, H, p, rp, H3);
    //R.y -= U1;
    dev_mcl_sub(R.y, U1, p, R.y);
    //printInt256(R.y, "R.y");

    //R.y -= U1;
    dev_mcl_sub(R.y, U1, p, R.y);
    //printInt256(R.y, "R.y");
    //printInt256(H3, "H3");
    //Fp::sub(R.x, R.y, H3);
    dev_mcl_sub(R.y, H3, p, R.x);
    //printInt256(R.x, "R.x");
    //U1 -= R.x;
    dev_mcl_sub(U1, R.x, p, U1);
    //U1 *= r;
    dev_mcl_mul(U1, r, p, rp, U1);
    //H3 *= S1;
    dev_mcl_mul(H3, S1, p, rp, H3);
    //Fp::sub(R.y, U1, H3);
    dev_mcl_sub(U1, H3, p, R.y);
}

inline __device__ void add(Ect& R, const Ect& P, const Ect& Q,
        const Int* one, const Int* p, const int specialA_, const Int* a_, 
        const int mode_, const uint64_t rp, const bool is_prefix_sum = false){
    if(P.is_zero()){
        R.set(Q); 
        return;
    }
    if(Q.is_zero()){
        R.set(P);
        return;
    }
    if(&P == &Q){
        R.dev_dblNoVerifyInfJacobi(P, one, p, specialA_, a_, rp);
        return;
    }
    int isPzOne = dev_is_one(P.z, one);//P.z.is_one(one);
    int isQzOne = dev_is_one(Q.z, one);//Q.z.is_one(one);
    //printf("isPzOne=%d, isQzOne=%d\n", isPzOne, isQzOne);
    dev_addJacobi(R, P, Q, isPzOne, isQzOne, one, p, specialA_, a_, mode_, rp);
}



inline __device__ void dev_mcl_mul_debug(const Int* a, const Int* b, const Int* p, const Int rp, Int* c){
    Int buf[N*2+1];
    Int* ptr = buf;
    dev_mul_1(a, b[0], ptr);
    Int q = ptr[0] * rp;
    Int t[N+1];
    dev_mul_1(p, q, t);
    Int carry = dev_add_n_1(ptr, t, ptr);
    ++ptr;
    ptr[N] = 0;
    for(int i = 1; i < N; i++){
        ptr[N+1] = 0;
        dev_mul_1(a, b[i], t);
        carry = dev_add_n_1(ptr, t, ptr);
        q = ptr[0] * rp;
        dev_mul_1(p, q, t);
        carry = dev_add_n_1(ptr, t, ptr);
        ++ptr;
    }
    if(dev_sub(ptr, p, c)){
        memcpy(c, ptr, sizeof(Int256));
    }
}

inline __device__ int dev_is_zero_g2(const Int* a, const Int* b){
    return dev_is_zero(a) && dev_is_zero(b);
}

inline __device__ int dev_is_one_g2(const Int* a, const Int* b, const Int* one){
    return dev_is_one(a, one) && dev_is_one(b, one);
}

struct Point2 {
    //Int* c0, *c1;
    Int c0[N*2], c1[N*2];

    inline __device__ void clear(){
        dev_clear_2(c0);
        dev_clear_2(c1);
    }
};

inline __device__ void printPoint(const Point& x, const char* desc){
    for(int i = 0; i < 4; i++){
        printf("%lu ", x.c0[i]);
    }
    printf("\n");
    for(int i = 0; i < 4; i++){
        printf("%lu ", x.c1[i]);
    }
    printf("\n");
}


inline __device__ void dev_mcl_add_g2(const Point& x, const Point& y, const Int* p, Point& z){
    dev_mcl_add(x.c0, y.c0, p, z.c0);
    dev_mcl_add(x.c1, y.c1, p, z.c1);
}

inline __device__ void dev_mcl_sub_g2(const Point& x, const Point& y, const Int* p, Point& z){
    dev_mcl_sub(x.c0, y.c0, p, z.c0);
    dev_mcl_sub(x.c1, y.c1, p, z.c1);
}

//z.c0[N*2], z.c1[N*2] 
//x.c0[N], x.c1[N]
//y.c0[N], y.c1[N]
inline __device__ void dev_fp2Dbl_mulPreW(const Point& x, const Point& y, const Int* p, Point2& z){
    dev_add(x.c0, x.c1, z.c0);
    dev_add(y.c0, y.c1, z.c0 + N);
    dev_clear_2(z.c1);
    dev_mul_wide(z.c0, z.c0+N, z.c1);
    Int d2[2*N] = {0};
    dev_mul_wide(x.c1, y.c1, d2);
    dev_clear_2(z.c0);
    dev_mul_wide(x.c0, y.c0, z.c0);

    dev_sub_wide(z.c1, d2, z.c1);
    dev_sub_wide(z.c1, z.c0, z.c1);

    if(dev_sub_wide(z.c0, d2, z.c0)){
        dev_add(z.c0 + N, p, z.c0 + N);
    }
}

inline __device__ void dev_mcl_sqr_g2(const Point& x, const Int* p, const Int rp, Point& y){
    Int256 t1, t2, t3;
    dev_mcl_add(x.c1, x.c1, p, t1);
    dev_mcl_mul(t1, x.c0, p, rp, t1);
    dev_mcl_add(x.c0, x.c1, p, t2);
    dev_mcl_sub(x.c0, x.c1, p, t3);
    dev_mcl_mul(t2, t3, p, rp, y.c0);
    memcpy(y.c1, t1, sizeof(Int256));
}

inline __device__ void dev_mcl_sqr_g2_debug(const Point& x, const Int* p, const Int rp, Point& y, Point& z){
    Int256 t1, t2, t3;
    printf("sqr 1\n");
    printPoint(z,"");
    dev_mcl_add(x.c1, x.c1, p, t1);
    printf("sqr 2\n");
    printPoint(z,"");
    dev_mcl_mul(t1, x.c0, p, rp, t1);
    printf("sqr 3\n");
    printPoint(z,"");
    dev_mcl_add(x.c0, x.c1, p, t2);
    printf("sqr 4\n");
    printPoint(z,"");
    dev_mcl_sub(x.c0, x.c1, p, t3);
    printf("sqr 5\n");
    printPoint(z,"");
    dev_mcl_mul_debug(t2, t3, p, rp, y.c0);
    printf("sqr 6\n");
    printPoint(z,"");
    memcpy(y.c1, t1, sizeof(Int256));
    printf("sqr 7\n");
    printPoint(z,"");
}

inline __device__ void dev_mcl_mul_g2(const Point& x, const Point& y, const Int* p, const Int rp, Point& z){
    Point2 d;
    d.clear();
    dev_fp2Dbl_mulPreW(x, y, p, d);
    dev_mont_red(d.c0, p, rp, z.c0);
    dev_mont_red(d.c1, p, rp, z.c1);
}

inline __device__ void dev_mcl_mul_g2_print(const Point& x, const Point& y, const Int* p, const Int rp, Point& z){
    Point2 d;
    d.clear();
    dev_fp2Dbl_mulPreW(x, y, p, d);
    printf("d\n");
    for(int i = 0; i < 8; i++){
        printf("%lu ", d.c0[i]);
        if(i == 3) printf("\n");
    }
    printf("\n");
    for(int i = 0; i < 8; i++){
        printf("%lu ", d.c1[i]);
        if(i == 3) printf("\n");
    }
    printf("\n");
    dev_mont_red(d.c0, p, rp, z.c0);
    dev_mont_red(d.c1, p, rp, z.c1);
}

struct Ect2{
    Point x, y, z;
    inline __device__ int is_zero() const {
        return dev_is_zero_g2(z.c0, z.c1);
    }

    inline __device__ void set(const Ect2& other){
        x.set(other.x);
        y.set(other.y);
        z.set(other.z);
    }
    inline __device__ void clear(){
        x.clear();
        y.clear();
        z.clear();
    }

    inline __device__ void dev_dblNoVerifyInfJacobi(
        const Ect2& P,  const Int* one, const Int* p, const int specialA_, const Point& a_, const Int rp){
        Point S, M, t, y2;
        S.clear();
        M.clear();
        t.clear();
        y2.clear();
        //dev_mcl_sqr_g2(bn_env, y2, P.y, p.mont, p.ptr, cache_buf, cache_t, rp); 
        dev_mcl_sqr_g2(P.y, p, rp, y2); 
        //dev_mcl_mul_g2(bn_env, S, P.x, y2, p.mont, p.ptr, cache_buf, cache_t, rp);
        //printf("S = P.x*y2\n");
        //printPoint(P.x, "");
        //printPoint(y2, "");
        dev_mcl_mul_g2(P.x, y2, p, rp, S); 

        const int isPzOne = P.z.is_one(one);
        //printf("isPzOne=%d\n", isPzOne);
        dev_mcl_add_g2(S, S, p, S);
        dev_mcl_add_g2(S, S, p, S);
        dev_mcl_sqr_g2(P.x, p, rp, M); 
        switch(specialA_){
            case 0:
                dev_mcl_add_g2(M, M, p, t);
                dev_mcl_add_g2(M, t, p, M);
                break;
            case 1:
                if(isPzOne){
                    dev_mcl_sub_g2(M, P.z, p, M);
                }else{
                    dev_mcl_sqr_g2(P.z, p, rp, t);
                    dev_mcl_sqr_g2(t, p, rp, t);
                    dev_mcl_sub_g2(M, t, p, M);
                }
                dev_mcl_add_g2(M, M, p, t);
                dev_mcl_add_g2(M, t, p, M);
                break;
            case 2:
            default:
                if (isPzOne) {
                    t.set(a_);
                }else{
                    dev_mcl_sqr_g2(P.z, p, rp, t);
                    dev_mcl_sqr_g2(t, p, rp, t);
                    dev_mcl_mul_g2(t, a_, p, rp, t);
                }
                dev_mcl_add_g2(t, M, p, t);
                dev_mcl_add_g2(M, M, p, M);
                dev_mcl_add_g2(M, t, p, M);
                break;
        }
        dev_mcl_sqr_g2(M, p, rp, this->x);
        dev_mcl_sub_g2(this->x, S, p, this->x);
        dev_mcl_sub_g2(this->x, S, p, this->x);
        if(isPzOne){
            this->z.set(P.y);
        }else{
            dev_mcl_mul_g2(P.y, P.z, p, rp, this->z);
        }
        dev_mcl_add_g2(this->z, this->z, p, this->z);
        //printf("y2 = y2^2\n");
        //printPoint(y2, "");
        dev_mcl_sqr_g2(y2, p, rp, y2);
        //printPoint(y2, "");

        dev_mcl_add_g2(y2, y2, p, y2);
        dev_mcl_add_g2(y2, y2, p, y2);
        dev_mcl_add_g2(y2, y2, p, y2);

        //printf("R.y = S-R.x\n");
        //printPoint(S, "");
        //printPoint(this->x, "");
        dev_mcl_sub_g2(S,  this->x, p, this->y);
        //printPoint(this->y, "");
        //printf("R.y*m\n");
        //printPoint(M, "");
        dev_mcl_mul_g2(this->y, M, p, rp, this->y);
        //printPoint(this->y, "");
        //printPoint(y2, "");
        dev_mcl_sub_g2(this->y, y2, p, this->y);
        //printPoint(this->y, "");
    }

};

inline __device__ void dev_addJacobi_NoPzAndNoQzOne_g2(
        const Ect2&P, const Ect2& Q, const int isPzOne, const int isQzOne,
        const Int* one, const Int* p, const int specialA_, const Point& a_, const int mode_, const Int rp, Ect2& R){
    Point r, U1, H3, S1;
    r.clear();
    U1.clear();
    H3.clear();
    S1.clear();
    //dev_mcl_mul_g2(P.z, P.z, p, rp, r);
    dev_mcl_sqr_g2(P.z, p, rp, r);	
    //dev_mcl_mul_g2(Q.z, Q.z, p, rp, S1);
    dev_mcl_sqr_g2(Q.z, p, rp, S1);

    dev_mcl_mul_g2(P.x, S1, p, rp, U1);
    dev_mcl_mul_g2(Q.x, r, p, rp, R.x);
    dev_mcl_mul_g2(S1, Q.z, p, rp, S1);

    dev_mcl_sub_g2(R.x, U1, p, R.x);

    dev_mcl_mul_g2(r, P.z, p, rp, r);
    dev_mcl_mul_g2(S1, P.y, p, rp, S1);
    dev_mcl_mul_g2(r, Q.y, p, rp, r);

    dev_mcl_sub_g2(r, S1, p, r);

    if(R.x.is_zero()){
        if (r.is_zero()) {
            R.dev_dblNoVerifyInfJacobi(P, one, p, specialA_, a_, rp);
        } else {
            R.clear();
        }
        return;
    }
    dev_mcl_mul_g2(P.z, Q.z, p, rp, R.z);

    //dev_mcl_mul_g2(R.x, R.x, p, rp, H3);
    dev_mcl_sqr_g2(R.x, p, rp, H3);

    dev_mcl_mul_g2(R.z, R.x, p, rp, R.z);
    //printInt256(R.z.c0, "R.z.c0");
    dev_mcl_mul_g2(U1, H3, p, rp, U1);

    //dev_mcl_mul_g2(r, r, p, rp, R.y);
    dev_mcl_sqr_g2(r, p, rp, R.y);
    //printInt256(R.z.c0, "R.z.c0");

    dev_mcl_mul_g2(H3, R.x, p, rp, H3);

    dev_mcl_sub_g2(R.y, U1, p, R.y);
    dev_mcl_sub_g2(R.y, U1, p, R.y);
    dev_mcl_sub_g2(R.y, H3, p, R.x);
    dev_mcl_sub_g2(U1, R.x, p, U1);

    dev_mcl_mul_g2(H3, S1, p, rp, H3);
    dev_mcl_mul_g2(U1, r, p, rp, U1);

    dev_mcl_sub_g2(U1, H3, p, R.y);
}


inline __device__ void dev_addJacobi_g2(Ect2& R, const Ect2& P, const Ect2& Q, const bool isPzOne, const bool isQzOne,
        const Int* one, const Int* p, const int specialA_, const Point& a_, const int mode_, const uint64_t rp){
    Point r, U1, S1, H, H3;
    r.clear();
    U1.clear();
    S1.clear();
    H3.clear();
    H.clear();
    if (isPzOne) {
        // r = 1;
    } else {
        //Fp::sqr(r, P.z);
        //dev_mcl_sqr_g2(bn_env, r, P.z, p.mont, p.ptr, cache_buf, cache_t, rp);
        //dev_mcl_mul_g2(P.z, P.z, p, rp, r);
        dev_mcl_sqr_g2(P.z, p, rp, r);
    }
    if (isQzOne) {
        //U1 = P.x;
        //U1.set(bn_env, P.x);
        //U1.copy_from(bn_env, P.x);
        U1.set(P.x);
        if (isPzOne) {
            //H = Q.x;
            //H.set(bn_env, Q.x);
            //R.x.copy_from(bn_env, Q.x);
            H.set(Q.x);
        } else {
            //Fp::mul(H, Q.x, r);
            //dev_mcl_mul_g2(bn_env, R.x, Q.x, r, p.mont, p.ptr, cache_buf, cache_t, rp);
            dev_mcl_mul_g2(Q.x, r, p, rp, H);
        }
        //H -= U1;
        //dev_mcl_sub_g2(bn_env, R.x, R.x, U1, p.mont);
        dev_mcl_sub_g2(H, U1, p, H);
        //S1 = P.y;
        //S1.set(bn_env, P.y);
        //S1.copy_from(bn_env, P.y);
        S1.set(P.y);
    } else {
        //Fp::sqr(S1, Q.z);
        //dev_mcl_sqr_g2(bn_env, S1, Q.z, p.mont, p.ptr, cache_buf, cache_t, rp);
        dev_mcl_sqr_g2(Q.z, p, rp, S1);
        //Fp::mul(U1, P.x, S1);
        //dev_mcl_mul_g2(bn_env, U1, P.x, S1, p.mont, p.ptr, cache_buf, cache_t, rp);
        dev_mcl_mul_g2(P.x, S1, p, rp, U1);
        if (isPzOne) {
            //H = Q.x;
            //H.set(bn_env, Q.x);
            //R.x.copy_from(bn_env, Q.x);
            H.set(Q.x);
        } else {
            //Fp::mul(H, Q.x, r);
            //dev_mcl_mul_g2(bn_env, R.x, Q.x, r, p.mont, p.ptr, cache_buf, cache_t, rp);
            dev_mcl_mul_g2(Q.x, r, p, rp, H);
        }
        //H -= U1;
        //dev_mcl_sub_g2(bn_env, R.x, R.x, U1, p.mont);
        dev_mcl_sub_g2(H, U1, p, H);
        //S1 *= Q.z;
        //dev_mcl_mul_g2(bn_env, S1, S1, Q.z, p.mont, p.ptr, cache_buf, cache_t, rp);
        dev_mcl_mul_g2(S1, Q.z, p, rp, S1);
        //S1 *= P.y;
        //dev_mcl_mul_g2(bn_env, S1, S1, P.y, p.mont, p.ptr, cache_buf, cache_t, rp);
        dev_mcl_mul_g2(S1, P.y, p, rp, S1);
    }
    if (isPzOne) {
        //r = Q.y;
        //r.set(bn_env, Q.y);
        //r.copy_from(bn_env, Q.y);
        r.set(Q.y);
    } else {
        //r *= P.z;
        //dev_mcl_mul_g2(bn_env, r, r, P.z, p.mont, p.ptr, cache_buf, cache_t, rp);
        dev_mcl_mul_g2(r, P.z, p, rp, r);
        //r *= Q.y;
        //dev_mcl_mul_g2(bn_env, r, r, Q.y, p.mont, p.ptr, cache_buf, cache_t, rp);
        dev_mcl_mul_g2(r, Q.y, p, rp, r);
    }
    //r -= S1;
    //dev_mcl_sub_g2(bn_env, r, r, S1, p.mont);
    dev_mcl_sub_g2(r, S1, p, r);
    if (H.is_zero()) {
        if (r.is_zero()) {
            //R.dev_dblNoVerifyInfJacobi(bn_env, P, one, p, specialA_, cache_buf, cache_t, a_, rp);
            R.dev_dblNoVerifyInfJacobi(P, one, p, specialA_, a_, rp);
        } else {
            //R.clear(bn_env);
            R.clear();
        }
        return;
    }
    if (isPzOne) {
        if (isQzOne) {
            //R.z = H;
            //R.z.set(bn_env, H);
            //R.z.copy_from(bn_env, R.x);
            R.z.set(H);
        } else {
            //Fp::mul(R.z, H, Q.z);
            //dev_mcl_mul_g2(bn_env, R.z, R.x, Q.z, p.mont, p.ptr, cache_buf, cache_t, rp);
            dev_mcl_mul_g2(H, Q.z, p, rp, R.z);
        }
    } else {
        if (isQzOne) {
            //Fp::mul(R.z, P.z, H);
            //dev_mcl_mul_g2(bn_env, R.z, P.z, R.x, p.mont, p.ptr, cache_buf, cache_t, rp);
            dev_mcl_mul_g2(P.z, H, p, rp, R.z);
        } else {
            //Fp::mul(R.z, P.z, Q.z);
            //dev_mcl_mul_g2(bn_env, R.z, P.z, Q.z, p.mont, p.ptr, cache_buf, cache_t, rp);
            dev_mcl_mul_g2(P.z, Q.z, p, rp, R.z);
            //R.z *= H;
            //dev_mcl_mul_g2(bn_env, R.z, R.z, R.x, p.mont, p.ptr, cache_buf, cache_t, rp);
            dev_mcl_mul_g2(R.z, H, p, rp, R.z);
        }
    }
    //Fp::sqr(H3, H); // H^2
    //dev_mcl_sqr_g2(bn_env, H3, R.x, p.mont, p.ptr, cache_buf, cache_t, rp);
    dev_mcl_sqr_g2(H, p, rp, H3);
    //Fp::sqr(R.y, r); // r^2
    //dev_mcl_sqr_g2(bn_env, R.y, r, p.mont, p.ptr, cache_buf, cache_t, rp);
    dev_mcl_sqr_g2(r, p, rp, R.y);
    ///U1 *= H3; // U1 H^2
    //dev_mcl_mul_g2(bn_env, U1, U1, H3, p.mont, p.ptr, cache_buf, cache_t, rp);
    dev_mcl_mul_g2(U1, H3, p, rp, U1);
    //H3 *= H; // H^3
    //dev_mcl_mul_g2(bn_env, H3, H3, R.x, p.mont, p.ptr, cache_buf, cache_t, rp);
    dev_mcl_mul_g2(H3, H, p, rp, H3);
    //R.y -= U1;
    //dev_mcl_sub_g2(bn_env, R.y, R.y, U1, p.mont);
    dev_mcl_sub_g2(R.y, U1, p, R.y);
    //R.y -= U1;
    //dev_mcl_sub_g2(bn_env, R.y, R.y, U1, p.mont);
    dev_mcl_sub_g2(R.y, U1, p, R.y);
    //Fp::sub(R.x, R.y, H3);
    //dev_mcl_sub_g2(bn_env, R.x, R.y, H3, p.mont);
    dev_mcl_sub_g2(R.y, H3, p, R.x);
    //U1 -= R.x;
    //dev_mcl_sub_g2(bn_env, U1, U1, R.x, p.mont);
    dev_mcl_sub_g2(U1, R.x, p, U1);
    //U1 *= r;
    //dev_mcl_mul_g2(bn_env, U1, U1, r, p.mont, p.ptr, cache_buf, cache_t, rp);
    dev_mcl_mul_g2(U1, r, p, rp, U1);
    //H3 *= S1;
    //dev_mcl_mul_g2(bn_env, H3, H3, S1, p.mont, p.ptr, cache_buf, cache_t, rp);
    dev_mcl_mul_g2(H3, S1, p, rp, H3);
    //Fp::sub(R.y, U1, H3);
    //dev_mcl_sub_g2(bn_env, R.y, U1, H3, p.mont);
    dev_mcl_sub_g2(U1, H3, p, R.y);
}

inline __device__ void add_g2(Ect2& R, const Ect2& P, const Ect2& Q,
        const Int* one, const Int* p, const int specialA_, const Point& a_, 
        const int mode_, const uint64_t rp, const bool is_prefix_sum = false){
    if(P.is_zero()){
        R.set(Q); 
        return;
    }
    if(Q.is_zero()){
        R.set(P);
        return;
    }
    if(&P == &Q){
        R.dev_dblNoVerifyInfJacobi(P, one, p, specialA_, a_, rp);
        return;
    }
    int isPzOne = P.z.is_one(one);//dev_is_one(bn_env, P.z.mont, one.mont);
    int isQzOne = Q.z.is_one(one);//dev_is_one(bn_env, Q.z.mont, one.mont);
    dev_addJacobi_g2(R, P, Q, isPzOne, isQzOne, one, p, specialA_, a_, mode_, rp);
}

} // namespace BigInt256

#endif
