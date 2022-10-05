
#include "cgbn_alt_bn128_g1.h"
#include "bigint_256.cuh"

#include <algorithm>


namespace gpu{

alt_bn128_g1::alt_bn128_g1(const int count){
  init(count);
}
void alt_bn128_g1::init(const int count){
  x.init(count);
  y.init(count);
  z.init(count);
}
void alt_bn128_g1::init_host(const int count){
  x.init_host(count);
  y.init_host(count);
  z.init_host(count);
}
void alt_bn128_g1::resize(const int count){
  x.resize(count);
  y.resize(count);
  z.resize(count);
}
void alt_bn128_g1::resize_host(const int count){
  x.resize_host(count);
  y.resize_host(count);
  z.resize_host(count);
}
void alt_bn128_g1::release(){
  x.release();
  y.release();
  z.release();
}
void alt_bn128_g1::release_host(){
  x.release_host();
  y.release_host();
  z.release_host();
}
void alt_bn128_g1::copy_from_cpu(const alt_bn128_g1& g1, CudaStream stream){
  x.copy_from_cpu(g1.x, stream);
  y.copy_from_cpu(g1.y, stream);
  z.copy_from_cpu(g1.z, stream);
}
void alt_bn128_g1::copy_from_gpu(const alt_bn128_g1& g1, CudaStream stream){
  x.copy_from_gpu(g1.x, stream);
  y.copy_from_gpu(g1.y, stream);
  z.copy_from_gpu(g1.z, stream);
}
void alt_bn128_g1::copy_to_cpu(alt_bn128_g1& g1, CudaStream stream){
  g1.x.copy_to_cpu(x, stream);
  g1.y.copy_to_cpu(y, stream);
  g1.z.copy_to_cpu(z, stream);
}
void alt_bn128_g1::clear(CudaStream stream ){
  this->x.clear(stream);
  this->y.clear(stream);
  this->z.clear(stream);
}

__global__ void kernel_update_seconds(const uint32_t *firsts, uint32_t* seconds, const int range_size){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < range_size){
		int first = firsts[tid];
		int second = seconds[tid];
		seconds[tid] = first + (second - first + 1) / 2;
	}
}

__global__ void kernel_elementwise_mul_scalar(
    Fp_model datas, 
    Fp_model sconst, 
    const uint32_t n,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv){
  const int instance = blockIdx.x * blockDim.x + threadIdx.x;
  const int local_instances = blockDim.x;
  if(instance >= n) return;
  using namespace BigInt256;

  Fp local_modulus;
  local_modulus.load((Int*)modulus);
  Fp local_sconst;
  local_sconst.load((Int*)sconst.mont_repr_data);
  for(int i = instance; i < n; i += gridDim.x * local_instances){
    Fp a;
    a.load((Int*)(datas.mont_repr_data + i));
    a = a.mul(local_sconst, local_modulus, inv);
    a.store((Int*)(datas.mont_repr_data + i));
  }
}

void alt_bn128_g1_elementwise_mul_scalar(
    Fp_model datas, 
    Fp_model sconst, 
    const uint32_t n,
    cgbn_mem_t<BITS>* modulus, const uint64_t inv){
  const int instances = 64;
  const int threads = instances;
  const int blocks = (n + instances - 1) / instances;

  kernel_elementwise_mul_scalar<<<blocks, threads>>>(datas, sconst, n, modulus, inv); 
}


__global__ void kernel_fft_copy(
    Fp_model in,
    Fp_model out,
    const int *in_offsets,
    const int *out_offsets,
    const int *strides,
    const int n,
    const int radix){
    const int instance = threadIdx.x + blockIdx.x * blockDim.x;
    if(instance >= n) return;

    const int in_offset = in_offsets[instance];
    const int out_offset = out_offsets[instance];
    const int stride = strides[instance];
    using namespace BigInt256;

    for(int i = 0; i < radix; i++){
        Fp a;
        a.load((Int*)(in.mont_repr_data + in_offset + i * stride)); 
        a.store((Int*)(out.mont_repr_data + out_offset + i));
    }
}

void fft_copy(
    Fp_model in,
    Fp_model out,
    const int *in_offsets,
    const int *out_offsets,
    const int *strides,
    const int n,
    const int radix){
  //cgbn_error_report_t *report;
  //CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  const int instances = 64;
  int threads = instances;
  int blocks = (n + instances - 1) / instances;
  kernel_fft_copy<<<blocks, threads>>>(in, out, in_offsets, out_offsets, strides, n, radix);
  //CUDA_CHECK(cudaDeviceSynchronize());
}

template<int BlockInstances>
__global__ void kernel_butterfly_2_new(
        Fp_model out,
        Fp_model twiddles, 
        const int twiddle_offset,
        const int *strides, 
        const uint32_t stage_length, 
        const int* out_offsets, 
        const int n,
        cgbn_mem_t<BITS>* max_value, 
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv){
    const int instance = threadIdx.x + blockIdx.x * blockDim.x;
    if(instance >= n) return;

    using namespace BigInt256;
    Fp local_max_value, local_modulus;
    local_max_value.load((Int*)max_value);
    local_modulus.load((Int*)modulus);

    uint32_t out_offset = out_offsets[instance];
    uint32_t out_offset2 = out_offset + stage_length;
    Fp out1, out2;
    //FieldT t = out[out_offset2];
    out2.load((Int*)(out.mont_repr_data + out_offset2));
    out1.load((Int*)(out.mont_repr_data + out_offset));
    //out[out_offset2] = out[out_offset] - t;
    Fp tmp_out2 = out1.sub(out2, local_modulus);
    tmp_out2.store((Int*)(out.mont_repr_data + out_offset2));
    //out[out_offset] += t;
    Fp tmp_out = out1.add(out2, local_modulus); 
    tmp_out.store((Int*)(out.mont_repr_data + out_offset));
    out_offset2++;
    out_offset++;
    for (unsigned int k = 1; k < stage_length; k++){
        //FieldT t = twiddles[k] * out[out_offset2];
        out2.load((Int*)(out.mont_repr_data + out_offset2));
        out1.load((Int*)(out.mont_repr_data + out_offset));
        Fp twiddle;
        twiddle.load((Int*)(twiddles.mont_repr_data + twiddle_offset + k));
        Fp t = twiddle.mul(out2, local_modulus, inv);
        //out[out_offset2] = out[out_offset] - t;
        tmp_out2 = out1.sub(t, local_modulus);
        tmp_out2.store((Int*)(out.mont_repr_data + out_offset2));
        //out[out_offset] += t;
        tmp_out = out1.add(t, local_modulus);
        tmp_out.store((Int*)(out.mont_repr_data + out_offset));
        out_offset2++;
        out_offset++;
    }
}


void butterfly_2(
        Fp_model out,
        Fp_model twiddles, 
        const int twiddle_offset,
        const int *strides, 
        const uint32_t stage_length, 
        const int* out_offsets, 
        const int n,
        cgbn_mem_t<BITS>* max_value, 
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv){
    const int instances = 64;
    int threads = instances;
    int blocks = (n + instances - 1) / instances;
    kernel_butterfly_2_new<instances><<<blocks, threads>>>(
            out, 
            twiddles, 
            twiddle_offset,
            strides, 
            stage_length, 
            out_offsets, n, max_value, modulus, inv);
}

template<int BlockInstances>
__global__ void kernel_butterfly_4_new(
        Fp_model out,
        Fp_model twiddles, 
        const int twiddles_len,
        const int twiddle_offset,
        const int* strides, 
        const uint32_t stage_length, 
        const int* out_offsets, 
        const int n,
        cgbn_mem_t<BITS>* max_value, 
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv){
    const int instance = threadIdx.x + blockIdx.x * blockDim.x;
    if(instance >= n) return;
    using namespace BigInt256;
    Fp local_modulus;
    local_modulus.load((Int*)modulus);

    const int real_instance = instance / stage_length;
    const int stage_instance = instance % stage_length;
    uint32_t out_offset = out_offsets[real_instance] + stage_instance;
    //uint32_t stride = strides[instance];

    Fp j;
    j.load((Int*)(twiddles.mont_repr_data + twiddle_offset + twiddles_len-1));
    uint32_t tw = stage_instance * 3;
    /* Case twiddle == one */
    if(false){
		const unsigned i0  = out_offset;
        const unsigned i1  = out_offset + stage_length;
        const unsigned i2  = out_offset + stage_length*2;
        const unsigned i3  = out_offset + stage_length*3;

		Fp z0, z1, z2, z3;
        //const FieldT z0  = out[i0];
        z0.load((Int*)(out.mont_repr_data + i0));
        //const FieldT z1  = out[i1];
        z1.load((Int*)(out.mont_repr_data + i1));
        //const FieldT z2  = out[i2];
        z2.load((Int*)(out.mont_repr_data + i2));
        //const FieldT z3  = out[i3];
        z3.load((Int*)(out.mont_repr_data + i3));

        Fp t1, t2, t3, t4, t4j;
        //const FieldT t1  = z0 + z2;
        t1 = z0.add(z2, local_modulus);
        //const FieldT t2  = z1 + z3;
        t2 = z1.add(z3, local_modulus);
        //const FieldT t3  = z0 - z2;
        t3 = z0.sub(z2, local_modulus);
        //const FieldT t4j = j * (z1 - z3);
        t4 = z1.sub(z3, local_modulus);
        t4j = j.mul(t4, local_modulus, inv);

        Fp out0, out1, out2, out3;
        //out[i0] = t1 + t2;
        out0 = t1.add(t2, local_modulus);
        out0.store((Int*)(out.mont_repr_data + i0));
        //out[i1] = t3 - t4j;
        out1 = t3.sub(t4j, local_modulus);
        out1.store((Int*)(out.mont_repr_data + i1));
        //out[i2] = t1 - t2;
        out2 = t1.sub(t2, local_modulus);
        out2.store((Int*)(out.mont_repr_data + i2));
        //out[i3] = t3 + t4j;
        out3 = t3.add(t4j, local_modulus);
        out3.store((Int*)(out.mont_repr_data + i3));

        out_offset++;
        tw += 3;
    }

	//for (unsigned int k = 0; k < stage_length; k++)
	{
		const unsigned i0  = out_offset;
		const unsigned i1  = out_offset + stage_length;
		const unsigned i2  = out_offset + stage_length*2;
		const unsigned i3  = out_offset + stage_length*3;

        Fp z0, z1, z2, z3;
        Fp out0, out1, out2, out3;
		//const FieldT z0  = out[i0];
        z0.load((Int*)(out.mont_repr_data + i0));
        out1.load((Int*)(out.mont_repr_data + i1));
        out2.load((Int*)(out.mont_repr_data + i2));
        out3.load((Int*)(out.mont_repr_data + i3));
        Fp tw0, tw1, tw2;
        tw0.load((Int*)(twiddles.mont_repr_data + twiddle_offset + tw));
        tw1.load((Int*)(twiddles.mont_repr_data + twiddle_offset + tw+1));
        tw2.load((Int*)(twiddles.mont_repr_data + twiddle_offset + tw+2));
		//const FieldT z1  = out[i1] * twiddles[tw];
        z1 = out1.mul(tw0, local_modulus, inv);
		//const FieldT z2  = out[i2] * twiddles[tw+1];
        z2 = out2.mul(tw1, local_modulus, inv);
		//const FieldT z3  = out[i3] * twiddles[tw+2];
        z3 = out3.mul(tw2, local_modulus, inv);

        Fp t1, t2, t3, t4, t4j;
		//const FieldT t1  = z0 + z2;
        t1 = z0.add(z2, local_modulus);
		//const FieldT t2  = z1 + z3;
        t2 = z1.add(z3, local_modulus);
		//const FieldT t3  = z0 - z2;
        t3 = z0.sub(z2, local_modulus);
		//const FieldT t4j = j * (z1 - z3);
        t4 = z1.sub(z3, local_modulus);
        t4j = j.mul(t4, local_modulus, inv);

		//out[i0] = t1 + t2;
        out0 = t1.add(t2, local_modulus);
        out0.store((Int*)(out.mont_repr_data + i0));
		//out[i1] = t3 - t4j;
        out1 = t3.sub(t4j, local_modulus);
        out1.store((Int*)(out.mont_repr_data + i1));
		//out[i2] = t1 - t2;
        out2 = t1.sub(t2, local_modulus);
        out2.store((Int*)(out.mont_repr_data + i2));
		//out[i3] = t3 + t4j;
        out3 = t3.add(t4j, local_modulus);
        out3.store((Int*)(out.mont_repr_data + i3));

		//out_offset++;
		//tw += 3;
	}

}

void butterfly_4(
        Fp_model out,
        Fp_model twiddles, 
        const int twiddles_len,
        const int twiddle_offset,
        const int* strides, 
        const uint32_t stage_length, 
        const int* out_offsets, 
        const int n,
        cgbn_mem_t<BITS>* max_value, 
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv){
    const int instances = 64;
    int threads = instances;
    const int total_n = n * stage_length;
    int blocks = (total_n + instances - 1) / instances;
    kernel_butterfly_4_new<instances><<<blocks, threads>>>(
            out, 
            twiddles, 
            twiddles_len,
            twiddle_offset,
            strides, 
            stage_length, 
            out_offsets, total_n, max_value, modulus, inv);
    //CUDA_CHECK(cudaDeviceSynchronize());
}

template<int BlockInstances>
__global__ void kernel_multiply_by_coset_and_constant_new(
        Fp_model inputs,
        const int n,
        Fp_model g,
        Fp_model c, 
        Fp_model one,
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv,
        const int gmp_num_bits){
    const int instance = threadIdx.x + blockIdx.x * blockDim.x;
    if(instance >= n) return;

    using namespace BigInt256;
    Fp local_modulus;
    local_modulus.load((Int*)modulus);

    Fp dev_c, dev_g, dev_one;
    dev_c.load((Int*)c.mont_repr_data);
    dev_g.load((Int*)g.mont_repr_data);
    dev_one.load((Int*)one.mont_repr_data);
    if(instance == 0){
        Fp a0;
        a0.load((Int*)inputs.mont_repr_data);
        a0 = a0.mul(dev_c, local_modulus, inv);
        a0.store((Int*)(inputs.mont_repr_data));
    }else{
        Fp tmp = dev_g.power(dev_one, instance, local_modulus, inv, gmp_num_bits);
        Fp u = dev_c.mul(tmp, local_modulus, inv);
        Fp ai;
        ai.load((Int*)(inputs.mont_repr_data + instance));
        ai = ai.mul(u, local_modulus, inv);
        ai.store((Int*)(inputs.mont_repr_data + instance));
    }
}

void multiply_by_coset_and_constant(
        Fp_model inputs,
        const int n,
        Fp_model g,
        Fp_model c, 
        Fp_model one,
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv,
        const int gmp_num_bits){
    //CUDA_CHECK(cgbn_error_report_alloc(&report)); 
    const int instances = 64;
    int threads = instances;
    int blocks = (n + instances - 1) / instances;
    kernel_multiply_by_coset_and_constant_new<instances><<<blocks, threads>>>(inputs, n, g, c, one, modulus, inv, gmp_num_bits); 
    //CUDA_CHECK(cudaDeviceSynchronize());
}

template<int BlockInstances>
__global__ void kernel_calc_xor_new(
        Fp_model xor_results,
        const int n,
        const int offset,
        Fp_model g,
        Fp_model one,
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv,
        const int gmp_num_bits){
    const int instance = threadIdx.x + blockIdx.x * blockDim.x;
    if(instance >= n-1) return;

    using namespace BigInt256;
    Fp local_modulus;
    local_modulus.load((Int*)modulus);

    Fp dev_g, dev_one;
    dev_g.load((Int*)(g.mont_repr_data));
    dev_one.load((Int*)one.mont_repr_data);
    Fp xor_result = dev_g.power(dev_one, instance + offset, local_modulus, inv, gmp_num_bits);
    xor_result.store((Int*)(xor_results.mont_repr_data + instance+offset));
}

//xor_result = g^i
void calc_xor(
        Fp_model xor_results,
        const int n,
        const int offset,
        Fp_model g,
        Fp_model one,
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv,
        const int gmp_num_bits){
    const int instances = 64;
    int threads = instances;
    int blocks = (n + instances - 1) / instances;
    kernel_calc_xor_new<instances><<<blocks, threads>>>(xor_results, n, offset, g, one, modulus, inv, gmp_num_bits); 
}

template<int BlockInstances>
__global__ void kernel_multiply_new(
        Fp_model inputs,
        Fp_model xor_results,
        const int n,
        const int offset,
        Fp_model c, 
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv){
    const int instance = threadIdx.x + blockIdx.x * blockDim.x;
    if(instance >= n) return;

    using namespace BigInt256;
    Fp local_modulus;
    local_modulus.load((Int*)modulus);

    Fp dev_c;
    dev_c.load((Int*)(c.mont_repr_data));
    if(instance == 0){
        Fp a0;
        a0.load((Int*)inputs.mont_repr_data);
        a0 = a0.mul(dev_c, local_modulus, inv);
        a0.store((Int*)inputs.mont_repr_data);
    }else{
        Fp xor_result;
        xor_result.load((Int*)(xor_results.mont_repr_data + instance));
        Fp u = dev_c.mul(xor_result, local_modulus, inv);
        Fp ai;
        ai.load((Int*)(inputs.mont_repr_data + instance));
        ai = ai.mul(u, local_modulus, inv);
        ai.store((Int*)(inputs.mont_repr_data + instance));
    }
}

//inputs[0] *= c
//inputs[i] *= xor_results[i]
void multiply(
        Fp_model inputs,
        Fp_model xor_results,
        const int n,
        const int offset,
        Fp_model c, 
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv){
    const int instances = 64;
    int threads = instances;
    int blocks = (n + instances - 1) / instances;
    kernel_multiply_new<instances><<<blocks, threads>>>(inputs, xor_results, n, offset, c, modulus, inv); 
}


template<int BlockInstances>
__global__ void kernel_calc_H_new(
        Fp_model A,
        Fp_model B,
        Fp_model C,
        Fp_model out,
        Fp_model Z_inverse_at_coset,
        const int n,
        cgbn_mem_t<BITS>* max_value, 
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv){
    const int instance = threadIdx.x + blockIdx.x * blockDim.x;
    if(instance >= n) return;

    using namespace BigInt256;
    Fp local_modulus; 
    local_modulus.load((Int*)modulus);

    Fp dev_a, dev_b, dev_c, dev_out, dev_coset;
    dev_coset.load((Int*)Z_inverse_at_coset.mont_repr_data);
    dev_a.load((Int*)(A.mont_repr_data + instance));
    dev_b.load((Int*)(B.mont_repr_data + instance));
    dev_c.load((Int*)(C.mont_repr_data + instance));
    Fp tmp = dev_a.mul(dev_b, local_modulus, inv);
    dev_out = tmp.sub(dev_c, local_modulus);
    dev_out = dev_out.mul(dev_coset, local_modulus, inv);
    dev_out.store((Int*)(out.mont_repr_data + instance));
}

//out[i] = ((A[i] * B[i]) - C[i]) * Z_inverse_at_coset
void calc_H(
        Fp_model A,
        Fp_model B,
        Fp_model C,
        Fp_model out,
        Fp_model Z_inverse_at_coset,
        const int n,
        cgbn_mem_t<BITS>* max_value, 
        cgbn_mem_t<BITS>* modulus, 
        const uint64_t inv){
    const int instances = 64;
    int threads = instances;
    int blocks = (n + instances - 1) / instances;
    kernel_calc_H_new<instances><<<blocks, threads>>>(A, B, C, out, Z_inverse_at_coset, n, max_value, modulus, inv); 
}



__global__ void kernel_warmup(){
  int sum = 0;
  for(int i = 0; i < 1000; i++){
    sum += i;
  }
}
void warm_up(){
  //kernel_warmup<<<1, 1>>>();
  //cuda_check(cudadevicesynchronize());
  cudaSetDevice(0);
  cudaFree(0);
}

} //gpu
