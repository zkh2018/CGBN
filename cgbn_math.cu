#include "cgbn_math.h"
#include <cuda_runtime.h>
#include <cuda.h>

#include "gpu_support.h"

namespace gpu{

void gpu_malloc(void** ptr, size_t size){
  CUDA_CHECK(cudaMalloc(ptr, size));
  //CUDA_CHECK(cudaMemsetAsync(*ptr, 0, size));
  //CUDA_CHECK(cudaDeviceSynchronize());
}
void gpu_malloc_host(void** ptr, size_t size){
  CUDA_CHECK(cudaMallocHost(ptr, size));
}
void gpu_set_zero(void* ptr, size_t size, CudaStream stream){
  CUDA_CHECK(cudaMemsetAsync(ptr, 0, size, stream));
  //CUDA_CHECK(cudaMemset(ptr, 0, size));
  //CUDA_CHECK(cudaDeviceSynchronize());
}
void gpu_free(void* ptr){
  CUDA_CHECK(cudaFree(ptr));
  //CUDA_CHECK(cudaDeviceSynchronize());
}
void gpu_free_host(void* ptr){
  CUDA_CHECK(cudaFreeHost(ptr));
}
void copy_cpu_to_gpu(void* dst, const void* src, size_t size, CudaStream stream){
  CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
  //CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
  //CUDA_CHECK(cudaDeviceSynchronize());
}
void copy_gpu_to_cpu(void* dst, const void* src, size_t size, CudaStream stream){
  CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
  //CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
  //CUDA_CHECK(cudaDeviceSynchronize());
}
void copy_gpu_to_gpu(void* dst, const void* src, size_t size, CudaStream stream){
  CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream));
  //CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
  //CUDA_CHECK(cudaDeviceSynchronize());
}

void create_stream(CudaStream* stream){
  cudaStreamCreate(stream);
  //CUDA_CHECK(cudaDeviceSynchronize());
}
void release_stream(CudaStream& stream){
  cudaStreamDestroy(stream);
  //CUDA_CHECK(cudaDeviceSynchronize());
}

void sync(CudaStream stream){
  cudaStreamSynchronize(stream);
  //CUDA_CHECK(cudaDeviceSynchronize());
}
void sync_device(){
  CUDA_CHECK(cudaDeviceSynchronize());
}

void gpu_reset(){
  cudaDeviceReset();
}

void gpu_meta::resize(const size_t _size){
  if(_size > this->size){
    if(this->size > 0){
      gpu_free(this->ptr);
    }
    gpu_malloc(&ptr, _size);
    this->size = _size;
  }
}
void gpu_meta::release(){
  if(size > 0 && ptr != nullptr)
      gpu_free(ptr);
}

void gpu_buffer::resize(int new_n){
  if(total_n == 0){
    total_n = new_n;
    n = new_n;
    gpu_malloc((void**)&ptr, n * sizeof(cgbn_mem_t<BITS>));
  }else if(total_n < new_n){
    gpu_free((void*)ptr);
    total_n = new_n;
    n = new_n;
    gpu_malloc((void**)&ptr, n * sizeof(cgbn_mem_t<BITS>));
  }else{
    n = new_n;
  }
}
void gpu_buffer::resize_host(int new_n){
  if(total_n == 0){
    total_n = new_n;
    n = new_n;
    //ptr = (cgbn_mem_t<BITS>*)malloc(n * sizeof(cgbn_mem_t<BITS>)); 
    gpu_malloc_host((void**)&ptr, n * sizeof(cgbn_mem_t<BITS>));
  }else if(total_n < n){
    free(ptr);
    total_n = n;
    n = new_n;
    //ptr = (cgbn_mem_t<BITS>*)malloc(n * sizeof(cgbn_mem_t<BITS>));
    gpu_malloc_host((void**)&ptr, n * sizeof(cgbn_mem_t<BITS>));
  }else{
    n = new_n;
  }
  //memset(ptr, 0, sizeof(cgbn_mem_t<BITS>) * new_n);
}

void gpu_buffer::release(){
    if(n > 0 && ptr != nullptr)
        cudaFree(ptr);
}
void gpu_buffer::release_host(){
  //free(ptr);
  gpu_free_host(ptr);
}

void gpu_buffer::copy_from_host(gpu_buffer& buf, CudaStream stream){
  copy_cpu_to_gpu(ptr, buf.ptr, n * sizeof(cgbn_mem_t<BITS>), stream);
}
void gpu_buffer::copy_to_host(gpu_buffer& buf, CudaStream stream){
  copy_gpu_to_cpu(buf.ptr, ptr, n * sizeof(cgbn_mem_t<BITS>), stream);
}
void gpu_buffer::copy_from_host(const uint32_t* data, const uint32_t n, CudaStream stream){
  copy_cpu_to_gpu((void*)ptr, (void*)data, n * sizeof(uint32_t), stream);
}
void gpu_buffer::copy_to_host(uint32_t* data, const uint32_t n, CudaStream stream){
  copy_cpu_to_gpu((void*)data, (void*)ptr, n * sizeof(uint32_t), stream);
}

}
