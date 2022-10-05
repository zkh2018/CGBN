#include "cgbn_alt_bn128_g2.h"
//#include "cgbn_alt_bn128_g2.cuh"

namespace gpu{

alt_bn128_g2::alt_bn128_g2(const int count){
  init(count);
}
void alt_bn128_g2::init(const int count){
  x.init(count);
  y.init(count);
  z.init(count);
}
void alt_bn128_g2::init_host(const int count){
  x.init_host(count);
  y.init_host(count);
  z.init_host(count);
}
void alt_bn128_g2::resize(const int count){
  x.resize(count);
  y.resize(count);
  z.resize(count);
}
void alt_bn128_g2::resize_host(const int count){
  x.resize_host(count);
  y.resize_host(count);
  z.resize_host(count);
}
void alt_bn128_g2::release(){
  x.release();
  y.release();
  z.release();
}
void alt_bn128_g2::release_host(){
  x.release_host();
  y.release_host();
  z.release_host();
}
void alt_bn128_g2::copy_from_cpu(const alt_bn128_g2& host, CudaStream stream){
  x.copy_from_cpu(host.x, stream);
  y.copy_from_cpu(host.y, stream);
  z.copy_from_cpu(host.z, stream);
}
void alt_bn128_g2::copy_from_gpu(const alt_bn128_g2& gpu, CudaStream stream){
  x.copy_from_gpu(gpu.x, stream);
  y.copy_from_gpu(gpu.y, stream);
  z.copy_from_gpu(gpu.z, stream);
}
void alt_bn128_g2::copy_to_cpu(alt_bn128_g2& host, CudaStream stream){
  host.x.copy_to_cpu(x, stream);
  host.y.copy_to_cpu(y, stream);
  host.z.copy_to_cpu(z, stream);
}
void alt_bn128_g2::clear(CudaStream stream ){
  this->x.clear(stream);
  this->y.clear(stream);
  this->z.clear(stream);
}


}//namespace gpu
