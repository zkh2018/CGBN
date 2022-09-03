#include <cuda_runtime.h>

#include <stdio.h>


__global__ void child_kernel(int *data, const int n){
    int tid = threadIdx.x;
    if(tid < n){
       data[tid] = tid; 
    }
}

__global__ void kernel(int *data, const int n){
    child_kernel<<<1, n>>>(data, n);
}

int main(){
    const int n = 64;
    int * data;
    cudaMalloc((void**)&data, n*sizeof(int));
    kernel<<<1, 1>>>(data, n);
    int hdata[n];
    cudaMemcpy(hdata, data, n * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0; i < n; i++){
        printf("%d ", hdata[i]);
    }
    printf("\n");
    return 0;
}
