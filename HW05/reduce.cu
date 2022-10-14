#include "reduce.cuh"

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n){
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    if(i + blockDim.x < n){
        sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    }
    else if(i < n){
        sdata[tid] = g_idata[i];
    }
    else{
        sdata[tid] = 0;
    }
    __syncthreads();
    for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if(tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if(tid == 0){
        g_odata[blockIdx.x] = sdata[0];
    }
}

__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block){
    int n = N;
    int k = (n - 1) / threads_per_block + 1; 
    reduce_kernel<<<k, threads_per_block, sizeof(float)*threads_per_block>>>(*input, *output, n);
    while(n > 1){
        n = k;
        k = (n - 1) / threads_per_block + 1;
        reduce_kernel<<<k, threads_per_block, sizeof(float)*threads_per_block>>>(*output, *output, n);
    }
}
                     