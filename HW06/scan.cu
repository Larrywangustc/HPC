#include "scan.cuh"

__global__ void hillis_steele(const float *g_idata, float *g_odata, int n) {
    extern __shared__ float temp[];
    int thid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pout = 0, pin = 1;
    //temp[thid] = (thid == 0) ? 0: g_idata[thid-1];
    if (i < n){
        temp[thid] = g_idata[i];
    }
    else{
        temp[thid] = 0.0f;
    }
    __syncthreads();
    for(int offset = 1; offset < blockDim.x; offset *= 2 ) {
        pout = 1 - pout; 
        pin  = 1 - pout;
        if(thid >= offset){
            temp[pout * blockDim.x + thid] = temp[pin * blockDim.x + thid] + temp[pin * blockDim.x + thid - offset];
        }
        else{
            temp[pout * blockDim.x + thid] = temp[pin * blockDim.x + thid];
        }    
        __syncthreads();
    }
    __syncthreads();
    if(thid == blockDim.x - 1){
        g_odata[blockIdx.x] = temp[thid];
    }
}

__host__ void scan(const float* input, float* output, unsigned int N, unsigned int threads_per_block){
    int n = N;
    int k = (n - 1) / threads_per_block + 1; 
    hillis_steele<<<k, threads_per_block, 2 * sizeof(float) * threads_per_block>>>(input, output, n);
    while(n > 1){
        n = k;
        k = (n - 1) / threads_per_block + 1;
        hillis_steele<<<k, threads_per_block, 2 * sizeof(float) * threads_per_block>>>(output, output, n);
    }
}