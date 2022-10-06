#include "matmul.cuh"
void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block){
    int k = (n * n - 1) / threads_per_block + 1;
    matmul_kernel<<<k, threads_per_block>>>(A, B, C, n);
}

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if(tx < n * n){
        float Cvalue= 0;
        int i = tx / n;
        int j = tx % n;
        for(int k = 0; k < n; ++k)  { 
            Cvalue += A[i * n + k] * B[k * n + j];
        }
        C[i * n + j] = Cvalue;
    }
}