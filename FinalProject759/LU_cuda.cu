#include <stdlib.h>             
#include <stdio.h>              
#include <fstream>              
#include <cuda.h>               
#include <iostream>             
#include <iomanip>              
#include <time.h>               
#include <cuda_runtime.h>       
using namespace std;            
#define TILE 32                 

__global__ void luDecompositionKernel(float* L, float* U, const float* A, int n){
    // Compute the updated lower and upper triangular matrices
    // at the current thread's index.
    int i = threadIdx.x;
    int j = threadIdx.y;

    for (int k = 0; k < i; ++k){
        U[i * n + j] -= L[i * n + k] * U[k * n + j];
        L[i * n + j] -= L[i * n + k] * L[k * n + j];
    }
    if (i == j){
        L[i * n + i] = 1;
    }
    else{
        L[i * n + j] /= U[j * n + j];
        U[i * n + j] /= U[j * n + j];
    }
}


// Perform LU decomposition using CUDA

void luDecompositionCuda(float* L, float* U, const float* A, int n)
{
    // Allocate device memory for the lower and upper triangular matrices
    float* d_L, *d_U;
    cudaMalloc(&d_L, n * n * sizeof(float));
    cudaMalloc(&d_U, n * n * sizeof(float));

    // Copy the input matrix to the device
    cudaMemcpy(d_L, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, A, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    luDecompositionKernel<<<dim3(n, n), dim3(1, 1), 0>>>(d_L, d_U, A, n);


    // Copy the updated lower and upper triangular matrices from the device
    cudaMemcpy(L, d_L, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(U, d_U, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_L);
    cudaFree(d_U);

}
