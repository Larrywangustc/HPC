#include<cuda.h>
#include<iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "mmul.h"


int main(int argc, char **argv){
    const int n = atoi(argv[1]);
    const int n_tests = atoi(argv[2]);
    float *A, *B, *C;
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMallocManaged(&A, n * n *sizeof(float));
    cudaMallocManaged(&B, n * n *sizeof(float));
    cudaMallocManaged(&C, n * n *sizeof(float));
    srand((unsigned)time(0));
    for(int i=0; i < n * n; i++){
        A[i] = float(-1.0) + (rand()) / ( static_cast <float> (RAND_MAX/2.0));
        B[i] = float(-1.0) + (rand()) / ( static_cast <float> (RAND_MAX/2.0));
    }

    float ms = 0;
    float sum = 0;
    cudaEvent_t start, stop;
    for(int i=0; i<n_tests; i++){
        
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
   
        mmul(handle, A, B, C, n);
   
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop); 
        sum += ms;
    }
    sum = sum / n_tests;
    std::cout << sum << std::endl;
    std::cout << std::endl;
    // printf("cudaMalloc function : %s\n",cudaGetErrorString(cudaGetLastError()));
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
