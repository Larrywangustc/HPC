#include<cuda.h>
#include<iostream>
#include <cuda_runtime.h>
#include "scan.cuh"


int main(int argc, char **argv){
    const int n = atoi(argv[1]);
    const int threads_per_block = atoi(argv[2]);
    float *A, *B;

    cudaMallocManaged(&A, n * sizeof(float));
    cudaMallocManaged(&B, n * sizeof(float));
    srand((unsigned)time(0));
    for(int i=0; i < n; i++){
        A[i] = float(-1.0) + (rand()) / ( static_cast <float> (RAND_MAX/2.0));
    }

    float ms = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    scan(A, B, n, threads_per_block);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop); 
    std::cout << B[0] << std::endl;
    std::cout << ms << std::endl;
    std::cout << std::endl;
    //printf("cudaMalloc function : %s\n",cudaGetErrorString(cudaGetLastError()));
    cudaFree(A);
    cudaFree(B);
    return 0;
}
