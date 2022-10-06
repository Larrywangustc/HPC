#include<cuda.h>
#include<iostream>
#include "matmul.cuh"

int main(int argc, char **argv){
    const int n = atoi(argv[1]);
    const int threads_per_block = atoi(argv[2]);
    float *A, *B, *C;

    A = (float*)malloc(sizeof(float)* n * n);
    B = (float*)malloc(sizeof(float)* n * n);
    C = (float*)malloc(sizeof(float)* n * n);
    srand((unsigned)time(0));
    for(int i=0; i < n * n; i++){
        A[i] = float(-1) + (rand()) / ( static_cast <float> (RAND_MAX/2));
        B[i] = float(-1) + (rand()) / ( static_cast <float> (RAND_MAX/2));
    }

    float *dA, *dB, *dC;
    cudaMalloc((void**)&dA, sizeof(float)* n * n);
    cudaMalloc((void**)&dB, sizeof(float)* n * n);
    cudaMalloc((void**)&dC, sizeof(float)* n * n);
    cudaMemset(dC, 0, n * n * sizeof(float));
    cudaMemcpy(dA, A, sizeof(float) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float) * n * n, cudaMemcpyHostToDevice);
    cudaEvent_t start;
    cudaEvent_t stop;
    float ms;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
   
    matmul(dA, dB, dC, n, threads_per_block); 
   
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop); 

    cudaMemcpy(C, dC, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
    
    std::cout << C[n * n - 1] << std::endl;
    std::cout << ms << std::endl;
    std::cout << std::endl;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(A);
    free(B);
    free(C);
    return 0;
}





