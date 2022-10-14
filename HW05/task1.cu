#include<cuda.h>
#include<iostream>
#include "reduce.cuh"

int main(int argc, char **argv){
    const int n = atoi(argv[1]);
    const int threads_per_block = atoi(argv[2]);
    float *A, *B;
    int k = (n - 1) / threads_per_block + 1; 
    A = (float*)malloc(sizeof(float) * n);
    B = (float*)malloc(sizeof(float) * k);
    srand((unsigned)time(0));
    //float sum = 0;
    for(int i=0; i < n; i++){
        A[i] = float(-1.0) + (rand()) / ( static_cast <float> (RAND_MAX/2.0));
        //sum += A[i];
    }

    float *dA, *dB;
    cudaMalloc((void**)&dA, sizeof(float) * n);
    cudaMalloc((void**)&dB, sizeof(float) * k);
    cudaMemset(dB, 0, k * sizeof(float));
    cudaMemcpy(dA, A, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaEvent_t start;
    cudaEvent_t stop;
    float ms;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
   
    reduce(&dA, &dB, n, threads_per_block); 
   
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop); 

    cudaMemcpy(B, dB, sizeof(float) * k, cudaMemcpyDeviceToHost);
    
    std::cout << B[0] << std::endl;
    std::cout << ms << std::endl;
    std::cout << std::endl;
    cudaFree(dA);
    cudaFree(dB);
    free(A);
    free(B);
    return 0;
}