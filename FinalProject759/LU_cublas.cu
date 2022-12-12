#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(int argc, char **argv){
    const int n = atoi(argv[1]);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float *A;
    cudaMalloc((void**)&A, n*n*sizeof(float));
    int *ipiv;
    cudaMalloc((void**)&ipiv, n*sizeof(int));
    int info;

    srand((unsigned)time(0));
    
    for(int i=0; i < n * n; i++){
        A[i] = float(-1.0) + (rand()) / ( static_cast <float> (RAND_MAX/2.0));
    }

    
    cudaEvent_t start;
    cudaEvent_t stop;
    float ms;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Perform the LU decomposition
    cublasSgetrfBatched(handle, n, A, n, ipiv, info, 1);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop); 

    std::cout << A[0] << std::endl;
    std::cout << ms << std::endl;
    std::cout << std::endl;

    cudaFree(A);
    cudaFree(ipiv);
    cublasDestroy(handle);

    return 0;
}