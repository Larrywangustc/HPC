#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(int argc, char **argv){

    const int n = atoi(argv[1]);
    int batchSize = atoi(argv[2]);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaError_t error;
    cublasStatus_t status;
    float *A;
    error = cudaMallocManaged(&A, batchSize * n * n * sizeof(float));
    if (error != cudaSuccess) fprintf(stderr,"\nError: %s\n",cudaGetErrorString(error));
    srand((unsigned)time(0));

    for(int i=0; i < batchSize * n * n; i++){
        A[i] = float(-1.0) + (rand()) / ( static_cast <float> (RAND_MAX/2.0));
    }

    float **Aarray;
    error = cudaMallocManaged(&Aarray, batchSize * sizeof(float));
    int *ipiv;
    error = cudaMallocManaged(&ipiv, batchSize * n * sizeof(int));
    if (error != cudaSuccess) fprintf(stderr,"\nError: %s\n",cudaGetErrorString(error));
    int *info;

    error = cudaMallocManaged(&info, batchSize * sizeof(int));
    if (error != cudaSuccess) fprintf(stderr,"\nError: %s\n",cudaGetErrorString(error));
    for (int i = 0; i < batchSize; i++) {
        info[i] = 0;
    }

    for (int i = 0; i < batchSize; i++) {
      Aarray[i] = A + i * n * n;
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    float ms;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Perform the LU decomposition
    status = cublasSgetrfBatched(handle, n, (float **)Aarray, n, (int *)ipiv, info, batchSize);
    if (status != CUBLAS_STATUS_SUCCESS) fprintf(stderr,"error in dgetrf %i\n",status);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop); 

    std::cout << A[0] << std::endl;
    std::cout << ms << std::endl;
    std::cout << std::endl;

    cudaFree(A);
    cudaFree(Aarray);
    cudaFree(ipiv);
    cudaFree(info);
    cublasDestroy(handle);

    return 0;
}