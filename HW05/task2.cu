#include<cuda.h>
#include<iostream>
#include "matmul.cuh"

int main(int argc, char **argv){
    const int n = atoi(argv[1]);
    const int block_dim = atoi(argv[2]);
    float *A_float, *B_float, *C_float;
    int *A_int, *B_int, *C_int;
    double *A_double, *B_double, *C_double;

    A_float = (float*)malloc(sizeof(float) * n * n);
    B_float = (float*)malloc(sizeof(float) * n * n);
    C_float = (float*)malloc(sizeof(float) * n * n);
    A_int = (int*)malloc(sizeof(int) * n * n);
    B_int = (int*)malloc(sizeof(int) * n * n);
    C_int = (int*)malloc(sizeof(int) * n * n);
    A_double = (double*)malloc(sizeof(double) * n * n);
    B_double = (double*)malloc(sizeof(double) * n * n);
    C_double = (double*)malloc(sizeof(double) * n * n);
    srand((unsigned)time(0));
    for(int i = 0; i < n * n; i++){
        A_float[i] = 1;
        B_float[i] = 1;
        A_int[i] = 1;
        B_int[i] = 1;
        A_double[i] = 1;
        B_double[i] = 1;
    }

    float *dA_float, *dB_float, *dC_float;
    int *dA_int, *dB_int, *dC_int;
    double *dA_double, *dB_double, *dC_double;
    cudaMalloc((void**)&dA_float, sizeof(float) * n * n);
    cudaMalloc((void**)&dB_float, sizeof(float) * n * n);
    cudaMalloc((void**)&dC_float, sizeof(float) * n * n);
    cudaMalloc((void**)&dA_int, sizeof(int) * n * n);
    cudaMalloc((void**)&dB_int, sizeof(int) * n * n);
    cudaMalloc((void**)&dC_int, sizeof(int) * n * n);
    cudaMalloc((void**)&dA_double, sizeof(double) * n * n);
    cudaMalloc((void**)&dB_double, sizeof(double) * n * n);
    cudaMalloc((void**)&dC_double, sizeof(double) * n * n);
    
    cudaMemcpy(dA_float, A_float, sizeof(float) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB_float, B_float, sizeof(float) * n * n, cudaMemcpyHostToDevice);
    cudaMemset(dC_float, 0, sizeof(float) * n * n);
    cudaMemcpy(dA_int, A_int, sizeof(int) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB_int, B_int, sizeof(int) * n * n, cudaMemcpyHostToDevice);
    cudaMemset(dC_int, 0, sizeof(int) * n * n);
    cudaMemcpy(dA_double, A_double, sizeof(double) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB_double, B_double, sizeof(double) * n * n, cudaMemcpyHostToDevice);
    cudaMemset(dC_double, 0, sizeof(double) * n * n);

    cudaEvent_t start;
    cudaEvent_t stop;
    float ms_int;
    float ms_float;
    float ms_double;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
   
    matmul_2(dA_float, dB_float, dC_float, n, block_dim);  
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_float, start, stop); 

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
   
    matmul_1(dA_int, dB_int, dC_int, n, block_dim); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_int, start, stop); 

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
   
    matmul_3(dA_double, dB_double, dC_double, n, block_dim); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_double, start, stop); 
    
    cudaMemcpy(C_int, dC_int, sizeof(int) * n * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(C_float, dC_float, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(C_double, dC_double, sizeof(double) * n * n, cudaMemcpyDeviceToHost);


    /*std::cout << ms_int << ", ";
    std::cout << ms_float << ", ";
    std::cout << ms_double << ", ";
    std::cout << std::endl;*/

    std::cout << C_int[0] << std::endl;
    std::cout << C_int[n * n - 1] << std::endl;
    std::cout << ms_int << std::endl;
    std::cout << C_float[0] << std::endl;
    std::cout << C_float[n * n - 1] << std::endl;
    std::cout << ms_float << std::endl;
    std::cout << C_double[0] << std::endl;
    std::cout << C_double[n * n - 1] << std::endl;
    std::cout << ms_double << std::endl;
    std::cout << std::endl;

    free(A_float);
    free(B_float);
    free(C_float);
    free(A_int);
    free(B_int);
    free(C_int);
    free(A_double);
    free(B_double);
    free(C_double);
    cudaFree(dA_int);
    cudaFree(dB_int);
    cudaFree(dC_int);
    cudaFree(dA_float);
    cudaFree(dB_float);
    cudaFree(dC_float);
    cudaFree(dA_double);
    cudaFree(dB_double);
    cudaFree(dC_double);
    
    return 0;
}

