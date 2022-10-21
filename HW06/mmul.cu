#include<cuda.h>
#include<iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n){
    float a = 1, b = 0;
    cublasSgemm(
        handle,
        CUBLAS_OP_N,//
        CUBLAS_OP_N,//
        n,
        n,
        n,
        &a,
        A,
        n,
        B,
        n,
        &b,
        C,
        n
        );
    cudaDeviceSynchronize();
}