#include<cuda.h>
#include<iostream>
#include "matmul.cuh"

__global__ void Matmul_int(const int * A, const int * B, int * C, int N, unsigned int block_dim){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int aBegin = N * block_dim * by;
    int aEnd = aBegin + N - 1;
    int aStep = block_dim;
    int bBegin = block_dim * bx;
    int bStep = block_dim * N;
    int Csub = 0;
    int ai, bi;
    extern __shared__ int S[];
    int *As = S;
    int *Bs = (int*)&As[block_dim * block_dim];
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        ai = a + N * ty + tx;
        bi = b + N * ty + tx;
        if ((ai / N >= N) || (ai % N < a % N) || (a / N + ty >= N) || (a % N + tx >= N)){
            As[tx + ty * block_dim] = 0;
        }
        else{
            As[tx + ty * block_dim] = A[ai];
        }
        if ((bi / N >= N) || (bi % N < b % N) || (b / N + ty >= N) || (b % N + tx >= N)){
            Bs[tx + ty * block_dim] = 0;
        }
        else{
            Bs[tx + ty * block_dim] = B[bi];
        }
        __syncthreads();
        for (int k = 0; k < block_dim; ++k) {
            Csub += As[ty * block_dim + k] * Bs[k * block_dim + tx];
        }
        __syncthreads();
    }
    int c = N * block_dim * by + block_dim * bx;
    if ((aBegin % N + tx < N) && (aBegin / N + ty < N) && (bBegin % N + tx < N) && (bBegin / N + ty < N) && (c + N * ty + tx < N * N)){
        C[c + N * ty + tx] = Csub;
    }
}

__global__ void Matmul_float(const float * A, const float * B, float * C, int N, unsigned int block_dim){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int aBegin = N * block_dim * by;
    int aEnd = aBegin + N - 1;
    int aStep = block_dim;
    int bBegin = block_dim * bx;
    int bStep = block_dim * N;
    float Csub= 0;
    int ai, bi;
    extern __shared__ float S_float[];
    float *As = S_float;
    float *Bs = (float*)&As[block_dim * block_dim];
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        ai = a + N * ty + tx;
        bi = b + N * ty + tx;
        if ((ai / N >= N) || (ai % N < a % N) || (a / N + ty >= N) || (a % N + tx >= N)){
            As[tx + ty * block_dim] = 0;
        }
        else{
            As[tx + ty * block_dim] = A[ai];
        }
        if ((bi / N >= N) || (bi % N < b % N) || (b / N + ty >= N) || (b % N + tx >= N)){
            Bs[tx + ty * block_dim] = 0;
        }
        else{
            Bs[tx + ty * block_dim] = B[bi];
        }
        __syncthreads();
        for (int k = 0; k < block_dim; ++k) {
            Csub += As[ty * block_dim + k] * Bs[k * block_dim + tx];
        }
        __syncthreads();
    }
    int c = N * block_dim * by + block_dim * bx;
    if ((aBegin % N + tx < N) && (aBegin / N + ty < N) && (bBegin % N + tx < N) && (bBegin / N + ty < N) && (c + N * ty + tx < N * N)){
        C[c + N * ty + tx] = Csub;
    }
}

__global__ void Matmul_double(const double * A, const double * B, double * C, int N, unsigned int block_dim){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int aBegin = N * block_dim * by;
    int aEnd = aBegin + N - 1;
    int aStep = block_dim;
    int bBegin = block_dim * bx;
    int bStep = block_dim * N;
    int ai, bi;
    double Csub= 0;
    extern __shared__ double S_double[];
    double *As = S_double;
    double *Bs = (double*)&As[block_dim * block_dim];
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        ai = a + N * ty + tx;
        bi = b + N * ty + tx;
        if ((ai / N >= N) || (ai % N < a % N) || (a / N + ty >= N) || (a % N + tx >= N)){
            As[tx + ty * block_dim] = 0;
        }
        else{
            As[tx + ty * block_dim] = A[ai];
        }
        if ((bi / N >= N) || (bi % N < b % N) || (b / N + ty >= N) || (b % N + tx >= N)){
            Bs[tx + ty * block_dim] = 0;
        }
        else{
            Bs[tx + ty * block_dim] = B[bi];
        }
        __syncthreads();
        for (int k = 0; k < block_dim; ++k) {
            Csub += As[ty * block_dim + k] * Bs[k * block_dim + tx];
        }
        __syncthreads();
    }
    int c = N * block_dim * by + block_dim * bx;
    if ((aBegin % N + tx < N) && (aBegin / N + ty < N) && (bBegin % N + tx < N) && (bBegin / N + ty < N) && (c + N * ty + tx < N * N)){
        C[c + N * ty + tx] = Csub;
    }
    
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim){
    dim3 dimBlock(block_dim, block_dim);
    int k = (n - 1) / block_dim + 1;
    dim3 dimGrid(k, k);
    Matmul_int<<<dimGrid, dimBlock, block_dim * block_dim * 2 * sizeof(int)>>>(A, B, C, n, block_dim);
}


__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim){
    dim3 dimBlock(block_dim, block_dim);
    int k = (n - 1) / block_dim + 1;
    dim3 dimGrid(k, k);
    Matmul_float<<<dimGrid, dimBlock, block_dim * block_dim * 2 * sizeof(float)>>>(A, B, C, n, block_dim);
}

__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim){
    dim3 dimBlock(block_dim, block_dim);
    int k = (n - 1) / block_dim + 1;
    dim3 dimGrid(k, k);
    Matmul_double<<<dimGrid, dimBlock, block_dim * block_dim * 2 * sizeof(double)>>>(A, B, C, n, block_dim);
}