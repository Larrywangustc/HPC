#include <stdlib.h>             
#include <stdio.h>              
#include <fstream>              
#include <cuda.h>               
#include <iostream>             
#include <iomanip>              
#include <time.h>               
#include <cuda_runtime.h>       
using namespace std;              
#define Tile_Width 32
#define BSZ 32

__global__ void lu_decomposition_kernel(const float *A, float *L, float *U, const int N){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int y = by * blockDim.y + ty;
    int x = bx * blockDim.x + tx;
    __shared__ float part_sum[BSZ * BSZ];

    for (int k = 0; k < N; ++k)
    {
        part_sum[y * BSZ + x] = 0;
        for (int i = 0; i < k; ++i)
        {
            part_sum[y * BSZ + x] += A[k * N + i] * A[i * N + y];
        }

        __syncthreads();

        L[k * N + y] = (y == k) ? 1 : A[k * N + y] - part_sum[y * BSZ + x];
        U[k * N + y] = (y == k) ? A[k * N + k] - part_sum[y * BSZ + x] : 0;
    }
}


__global__ void luDecompositionOptimized_kernel(const float* d_A, float* d_L, float* d_U, int N)
{
    // Declare shared memory for the tile of the input matrix
    __shared__ float tile[Tile_Width][Tile_Width];

    // Load the tile of the input matrix into shared memory
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    if (row < N && col < N) {
        tile[ty][tx] = d_A[row * N + col];
    }

    for (int k = 0; k < N; k++) {
        if (row == k) {
            d_L[row * N + col] = 1.0f; 
        }
        if (row > k) {
            d_L[row * N + col] = tile[row][k] / tile[k][k];
            for (int i = k+1; i < Tile_Width; i++) {
                tile[row][i] -= d_L[row * N + col] * tile[k][i];
            }
        }

        if (row <= k) {
            d_U[row * N + col] = tile[row][col];
        }
    }
}


__global__ void lu_decomposition_kernel(const float *A, float *L, float *U, const int N){
    int ty = threadIdx.y;
    int by = blockIdx.y;
    int y = by * blockDim.y + ty;

    if (y >= N) return;  // Check that y is within bounds

    for (int k = 0; k < N; ++k)
    {
        // Calculate the elements of the lower triangular matrix
        for (int i = 0; i < k; ++i)
        {
            L[k * N + y] -= A[k * N + i] * L[i * N + y];
        }

        if (k == y)
        {
            L[k * N + y] = 1;  // Diagonal elements are 1
        }
        else
        {
            L[k * N + y] = A[k * N + y];
        }

        // Calculate the elements of the upper triangular matrix
        for (int i = 0; i < k; ++i)
        {
            U[k * N + y] -= A[k * N + i] * U[i * N + y];
        }

        if (k == y)
        {
            U[k * N + y] = A[k * N + y] / L[k * N + y];  // Diagonal elements are the original value divided by the corresponding element in the lower triangular matrix
        }
        else
        {
            U[k * N + y] = A[k * N + y] / L[k * N + y];
        }
    }
}


void luDecompositionCuda(float* L, float* U, const float* A, int n, int block_dim){
    int k = (n - 1) / block_dim + 1;
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(k, k);
    lu_decomposition_kernel<<<dimGrid, dimBlock>>>(A, L, U, n);
    //luDecompositionOptimized_kernel<<<dimGrid, dimBlock, Tile_Width * Tile_Width * sizeof(float)>>>(A, L, U, n);

}


int main(int argc, char **argv){
    const int n = atoi(argv[1]);
    const int threads_per_block = atoi(argv[2]);
    float *A, *L, *U;

    A = (float*)malloc(sizeof(float) * n * n);
    L = (float*)malloc(sizeof(float) * n * n);
    U = (float*)malloc(sizeof(float) * n * n);

    float* d_L, *d_U, *d_A;
    cudaMalloc(&d_L, n * n * sizeof(float));
    cudaMalloc(&d_U, n * n * sizeof(float));
    cudaMalloc(&d_A, n * n * sizeof(float));

    srand((unsigned)time(0));
    
    for(int i=0; i < n * n; i++){
        A[i] = float(-1.0) + (rand()) / ( static_cast <float> (RAND_MAX/2.0));
    }

    cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start;
    cudaEvent_t stop;
    float ms;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
   
    luDecompositionCuda(d_L, d_U, d_A, n, threads_per_block);
   
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop); 
    
    cudaMemcpy(L, d_L, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(U, d_U, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_L);
    cudaFree(d_U);
    cudaFree(d_A);

    std::cout << L[0] << std::endl;
    std::cout << ms << std::endl;
    std::cout << std::endl;
    free(A);
    free(L);
    free(U);
    return 0;
}