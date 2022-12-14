#include <stdlib.h>             
#include <stdio.h>              
#include <fstream>              
#include <cuda.h>               
#include <iostream>             
#include <iomanip>                       
#include <cuda_runtime.h>       
#include <assert.h>
using namespace std;              
#define Tile_Width 32

__global__ void jacobiUnOptimizedOnDevice(float* x_next_u, float* A_u, float* x_now_u, float* b_u, int N){
    // Optimization step 1: tiling
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (idx < N)
    {
        float sigma = 0.0;
        
        int idx_Ai = idx*N;
        
        for (int j=0; j<N; j++)
            if (idx != j)
                sigma += A_u[idx_Ai + j] * x_now_u[j];
        x_next_u[idx] = (b_u[idx] - sigma) / A_u[idx_Ai + idx];
    }
}


__constant__ float b_s[512];
__global__ void jacobiOptimizedOnDevice(float* d_x_next, float* d_A, float* d_x_now,  int N){
    __shared__ float xdsn[Tile_Width];
    __shared__ float xdsx[Tile_Width];

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int xIndex = bx * Tile_Width + tx;
    
    int idx = xIndex * Tile_Width + threadIdx.x;
    if (idx < N) {
        float sigma = 0.0;
        int idx_Ai = idx * N;
        for (int j=0; j<Tile_Width; j++) {
            if (idx != j) {
                xdsn[tx] = d_x_now[idx*Tile_Width];
                xdsx[tx] = d_x_next[idx * Tile_Width];
                sigma += d_A[idx_Ai + j] * xdsn[tx];           
                xdsx[tx] = (b_s[idx] - sigma) / d_A[idx_Ai + idx];
            }
        }
        for (int k=0; k<N; k++) {
            d_x_next[k* N] = xdsx[tx];
        }
    }
}

int main(int argc, char **argv){
    const int n = atoi(argv[1]);
    const int threads_per_block = atoi(argv[2]);

    float *A, *b_h, *x_d, *x_next, *x_now;
    float *x_next_u, *x_now_u, *A_u, *b_u;
    float *d_x_now, *d_x_next, *d_A, *b;
    
    int iter = 1000, tileSize = 32;
    
    x_next = (float *) malloc(n * sizeof(float));
    A = (float *) malloc(n * n * sizeof(float));
    x_now = (float *) malloc(n * sizeof(float));
    b_h = (float *) malloc(n * sizeof(float));
    x_d = (float *) malloc(n * sizeof(float));

    for(int i=0; i < n * n; i++){
        A[i] = float(-1.0) + (rand()) / ( static_cast <float> (RAND_MAX/2.0));
    }
    for(int i=0; i < n; i++){
        b_h[i] = float(-1.0) + (rand()) / ( static_cast <float> (RAND_MAX/2.0));
        x_now[i] = 0;
        x_next[i] = 0;
    }

    assert(cudaSuccess == cudaMalloc((void **) &x_next_u, n * sizeof(float)));
    assert(cudaSuccess == cudaMalloc((void **) &A_u, n * n * sizeof(float)));
    assert(cudaSuccess == cudaMalloc((void **) &x_now_u, n * sizeof(float)));
    assert(cudaSuccess == cudaMalloc((void **) &b_u, n * sizeof(float)));

    assert(cudaSuccess == cudaMalloc((void **) &d_x_next, n * sizeof(float)));
    assert(cudaSuccess == cudaMalloc((void **) &d_A, n * n * sizeof(float)));
    assert(cudaSuccess == cudaMalloc((void **) &d_x_now, n * sizeof(float)));
    assert(cudaSuccess == cudaMalloc((void **) &b, n * sizeof(float)));
    
    cudaMemcpy(x_next_u, x_next, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(A_u, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x_now_u, x_now, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_u, b_h, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_x_next, x_next, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_now, x_now, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b, b_h, n * sizeof(float), cudaMemcpyHostToDevice);
    
    int nTiles = n/tileSize + (n%tileSize == 0?0:1);
    int grid = n/tileSize + (n%tileSize == 0?0:1);

    dim3 dGrid(grid, grid),
    dBlock(tileSize, tileSize);
    
    dim3 dimGrid(64,16);
    dim3 dimBlock(16,1);

    cudaEvent_t start;
    cudaEvent_t stop;
    float ms1;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int k=0; k<iter; k++){
        if (k%2)
            jacobiUnOptimizedOnDevice <<< nTiles, tileSize >>> (x_now_u, A_u, x_next_u, b_u, n);
        else     
            jacobiUnOptimizedOnDevice <<< nTiles, tileSize >>> (x_now_u, A_u, x_next_u, b_u, n);
        cudaMemcpy(x_now_u, x_next_u, sizeof(float)*n, cudaMemcpyDeviceToDevice);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms1, start, stop); 

    float ms2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int k=0; k<iter; k++){
        if (k%2)
            jacobiOptimizedOnDevice <<< dimGrid, dimBlock >>> (d_x_now,  d_A, d_x_next, n);
        else
            jacobiOptimizedOnDevice <<< dimGrid, dimBlock >>> (d_x_now,  d_A, d_x_next, n);
        cudaMemcpy(d_x_now, d_x_next, sizeof(float)*n, cudaMemcpyDeviceToDevice);
    }    
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms2, start, stop); 
    
    cudaMemcpy(x_d, d_x_next, sizeof(float) * n, cudaMemcpyDeviceToHost);

    free(x_next); free(A); free(x_now); free(b_h);
    cudaFree(d_x_next); cudaFree(d_A); cudaFree(d_x_now); cudaFree(b);
    cudaFree(x_now_u); cudaFree(x_next_u); cudaFree(A_u); cudaFree(b_u);
    

    std::cout << ms1 << std::endl;
    std::cout << ms2 << std::endl;
    std::cout << std::endl;
    return 0;
}
