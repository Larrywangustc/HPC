#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <cuda_runtime.h>

void jacobiOnHost(float* x_next, float* A, float* x_now, float* b_h, int Ni, int Nj)
{
    int i,j;
    float sigma;
    
    for (i=0; i<Ni; i++)
    {
        sigma = 0.0;
        for (j=0; j<Nj; j++)
        {
            if (i != j)
                sigma += A[i*Nj + j] * x_now[j]; // From the
            // argothum sigma is the Rx, and also matrix A is
            // seperated into the Nj + j and Ni + i
        }
        x_next[i] = (b_h[i] - sigma) / A[i*Ni + i];
    }
}

__constant__ float b[512];
__global__ void jacobiConstantOnDevice(float* d_x_next, float* d_A, float* d_x_now,  int Ni, int Nj)
{
    // Optimization step 1: tiling
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (i < Ni)
    {
        float sigma = 0.0;
        
        int idx_Aci = i*Nj;
        
        for (int j=0; j<Nj; j++)
            if (i != j)
                sigma += d_A[idx_Aci + j] * d_x_now[j];
        d_x_next[i] = (b[i] - sigma) / d_A[idx_Aci + i];
        
    }
}

#define Tile_Width 32
__constant__ float b_s[512];
__global__ void jacobiOptimizedOnDevice(float* d_x_next, float* d_A, float* d_x_now,  int Ni, int Nj)
{
    __shared__ float xdsn[Tile_Width];
    __shared__ float xdsx[Tile_Width];
    // Optimization step 1: tiling
    //read the matrix tile into shared memory
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int xIndex = bx * Tile_Width + tx;
    
    int idx = xIndex * Tile_Width + threadIdx.x;
    if (idx < Ni) {
        float sigma = 0.0;
        int idx_Ai = idx * Nj;
        for (int j=0; j<Tile_Width; j++) {
            if (idx != j) {
                xdsn[tx] = d_x_now[idx*Tile_Width];
                xdsx[tx] = d_x_next[idx * Tile_Width];

                sigma += d_A[idx_Ai + j] * xdsn[tx];
                    
                xdsx[tx] = (b_s[idx] - sigma) / d_A[idx_Ai + idx];
                
  
            }
            
        }
        for (int k=0; k<Ni; k++) {
            d_x_next[k* Ni] = xdsx[tx];
        }   
    }
}