#include "stencil.cuh"

__host__ void stencil(const float* image,
                      const float* mask,
                      float* output,
                      unsigned int n,
                      unsigned int R,
                      unsigned int threads_per_block){
    int k = (n - 1) / threads_per_block + 1;
    int l = threads_per_block * 2 + 4 * R + 1; 
    stencil_kernel<<<k, threads_per_block, l>>>(image, mask, output, n, R);
}

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R){
    extern__shared__ int s[];
    float *Image = s;
    float *Mask = (float*)&Image[threads_per_block + 2 * R];
    float *Output = (float*)&Mask[2 * R + 1];  
    int bx = blockIdx.x;
    int tx= threadIdx.x; 
    

    float c = 0;


    // Write the block sub-matrix to global memory;
    // each thread writes one element
    int c = wB* BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB* ty + tx] = Csub;

}