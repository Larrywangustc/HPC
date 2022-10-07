#include "stencil.cuh"

__host__ void stencil(const float* image,
                      const float* mask,
                      float* output,
                      unsigned int n,
                      unsigned int R,
                      unsigned int threads_per_block){
    int k = (n - 1) / threads_per_block + 1;
    int l = threads_per_block + 4 * R + 1; 
    stencil_kernel<<<k, threads_per_block, l*sizeof(float)>>>(image, mask, output, n, R);
}

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R){
    extern __shared__ float s[];
    float *Image = s;
    float *Mask = (float*)&Image[blockDim.x + 2 * R];

    int bx = blockIdx.x;
    int tx = threadIdx.x; 
    int bd = blockDim.x;
    int N = n;
    int r = R;
    
    Image[tx + r] = image[bx * bd + tx];
    if(tx < r){
        if(bx * bd + tx - r >= 0){
            Image[tx] = image[bx * bd + tx - r];
        }
        else{
            Image[tx] = 1;
        }
        if(bx * bd + tx + bd < N){
            Image[tx + r + bd] = image[bx * bd + tx + bd];
        }
        else{
            Image[tx + r + bd] = 1;
        }
    }
    if((0 <= tx) & (tx < 2 * r + 1)){
        Mask[tx] = mask[tx];
    }
    __syncthreads();
    float c = 0;
    for(int j=-r;j<r+1;j++){
        c += Image[tx + r + j] * Mask[j + r];
    }
    /*for(int j=-r;j<r+1;j++){
        c += image[bx * bd + tx + j] * mask[j + r];
    }*/
    output[bx * bd + tx] = c;
}