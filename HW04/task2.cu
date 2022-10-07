#include<cuda.h>
#include<iostream>
#include "stencil.cuh"

int main(int argc, char **argv){
    const int n = atoi(argv[1]);
    const int R = atoi(argv[2]);
    const int threads_per_block = atoi(argv[3]);
    float *image, *mask, *output;

    image = (float*)malloc(sizeof(float)* n);
    mask = (float*)malloc(sizeof(float)* (2 * R + 1));
    output = (float*)malloc(sizeof(float)* n);
    srand((unsigned)time(0));
    for(int i = 0; i < n; i++){
        image[i] = float(-1) + (rand()) / ( static_cast <float> (RAND_MAX/2));
    }
    for(int i = 0; i < (2 * R + 1); i++){
        mask[i] = float(-1) + (rand()) / ( static_cast <float> (RAND_MAX/2));
    }

    float *dA, *dB, *dC;
    cudaMalloc((void**)&dA, sizeof(float)* n);
    cudaMalloc((void**)&dB, sizeof(float)* (2 * R + 1));
    cudaMalloc((void**)&dC, sizeof(float)* n);
    cudaMemset(dC, 0, sizeof(float) * n);
    cudaMemcpy(dA, image, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, mask, sizeof(float) * (2 * R + 1), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEvent_t start;
    cudaEvent_t stop;
    float ms;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
   
    stencil(dA, dB, dC, n, R, threads_per_block);
   
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop); 

    cudaMemcpy(output, dC, sizeof(float) * n, cudaMemcpyDeviceToHost);
    
    std::cout << output[n - 1] << std::endl;
    std::cout << ms << std::endl;
    std::cout << std::endl;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(image);
    free(mask);
    free(output);
    return 0;
}





