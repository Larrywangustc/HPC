#include<cuda.h>
#include<iostream>
#include "vscale.cuh"

int main(int argc, char **argv){
    const int n = atoi(argv[1]);
    float hA[n], hB[n];
    //float *hB = new float[n];
    float *dA, *dB;
    cudaMalloc((void**)&dA, sizeof(float)* n);
    cudaMalloc((void**)&dB, sizeof(float)* n);
    cudaMemset(dA, 0, n * sizeof(float));
    cudaMemset(dB, 0, n * sizeof(float));

    srand((unsigned)time(0));
    for(int i=0;i<n;i++){
        hA[i] = float(-10) + (rand()) / ( static_cast <float> (RAND_MAX/20));
        hB[i] = (rand()) / ( static_cast <float> (RAND_MAX));
    }
c
    int k = (n - 1) / 512;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
   
    vscale<<<k + 1,512>>>(dA, dB, n); 
   
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(&hB, dB, sizeof(float) * n, cudaMemcpyDeviceToHost);

    std::cout << ms << std::endl;
    std::cout << hB[0] << std::endl;
    std::cout << hB[n - 1] << std::endl;
    std::cout << std::endl;
    cudaFree(dA);
    cudaFree(dB);
    return 0;
}


