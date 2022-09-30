#include<cuda.h>
#include<iostream>
#include "vscale.cuh"

int main(int argc, char **argv){
    unsigned int n = atoi(argv[1]);
    float *a = new float[n];
    float *b = new float[n];

    srand((unsigned)time(0));
    for(int i=0;i<n;i++){
        a[i] = float(-10) + (rand()) / ( static_cast <float> (RAND_MAX/20));
        b[i] = (rand()) / ( static_cast <float> (RAND_MAX));
    }

    int k = (n - 1) / 512;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
   
    vscale<<<k + 1,512>>>(a, b, n); 
   
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << ms << std::endl;
    std::cout << b[0] << std::endl;
    std::cout << b[n - 1] << std::endl;
    std::cout << std::endl;
    //release the memory allocated on the GPU 
    return 0;
}


