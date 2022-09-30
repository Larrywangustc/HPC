#include<cuda.h>
#include<iostream>
#include <chrono>
#include <ratio>
#include "vscale.cuh"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char **argv){
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    unsigned int n = atoi(argv[1]);
    float *a = new float[n];
    float *b = new float[n];

    srand((unsigned)time(0));
    for(int i;i<n;i++){
        a[i] = float(-10) + (rand()) / ( static_cast <float> (RAND_MAX/20));
        b[i] = (rand()) / ( static_cast <float> (RAND_MAX));
    }

    int k = (n - 1) / 512;

    start = high_resolution_clock::now();
    vscale<<<k + 1,512>>>(a, b, n);
    end = high_resolution_clock::now();
    
    std::cout << duration_sec.count() << std::endl;
    std::cout << b[0] << std::endl;
    std::cout << b[n - 1] << std::endl;

    //release the memory allocated on the GPU 
    return 0;
}


