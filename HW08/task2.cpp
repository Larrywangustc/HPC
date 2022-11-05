#include <iostream>
#include <chrono>
#include <ratio>
#include "convolution.h"
#include <omp.h>

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char **argv){
    const int n = atoi(argv[1]);
    int t = atoi(argv[2]);
    
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    float* image = new float[n * n];
    int m = 3;
    float* output = new float[n * n];
    float* mask = new float[m * m];;
    srand((unsigned)time(0));
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            image[n * i + j] = float(-10.0) + (rand()) / ( static_cast <float> (RAND_MAX/20.0));
        }
    }
    for(int i = 0; i < m; i++){
        for(int j = 0; j < m; j++){
            mask[m * i + j] = float(-1.0) + (rand()) / ( static_cast <float> (RAND_MAX/2.0));
        }
    }
    omp_set_num_threads(t);
    start = high_resolution_clock::now();
    convolve(image, output, n, mask, m);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    
    cout << output[0] << std::endl;
    cout << output[n * n - 1] << std::endl;
    cout << duration_sec.count() << std::endl;
    cout << std::endl;
    delete [] output;
    delete [] image;
    
    return 0;
}
