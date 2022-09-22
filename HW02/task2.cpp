#include <iostream>
#include <chrono>
#include <ratio>
#include <cmath>
#include "convolution.h"


using std::chrono::high_resolution_clock;
using std::chrono::duration;


int main(int argc, char **argv){
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    //cout << "yy" << "\n";
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    float* image = new float[n * n];
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
    start = high_resolution_clock::now();
    convolve(image, output, n, mask, m);
    end = high_resolution_clock::now();
    
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    
    std::cout << duration_sec.count() << std::endl;
    std::cout << output[0] << std::endl;
    std::cout << output[n * n - 1] << std::endl;
    delete [] image;
    delete [] mask;
    delete [] output;
    return 0;
}


