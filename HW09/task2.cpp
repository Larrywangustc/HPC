#include <iostream>
#include <algorithm>
#include <chrono>
#include <ratio>
#include <omp.h>
#include "montecarlo.h"
#include <stdlib.h>
#include <string.h>

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char **argv){
    const int n = atoi(argv[1]);
    int t = atoi(argv[2]);
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    float* X = new float[n];
    float* Y = new float[n];
    srand((unsigned)time(0));
    for(int i = 0; i < n; i++){
        X[i] = float(-1.0) + (rand()) / ( static_cast <float> (RAND_MAX/2.0));
        Y[i] = float(-1.0) + (rand()) / ( static_cast <float> (RAND_MAX/2.0));
    }

    start = high_resolution_clock::now();
    omp_set_num_threads(t);
    int incircle = montecarlo(n, X, Y, 1.0);
    
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    float pi = float(4.0 * incircle) / float(n);
    cout << pi << std::endl;
    cout << duration_sec.count() << std::endl;
    cout << std::endl;
    delete [] X;
    delete [] Y;
    return 0;
}
