#include <cstddef>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <stdio.h>
#include <omp.h>
#include "reduce.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char *argv[]) {

    int n = atol(argv[1]);
    int t = atol(argv[2]);
    omp_set_num_threads(t);

    float *arr = new float[n];
    for (int i = 0; i < n; i++) {
        arr[i] = float(-1.0) + (rand()) / ( static_cast <float> (RAND_MAX/2.0));
    }

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    
    start = high_resolution_clock::now();
    float result = reduce(arr, 0, n);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);


    cout << global_res << endl;
    cout << duration_sec.count() << endl;
    cout << endl;

}