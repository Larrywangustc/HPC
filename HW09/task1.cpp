#include <iostream>
#include <algorithm>
#include <chrono>
#include <ratio>
#include <omp.h>
#include "cluster.h"
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
    float* arr = new float[n];
    float* centers = new float[t];
    float* dists = new float[t];
    srand((unsigned)time(0));
    for(int i = 0; i < n; i++){
        arr[i] = (rand()) / ( static_cast <float> (RAND_MAX/n));
    }
    for(int i = 0; i < t; i++){
        centers[i] = float(n * (2 * i + 1)) / float(2 * t);
    }
    std::sort(arr, arr+n);
    start = high_resolution_clock::now();
    omp_set_num_threads(t);
    cluster(n, t, arr, centers, dists);
    
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    int k = 0;
    float max = dists[0];
    for(int i=1; i<t; i++){
        if(dists[i]>max){
            k = i;
            max = dists[i];
        }
    }
    cout << max << std::endl;
    cout << k << std::endl;
    cout << duration_sec.count() << std::endl;
    cout << std::endl;
    delete [] arr;
    return 0;
}
