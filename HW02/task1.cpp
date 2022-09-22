#include <iostream>
#include <chrono>
#include <ratio>
#include <cmath>
#include "scan.h"

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;


int main(int argc, char **argv){
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    int n = atoi(argv[1]);
    float* a=new float[n];
    float* b=new float[n];
    srand((unsigned)time(0));
    for(int i=0;i<n;i++){
        a[i] = float(-1.0) + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/2.0));
    }
    
    start = high_resolution_clock::now();
    scan(a, b, n);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << duration_sec.count() << "\n";
    cout << b[0] << "\n";
    cout << b[n - 1] << "\n";
    free(a);
    free(b);
    
    return 0;
}


