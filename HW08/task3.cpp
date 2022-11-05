#include <iostream>
#include <chrono>
#include <ratio>
#include <omp.h>
#include "msort.h"
#include <stdlib.h>
#include <string.h>

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char **argv){
    const int n = atoi(argv[1]);
    int t = atoi(argv[2]);
    int ts = atoi(argv[3]);

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    int* arr = new int[n];
    srand((unsigned)time(0));
    for(int i = 0; i < n; i++){
        arr[i] = int(-1000) + (rand()) / ( static_cast <int> (RAND_MAX/2000));
    }
    
    start = high_resolution_clock::now();
    omp_set_num_threads(t);
    msort(arr, n, ts);

    
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    
    int k = 0;
    for(int i=0;i<n-1;i++){
        if(arr[i]>arr[i+1]){
            k++;
        }
    }
    cout << arr[0] << std::endl;
    cout << arr[n - 1] << std::endl;
    cout << duration_sec.count() << std::endl;
    cout << std::endl;
    delete [] arr;
    return 0;
}
