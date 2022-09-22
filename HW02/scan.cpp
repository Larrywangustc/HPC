#include <iostream>
#include "scan.h"

void scan(const float *arr, float *output, std::size_t n){
    int N = n;
    output[0] = arr[0];
    for(int i=1;i<N;i++){
        output[i] = arr[i] + output[i - 1];
    }
}
