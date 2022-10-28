#include "count.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <iostream>

int main(int argc, char **argv){
    const int n = atoi(argv[1]);
    thrust::host_vector<int> h_vec(n);
    thrust::device_vector<int> counts(n);
    thrust::device_vector<int> values(n);
    srand((unsigned)time(0));
    for(int i=0; i < n; i++){
        h_vec[i] = (rand() % (501));
    }

    thrust::device_vector<int> d_vec(n);
    d_vec = h_vec;
    
    float ms = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    count(d_vec, values, counts);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop); 
    size_t N = values.size();
    std::cout << values[N - 1] << std::endl;
    std::cout << counts[N - 1] << std::endl;
    std::cout << ms << std::endl;
    std::cout << std::endl;



}