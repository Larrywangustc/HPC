#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/transform.h>
#include<thrust/reduce.h>
#include<thrust/functional.h>
#include<iostream>

int main(int argc, char **argv){
    const int n = atoi(argv[1]);
    thrust::host_vector<float> h_vec(n);
    srand((unsigned)time(0));
    for(int i=0; i < n; i++){
        h_vec[i] = float(-1.0) + (rand()) / ( static_cast <float> (RAND_MAX/2.0));
    }

    thrust::device_vector<float> d_vec(n);
    d_vec = h_vec;
    
    
    float ms = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    float result = thrust::reduce(d_vec.begin(), d_vec.end(), (float) 0, thrust::plus<float>());
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop); 
    std::cout << result << std::endl;
    std::cout << ms << std::endl;
    std::cout << std::endl;

}