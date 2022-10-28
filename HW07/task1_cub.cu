#define CUB_STDERR // print CUDA runtime errors to console
#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include "cub/util_debug.cuh"
using namespace cub;
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

int main(int argc, char **argv){
    const int n = atoi(argv[1]);
    // Set up host arrays
    float h_in[n];
    srand((unsigned)time(0));
    for(int i=0; i < n; i++){
        h_in[i] = float(-1.0) + (rand()) / ( static_cast <float> (RAND_MAX/2.0));
    }

    // Set up device arrays
    float* d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_in, sizeof(float) * n));
    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice));
    // Setup device output array
    float* d_sum = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_sum, sizeof(float) * 1));
    // Request and allocate temporary storage
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Do the actual reduce operation
    float ms = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop); 

    float gpu_sum;
    CubDebugExit(cudaMemcpy(&gpu_sum, d_sum, sizeof(float) * 1, cudaMemcpyDeviceToHost));

    std::cout << gpu_sum << std::endl;
    std::cout << ms << std::endl;
    std::cout << std::endl;

    // Cleanup
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_sum) CubDebugExit(g_allocator.DeviceFree(d_sum));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    
    return 0;
}