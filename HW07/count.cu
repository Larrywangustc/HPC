#include "count.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/distance.h>
#include <iostream>

void count(const thrust::device_vector<int>& d_in, thrust::device_vector<int>& values, thrust::device_vector<int>& counts){
    size_t N = d_in.size();
    thrust::device_vector<int> d_in_copy(N);
    thrust::copy(d_in.begin(), d_in.end(), d_in_copy.begin());
    thrust::sort(d_in_copy.begin(), d_in_copy.end());
    thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end = thrust::reduce_by_key(thrust::device, d_in_copy.begin(), d_in_copy.end(), d_in_copy.begin(), values.begin(), counts.begin());
    values.resize(thrust::distance(values.begin(), new_end.first));
    counts.resize(thrust::distance(counts.begin(), new_end.second));
    thrust::transform(counts.begin(), counts.end(), values.begin(), counts.begin(), thrust::divides<float>());
}