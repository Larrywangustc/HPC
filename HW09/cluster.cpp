#include "cluster.h"
#include <cmath>
#include <iostream>

void cluster(const size_t n, const size_t t, const float *arr,
             const float *centers, float *dists) {
#pragma omp parallel num_threads(t)
  {
    unsigned int tid = omp_get_thread_num();
    float* dists_private = new float[t];
#pragma omp for
    for (size_t i = 0; i < n; i++) {
      dists_private[tid] += std::fabs(arr[i] - centers[tid]);
    }
#pragma omp critical
    for (size_t i = 0; i < t; ++i) {
      dists[i] += dists_private[i];
    }
  }
}