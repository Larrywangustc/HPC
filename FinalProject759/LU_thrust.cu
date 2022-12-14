#include <iostream>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <cuda.h>                                      
#include <time.h>               
#include <cuda_runtime.h>
using namespace std;              

int main(int argc, char **argv){
    const int n = atoi(argv[1]);

    float *A = new float[n * n];
    float *U = new float[n * n];
    float *L = new float[n * n];

    srand((unsigned)time(0));
    
    for(int i=0; i < n * n; i++){
        A[i] = float(-1.0) + (rand()) / ( static_cast <float> (RAND_MAX/2.0));
        L[i] = 0;
        U[i] = 0;
    }

    thrust::device_vector<float> d_A(n * n);
    //float *d_U;
    //float *d_L;
    thrust::copy(A, A + n * n, d_A.begin());
    thrust::device_vector<float> d_L(n * n);
    thrust::device_vector<float> d_U(n * n);
    thrust::copy(L, L + n * n, d_L.begin());
    thrust::copy(U, U + n * n, d_U.begin());
    //cudaMalloc(&d_L, n * n * sizeof(float));
    //cudaMalloc(&d_U, n * n * sizeof(float));

    cudaEvent_t start;
    cudaEvent_t stop;
    float ms;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(d_A.begin(), d_A.begin() + n * n)),
        thrust::make_zip_iterator(thrust::make_tuple(d_A.end(), d_A.end() + n * n)),
        d_A.begin(), 
        d_L.begin(), d_U.begin(),
        [n, &d_L, &d_U, &d_A](thrust::tuple<float, float> t) {
            int i = thrust::get<0>(t);
            int j = thrust::get<1>(t);

            if (i == j) {
                return d_L[i * n + j] * d_U[i * n + j];
            } 
            else if (i < j) {
                float sum = 0;
                for (int k = 0; k < i; k++) {
                    sum += d_L[i * n + k] * d_U[k * n + j];
                }
                return (d_A[i * n + j] - sum) / d_U[i * n + i];
            }
            else {
                float sum = 0;
                for (int k = 0; k < j; k++) {
                    sum += d_L[i * n + k] * d_U[k * n + j];
                }
                    return d_A[i * n + j] - sum;
            }
        }
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop); 

    std::cout << ms << std::endl;
    std::cout << std::endl;
    return 0;
}