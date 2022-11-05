#include <omp.h>
#include "matmul.h"


void mmul(const float* A, const float* B, float* C, const std::size_t n){
    int N = n;
    int i, k, j;
    #pragma omp parallel shared(A, B, C) private(i, j, k)
    {
        #pragma omp for schedule (dynamic)
        for(i=0; i<N; i++){
            for(k=0; k<N; k++){
                for(j=0; j<N; j++){  
                    C[n * i + j] += A[n * i + k] * B[n * k + j];
                }
            }
        }
    }
}
