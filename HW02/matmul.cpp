#include <iostream>
#include "matmul.h"

void mmul1(const double* A, const double* B, double* C, const unsigned int n){
    int N = n;
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            for(int k=0;k<N;k++){
                C[n * i + j] += A[n * i + k] * B[n * k + j];
            }
        }
    }
}

void mmul2(const double* A, const double* B, double* C, const unsigned int n){
    int N = n;
    for(int i=0;i<N;i++){
        for(int k=0;k<N;k++){
            for(int j=0;j<N;j++){  
                C[n * i + j] += A[n * i + k] * B[n * k + j];
            }
        }
    }
}

void mmul3(const double* A, const double* B, double* C, const unsigned int n){
    int N = n;
    for(int j=0;j<N;j++){
        for(int k=0;k<N;k++){
            for(int i=0;i<N;i++){
                C[n * i + j] += A[n * i + k] * B[n * k + j];
            }
        }
    }
}

void mmul4(const std::vector<double>& A, const std::vector<double>& B, double* C, const unsigned int n){
    int N = n;
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            for(int k=0;k<N;k++){
                C[n * i + j] += A[n * i + k] * B[n * k + j];
            }
        }
    }
}
