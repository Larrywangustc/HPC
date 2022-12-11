#include <iostream>
#include <omp.h>
#include <stdlib.h>             
#include <stdio.h>         
#include <chrono>
#include <ratio>     
#include <fstream>                                    
#include <iomanip>              
#include <time.h>      

using namespace std;              
using std::chrono::high_resolution_clock;
using std::chrono::duration;

void luDecompositionOpenMP(float* L, float* U, const float* A, int n){
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j < i) {
                L[j * n + i] = 0;
            } else {
                L[j * n + i] = A[j * n + i];
                for (int k = 0; k < i; k++) {
                    L[j * n + i] = L[j * n + i] - L[j * n + k] * U[n * k + i];
                }
            }
            if (j < i) {
                U[n * i + j] = 0;
            } else if (j == i) {
                U[n * i + j] = 1;
            } else {
                U[n * i + j] = A[n * i + j] / L[n * i + i];
                for (int k = 0; k < i; k++) {
                    U[n * i + j] = U[n * i + j] - ((L[n * i + k] * U[n * k + j]) / L[n * i + i]);
                }
            }
        }
    }



}

int main(int argc, char **argv){
    const int n = atoi(argv[1]);
    const int t = atoi(argv[2]);

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    float *A, *L, *U;

    A = (float*)malloc(sizeof(float) * n * n);
    L = (float*)malloc(sizeof(float) * n * n);
    U = (float*)malloc(sizeof(float) * n * n);
    srand((unsigned)time(0));
    
    for(int i=0; i < n * n; i++){
        A[i] = float(-1.0) + (rand()) / ( static_cast <float> (RAND_MAX/2.0));
    }



    start = high_resolution_clock::now();
    
    omp_set_num_threads(t);
    luDecompositionOpenMP(L, U, A, n);

    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    std::cout << duration_sec.count() << "\n";
    std::cout << L[n * n - 1] << std::endl;
    std::cout << std::endl;
    free(A);
    free(L);
    free(U);
    return 0;
}