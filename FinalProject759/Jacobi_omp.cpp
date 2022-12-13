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

void jacobiOnHost(float* x_next, float* A, float* x_now, float* b, int N){
    float sigma;
    #pragma omp parallel for
    for (int i=0; i<N; i++){
        sigma = 0.0;
        for (int j=0; j<N; j++)
        {
            if (i != j)
                sigma += A[i*N + j] * x_now[j];
        }
        x_next[i] = (b[i] - sigma) / A[i*N + i];
    }
}


int main(int argc, char **argv){
    const int n = atoi(argv[1]);
    const int t = atoi(argv[2]);
    int iter = 1000;
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

    float *A, *b, *x_next, *x_now;

    A = (float*)malloc(sizeof(float) * n * n);
    b = (float*)malloc(sizeof(float) * n);
    x_next = (float*)malloc(sizeof(float) * n);
    x_now = (float*)malloc(sizeof(float) * n);
    srand((unsigned)time(0));
    
    for(int i=0; i < n * n; i++){
        A[i] = float(-1.0) + (rand()) / ( static_cast <float> (RAND_MAX/2.0));
    }
    for(int i=0; i < n; i++){
        b[i] = float(-1.0) + (rand()) / ( static_cast <float> (RAND_MAX/2.0));
        x_now[i] = 0;
        x_next[i] = 0;
    }

    start = high_resolution_clock::now();
    
    omp_set_num_threads(t);
    #pragma omp parallel for
    for (int k=0; k<iter; k++){
        jacobiOnHost(x_next, A, x_now, b, n);
        for (int i=0; i<n; i++)
            x_now[i] = x_next[i];
    }

    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    std::cout << duration_sec.count() << std::endl;
    std::cout << std::endl;
    free(A);
    free(x_next);
    free(x_now);
    free(b);
    return 0;
}