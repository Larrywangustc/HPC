#include <iostream>
#include <chrono>
#include <ratio>
#include <cmath>
#include <vector>
#include "matmul.h"

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(){
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec1;
    duration<double, std::milli> duration_sec2;
    duration<double, std::milli> duration_sec3;
    duration<double, std::milli> duration_sec4;
    int n = 1024;
    double* A = new double[n * n];
    double* B = new double[n * n];
    std::vector<double>A_vec(n * n);
    std::vector<double>B_vec(n * n);

    srand((unsigned)time(0));
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            A[n * i + j] = double(-1.0) + (rand()) / ( static_cast <double> (RAND_MAX/2.0));
            B[n * i + j] = double(-1.0) + (rand()) / ( static_cast <double> (RAND_MAX/2.0));
            A_vec[n * i + j] = A[n * i + j];
            B_vec[n * i + j] = B[n * i + j];
        }
    }
    cout << n << std::endl;
    double* C = new double[n * n];
    start = high_resolution_clock::now();
    mmul1(A, B, C, n);
    end = high_resolution_clock::now();
    duration_sec1 = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << duration_sec1.count() << "\n";
    cout << C[n * n - 1] << "\n";
    
    for(int i = 0; i < n * n; i++){
        C[i] = 0;
    }
    start = high_resolution_clock::now();
    mmul2(A, B, C, n);
    end = high_resolution_clock::now();
    duration_sec2 = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << duration_sec2.count() << "\n";
    cout << C[n * n - 1] << "\n";

    for(int i = 0; i < n * n; i++){
        C[i] = 0;
    }
    start = high_resolution_clock::now();
    mmul3(A, B, C, n);
    end = high_resolution_clock::now();
    duration_sec3 = std::chrono::duration_cast<duration<double, std::milli>>(end - start);\
    cout << duration_sec3.count() << "\n";
    cout << C[n * n - 1] << "\n";

    for(int i = 0; i < n * n; i++){
        C[i] = 0;
    }
    start = high_resolution_clock::now();
    mmul4(A_vec, B_vec, C, n);
    end = high_resolution_clock::now();
    duration_sec4 = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << duration_sec4.count() << "\n";
    cout << C[n * n - 1] << "\n";
    
    delete [] C;
    delete [] A;
    delete [] B;
    return 0;
}











