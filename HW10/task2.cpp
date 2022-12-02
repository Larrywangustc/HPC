#include <cstddef>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <sstream>
#include <stdio.h>
#include <omp.h>
#include "reduce.h"
#include <mpi.h>

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration;


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int n = atol(argv[1]);
    int t = atol(argv[2]);
    omp_set_num_threads(t);

    float *arr = new float[n];
    for (int i = 0; i < n; i++) {
        arr[i] = float(-1.0) + (rand()) / ( static_cast <float> (RAND_MAX/2.0));
    }

    float global_res = 0;
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    reduce(arr, 0, n);
    MPI_Barrier(MPI_COMM_WORLD);

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    
    start = high_resolution_clock::now();
    float result = reduce(arr, 0, n);
    MPI_Reduce(&result, &global_res, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    if (rank == 0) {
        cout << global_res << endl;
        cout << duration_sec.count() << endl;
        cout << endl;
    }

    MPI_Finalize();
}