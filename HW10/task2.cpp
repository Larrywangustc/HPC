#include <cstddef>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <sstream>
#include <stdio.h>
#include <omp.h>
#include "reduce.h"

using namespace std;
using namespace std::chrono;


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

    double start = high_resolution_clock::now();
    float result = reduce(arr, 0, n);
    MPI_Reduce(&result, &global_res, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    double end = high_resolution_clock::now();
    double ms = duration_cast<duration<double, std::milli>>(end - start).count();

    if (rank == 0) {
        cout << global_res << endl;
        cout << ms << endl;
        cout << endl;
    }

    MPI_Finalize();
}