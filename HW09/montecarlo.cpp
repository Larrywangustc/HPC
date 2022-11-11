#include <omp.h>
#include "montecarlo.h"
#include <math.h>
int montecarlo(const size_t n, const float *x, const float *y, const float radius){
    int incircle = 0;
    #pragma omp simd{
    for (size_t i = 0; i < n; i++) {
        incircle += (x[i] * x[i] + y[i] * y[i] < radius * radius) ? 1 : 0;
        //incircle += (pow(x[i], 2)+pow(y[i], 2)< pow(radius, 2)) ? 1 : 0;
    }
    }
    return incircle;
}

/*int montecarlo(const size_t n, const float *x, const float *y, const float radius){
    int incircle = 0;
    #pragma omp parallel reduction(+:incircle)
    {
    #pragma omp for
    for (size_t i = 0; i < n; i++) {
        incircle += (x[i] * x[i] + y[i] * y[i] < radius * radius) ? 1 : 0;
    }
    }
    return incircle;
}*/
