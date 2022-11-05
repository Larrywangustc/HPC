#include <omp.h>
#include "msort.h"
#include <stdlib.h>
#include <string.h>


void merge(int * X, int n) {
    int *tmp = new int[n];
    int i = 0;
    int j = n/2;
    int ti = 0;

    while (i<n/2 && j<n) {
        if (X[i] <= X[j]) {
            tmp[ti] = X[i];
            ti++; i++;
        } else {
            tmp[ti] = X[j];
            ti++; j++;
        }
    }
    while (i<n/2) { 
        tmp[ti] = X[i];
        ti++; i++;
    }
    while (j<n) {
        tmp[ti] = X[j];
        ti++; j++;
    }
    memcpy(X, tmp, n*sizeof(int));
    delete [] tmp;
}


void mergesort(int* arr, const std::size_t n, const std::size_t threshold){
    int k = threshold;
    if (n < k){
        int i, j, temp;
        for(i=0; i<(n-1); i++){
            for(j=0; j<(n-i-1); j++){
                if(arr[j]>arr[j+1]){
                    temp = arr[j];
                    arr[j] = arr[j+1];
                    arr[j+1] = temp;
                }
            }
        }
        return;
    }
    #pragma omp task firstprivate (arr, n)
    mergesort(arr, n/2, threshold);
    #pragma omp task firstprivate (arr, n)
    mergesort(arr+n/2, n-n/2, threshold);
    #pragma omp taskwait
    merge(arr, n);
}


void msort(int* arr, const std::size_t n, const std::size_t threshold){
    #pragma omp parallel
    {
        #pragma omp single
        mergesort(arr, n, threshold);
    }
}
