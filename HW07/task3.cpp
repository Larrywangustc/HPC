#include <iostream>
#include <omp.h>
using namespace std;


int main() {
    omp_set_num_threads(4);
    int n = 8;
    #pragma omp parallel
    #pragma omp master
    {
        int nThreads = omp_get_num_threads();
        printf("Number of threads:  %d \n", nThreads);
    }
    #pragma omp parallel
    {
        int myId = omp_get_thread_num();
        printf("I'm thread No.  %d\n", myId);
    }
    #pragma omp parallel for
        for(int i=1; i<=n; i++){ 
            int factorial = 1;
            for(int j=1; j<=i; j++){
                factorial = factorial * j;
            } 
            printf("%d!=%d\n", i, factorial);
        }
}