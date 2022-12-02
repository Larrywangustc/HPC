#include <iostream>
#include <chrono>
#include <ratio>
#include <omp.h>
#include "optimize.h"
#include <stdlib.h>

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char **argv){
    const int n = atoi(argv[1]);

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    vec v(n);
    data_t dest;
    data_t *A = new data_t[n];
    v.data = A;
    data_t *d = v.data;
    srand((unsigned)time(0));
    for(int i = 0; i < n; i++){
        d[i] = 1;
    }


    start = high_resolution_clock::now();
    
    optimize1(&v, &dest);
    
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    cout << dest << std::endl;
    cout << duration_sec.count() << std::endl;
    cout << std::endl;

    start = high_resolution_clock::now();
    
    optimize2(&v, &dest);
    
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    cout << dest << std::endl;
    cout << duration_sec.count() << std::endl;
    cout << std::endl;

    start = high_resolution_clock::now();
    
    optimize3(&v, &dest);
    
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    cout << dest << std::endl;
    cout << duration_sec.count() << std::endl;
    cout << std::endl;

    start = high_resolution_clock::now();
    
    optimize4(&v, &dest);
    
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    cout << dest << std::endl;
    cout << duration_sec.count() << std::endl;
    cout << std::endl;

    start = high_resolution_clock::now();
    
    optimize5(&v, &dest);
    
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    cout << dest << std::endl;
    cout << duration_sec.count() << std::endl;
    cout << std::endl;
    return 0;
}
