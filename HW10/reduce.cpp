#include <cstddef>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <sstream>
#include <stdio.h>
#include <cstring>
#include <algorithm>
#include <omp.h>
#include "reduce.h"

float reduce(const float* arr, const size_t l, const size_t r) {
    float output = 0;
#pragma omp parallel for simd reduction(+ : output)
    for (size_t i = l; i < r; i++)
        output += arr[i];
    return output;
}