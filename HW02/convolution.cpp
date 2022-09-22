#include <iostream>
#include "convolution.h"


void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m){
    std::size_t i, j, k, l;
    std::size_t a = (m - 1) / 2;
    for(i=0;i<n;i++){
        for(j=0;j<n;j++){
            for(k=0;k<m;k++){
                for(l=0;l<m;l++){
                    if((0 <= i + k - a && i + k - a < n) && (0 <= j + l - a && j + l - a < n)){
                        output[n * i + j] += mask[m * k + l] * image[n * i + k - a + j + l - a];
                    }
                    else if((0 <= i + k - a && i + k - a < n) || (0 <= j + l - a && j + l - a < n)){
                        output[n * i + j] += mask[m * k + l];
                    }
                    else{
                        output[n * i + j] += 0;
                    }
                }
            }
        }
    }
}