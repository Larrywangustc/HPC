#include<cuda.h>
#include<iostream>

__global__ void simpleKernel(int* data){
    int n = 1;
    for(int i=1;i<=threadIdx.x+1;i++){
        n *= i;
    }
    printf("%d!=%d\n", threadIdx.x+1, n);
}

int main(){
    const int numElems= 8;
    int Array[numElems];

    simpleKernel<<<1,numElems>>>(Array);
    cudaDeviceSynchronize();
}
