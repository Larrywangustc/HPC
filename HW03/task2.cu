#include<cuda.h>
#include<iostream>

__global__ void simpleKernel(int* data, int a){
    //this adds a value to a variable stored in global memory
    data[threadIdx.x] = a * threadIdx.x + blockIdx.x;
}

int main(){
    const int numElems= 16;
    int hA[numElems], *dA;
    cudaMalloc((void**)&dA, sizeof(int) * numElems);
    cudaMemset(dA, 0, numElems* sizeof(int));
    
    
    srand((unsigned)time(0));
    int a = int(-10) + (rand()) / ( static_cast <int> (RAND_MAX/20));

    simpleKernel<<<2,8>>>(dA, a);
    //bring the result back from the GPU into the hostArray
    cudaMemcpy(&hA, dA, sizeof(int) * numElems, cudaMemcpyDeviceToHost);

    for (int i = 0; i < numElems; i++){
        std::cout<< hA[i] << " ";
    }
    //release the memory allocated on the GPU 
    cudaFree(dA);
    return 0;
}


