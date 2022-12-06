#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <cuda_runtime.h>
using namespace std;
#define TILE 32


// LU Decomposition using Shared Memory 


//Initialize a 2D matrix   
void initialize_matrices(double** a, double** l, double** u, int size){ 
    for (int i = 0; i < size; ++i)
    {
        a[i] = new double[size];
        l[i] = new double[size];
        u[i] = new double[size];
    }
}


// Scale the index for threads to get pivot starting and ending points
__global__ void scaleIndex(double *matrix, int n, int index){
    int start = (index * n + index);
	int end = (index * n + n);
	
	for(int i = start+1; i < end; ++i){
		matrix[i] = (matrix[i] / matrix[start]);
	}

}


// Row elimination Kernel - takes matrix, dimension, currect row index, and block size

__global__ void elim(double *A, int n, int index, int bsize){
	extern __shared__ double pivot[];

	int idThread = threadIdx.x;
	int idBlock = blockIdx.x;
	int blockSize = bsize;

	if(idThread == 0){
	    for(int i = index; i < n; i++){
            pivot[i] = A[(index * n) + i];
        }
	}

	__syncthreads();
    //Variables for pivot, row, start and end
	int pivotRow = (index * n);
	int currentRow = (((blockSize * idBlock) + idThread) * n);
	int start = currentRow + index;
	int end = currentRow + n;
    //If greater than pivot row, loop from start index + 1(next row) to end of column
	if(currentRow > pivotRow){
        for(int i = start+1; i<end; ++i){
            //Set the matrix value of next row and its column - pivot
            A[i] = A[i] - (A[start] * pivot[i-currentRow]);
        }
    }
}





int main(int argc, char** argv){
    int n = atoi(argv[1]);

    //Allocate A matrix, U, and L  for CPU
    double *a = new double[n*n];
    double *ret = new double[n*n];

    srand((unsigned)time(0));
    for(int i=0; i < n * n; i++){
        a[i] = double(-1.0) + (rand()) / ( static_cast <double> (RAND_MAX/2.0));
    }

    
    double *da;
    int numblock = n/TILE + ((n % TILE)?1:0);

    cudaEvent_t start;
    cudaEvent_t stop;
    float ms;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
   
    cudaMalloc(&da, n*n* sizeof (double));
    cudaMemcpy(da, a, n*n* sizeof(double), cudaMemcpyHostToDevice);
    for(int i=0; i<n; ++i){
        scaleIndex<<<1,1>>>(da,n,i);
	    elim<<<numblock,TILE,n*sizeof(double)>>>(da,n,i,TILE);
    }
    cudaMemcpy(ret, da, n*n*sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop); 


    cout << "Runtime for LU Decomposition is: " << ms << endl;

    double** A = new double* [n];
    double** u = new double* [n];
    double** l = new double* [n];
 

    initialize_matrices(A, u, l, n);
    for(int i = 0 ;i < n ; ++i){
        for(int j = 0; j < n; ++j){
            A[i][j] = ret[i*n+j];
        }
    }
    //Take values diagonal values from returned array and pull L and U
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            //Find diagonals
            for(int k=0; k<n; k++){
                //If the outermost for loop is larger or equal to k, then grab L values
                if(i>=k)
                    l[i][k] = A[i][k];
                //Else the rest of the array is zeroes
                else l[i][k] = 0;
                //If loops at diagonal then enter 1 for U, if j > k then we're on upper part of Matrix so fill in values, 
                if(k==j) u[k][j] = 1;
                else if(k<j) u[k][j] = A[k][j];
                else u[k][j] = 0.0;
            }
        }
    }



    cudaFree(da);
    cudaFree(ret);
    delete[] a;
    delete[] ret; 

    return 0;
}