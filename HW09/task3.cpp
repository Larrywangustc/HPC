#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <stdio.h>
#include <mpi.h>

using namespace std;
using chrono::high_resolution_clock;
using chrono::duration;

int main(int argc, char* argv[]){
    int my_rank, p;
    int dest1 = 1, dest2 = 0;
    int source1 = 1, source2 = 0;
    int tag1 = 0, tag2 = 1;
    int n = atoi(argv[1]);
    float* A_send = (float *)malloc(n * sizeof(float));
    float* B_send = (float *)malloc(n * sizeof(float));
    float* A_recieve = (float *)malloc(n * sizeof(float));
    float* B_recieve = (float *)malloc(n * sizeof(float));
    for(int i=0; i<n; i++){
        A_send[i] = 1.0;
        B_send[i] = -1.0;
    }
    MPI_Status status; /* return status for receive  */
    MPI_Init(&argc, &argv); // Start up MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); // Find out process rank
    MPI_Comm_size(MPI_COMM_WORLD, &p); // Find out number of processes

    high_resolution_clock::time_point start1;
    high_resolution_clock::time_point start2;
    high_resolution_clock::time_point end1;
    high_resolution_clock::time_point end2;
    duration<double, std::milli> duration_sec1; 
    duration<double, std::milli> duration_sec2; 
    double time;
    //double t1=0, t2=0, t3=0, t4=0;

    if(my_rank==0){
        start1 = high_resolution_clock::now();
        //t1 = MPI_Wtime(); 
        MPI_Send(A_send,n,MPI_FLOAT,dest1,tag1,MPI_COMM_WORLD);
        MPI_Recv(A_recieve,n,MPI_FLOAT,source1,tag2,MPI_COMM_WORLD,&status);
        end1 = high_resolution_clock::now();
        //t2 = MPI_Wtime(); 
    } 
    else if(my_rank==1){
        start2 = high_resolution_clock::now();
        //t3 = MPI_Wtime(); 
        MPI_Recv(B_recieve,n,MPI_FLOAT,source2,tag1,MPI_COMM_WORLD,&status);
        MPI_Send(B_send,n,MPI_FLOAT,dest2,tag2,MPI_COMM_WORLD);
        end2 = high_resolution_clock::now();
        //t4 = MPI_Wtime(); 
    }
    duration_sec1 = std::chrono::duration_cast<duration<double, std::milli>>(end1 - start1);
    duration_sec2 = std::chrono::duration_cast<duration<double, std::milli>>(end2 - start2);
    time = duration_sec1.count() + duration_sec2.count(); 
    if(my_rank==0){
        cout << time << std::endl;
    }
    //cout << n << std::endl;
    //cout << A_recieve[n-1] << std::endl;
    //cout << B_recieve[n-1] << std::endl;
    //cout << t2 - t1 + t4 - t3 << std::endl;
    
    return 0;

}
