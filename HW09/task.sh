#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:05:00
#SBATCH -J Slurm
#SBATCH --cpus-per-task=10
#SBATCH -o Slurm.out -e Slurm.err

module load mpi/mpich/4.0.2

g++ task1.cpp cluster.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

for i in {1..10}
do
   ./task1 5040000 $i
done

g++ task2.cpp montecarlo.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec

for i in {1..2}
do
   ./task2 1000000 $i
done

mpicxx task3.cpp -Wall -O3 -o task3

for i in {1..2}
do
   n=$((2**i))
   srun -n 2 task3 $n
done