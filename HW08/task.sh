#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:05:00
#SBATCH -J Slurm
#SBATCH --ntasks=1 --cpus-per-task=20
#SBATCH --cpus-per-task=20
#SBATCH -o Slurm.out -e Slurm.err

g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

for i in {1..2}
do
   ./task1 1024 $i
done

g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp

for i in {1..2}
do
   ./task2 1024 $i
done

g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

for i in {1..2}
do
   ts=$((2**i))
   ./task3 1000000 8 $ts
done

for i in {1..20}
do
   ./task3 1000000 $i 64
done