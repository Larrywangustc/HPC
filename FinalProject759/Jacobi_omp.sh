#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:05:00
#SBATCH -J Slurm
#SBATCH --ntasks=1 --cpus-per-task=20
#SBATCH --cpus-per-task=20
#SBATCH -o Slurm_omp_j.out -e Slurm.err

g++ Jacobi_omp.cpp -Wall -O3 -std=c++17 -o Jacobi_omp -fopenmp

./Jacobi_omp 2048 10

for i in {5..11}
do
   n=$((2**i))
   ./Jacobi_omp $n 10
done

for i in {1..20}
do
   ./Jacobi_omp 2048 $i
done
