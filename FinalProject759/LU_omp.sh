#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:05:00
#SBATCH -J Slurm
#SBATCH --ntasks=1 --cpus-per-task=10
#SBATCH --cpus-per-task=10
#SBATCH -o Slurm_omp.out -e Slurm.err

g++ LU_omp.cpp -Wall -O3 -std=c++17 -o LU_omp -fopenmp

./LU_omp 2048 10

