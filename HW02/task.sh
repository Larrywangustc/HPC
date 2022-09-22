#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:01:00
#SBATCH -J Slurm
#SBATCH -o Slurm.out -e Slurm.err


g++ convolution.cpp task2.cpp -Wall -O3 -std=c++17 -o task2
./task2 10 3

