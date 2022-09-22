#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:05:00
#SBATCH -J Slurm
#SBATCH -o Slurm.out -e Slurm.err

g++ scan.cpp task1.cpp -Wall -O3 -std=c++17 -o task1
for i in {10..30}
do
   n=$((2**i))
   ./task1 n
done

g++ convolution.cpp task2.cpp -Wall -O3 -std=c++17 -o task2
./task2 100 9

g++ task3.cpp matmul.cpp -Wall -O3 -std=c++17 -o task3
./task3