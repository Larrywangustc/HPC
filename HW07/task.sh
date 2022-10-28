#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:05:00
#SBATCH -J Slurm
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --nodes=1 --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH -o Slurm.out -e Slurm.err

module load nvidia/cuda/11.6.0.lua
module load nvidia/cuda gcc/9.4.0

nvcc task1_thrust.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1_thrust

for i in {10..30}
do
   n=$((2**i))
   ./task1_thrust $n
done

nvcc task1_cub.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1_cub

for i in {10..30}
do
   n=$((2**i))
   ./task1_cub $n
done

nvcc task2.cu count.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2

for i in {5..24}
do
   n=$((2**i))
   ./task2 $n
done

g++ task3.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

./task3