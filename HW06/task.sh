#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:03:00
#SBATCH -J Slurm
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH -o Slurm.out -e Slurm.err

module load nvidia/cuda/11.6.0

nvcc task1.cu mmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std c++17 -o task1
###for i in {5..14}
###do
###   n=$((2**i))
###   ./task1 $n 64
###done

nvcc task2.cu scan.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2
cuda-memcheck ./task2 100000 1024
for i in {10..20}
do
   n=$((2**i))
   ./task2 $n 1024
done