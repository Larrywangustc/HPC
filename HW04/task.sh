#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:03:00
#SBATCH -J Slurm
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH -o Slurm.out -e Slurm.err

module load nvidia/cuda

nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1
for i in {5..14}
do
   n=$((2**i))
   ./task1 $n 1024
done
