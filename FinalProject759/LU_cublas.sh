#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:02:00
#SBATCH -J Slurm
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH -o Slurm_cublas.out -e Slurm.err

module load nvidia/cuda/11.6.0

nvcc LU_cublas.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std c++17 -o LU_cublas

./LU_cublas 2048