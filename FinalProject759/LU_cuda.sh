#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:02:00
#SBATCH -J Slurm
#SBATCH --ntasks=1 --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH -o Slurm_cuda.out -e Slurm.err

module load nvidia/cuda/11.6.0

nvcc LU_cuda.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o LU_cuda

for i in {5..12}
do
   n=$((2**i))
   ./LU_cuda $n 32
done