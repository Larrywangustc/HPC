#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:06:00
#SBATCH -J Slurm
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH -o Slurm.out -e Slurm.err

module load nvidia/cuda/11.6.0

nvcc task1.cu reduce.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1
###for i in {10..30}
###do
###   n=$((2**i))
###   ./task1 $n 16
###done

nvcc task2.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2
for i in {5..14}
do
   n=$((2**i))
   ./task2 $n 16
done