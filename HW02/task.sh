#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:05:00
#SBATCH -J Slurm
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH -o Slurm.out -e Slurm.err


###g++ scan.cpp task1.cpp -Wall -O3 -std=c++17 -o task1
###for i in {10..30}
###do
###   n=$((2**i))
###   ./task1 $n
###done

g++ convolution.cpp task2.cpp -Wall -O3 -std=c++17 -o task2
for i in {9..13}
do
   n=$((2**i))
   ./task2 $n 16
done

g++ task3.cpp matmul.cpp -Wall -O3 -std=c++17 -o task3
for i in {5..12}
do
   n=$((2**i))
   ./task3 $n
done