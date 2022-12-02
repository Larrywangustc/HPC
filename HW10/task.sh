#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:05:00
#SBATCH -J Slurm
#SBATCH --nodes=2 --cpus-per-task=20 --ntasks-per-node=1
#SBATCH -o Slurm.out -e Slurm.err

module load mpi/mpich/4.0.2

mpicxx task2.cpp reduce.cpp -Wall -O3 -o task2 -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec

for i in {1..20}
do
   srun -n 2 --cpu-bind=none ./task2 10000000 $i
done


