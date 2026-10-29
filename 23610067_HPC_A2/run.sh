#!/bin/bash
# A2 — OpenMP Parallel Loops & Scheduling
# Run from inside 23610067_HPC_A2/
#
# Prerequisites (Ubuntu):
#   sudo apt update
#   sudo apt install gcc libgomp1

set -e
cd "$(dirname "$0")"
mkdir -p bin

echo "=== Compiling ==="
gcc -fopenmp code/t1_matrix-addition.c       -o bin/t1_matrix-addition
gcc -fopenmp code/t2_sum-of-numbers.c        -o bin/t2_sum-of-numbers
gcc -fopenmp code/t3_matrix-multiplication.c -o bin/t3_matrix-multiplication
gcc -fopenmp code/t4_scheduling-strategies.c -o bin/t4_scheduling-strategies
gcc -fopenmp code/t5_loop-collapse-optim.c   -o bin/t5_loop-collapse-optim

echo ""
echo "=== T1: Matrix Addition (4 threads) ==="
./bin/t1_matrix-addition 4

echo ""
echo "=== T2: Sum of N Numbers (N=10M, 4 threads) ==="
./bin/t2_sum-of-numbers 10000000 4

echo ""
echo "=== T3: Matrix Multiplication (500x500, 4 threads) ==="
./bin/t3_matrix-multiplication 500 4

echo ""
echo "=== T4: Scheduling Strategies (500x500, 4 threads) ==="
./bin/t4_scheduling-strategies 500 4

echo ""
echo "=== T5: Loop Collapse Optimization (500x500, 4 threads) ==="
./bin/t5_loop-collapse-optim 500 4
