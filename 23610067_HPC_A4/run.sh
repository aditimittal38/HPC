#!/bin/bash
# A4 — Data Decomposition Strategies
# Run from inside 23610067_HPC_A4/
#
# Prerequisites (Ubuntu):
#   sudo apt update
#   sudo apt install gcc libgomp1

set -e
cd "$(dirname "$0")"
mkdir -p bin

echo "=== Compiling ==="
gcc -fopenmp code/t1_matrix-vector-mul.c      -o bin/t1_matrix-vector-mul
gcc -fopenmp code/t2_matrix-matrix-mul.c      -o bin/t2_matrix-matrix-mul
gcc -fopenmp code/t3_decomp-row-col.c         -o bin/t3_decomp-row-col
gcc -fopenmp code/t4_decomp-all-strategies.c  -o bin/t4_decomp-all-strategies
gcc -fopenmp code/c3_recursive-merge-sort.c   -o bin/c3_recursive-merge-sort

echo ""
echo "=== T1: Matrix-Vector Mul — Input Decomposition (n=1000, 4 threads) ==="
./bin/t1_matrix-vector-mul 1000 4

echo ""
echo "=== T2: Matrix-Matrix Mul — Output Decomposition (n=300, 4 threads) ==="
./bin/t2_matrix-matrix-mul 300 4

echo ""
echo "=== T3: Row vs Column Decomposition (n=500, 4 threads) ==="
./bin/t3_decomp-row-col 500 4

echo ""
echo "=== T4: All Decomposition Strategies (n=500, 4 threads, block=64) ==="
./bin/t4_decomp-all-strategies 500 4 64

echo ""
echo "=== C3: Recursive Merge Sort with Tasks (N=2M, 4 threads) ==="
./bin/c3_recursive-merge-sort 2000000 4
