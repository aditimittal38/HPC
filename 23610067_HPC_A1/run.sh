#!/bin/bash
# A1 — Environment Setup & OpenMP Basics
# Run from inside 23610067_HPC_A1/
#
# Prerequisites (Ubuntu):
#   sudo apt update
#   sudo apt install gcc libgomp1 python3

set -e
cd "$(dirname "$0")"
mkdir -p bin

echo "=== Compiling ==="
gcc -fopenmp code/t1_gcc-hello.c          -o bin/t1_gcc-hello
gcc -fopenmp code/t2_openmp-hello.c       -o bin/t2_openmp-hello
gcc -fopenmp code/t3_parallel-square.c    -o bin/t3_parallel-square

echo ""
echo "=== T1: GCC Hello World ==="
./bin/t1_gcc-hello

echo ""
echo "=== T2: OpenMP Hello (4 threads) ==="
OMP_NUM_THREADS=4 ./bin/t2_openmp-hello

echo ""
echo "=== T3: Parallel Square (first 10 of 1000 iterations, 4 threads) ==="
OMP_NUM_THREADS=4 ./bin/t3_parallel-square 2>&1 | head -15

echo ""
echo "=== T4: Theoretical Peak FLOPS ==="
python3 code/t4_flops-calculator.py
