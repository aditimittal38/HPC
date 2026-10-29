#!/bin/bash
# A3 — OpenMP Reduction & Parallel Scan
# Run from inside 23610067_HPC_A3/
#
# Prerequisites (Ubuntu):
#   sudo apt update
#   sudo apt install gcc libgomp1

set -e
cd "$(dirname "$0")"
mkdir -p bin

echo "=== Compiling ==="
gcc -fopenmp code/t1_array-sum-reduction.c      -o bin/t1_array-sum-reduction
gcc -fopenmp code/t2_dot-product.c              -o bin/t2_dot-product
gcc -fopenmp code/t3_prefix-sum.c               -o bin/t3_prefix-sum
gcc -fopenmp code/t4_perf-compare.c             -o bin/t4_perf-compare
gcc -fopenmp code/c1_dot-product-sync-methods.c -o bin/c1_dot-product-sync-methods

echo ""
echo "=== T1: Array Sum — critical vs atomic vs reduction (N=1M, 4 threads) ==="
./bin/t1_array-sum-reduction 1000000 4

echo ""
echo "=== T2: Dot Product (N=10M, 4 threads) ==="
./bin/t2_dot-product 10000000 4

echo ""
echo "=== T3: Prefix Sum (N=10M, 4 threads) ==="
./bin/t3_prefix-sum 10000000 4

echo ""
echo "=== T4: Performance Comparison (N=10M, 4 threads) ==="
./bin/t4_perf-compare 10000000 4

echo ""
echo "=== C1: Dot Product Sync Methods (N=5M, 4 threads) ==="
./bin/c1_dot-product-sync-methods 5000000 4
