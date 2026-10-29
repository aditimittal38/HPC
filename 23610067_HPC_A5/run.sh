#!/bin/bash
# A5 — OpenMP Sections & Tasks
# Run from inside 23610067_HPC_A5/
#
# Prerequisites (Ubuntu):
#   sudo apt update
#   sudo apt install gcc libgomp1

set -e
cd "$(dirname "$0")"
mkdir -p bin

echo "=== Compiling ==="
gcc -fopenmp code/t1_array-ops-sections.c       -o bin/t1_array-ops-sections
gcc -fopenmp code/t2_parallel-section-compare.c -o bin/t2_parallel-section-compare
gcc -fopenmp code/t3_data-pipeline.c            -o bin/t3_data-pipeline -lm
gcc -fopenmp code/t4_matrix-ops-sections.c      -o bin/t4_matrix-ops-sections
gcc -fopenmp code/t5_merge-sort-tasks.c         -o bin/t5_merge-sort-tasks
gcc -fopenmp code/t6_task-dependency.c          -o bin/t6_task-dependency

echo ""
echo "=== T1: Array Ops via Sections (N=10M, 4 threads) ==="
./bin/t1_array-ops-sections 10000000 4

echo ""
echo "=== T2: parallel for vs sections (N=10M, 4 threads) ==="
./bin/t2_parallel-section-compare 10000000 4

echo ""
echo "=== T3: Data Pipeline (N=4M, 4 threads) ==="
./bin/t3_data-pipeline 4000000 4

echo ""
echo "=== T4: Matrix Ops via Sections (n=500, 4 threads) ==="
./bin/t4_matrix-ops-sections 500 4

echo ""
echo "=== T5: Merge Sort with Tasks (N=2M, 4 threads, threshold=10000) ==="
./bin/t5_merge-sort-tasks 2000000 4 10000

echo ""
echo "=== T6: Task Dependency Graph (A→{B,C}→D) ==="
./bin/t6_task-dependency
