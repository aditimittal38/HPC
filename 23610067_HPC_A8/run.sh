#!/bin/bash
# A8 — Hybrid Parallelism: OpenMP + MPI (Matrix Multiply & 2D Convolution)
# Run from inside 23610067_HPC_A8/
#
# Prerequisites (Ubuntu):
#   sudo apt update
#   sudo apt install gcc libgomp1 openmpi-bin libopenmpi-dev python3 python3-matplotlib python3-pandas
#
# Note: --oversubscribe lets you run more MPI processes than available CPU cores.
# If running as root (e.g. in a container), add --allow-run-as-root to mpirun.

set -e
cd "$(dirname "$0")"
mkdir -p bin

echo "=== Compiling ==="
gcc  -O2 code/t1_matrix-matrix-seq.c -o bin/t1_matmul-seq
gcc  -O2 -fopenmp code/t1_matrix-matrix-omp.c -o bin/t1_matmul-omp
mpicc -O2 code/t1_matrix-matrix-mpi.c -o bin/t1_matmul-mpi

gcc  -O2 code/t2_conv-seq.c -o bin/t2_conv-seq
gcc  -O2 -fopenmp code/t2_conv-omp.c -o bin/t2_conv-omp
mpicc -O2 code/t2_conv-mpi.c -o bin/t2_conv-mpi

# ── Task 1: Matrix-Matrix Multiply ──────────────────────────────────────────

echo ""
echo "=== T1 (seq):  Matrix-Matrix Sequential (N=500) ==="
./bin/t1_matmul-seq 500

echo ""
echo "=== T1 (omp):  Matrix-Matrix OpenMP (N=500, 4 threads) ==="
./bin/t1_matmul-omp 500 4

echo ""
echo "=== T1 (mpi):  Matrix-Matrix MPI (N=500, 4 processes) ==="
mpirun --oversubscribe -n 4 ./bin/t1_matmul-mpi 500

# ── Task 2: 2D Convolution ──────────────────────────────────────────────────

echo ""
echo "=== T2 (seq):  Convolution Sequential (N=1000) ==="
./bin/t2_conv-seq 1000

echo ""
echo "=== T2 (omp, static):  Convolution OpenMP static (N=1000, 4 threads) ==="
./bin/t2_conv-omp 1000 4 0

echo ""
echo "=== T2 (omp, dynamic): Convolution OpenMP dynamic (N=1000, 4 threads) ==="
./bin/t2_conv-omp 1000 4 1

echo ""
echo "=== T2 (omp, guided):  Convolution OpenMP guided (N=1000, 4 threads) ==="
./bin/t2_conv-omp 1000 4 2

echo ""
echo "=== T2 (mpi):  Convolution MPI (N=1000, 4 processes) ==="
mpirun --oversubscribe -n 4 ./bin/t2_conv-mpi 1000

echo ""
echo "=== Done. To run full benchmark suite and generate graphs: ==="
echo "    bash code/run_experiments.sh"
echo "    python3 code/generate_graphs.py"
