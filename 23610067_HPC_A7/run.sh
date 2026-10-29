#!/bin/bash
# A7 — MPI Collective Communication & Matrix Parallelism
# Run from inside 23610067_HPC_A7/
#
# Prerequisites (Ubuntu):
#   sudo apt update
#   sudo apt install gcc openmpi-bin libopenmpi-dev
#
# Note: --oversubscribe lets you run more MPI processes than available CPU cores.
# If running as root (e.g. in a container), add --allow-run-as-root to mpirun.

set -e
cd "$(dirname "$0")"
mkdir -p bin

echo "=== Compiling ==="
mpicc code/t1_broadcast.c             -o bin/t1_broadcast
mpicc code/t2_scatter-gather-sum.c    -o bin/t2_scatter-gather-sum
mpicc code/t3_mpi-reduce.c            -o bin/t3_mpi-reduce
mpicc code/t4_matrix-vector-mpi.c     -o bin/t4_matrix-vector-mpi
mpicc code/t4_matrix-vector-seq.c     -o bin/t4_matrix-vector-seq
mpicc code/t5_matrix-matrix-mpi.c     -o bin/t5_matrix-matrix-mpi
mpicc code/t5_matrix-matrix-seq.c     -o bin/t5_matrix-matrix-seq
mpicc code/t6_block-matrix-mul.c      -o bin/t6_block-matrix-mul
mpicc code/t6_column-matrix-mul.c     -o bin/t6_column-matrix-mul
mpicc code/t7_scatter-v.c             -o bin/t7_scatter-v

echo ""
echo "=== T1: Broadcast (4 processes) ==="
mpirun --oversubscribe -n 4 ./bin/t1_broadcast

echo ""
echo "=== T2: Scatter / Gather Sum (4 processes) ==="
mpirun --oversubscribe -n 4 ./bin/t2_scatter-gather-sum

echo ""
echo "=== T3: MPI Reduce SUM/MAX/MIN (4 processes) ==="
mpirun --oversubscribe -n 4 ./bin/t3_mpi-reduce

echo ""
echo "=== T4 (seq): Matrix-Vector Sequential (N=500) ==="
./bin/t4_matrix-vector-seq 500

echo ""
echo "=== T4 (MPI): Matrix-Vector MPI (N=500, 4 processes) ==="
mpirun --oversubscribe -n 4 ./bin/t4_matrix-vector-mpi 500

echo ""
echo "=== T5 (seq): Matrix-Matrix Sequential (N=200) ==="
./bin/t5_matrix-matrix-seq 200

echo ""
echo "=== T5 (MPI): Matrix-Matrix MPI (N=200, 4 processes) ==="
mpirun --oversubscribe -n 4 ./bin/t5_matrix-matrix-mpi 200

echo ""
echo "=== T6: Block Matrix Multiply (4 processes, N=8) ==="
mpirun --oversubscribe -n 4 ./bin/t6_block-matrix-mul 8

echo ""
echo "=== T6: Column Matrix Multiply (4 processes, N=8) ==="
mpirun --oversubscribe -n 4 ./bin/t6_column-matrix-mul 8

echo ""
echo "=== T7: Scatterv Uneven Rows (3 processes) ==="
mpirun --oversubscribe -n 3 ./bin/t7_scatter-v
