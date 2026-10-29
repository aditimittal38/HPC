#!/bin/bash
# A6 — MPI Basics: Point-to-Point Communication
# Run from inside 23610067_HPC_A6/
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
mpicc code/t2_hello-mpi.c            -o bin/t2_hello-mpi
mpicc code/t3_rank-size.c            -o bin/t3_rank-size
mpicc code/t4_group-comm.c           -o bin/t4_group-comm
mpicc code/t5_send-receive.c         -o bin/t5_send-receive
mpicc code/t6_integer-send.c         -o bin/t6_integer-send
mpicc code/t7_array-send.c           -o bin/t7_array-send
mpicc code/c1_ping-pong.c            -o bin/c1_ping-pong
mpicc code/ca_send-recv-nonblock.c   -o bin/ca_send-recv-nonblock
mpicc code/pa_communication-time.c   -o bin/pa_communication-time

echo ""
echo "=== T2: Hello MPI (4 processes) ==="
mpirun --oversubscribe -n 4 ./bin/t2_hello-mpi

echo ""
echo "=== T3: Rank and Size (4 processes) ==="
mpirun --oversubscribe -n 4 ./bin/t3_rank-size

echo ""
echo "=== T4: Group Communication (4 processes) ==="
mpirun --oversubscribe -n 4 ./bin/t4_group-comm

echo ""
echo "=== T5: Send/Receive string (2 processes) ==="
mpirun --oversubscribe -n 2 ./bin/t5_send-receive

echo ""
echo "=== T6: Integer Send (4 processes) ==="
mpirun --oversubscribe -n 4 ./bin/t6_integer-send

echo ""
echo "=== T7: Array Send (2 processes) ==="
mpirun --oversubscribe -n 2 ./bin/t7_array-send

echo ""
echo "=== C1: Ping-Pong Benchmark (2 processes) ==="
mpirun --oversubscribe -n 2 ./bin/c1_ping-pong

echo ""
echo "=== CA: SendRecv Non-blocking (4 processes) ==="
mpirun --oversubscribe -n 4 ./bin/ca_send-recv-nonblock

echo ""
echo "=== PA: Communication Time Profiler (2 processes) ==="
mpirun --oversubscribe -n 2 ./bin/pa_communication-time
