#!/bin/bash
# Run all A8 experiments and collect timing data
# Usage: bash run_experiments.sh

TASK1="$(dirname "$0")/../Task1"
TASK2="$(dirname "$0")/../Task2"
OUT="$(dirname "$0")/results.csv"

echo "app,model,param,size,time_sec" > "$OUT"

# Task 1: Matrix-Matrix Multiplication
echo "=== Task 1: Matrix-Matrix Multiplication ==="
for N in 500 1000 2000; do
    # Sequential
    t=$("$TASK1/matrix-matrix-seq" $N | grep "Sequential Time" | awk '{print $4}')
    echo "matmul,sequential,1,$N,$t"
    echo "matmul,sequential,1,$N,$t" >> "$OUT"

    # OpenMP
    for T in 2 4 8; do
        t=$("$TASK1/matrix-matrix-omp" $N $T | grep "OpenMP Time" | awk '{print $4}')
        echo "matmul,openmp,$T,$N,$t"
        echo "matmul,openmp,$T,$N,$t" >> "$OUT"
    done

    # MPI
    for P in 2 4 8; do
        t=$(mpirun -n $P "$TASK1/matrix-matrix-mpi" $N | grep "MPI Time" | awk '{print $4}')
        echo "matmul,mpi,$P,$N,$t"
        echo "matmul,mpi,$P,$N,$t" >> "$OUT"
    done
done

# Task 2: 2D Convolution
echo "=== Task 2: 2D Convolution ==="
for N in 1000 2000 4000; do
    # Sequential
    t=$("$TASK2/conv-seq" $N | grep "Sequential Time" | awk '{print $4}')
    echo "conv,sequential,1,$N,$t"
    echo "conv,sequential,1,$N,$t" >> "$OUT"

    # OpenMP (static schedule = 0, dynamic = 1, guided = 2)
    for T in 2 4 8; do
        t=$("$TASK2/conv-omp" $N $T 0 | grep "OpenMP Time" | awk '{print $4}')
        echo "conv,openmp_static,$T,$N,$t"
        echo "conv,openmp_static,$T,$N,$t" >> "$OUT"

        t=$("$TASK2/conv-omp" $N $T 1 | grep "OpenMP Time" | awk '{print $4}')
        echo "conv,openmp_dynamic,$T,$N,$t"
        echo "conv,openmp_dynamic,$T,$N,$t" >> "$OUT"

        t=$("$TASK2/conv-omp" $N $T 2 | grep "OpenMP Time" | awk '{print $4}')
        echo "conv,openmp_guided,$T,$N,$t"
        echo "conv,openmp_guided,$T,$N,$t" >> "$OUT"
    done

    # MPI
    for P in 2 4 8; do
        t=$(mpirun -n $P "$TASK2/conv-mpi" $N | grep "MPI Time" | awk '{print $4}')
        echo "conv,mpi,$P,$N,$t"
        echo "conv,mpi,$P,$N,$t" >> "$OUT"
    done
done

echo ""
echo "Results saved to: $OUT"
