// Section A - Task 2: Dot Product using OpenMP

// gcc -fopenmp Task2/dot-product.c -o dot-product

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double dot_sequential(double *A, double *B, long N) {
    double sum = 0.0;
    for (long i = 0; i < N; i++) sum += A[i] * B[i];
    return sum;
}

double dot_parallel(double *A, double *B, long N) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < N; i++) sum += A[i] * B[i];
    return sum;
}

int main(int argc, char *argv[]) {
    long N = 10000000;
    int num_threads = 4;
    if (argc > 1) N = atol(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);
    omp_set_num_threads(num_threads);

    double *A = malloc(N * sizeof(double));
    double *B = malloc(N * sizeof(double));
    for (long i = 0; i < N; i++) { A[i] = 1.0; B[i] = 2.0; }

    double t1 = omp_get_wtime();
    double seq = dot_sequential(A, B, N);
    double t2 = omp_get_wtime();
    double seq_time = t2 - t1;

    double t3 = omp_get_wtime();
    double par = dot_parallel(A, B, N);
    double t4 = omp_get_wtime();
    double par_time = t4 - t3;

    double speedup = seq_time / par_time;
    double efficiency = speedup / num_threads;

    printf("N = %ld, Threads = %d\n", N, num_threads);
    printf("Sequential dot = %.2f  time = %.6f s\n", seq, seq_time);
    printf("Parallel   dot = %.2f  time = %.6f s\n", par, par_time);
    printf("Results match: %s\n", (seq == par) ? "YES" : "NO (float rounding - check tolerance)");
    printf("Speedup = %.3f   Efficiency = %.3f\n", speedup, efficiency);

    free(A); free(B);
    return 0;
}