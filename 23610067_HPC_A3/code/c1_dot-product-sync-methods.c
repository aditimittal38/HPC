// Section C - Challenge 1: Dot Product using critical, atomic, reduction

// gcc -fopenmp Challenge1/dot-product-c1.c -o dot-product-c1

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double dot_critical(double *A, double *B, long N) {
    double sum = 0.0;
    #pragma omp parallel for
    for (long i = 0; i < N; i++) {
        double prod = A[i] * B[i];
        #pragma omp critical
        sum += prod;
    }
    return sum;
}

double dot_atomic(double *A, double *B, long N) {
    double sum = 0.0;
    #pragma omp parallel for
    for (long i = 0; i < N; i++) {
        double prod = A[i] * B[i];
        #pragma omp atomic
        sum += prod;
    }
    return sum;
}

double dot_reduction(double *A, double *B, long N) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < N; i++) sum += A[i] * B[i];
    return sum;
}

int main(int argc, char *argv[]) {
    long N = 5000000;
    int num_threads = 4;
    if (argc > 1) N = atol(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);
    omp_set_num_threads(num_threads);

    double *A = malloc(N*sizeof(double));
    double *B = malloc(N*sizeof(double));
    for (long i = 0; i < N; i++) { A[i] = 1.0; B[i] = 2.0; }

    volatile double sink;

    double t1 = omp_get_wtime();
    sink = dot_critical(A, B, N);
    double t2 = omp_get_wtime();
    double crit_time = t2 - t1;

    double t3 = omp_get_wtime();
    sink = dot_atomic(A, B, N);
    double t4 = omp_get_wtime();
    double atomic_time = t4 - t3;

    double t5 = omp_get_wtime();
    sink = dot_reduction(A, B, N);
    double t6 = omp_get_wtime();
    double red_time = t6 - t5;
    (void)sink;

    printf("N=%ld, Threads=%d\n", N, num_threads);
    printf("critical  : time=%.6f s\n", crit_time);
    printf("atomic    : time=%.6f s\n", atomic_time);
    printf("reduction : time=%.6f s\n", red_time);
    printf("\nRelative to reduction:\n");
    printf("critical  is %.1fx slower than reduction\n", crit_time / red_time);
    printf("atomic    is %.1fx slower than reduction\n", atomic_time / red_time);

    free(A); free(B);
    return 0;
}