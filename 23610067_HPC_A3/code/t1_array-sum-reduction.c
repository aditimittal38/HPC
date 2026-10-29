// Section A - Task 1: OpenMP Reduction
// Sum of an array: Sequential vs critical vs reduction

//gcc -fopenmp Task1/array-sum.c -o array-sum
// ./array-sum 1000000 4

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double sum_sequential(double *A, long N) {
    double sum = 0.0;
    for (long i = 0; i < N; i++) sum += A[i];
    return sum;
}

double sum_critical(double *A, long N) {
    double sum = 0.0;
    #pragma omp parallel for
    for (long i = 0; i < N; i++) {
        #pragma omp critical
        sum += A[i];
    }
    return sum;
}

double sum_reduction(double *A, long N) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < N; i++) sum += A[i];
    return sum;
}

int main(int argc, char *argv[]) {
    long N = 1000000;
    int num_threads = 4;
    if (argc > 1) N = atol(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);
    omp_set_num_threads(num_threads);

    double *A = malloc(N * sizeof(double));
    for (long i = 0; i < N; i++) A[i] = 1.0;

    double t1 = omp_get_wtime();
    double seq = sum_sequential(A, N);
    double t2 = omp_get_wtime();
    double seq_time = t2 - t1;

    double t3 = omp_get_wtime();
    double crit = sum_critical(A, N);
    double t4 = omp_get_wtime();
    double crit_time = t4 - t3;

    double t5 = omp_get_wtime();
    double red = sum_reduction(A, N);
    double t6 = omp_get_wtime();
    double red_time = t6 - t5;

    printf("N = %ld, Threads = %d\n", N, num_threads);
    printf("Sequential : sum=%.2f  time=%.6f s\n", seq, seq_time);
    printf("Critical   : sum=%.2f  time=%.6f s  speedup=%.3f\n", crit, crit_time, seq_time/crit_time);
    printf("Reduction  : sum=%.2f  time=%.6f s  speedup=%.3f\n", red, red_time, seq_time/red_time);
    printf("Correctness: critical==sequential? %s | reduction==sequential? %s\n",
           (crit == seq) ? "YES" : "NO", (red == seq) ? "YES" : "NO");

    free(A);
    return 0;
}