// Task 2: Parallel Loop Execution using OpenMP - Sum of N numbers

// gcc -fopenmp Task2/sum-of-numbers.c -o sum-of-numbers

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double sum_sequential(double *A, long N) {
    double sum = 0.0;
    for (long i = 0; i < N; i++) sum += A[i];
    return sum;
}

double sum_parallel(double *A, long N) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < N; i++) sum += A[i];
    return sum;
}

int main(int argc, char *argv[]) {
    long N = 10000000;
    int num_threads = 4;
    if (argc > 1) N = atol(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);
    omp_set_num_threads(num_threads);

    double *A = malloc(N * sizeof(double));
    for (long i = 0; i < N; i++) A[i] = 1.0;

    double t1 = omp_get_wtime();
    double seq_result = sum_sequential(A, N);
    double t2 = omp_get_wtime();
    double seq_time = t2 - t1;

    double t3 = omp_get_wtime();
    double par_result = sum_parallel(A, N);
    double t4 = omp_get_wtime();
    double par_time = t4 - t3;

    printf("N = %ld, Threads = %d\n", N, num_threads);
    printf("Sequential sum = %.2f, Time = %.6f s\n", seq_result, seq_time);
    printf("Parallel   sum = %.2f, Time = %.6f s\n", par_result, par_time);
    printf("Speedup = %.3f\n", seq_time / par_time);

    free(A);
    return 0;
}
