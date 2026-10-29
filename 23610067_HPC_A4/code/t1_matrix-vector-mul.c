/*
 * Task 1: Matrix-Vector Multiplication using Input Data Decomposition
 * Decomposition: each thread processes a contiguous subset of matrix ROWS.
 */

//  gcc -fopenmp Task1/matrix-vector-mul.c -o matrix-vector-mul

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void generate_matrix(double *A, int n) {
    for (int i = 0; i < n * n; i++) A[i] = (double)(rand() % 100) / 10.0;
}

void generate_vector(double *v, int n) {
    for (int i = 0; i < n; i++) v[i] = (double)(rand() % 100) / 10.0;
}

int main(int argc, char *argv[]) {
    int n = (argc > 1) ? atoi(argv[1]) : 1000;
    int threads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    omp_set_num_threads(threads);

    double *A     = malloc((size_t)n * n * sizeof(double));
    double *x     = malloc((size_t)n * sizeof(double));
    double *y_seq = malloc((size_t)n * sizeof(double));
    double *y_par = malloc((size_t)n * sizeof(double));

    srand(42);
    generate_matrix(A, n);
    generate_vector(x, n);

    /* ---------- Sequential version ---------- */
    double t0 = omp_get_wtime();
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) sum += A[(size_t)i * n + j] * x[j];
        y_seq[i] = sum;
    }
    double t1 = omp_get_wtime();
    double seq_time = t1 - t0;

    /* ---------- Parallel version: input data decomposition ----------
       Each thread gets a contiguous block of rows (schedule static).
       This is the natural "input decomposition" because the matrix A
       (the input) is split row-wise across threads; each thread reads
       only its own rows of A but the full vector x. */
    double t2 = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) sum += A[(size_t)i * n + j] * x[j];
        y_par[i] = sum;
    }
    double t3 = omp_get_wtime();
    double par_time = t3 - t2;

    /* ---------- Correctness check ---------- */
    int correct = 1;
    for (int i = 0; i < n; i++) {
        double diff = y_seq[i] - y_par[i];
        if (diff > 1e-6 || diff < -1e-6) { correct = 0; break; }
    }

    double speedup    = seq_time / par_time;
    double efficiency = speedup / threads;

    printf("========================================\n");
    printf("Matrix-Vector Multiplication (Input Decomposition)\n");
    printf("Matrix size : %d x %d\n", n, n);
    printf("Threads     : %d\n", threads);
    printf("Sequential time : %f s\n", seq_time);
    printf("Parallel time   : %f s\n", par_time);
    printf("Speedup         : %f\n", speedup);
    printf("Efficiency      : %f\n", efficiency);
    printf("Correctness     : %s\n", correct ? "PASSED" : "FAILED");
    printf("========================================\n");

    free(A); free(x); free(y_seq); free(y_par);
    return 0;
}