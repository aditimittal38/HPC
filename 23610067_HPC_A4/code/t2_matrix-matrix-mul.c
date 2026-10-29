/*
 * Task 2: Matrix-Matrix Multiplication using Output Data Decomposition
 * Decomposition: different blocks of the output matrix C are assigned
 * to different threads (each thread owns a set of (i,j) output cells).
 *
*/

// gcc -fopenmp Task2/matrix-matrix-mul.c -o matrix-matrix-mul

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void generate_matrix(double *M, int n) {
    for (int i = 0; i < n * n; i++) M[i] = (double)(rand() % 10);
}

void matmul_seq(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = sum;
        }
}

/* sched_type: 0 = static, 1 = dynamic */
void matmul_par(double *A, double *B, double *C, int n, int sched_type) {
    if (sched_type == 0) {
        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++)
                    sum += A[(size_t)i * n + k] * B[(size_t)k * n + j];
                C[(size_t)i * n + j] = sum;
            }
    } else {
        #pragma omp parallel for schedule(dynamic, 4) collapse(2)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++)
                    sum += A[(size_t)i * n + k] * B[(size_t)k * n + j];
                C[(size_t)i * n + j] = sum;
            }
    }
}

int main(int argc, char *argv[]) {
    int n       = (argc > 1) ? atoi(argv[1]) : 300;
    int threads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    omp_set_num_threads(threads);

    double *A         = malloc((size_t)n * n * sizeof(double));
    double *B         = malloc((size_t)n * n * sizeof(double));
    double *C_seq     = malloc((size_t)n * n * sizeof(double));
    double *C_static  = malloc((size_t)n * n * sizeof(double));
    double *C_dynamic = malloc((size_t)n * n * sizeof(double));

    srand(42);
    generate_matrix(A, n);
    generate_matrix(B, n);

    double t0 = omp_get_wtime();
    matmul_seq(A, B, C_seq, n);
    double t1 = omp_get_wtime();
    double seq_time = t1 - t0;

    double t2 = omp_get_wtime();
    matmul_par(A, B, C_static, n, 0);
    double t3 = omp_get_wtime();
    double static_time = t3 - t2;

    double t4 = omp_get_wtime();
    matmul_par(A, B, C_dynamic, n, 1);
    double t5 = omp_get_wtime();
    double dynamic_time = t5 - t4;

    printf("========================================\n");
    printf("Matrix-Matrix Multiplication (Output Decomposition)\n");
    printf("Matrix size : %d x %d | Threads: %d\n", n, n, threads);
    printf("Sequential time    : %f s\n", seq_time);
    printf("Parallel (static)  : %f s | Speedup: %f | Efficiency: %f\n",
           static_time, seq_time / static_time, (seq_time / static_time) / threads);
    printf("Parallel (dynamic) : %f s | Speedup: %f | Efficiency: %f\n",
           dynamic_time, seq_time / dynamic_time, (seq_time / dynamic_time) / threads);
    printf("========================================\n");

    free(A); free(B); free(C_seq); free(C_static); free(C_dynamic);
    return 0;
}