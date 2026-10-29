// Task 3: Parallel Matrix Multiplication using OpenMP
// C (MxP) = A (MxN) * B (NxP)


// gcc -fopenmp Task3/matrix-multiplication.c -o matrix-multiplication
// ./matrix-multiplication 500 4
// ./matrix-multiplication 1000 4
// ./matrix-multiplication 2000 4


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double *alloc_matrix(int rows, int cols) { return malloc(sizeof(double) * rows * cols); }

void fill_matrix(double *M, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) M[i] = (double)(i % 100) / 10.0;
}

void matmul_sequential(double *A, double *B, double *C, int M, int N, int P) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < P; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += A[i*N + k] * B[k*P + j];
            C[i*P + j] = sum;
        }
}

void matmul_parallel(double *A, double *B, double *C, int M, int N, int P) {
    #pragma omp parallel for
    for (int i = 0; i < M; i++)
        for (int j = 0; j < P; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += A[i*N + k] * B[k*P + j];
            C[i*P + j] = sum;
        }
}

int main(int argc, char *argv[]) {
    int SIZE = 500;
    int num_threads = 4;
    if (argc > 1) SIZE = atoi(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);
    omp_set_num_threads(num_threads);

    int M = SIZE, N = SIZE, P = SIZE;
    double *A = alloc_matrix(M, N);
    double *B = alloc_matrix(N, P);
    double *C_seq = alloc_matrix(M, P);
    double *C_par = alloc_matrix(M, P);
    fill_matrix(A, M, N);
    fill_matrix(B, N, P);

    double t1 = omp_get_wtime();
    matmul_sequential(A, B, C_seq, M, N, P);
    double t2 = omp_get_wtime();
    double seq_time = t2 - t1;

    double t3 = omp_get_wtime();
    matmul_parallel(A, B, C_par, M, N, P);
    double t4 = omp_get_wtime();
    double par_time = t4 - t3;

    double speedup = seq_time / par_time;
    double efficiency = speedup / num_threads;

    printf("Matrix size = %dx%d, Threads = %d\n", SIZE, SIZE, num_threads);
    printf("Sequential time = %.4f s\n", seq_time);
    printf("Parallel time   = %.4f s\n", par_time);
    printf("Speedup    = %.3f\n", speedup);
    printf("Efficiency = %.3f\n", efficiency);

    free(A); free(B); free(C_seq); free(C_par);
    return 0;
}

