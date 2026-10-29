// Task 5: Optimization using loop collapsing + thread affinity

// gcc -fopenmp Task5/optimization.c -o optimization

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
            for (int k = 0; k < N; k++) sum += A[i*N+k]*B[k*P+j];
            C[i*P+j] = sum;
        }
}

// collapse(2) merges the i and j loops into one iteration space of size M*P,
// giving OpenMP far more independent chunks to spread across threads than
// splitting on the outer loop (M) alone.
void matmul_collapsed(double *A, double *B, double *C, int M, int N, int P) {
    int i, j, k;
    #pragma omp parallel for collapse(2) private(k)
    for (i = 0; i < M; i++) {
        for (j = 0; j < P; j++) {
            double sum = 0.0;
            for (k = 0; k < N; k++) sum += A[i*N+k]*B[k*P+j];
            C[i*P+j] = sum;
        }
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
    double *Cseq = alloc_matrix(M, P);
    double *Cpar = alloc_matrix(M, P);
    fill_matrix(A, M, N);
    fill_matrix(B, N, P);

    double t1 = omp_get_wtime();
    matmul_sequential(A, B, Cseq, M, N, P);
    double t2 = omp_get_wtime();
    double seq_time = t2 - t1;

    double t3 = omp_get_wtime();
    matmul_collapsed(A, B, Cpar, M, N, P);
    double t4 = omp_get_wtime();
    double par_time = t4 - t3;

    double speedup = seq_time / par_time;
    double efficiency = speedup / num_threads;

    printf("Size=%d Threads=%d  Seq=%.4fs  Collapsed-Parallel=%.4fs  Speedup=%.3f  Efficiency=%.3f\n",
           SIZE, num_threads, seq_time, par_time, speedup, efficiency);

    free(A); free(B); free(Cseq); free(Cpar);
    return 0;
}