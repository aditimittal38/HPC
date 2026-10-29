/*
 * Task 4: Matrix Add / Subtract / Transpose(A) / Transpose(B) via sections
 * Compile: gcc -fopenmp Task4/matrix-ops.c -o matrix-ops
 * Run:     ./matrix-ops <n> [threads]
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void generate(double *M, int n) { for (int i = 0; i < n * n; i++) M[i] = (double)(rand() % 100); }
void mat_add(double *A, double *B, double *C, int n) { for (int i = 0; i < n * n; i++) C[i] = A[i] + B[i]; }
void mat_sub(double *A, double *B, double *C, int n) { for (int i = 0; i < n * n; i++) C[i] = A[i] - B[i]; }
void mat_transpose(double *M, double *T, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            T[j * n + i] = M[i * n + j];
}

int main(int argc, char *argv[]) {
    int n = (argc > 1) ? atoi(argv[1]) : 500;
    int threads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    omp_set_num_threads(threads);

    double *A = malloc((size_t)n * n * sizeof(double)), *B = malloc((size_t)n * n * sizeof(double));
    double *Cadd_s = malloc((size_t)n * n * sizeof(double)), *Csub_s = malloc((size_t)n * n * sizeof(double));
    double *Ta_s = malloc((size_t)n * n * sizeof(double)), *Tb_s = malloc((size_t)n * n * sizeof(double));
    double *Cadd_p = malloc((size_t)n * n * sizeof(double)), *Csub_p = malloc((size_t)n * n * sizeof(double));
    double *Ta_p = malloc((size_t)n * n * sizeof(double)), *Tb_p = malloc((size_t)n * n * sizeof(double));

    srand(42);
    generate(A, n); generate(B, n);

    double t0 = omp_get_wtime();
    mat_add(A, B, Cadd_s, n);
    mat_sub(A, B, Csub_s, n);
    mat_transpose(A, Ta_s, n);
    mat_transpose(B, Tb_s, n);
    double t1 = omp_get_wtime();
    double seq_time = t1 - t0;

    double t2 = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        mat_add(A, B, Cadd_p, n);
        #pragma omp section
        mat_sub(A, B, Csub_p, n);
        #pragma omp section
        mat_transpose(A, Ta_p, n);
        #pragma omp section
        mat_transpose(B, Tb_p, n);
    }
    double t3 = omp_get_wtime();
    double par_time = t3 - t2;

    printf("Matrix size %dx%d | Threads: %d\n", n, n, threads);
    printf("Sequential time: %f s\n", seq_time);
    printf("Sections time  : %f s\n", par_time);
    printf("Speedup        : %f\n", seq_time / par_time);

    free(A); free(B); free(Cadd_s); free(Csub_s); free(Ta_s); free(Tb_s);
    free(Cadd_p); free(Csub_p); free(Ta_p); free(Tb_p);
    return 0;
}