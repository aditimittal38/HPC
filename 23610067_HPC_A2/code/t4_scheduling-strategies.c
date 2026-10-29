// Task 4: Analyze OpenMP Scheduling Strategies (applied to matrix multiplication)

// gcc -fopenmp Task4/scheduling-strategies.c -o scheduling-strategies 

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double *alloc_matrix(int rows, int cols) { return malloc(sizeof(double) * rows * cols); }
void fill_matrix(double *M, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) M[i] = (double)(i % 100) / 10.0;
}

void matmul_scheduled(double *A, double *B, double *C, int M, int N, int P, int schedule_type) {
    int i, j, k;
    if (schedule_type == 0) {
        #pragma omp parallel for schedule(static) private(j,k)
        for (i = 0; i < M; i++)
            for (j = 0; j < P; j++) {
                double sum = 0.0;
                for (k = 0; k < N; k++) sum += A[i*N+k] * B[k*P+j];
                C[i*P+j] = sum;
            }
    } else if (schedule_type == 1) {
        #pragma omp parallel for schedule(dynamic) private(j,k)
        for (i = 0; i < M; i++)
            for (j = 0; j < P; j++) {
                double sum = 0.0;
                for (k = 0; k < N; k++) sum += A[i*N+k] * B[k*P+j];
                C[i*P+j] = sum;
            }
    } else {
        #pragma omp parallel for schedule(guided) private(j,k)
        for (i = 0; i < M; i++)
            for (j = 0; j < P; j++) {
                double sum = 0.0;
                for (k = 0; k < N; k++) sum += A[i*N+k] * B[k*P+j];
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
    double *C = alloc_matrix(M, P);
    fill_matrix(A, M, N);
    fill_matrix(B, N, P);

    double *Cseq = alloc_matrix(M, P);
    double ts1 = omp_get_wtime();
    for (int i = 0; i < M; i++)
        for (int j = 0; j < P; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) sum += A[i*N+k]*B[k*P+j];
            Cseq[i*P+j] = sum;
        }
    double ts2 = omp_get_wtime();
    double seq_time = ts2 - ts1;
    printf("Sequential time (baseline) = %.4f s\n\n", seq_time);

    const char *names[3] = {"static", "dynamic", "guided"};
    for (int s = 0; s < 3; s++) {
        double t1 = omp_get_wtime();
        matmul_scheduled(A, B, C, M, N, P, s);
        double t2 = omp_get_wtime();
        double par_time = t2 - t1;
        printf("Schedule = %-8s Time = %.4f s   Speedup = %.3f\n",
               names[s], par_time, seq_time / par_time);
    }

    free(A); free(B); free(C); free(Cseq);
    return 0;
}