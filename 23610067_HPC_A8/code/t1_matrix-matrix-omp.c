#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    int N = 500;
    int threads = 4;
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) threads = atoi(argv[2]);

    omp_set_num_threads(threads);

    double *A = malloc(N * N * sizeof(double));
    double *B = malloc(N * N * sizeof(double));
    double *C = malloc(N * N * sizeof(double));

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = 1.0;
            B[i * N + j] = 1.0;
            C[i * N + j] = 0.0;
        }

    double start = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }

    double end = omp_get_wtime();

    printf("Matrix Size = %d x %d\n", N, N);
    printf("OpenMP Threads = %d\n", threads);
    printf("OpenMP Time = %.6f seconds\n", end - start);
    printf("Sample Result C[0][0] = %.2f\n", C[0]);

    free(A); free(B); free(C);
    return 0;
}
