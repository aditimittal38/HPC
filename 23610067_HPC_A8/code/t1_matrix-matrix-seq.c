#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[])
{
    int N = 500;
    if (argc > 1)
        N = atoi(argv[1]);

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

    struct timespec ts, te;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }

    clock_gettime(CLOCK_MONOTONIC, &te);
    double elapsed = (te.tv_sec - ts.tv_sec) +
                     (te.tv_nsec - ts.tv_nsec) * 1e-9;

    printf("Matrix Size = %d x %d\n", N, N);
    printf("Sequential Time = %.6f seconds\n", elapsed);
    printf("Sample Result C[0][0] = %.2f\n", C[0]);

    free(A); free(B); free(C);
    return 0;
}
