#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[])
{
    int N = argc > 1 ? atoi(argv[1]) : 500;

    if (N <= 0)
    {
        fprintf(stderr, "Matrix size must be a positive integer.\n");
        return 1;
    }

    double *A = malloc(N * N * sizeof(double));
    double *X = malloc(N * sizeof(double));
    double *Y = malloc(N * sizeof(double));

    for (int i = 0; i < N; i++)
    {
        X[i] = 1.0;

        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = 1.0;
        }
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < N; i++)
    {
        Y[i] = 0.0;

        for (int j = 0; j < N; j++)
        {
            Y[i] += A[i * N + j] * X[j];
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double checksum = 0.0;
    for (int i = 0; i < N; i++)
    {
        checksum += Y[i];
    }

    printf(
        "Matrix Size = %d x %d\n",
        N,
        N
    );

    printf(
        "Sequential Time = %.9f seconds\n",
        (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) / 1e9
    );
    printf("Result checksum = %.2f\n", checksum);

    free(A);
    free(X);
    free(Y);

    return 0;
}