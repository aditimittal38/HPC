#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define KERNEL_SIZE 3

int main(int argc, char *argv[])
{
    int N = 1000;
    if (argc > 1) N = atoi(argv[1]);

    int M = N - KERNEL_SIZE + 1; /* valid convolution output size */

    double *input  = malloc(N * N * sizeof(double));
    double *output = malloc(M * M * sizeof(double));

    /* 3x3 averaging kernel */
    double kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        {1.0/9, 1.0/9, 1.0/9},
        {1.0/9, 1.0/9, 1.0/9},
        {1.0/9, 1.0/9, 1.0/9}
    };

    for (int i = 0; i < N * N; i++)
        input[i] = (double)(i % 256);

    struct timespec ts, te;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < M; j++)
        {
            double sum = 0.0;
            for (int u = 0; u < KERNEL_SIZE; u++)
                for (int v = 0; v < KERNEL_SIZE; v++)
                    sum += input[(i + u) * N + (j + v)] * kernel[u][v];
            output[i * M + j] = sum;
        }

    clock_gettime(CLOCK_MONOTONIC, &te);
    double elapsed = (te.tv_sec - ts.tv_sec) +
                     (te.tv_nsec - ts.tv_nsec) * 1e-9;

    printf("Input Size = %d x %d\n", N, N);
    printf("Output Size = %d x %d\n", M, M);
    printf("Sequential Time = %.6f seconds\n", elapsed);
    printf("Sample Output[0][0] = %.4f\n", output[0]);

    free(input); free(output);
    return 0;
}
