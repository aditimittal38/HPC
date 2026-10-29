#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define KERNEL_SIZE 3

int main(int argc, char *argv[])
{
    int N = 1000;
    int threads = 4;
    char schedule_type[16] = "static";
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) threads = atoi(argv[2]);
    if (argc > 3) {
        int s = atoi(argv[3]);
        if (s == 0) { snprintf(schedule_type, 16, "static"); }
        else if (s == 1) { snprintf(schedule_type, 16, "dynamic"); }
        else { snprintf(schedule_type, 16, "guided"); }
    }

    omp_set_num_threads(threads);

    int M = N - KERNEL_SIZE + 1;

    double *input  = malloc(N * N * sizeof(double));
    double *output = malloc(M * M * sizeof(double));

    double kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        {1.0/9, 1.0/9, 1.0/9},
        {1.0/9, 1.0/9, 1.0/9},
        {1.0/9, 1.0/9, 1.0/9}
    };

    for (int i = 0; i < N * N; i++)
        input[i] = (double)(i % 256);

    int sched = (argc > 3) ? atoi(argv[3]) : 0;

    double start = omp_get_wtime();

    if (sched == 1)
    {
        #pragma omp parallel for schedule(dynamic) collapse(2)
        for (int i = 0; i < M; i++)
            for (int j = 0; j < M; j++)
            {
                double sum = 0.0;
                for (int u = 0; u < KERNEL_SIZE; u++)
                    for (int v = 0; v < KERNEL_SIZE; v++)
                        sum += input[(i + u) * N + (j + v)] * kernel[u][v];
                output[i * M + j] = sum;
            }
    }
    else if (sched == 2)
    {
        #pragma omp parallel for schedule(guided) collapse(2)
        for (int i = 0; i < M; i++)
            for (int j = 0; j < M; j++)
            {
                double sum = 0.0;
                for (int u = 0; u < KERNEL_SIZE; u++)
                    for (int v = 0; v < KERNEL_SIZE; v++)
                        sum += input[(i + u) * N + (j + v)] * kernel[u][v];
                output[i * M + j] = sum;
            }
    }
    else
    {
        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = 0; i < M; i++)
            for (int j = 0; j < M; j++)
            {
                double sum = 0.0;
                for (int u = 0; u < KERNEL_SIZE; u++)
                    for (int v = 0; v < KERNEL_SIZE; v++)
                        sum += input[(i + u) * N + (j + v)] * kernel[u][v];
                output[i * M + j] = sum;
            }
    }

    double end = omp_get_wtime();

    printf("Input Size = %d x %d\n", N, N);
    printf("OpenMP Threads = %d\n", threads);
    printf("Schedule = %s\n", schedule_type);
    printf("OpenMP Time = %.6f seconds\n", end - start);
    printf("Sample Output[0][0] = %.4f\n", output[0]);

    free(input); free(output);
    return 0;
}
