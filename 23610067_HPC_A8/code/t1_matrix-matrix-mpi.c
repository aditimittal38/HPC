#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int N = 500;
    if (argc > 1) N = atoi(argv[1]);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Pad N to be divisible by size */
    int N_padded = ((N + size - 1) / size) * size;
    int rows_per_proc = N_padded / size;

    double *A = NULL;
    double *C = NULL;
    double *B = malloc(N_padded * N_padded * sizeof(double));
    double *local_A = malloc(rows_per_proc * N_padded * sizeof(double));
    double *local_C = malloc(rows_per_proc * N_padded * sizeof(double));

    if (rank == 0)
    {
        A = calloc(N_padded * N_padded, sizeof(double));
        C = calloc(N_padded * N_padded, sizeof(double));

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                A[i * N_padded + j] = 1.0;
                B[i * N_padded + j] = 1.0;
            }
    }
    else
    {
        for (int i = 0; i < N_padded * N_padded; i++)
            B[i] = 0.0;
    }

    MPI_Bcast(B, N_padded * N_padded, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Scatter(A, rows_per_proc * N_padded, MPI_DOUBLE,
                local_A, rows_per_proc * N_padded, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    for (int i = 0; i < rows_per_proc; i++)
        for (int j = 0; j < N_padded; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < N_padded; k++)
                sum += local_A[i * N_padded + k] * B[k * N_padded + j];
            local_C[i * N_padded + j] = sum;
        }

    MPI_Gather(local_C, rows_per_proc * N_padded, MPI_DOUBLE,
               C, rows_per_proc * N_padded, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == 0)
    {
        printf("Matrix Size = %d x %d\n", N, N);
        printf("MPI Processes = %d\n", size);
        printf("MPI Time = %.6f seconds\n", end - start);
        printf("Sample Result C[0][0] = %.2f\n", C[0]);

        free(A); free(C);
    }

    free(B); free(local_A); free(local_C);
    MPI_Finalize();
    return 0;
}
