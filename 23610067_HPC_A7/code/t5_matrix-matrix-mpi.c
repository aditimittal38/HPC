#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;

    int N = 8;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0)
    {
        if (rank == 0)
        {
            printf(
                "Matrix size must be divisible by number of processes.\n"
            );
        }

        MPI_Finalize();
        return 0;
    }

    int rows_per_process = N / size;

    double *A = NULL;
    double *B = NULL;
    double *C = NULL;

    double *local_A =
        malloc(rows_per_process * N * sizeof(double));

    double *local_C =
        malloc(rows_per_process * N * sizeof(double));

    /*
     * B is required by every process,
     * so every process allocates memory for B.
     */
    B = malloc(N * N * sizeof(double));

    if (rank == 0)
    {
        A = malloc(N * N * sizeof(double));
        C = malloc(N * N * sizeof(double));

        // Initialize A and B
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                A[i * N + j] = 1.0;
                B[i * N + j] = 1.0;
            }
        }
    }

    /*
     * Broadcast matrix B to all processes.
     */
    MPI_Bcast(
        B,
        N * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    /*
     * Distribute rows of A.
     */
    MPI_Scatter(
        A,
        rows_per_process * N,
        MPI_DOUBLE,
        local_A,
        rows_per_process * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    /*
     * Local matrix multiplication.
     *
     * local_C = local_A * B
     */
    for (int i = 0; i < rows_per_process; i++)
    {
        for (int j = 0; j < N; j++)
        {
            local_C[i * N + j] = 0.0;

            for (int k = 0; k < N; k++)
            {
                local_C[i * N + j] +=
                    local_A[i * N + k] *
                    B[k * N + j];
            }
        }
    }

    /*
     * Collect all rows of C at Process 0.
     */
    MPI_Gather(
        local_C,
        rows_per_process * N,
        MPI_DOUBLE,
        C,
        rows_per_process * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0)
    {
        printf("Matrix Size = %d x %d\n", N, N);
        printf("MPI Processes = %d\n", size);
        printf("MPI Time = %.6f seconds\n",
               end - start);

        printf("Sample Result C[0][0] = %.2f\n",
               C[0]);

        printf("Result Matrix:\n");

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                printf("%.2f ", C[i * N + j]);
            }

            printf("\n");
        }
    }

    free(B);
    free(local_A);
    free(local_C);

    if (rank == 0)
    {
        free(A);
        free(C);
    }

    MPI_Finalize();

    return 0;
}