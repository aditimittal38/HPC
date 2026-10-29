#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int N = argc > 1 ? atoi(argv[1]) : 500;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N <= 0)
    {
        if (rank == 0)
        {
            fprintf(stderr, "Matrix size must be a positive integer.\n");
        }

        MPI_Finalize();
        return 1;
    }

    int rows_per_process = N / size + (rank < N % size);

    double *A = NULL;
    double *X = malloc(N * sizeof(double));
    double *Y = NULL;
    int *matrix_counts = NULL;
    int *matrix_offsets = NULL;
    int *vector_counts = NULL;
    int *vector_offsets = NULL;

    double *local_A =
        malloc(rows_per_process * N * sizeof(double));

    double *local_Y =
        malloc(rows_per_process * sizeof(double));

    if (rank == 0)
    {
        A = malloc(N * N * sizeof(double));
        Y = malloc(N * sizeof(double));
        matrix_counts = malloc(size * sizeof(int));
        matrix_offsets = malloc(size * sizeof(int));
        vector_counts = malloc(size * sizeof(int));
        vector_offsets = malloc(size * sizeof(int));

        int offset = 0;
        for (int process = 0; process < size; process++)
        {
            int rows = N / size + (process < N % size);
            vector_counts[process] = rows;
            vector_offsets[process] = offset;
            matrix_counts[process] = rows * N;
            matrix_offsets[process] = offset * N;
            offset += rows;
        }

        for (int i = 0; i < N; i++)
        {
            X[i] = 1.0;

            for (int j = 0; j < N; j++)
            {
                A[i * N + j] = 1.0;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Bcast(
        X,
        N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );
        double broadcast_end = MPI_Wtime();

    MPI_Scatterv(
        A,
        matrix_counts,
        matrix_offsets,
        MPI_DOUBLE,
        local_A,
        rows_per_process * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );
        double scatter_end = MPI_Wtime();

    for (int i = 0; i < rows_per_process; i++)
    {
        local_Y[i] = 0.0;

        for (int j = 0; j < N; j++)
        {
            local_Y[i] +=
                local_A[i * N + j] * X[j];
        }
    }
        double compute_end = MPI_Wtime();

    MPI_Gatherv(
        local_Y,
        rows_per_process,
        MPI_DOUBLE,
        Y,
        vector_counts,
        vector_offsets,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    double end = MPI_Wtime();
    double elapsed = end - start;
    double mpi_time = 0.0;
        double broadcast_time = broadcast_end - start;
        double scatter_time = scatter_end - broadcast_end;
        double compute_time = compute_end - scatter_end;
        double gather_time = end - compute_end;
        double max_broadcast = 0.0, max_scatter = 0.0;
        double max_compute = 0.0, min_compute = 0.0, max_gather = 0.0;

    MPI_Reduce(&elapsed, &mpi_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&broadcast_time, &max_broadcast, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&scatter_time, &max_scatter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&compute_time, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&compute_time, &min_compute, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&gather_time, &max_gather, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        double checksum = 0.0;
        for (int i = 0; i < N; i++)
        {
            checksum += Y[i];
        }

        printf("Matrix Size = %d x %d\n", N, N);
        printf("MPI Processes = %d\n", size);
        printf("MPI Time = %.9f seconds\n", mpi_time);
        printf("Result checksum = %.2f\n", checksum);
            printf("Broadcast Time = %.9f seconds\n", max_broadcast);
            printf("Scatter Time = %.9f seconds\n", max_scatter);
            printf("Computation Time = %.9f seconds\n", max_compute);
            printf("Minimum Computation Time = %.9f seconds\n", min_compute);
            printf("Gather Time = %.9f seconds\n", max_gather);

        if (N <= 16)
        {
            printf("Result vector:\n");

            for (int i = 0; i < N; i++)
            {
                printf("%.2f ", Y[i]);
            }

            printf("\n");
        }
    }

    free(X);
    free(local_A);
    free(local_Y);

    if (rank == 0)
    {
        free(A);
        free(Y);
        free(matrix_counts);
        free(matrix_offsets);
        free(vector_counts);
        free(vector_offsets);
    }

    MPI_Finalize();

    return 0;
}