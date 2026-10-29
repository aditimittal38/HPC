#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;

    int N = 10;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 3)
    {
        if (rank == 0)
        {
            printf(
                "Please run this program with 3 processes.\n"
            );
        }

        MPI_Finalize();
        return 0;
    }

    /*
     * Number of rows assigned to each process.
     *
     * 10 rows:
     *
     * P0 -> 4 rows
     * P1 -> 3 rows
     * P2 -> 3 rows
     */
    int rows_per_process[3] = {4, 3, 3};

    /*
     * Number of matrix elements sent to
     * each process.
     *
     * Each row contains N = 10 elements.
     */
    int sendcounts[3] =
    {
        4 * N,
        3 * N,
        3 * N
    };

    /*
     * Starting positions in flattened matrix.
     */
    int displacements[3] =
    {
        0,
        4 * N,
        7 * N
    };

    int local_rows =
        rows_per_process[rank];

    double *A = NULL;
    double *X = NULL;
    double *Y = NULL;

    double *local_A =
        malloc(local_rows * N * sizeof(double));

    double *local_Y =
        malloc(local_rows * sizeof(double));

    /*
     * Every process needs the complete vector X.
     */
    X = malloc(N * sizeof(double));

    if (rank == 0)
    {
        A = malloc(N * N * sizeof(double));
        Y = malloc(N * sizeof(double));

        /*
         * Initialize A and X.
         */
        for (int i = 0; i < N; i++)
        {
            X[i] = 1.0;

            for (int j = 0; j < N; j++)
            {
                A[i * N + j] = 1.0;
            }
        }
    }

    /*
     * Broadcast vector X.
     */
    MPI_Bcast(
        X,
        N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    /*
     * Scatter unequal numbers of matrix elements.
     */
    MPI_Scatterv(
        A,
        sendcounts,
        displacements,
        MPI_DOUBLE,
        local_A,
        local_rows * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    /*
     * Local matrix-vector multiplication.
     */
    for (int i = 0; i < local_rows; i++)
    {
        local_Y[i] = 0.0;

        for (int j = 0; j < N; j++)
        {
            local_Y[i] +=
                local_A[i * N + j] * X[j];
        }
    }

    /*
     * Since the number of result elements is
     * different for each process, use MPI_Gatherv().
     */
    int recvcounts[3] =
    {
        4,
        3,
        3
    };

    int recvdisplacements[3] =
    {
        0,
        4,
        7
    };

    MPI_Gatherv(
        local_Y,
        local_rows,
        MPI_DOUBLE,
        Y,
        recvcounts,
        recvdisplacements,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0)
    {
        printf("MPI_Scatterv Matrix-Vector Multiplication\n");
        printf("Matrix Size = %d x %d\n", N, N);
        printf("MPI Processes = %d\n", size);

        printf("Row Distribution:\n");
        printf("Process 0 -> 4 rows\n");
        printf("Process 1 -> 3 rows\n");
        printf("Process 2 -> 3 rows\n");

        printf("\nResult Vector:\n");

        for (int i = 0; i < N; i++)
        {
            printf("%.2f ", Y[i]);
        }

        printf("\n");
    }

    free(X);
    free(local_A);
    free(local_Y);

    if (rank == 0)
    {
        free(A);
        free(Y);
    }

    MPI_Finalize();

    return 0;
}