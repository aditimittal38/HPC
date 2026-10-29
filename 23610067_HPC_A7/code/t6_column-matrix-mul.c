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

    int cols_per_process = N / size;

    double *A = NULL;
    double *B = NULL;
    double *C = NULL;

    double *local_B =
        malloc(N * cols_per_process * sizeof(double));

    double *local_C =
        malloc(N * cols_per_process * sizeof(double));

    /*
     * Every process needs A.
     */
    A = malloc(N * N * sizeof(double));

    if (rank == 0)
    {
        B = malloc(N * N * sizeof(double));
        C = malloc(N * N * sizeof(double));

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                A[i * N + j] = 1.0;
                B[i * N + j] = 1.0;
            }
        }
    }
    else
    {
        B = malloc(N * N * sizeof(double));
    }

    /*
     * Broadcast A to every process.
     */
    MPI_Bcast(
        A,
        N * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    /*
     * Distribute columns of B.
     *
     * B is stored row-wise, so MPI_Scatter
     * cannot directly send contiguous columns.
     *
     * Therefore, Process 0 creates a temporary
     * column-wise representation.
     */

    double *B_transposed = NULL;

    if (rank == 0)
    {
        B_transposed =
            malloc(N * N * sizeof(double));

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                B_transposed[j * N + i] =
                    B[i * N + j];
            }
        }
    }

    /*
     * Scatter columns of B after transposing it.
     */
    MPI_Scatter(
        B_transposed,
        cols_per_process * N,
        MPI_DOUBLE,
        local_B,
        cols_per_process * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    /*
     * Each process computes its assigned
     * columns of C.
     */
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < cols_per_process; j++)
        {
            local_C[i * cols_per_process + j] = 0.0;

            for (int k = 0; k < N; k++)
            {
                local_C[i * cols_per_process + j] +=
                    A[i * N + k] *
                    local_B[j * N + k];
            }
        }
    }

    /*
     * Gather the local column blocks.
     */
    double *C_transposed = NULL;

    if (rank == 0)
    {
        C_transposed =
            malloc(N * N * sizeof(double));
    }

    MPI_Gather(
        local_C,
        N * cols_per_process,
        MPI_DOUBLE,
        C_transposed,
        N * cols_per_process,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0)
    {
        /*
         * Convert gathered column representation
         * back to normal row-major C.
         */
        for (int p = 0; p < size; p++)
        {
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < cols_per_process; j++)
                {
                    int global_col =
                        p * cols_per_process + j;

                    C[i * N + global_col] =
                        C_transposed[
                            p * N * cols_per_process +
                            i * cols_per_process +
                            j
                        ];
                }
            }
        }

        printf("Column-wise Matrix Multiplication\n");
        printf("Matrix Size = %d x %d\n", N, N);
        printf("MPI Processes = %d\n", size);
        printf("C[0][0] = %.2f\n", C[0]);

        free(B_transposed);
        free(C_transposed);
        free(C);
    }

    free(A);
    free(B);
    free(local_B);
    free(local_C);

    MPI_Finalize();

    return 0;
}