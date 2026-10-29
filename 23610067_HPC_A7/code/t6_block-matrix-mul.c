#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;

    int N = argc > 1 ? atoi(argv[1]) : 8;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /*
     * This implementation uses 4 processes:
     *
     * P0 P1
     * P2 P3
     */
    if (size != 4)
    {
        if (rank == 0)
        {
            printf(
                "Please run this program with 4 processes.\n"
            );
        }

        MPI_Finalize();
        return 0;
    }

    if (N <= 0 || N % 2 != 0)
    {
        if (rank == 0)
        {
            fprintf(stderr, "Matrix size must be a positive even integer.\n");
        }

        MPI_Finalize();
        return 1;
    }

    int block_size = N / 2;

    double *A = malloc(N * N * sizeof(double));
    double *B = malloc(N * N * sizeof(double));
    double *C = NULL;

    double *local_C =
        calloc(block_size * block_size,
               sizeof(double));

    if (rank == 0)
    {
        C = malloc(N * N * sizeof(double));

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                A[i * N + j] = 1.0;
                B[i * N + j] = 1.0;
                C[i * N + j] = 0.0;
            }
        }
    }

    /*
     * For this simple demonstration:
     *
     * Every process receives the required
     * portions of A and B.
     *
     * A complete implementation for arbitrary
     * process grids would use MPI Cartesian
     * communicators and subarray datatypes.
     */

    MPI_Bcast(
        B,
        N * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    /*
     * Broadcast A as well.
     */
    MPI_Bcast(
        A,
        N * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    /*
     * Determine block position.
     *
     * P0 -> block (0,0)
     * P1 -> block (0,1)
     * P2 -> block (1,0)
     * P3 -> block (1,1)
     */
    int block_row = rank / 2;
    int block_col = rank % 2;

    /*
     * Each process computes its C block.
     *
     * For a block C(i,j), we calculate:
     *
     * C(i,j) = sum A(i,k) * B(k,j)
     */
    for (int i = 0; i < block_size; i++)
    {
        for (int j = 0; j < block_size; j++)
        {
            double sum = 0.0;

            for (int k = 0; k < N; k++)
            {
                int global_i =
                    block_row * block_size + i;

                int global_j =
                    block_col * block_size + j;

                sum +=
                    A[global_i * N + k] *
                    B[k * N + global_j];
            }

            local_C[i * block_size + j] = sum;
        }
    }

    /*
     * Gather blocks at Process 0.
     */
    double *all_blocks = NULL;

    if (rank == 0)
    {
        all_blocks =
            malloc(N * N * sizeof(double));
    }

    MPI_Gather(
        local_C,
        block_size * block_size,
        MPI_DOUBLE,
        all_blocks,
        block_size * block_size,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0)
    {
        /*
         * Rearrange blocks into C.
         */
        for (int p = 0; p < 4; p++)
        {
            int p_row = p / 2;
            int p_col = p % 2;

            for (int i = 0; i < block_size; i++)
            {
                for (int j = 0; j < block_size; j++)
                {
                    int global_i =
                        p_row * block_size + i;

                    int global_j =
                        p_col * block_size + j;

                    C[global_i * N + global_j] =
                        all_blocks[
                            p * block_size * block_size +
                            i * block_size +
                            j
                        ];
                }
            }
        }

        printf("Block-wise Matrix Multiplication\n");
        printf("Matrix Size = %d x %d\n", N, N);
        printf("MPI Processes = %d\n", size);
        printf("C[0][0] = %.2f\n", C[0]);

        free(C);
        free(all_blocks);
    }

    free(A);
    free(B);
    free(local_C);

    MPI_Finalize();

    return 0;
}