#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define KERNEL_SIZE 3
#define HALO        (KERNEL_SIZE - 1)

int main(int argc, char *argv[])
{
    int rank, size;
    int N = 1000;
    if (argc > 1) N = atoi(argv[1]);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int M = N - KERNEL_SIZE + 1; /* valid convolution output rows */

    /* base rows of INPUT assigned to each process */
    int base_rows  = N / size;
    int extra      = N % size;
    int my_rows    = base_rows + (rank < extra ? 1 : 0);
    int my_start   = rank * base_rows + (rank < extra ? rank : extra);

    /* how many input rows does this process need (including halo below) */
    int input_rows = my_rows + HALO;
    if (my_start + input_rows > N)
        input_rows = N - my_start;

    double *input     = malloc(N * N * sizeof(double));
    double *local_in  = malloc(input_rows * N * sizeof(double));

    /* output rows produced by this process */
    int out_rows = my_rows;
    if (my_start + my_rows + HALO > N)
        out_rows = (N - HALO) - my_start;
    if (out_rows < 0) out_rows = 0;

    double *local_out = malloc(out_rows * M * sizeof(double));
    double *output    = (rank == 0) ? malloc(M * M * sizeof(double)) : NULL;

    double kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        {1.0/9, 1.0/9, 1.0/9},
        {1.0/9, 1.0/9, 1.0/9},
        {1.0/9, 1.0/9, 1.0/9}
    };

    /* Process 0 initializes and broadcasts input */
    if (rank == 0)
        for (int i = 0; i < N * N; i++)
            input[i] = (double)(i % 256);

    MPI_Bcast(input, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Each process copies its slice (including halo rows) */
    for (int i = 0; i < input_rows; i++)
        for (int j = 0; j < N; j++)
            local_in[i * N + j] = input[(my_start + i) * N + j];

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int i = 0; i < out_rows; i++)
        for (int j = 0; j < M; j++)
        {
            double sum = 0.0;
            for (int u = 0; u < KERNEL_SIZE; u++)
                for (int v = 0; v < KERNEL_SIZE; v++)
                    sum += local_in[(i + u) * N + (j + v)] * kernel[u][v];
            local_out[i * M + j] = sum;
        }

    /* Gather output rows at rank 0 */
    int *recvcounts = NULL;
    int *displs     = NULL;
    if (rank == 0)
    {
        recvcounts = malloc(size * sizeof(int));
        displs     = malloc(size * sizeof(int));
        int off = 0;
        for (int r = 0; r < size; r++)
        {
            int r_rows  = N / size + (r < extra ? 1 : 0);
            int r_start = r * (N / size) + (r < extra ? r : extra);
            int r_out   = r_rows;
            if (r_start + r_rows + HALO > N)
                r_out = (N - HALO) - r_start;
            if (r_out < 0) r_out = 0;
            recvcounts[r] = r_out * M;
            displs[r]     = off;
            off += recvcounts[r];
        }
    }

    MPI_Gatherv(local_out, out_rows * M, MPI_DOUBLE,
                output, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == 0)
    {
        printf("Input Size = %d x %d\n", N, N);
        printf("Output Size = %d x %d\n", M, M);
        printf("MPI Processes = %d\n", size);
        printf("MPI Time = %.6f seconds\n", end - start);
        printf("Sample Output[0][0] = %.4f\n", output[0]);
        free(output);
        free(recvcounts);
        free(displs);
    }

    free(input); free(local_in); free(local_out);
    MPI_Finalize();
    return 0;
}
