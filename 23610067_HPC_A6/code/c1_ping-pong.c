#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank;
    int iterations = 1000;

    int sizes[] = {
        1024,
        10 * 1024,
        100 * 1024,
        1024 * 1024,
        10 * 1024 * 1024
    };

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int s = 0; s < 5; s++)
    {
        int bytes = sizes[s];

        char *buffer = (char *)malloc(bytes);

        MPI_Barrier(MPI_COMM_WORLD);

        double start = 0.0;
        double end = 0.0;

        if (rank == 0)
        {
            start = MPI_Wtime();

            for (int i = 0; i < iterations; i++)
            {
                MPI_Send(
                    buffer,
                    bytes,
                    MPI_BYTE,
                    1,
                    0,
                    MPI_COMM_WORLD
                );

                MPI_Recv(
                    buffer,
                    bytes,
                    MPI_BYTE,
                    1,
                    0,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE
                );
            }

            end = MPI_Wtime();

            double total_time = end - start;
            double average_rtt = total_time / iterations;

            printf(
                "Size = %d bytes | Iterations = %d | Total Time = %.6f s | Average RTT = %.9f s\n",
                bytes,
                iterations,
                total_time,
                average_rtt
            );
        }
        else if (rank == 1)
        {
            for (int i = 0; i < iterations; i++)
            {
                MPI_Recv(
                    buffer,
                    bytes,
                    MPI_BYTE,
                    0,
                    0,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE
                );

                MPI_Send(
                    buffer,
                    bytes,
                    MPI_BYTE,
                    0,
                    0,
                    MPI_COMM_WORLD
                );
            }
        }

        free(buffer);
    }

    MPI_Finalize();

    return 0;
}
