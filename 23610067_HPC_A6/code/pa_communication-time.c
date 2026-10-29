#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank;
    int sizes[] = {
        1024,
        10 * 1024,
        100 * 1024,
        1024 * 1024,
        10 * 1024 * 1024
    };

    int num_sizes = 5;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int s = 0; s < num_sizes; s++)
    {
        int bytes = sizes[s];
        char *buffer = (char *)malloc(bytes);

        if (rank == 0)
        {
            double start = MPI_Wtime();

            MPI_Send(
                buffer,
                bytes,
                MPI_BYTE,
                1,
                0,
                MPI_COMM_WORLD
            );

            double end = MPI_Wtime();

            printf(
                "Message Size = %d bytes, Communication Time = %.9f seconds\n",
                bytes,
                end - start
            );
        }
        else if (rank == 1)
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
        }

        free(buffer);
    }

    MPI_Finalize();

    return 0;
}


