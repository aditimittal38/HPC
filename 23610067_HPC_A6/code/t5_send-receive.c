#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank;
    char message[100];

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        strcpy(message, "Hello from Process 0");

        MPI_Send(
            message,
            strlen(message) + 1,
            MPI_CHAR,
            1,
            0,
            MPI_COMM_WORLD
        );

        printf("Process 0 sent: %s\n", message);
    }
    else if (rank == 1)
    {
        MPI_Recv(
            message,
            100,
            MPI_CHAR,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );

        printf("Process 1 received: %s\n", message);
    }

    MPI_Finalize();

    return 0;
}
