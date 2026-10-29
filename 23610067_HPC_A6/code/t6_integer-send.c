#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank;
    int value;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        value = 100;

        MPI_Send(
            &value,
            1,
            MPI_INT,
            1,
            0,
            MPI_COMM_WORLD
        );

        printf("Process 0: Value = %d\n", value);
    }
    else if (rank == 1)
    {
        MPI_Recv(
            &value,
            1,
            MPI_INT,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );

        printf("Process 1: Received Value = %d\n", value);
    }

    MPI_Finalize();

    return 0;
}
