#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank;
    int send_data;
    int recv_data;
    MPI_Status status;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        send_data = 100;

        MPI_Sendrecv(
            &send_data,
            1,
            MPI_INT,
            1,
            0,
            &recv_data,
            1,
            MPI_INT,
            1,
            0,
            MPI_COMM_WORLD,
            &status
        );

        printf(
            "Process 0 sent %d and received %d\n",
            send_data,
            recv_data
        );
    }
    else if (rank == 1)
    {
        send_data = 200;

        MPI_Sendrecv(
            &send_data,
            1,
            MPI_INT,
            0,
            0,
            &recv_data,
            1,
            MPI_INT,
            0,
            0,
            MPI_COMM_WORLD,
            &status
        );

        printf(
            "Process 1 sent %d and received %d\n",
            send_data,
            recv_data
        );
    }

    MPI_Finalize();

    return 0;
}
