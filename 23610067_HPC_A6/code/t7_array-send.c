#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank;
    int data[5] = {10, 20, 30, 40, 50};
    int received[5];

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        MPI_Send(
            data,
            5,
            MPI_INT,
            1,
            0,
            MPI_COMM_WORLD
        );

        printf("Process 0 sent array: ");

        for (int i = 0; i < 5; i++)
        {
            printf("%d ", data[i]);
        }

        printf("\n");
    }
    else if (rank == 1)
    {
        MPI_Recv(
            received,
            5,
            MPI_INT,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );

        int sum = 0;
        int max = received[0];

        for (int i = 0; i < 5; i++)
        {
            sum += received[i];

            if (received[i] > max)
            {
                max = received[i];
            }
        }

        double average = (double)sum / 5;

        printf("Process 1 received array: ");

        for (int i = 0; i < 5; i++)
        {
            printf("%d ", received[i]);
        }

        printf("\n");
        printf("Sum = %d\n", sum);
        printf("Average = %.2f\n", average);
        printf("Maximum = %d\n", max);
    }

    MPI_Finalize();

    return 0;
}
