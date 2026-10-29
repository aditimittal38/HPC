#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;

    int array[16];
    int local_array[16];

    int local_sum;
    int local_max;
    int local_min;

    int global_sum;
    int global_max;
    int global_min;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4)
    {
        if (rank == 0)
        {
            printf("Please run this program with 4 processes.\n");
        }

        MPI_Finalize();
        return 0;
    }

    if (rank == 0)
    {
        for (int i = 0; i < 16; i++)
        {
            array[i] = i + 1;
        }
    }

    MPI_Scatter(
        array,
        4,
        MPI_INT,
        local_array,
        4,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    local_sum = 0;
    local_max = local_array[0];
    local_min = local_array[0];

    for (int i = 0; i < 4; i++)
    {
        local_sum += local_array[i];

        if (local_array[i] > local_max)
            local_max = local_array[i];

        if (local_array[i] < local_min)
            local_min = local_array[i];
    }

    MPI_Reduce(
        &local_sum,
        &global_sum,
        1,
        MPI_INT,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &local_max,
        &global_max,
        1,
        MPI_INT,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &local_min,
        &global_min,
        1,
        MPI_INT,
        MPI_MIN,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0)
    {
        printf("Global Sum = %d\n", global_sum);
        printf("Global Maximum = %d\n", global_max);
        printf("Global Minimum = %d\n", global_min);
    }

    MPI_Finalize();

    return 0;
}