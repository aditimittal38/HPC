// Task 1: Thread Creation and Identification using OpenMP
// Matrix-Matrix Addition: C = A + B


// gcc -fopenmp Task1/matrix-addition.c -o matrix-addition
// ./task1_matrix_add 2
// ./task1_matrix_add 4
// ./task1_matrix_add 8


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define ROWS 4
#define COLS 4

int main(int argc, char *argv[]) {
    int A[ROWS][COLS], B[ROWS][COLS], C[ROWS][COLS];
    int num_threads = 4;

    if (argc > 1) num_threads = atoi(argv[1]);
    omp_set_num_threads(num_threads);

    for (int i = 0; i < ROWS; i++)
        for (int j = 0; j < COLS; j++) {
            A[i][j] = i + j;
            B[i][j] = i - j;
        }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int total = omp_get_num_threads();

        #pragma omp single
        printf("Total number of threads created: %d\n", total);

        #pragma omp for
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++)
                C[i][j] = A[i][j] + B[i][j];
            printf("Thread %d computed row %d\n", tid, i);
        }
    }

    printf("\nResultant Matrix C = A + B:\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++)
            printf("%4d ", C[i][j]);
        printf("\n");
    }
    return 0;
}
