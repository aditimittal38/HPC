/*
 * Task 3: Compare Decomposition Strategies for Matrix Multiplication
 *   - Row-wise decomposition    : parallelize outer loop over rows (i)
 *   - Column-wise decomposition : parallelize outer loop over columns (j)
 *   - Block (2D) decomposition  : divide C into tiles, assign tiles to threads
 *
 */
// gcc -fopenmp Task3/decomposition-strategies.c -o decomposition-strategies


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void generate_matrix(double *M, int n) {
    for (int i = 0; i < n * n; i++) M[i] = (double)(rand() % 10);
}

/* ---------- Row-wise decomposition ---------- */
void matmul_row(double *A, double *B, double *C, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = sum;
        }
}

/* ---------- Column-wise decomposition ---------- */
void matmul_col(double *A, double *B, double *C, int n) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = sum;
        }
}

/* ---------- Block (2D) decomposition ----------
   The output matrix C is divided into square tiles; each (block-row,
   block-col) tile is one unit of work handed to a thread. */
void matmul_block(double *A, double *B, double *C, int n, int block_size) {
    int num_blocks = (n + block_size - 1) / block_size;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < num_blocks; bi++) {
        for (int bj = 0; bj < num_blocks; bj++) {
            int i_start = bi * block_size;
            int i_end   = (i_start + block_size < n) ? i_start + block_size : n;
            int j_start = bj * block_size;
            int j_end   = (j_start + block_size < n) ? j_start + block_size : n;

            for (int i = i_start; i < i_end; i++)
                for (int j = j_start; j < j_end; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < n; k++)
                        sum += A[(size_t)i * n + k] * B[(size_t)k * n + j];
                    C[(size_t)i * n + j] = sum;
                }
        }
    }
}

int main(int argc, char *argv[]) {
    int n          = (argc > 1) ? atoi(argv[1]) : 500;
    int threads    = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    int block_size = (argc > 3) ? atoi(argv[3]) : 64;
    omp_set_num_threads(threads);

    double *A  = malloc((size_t)n * n * sizeof(double));
    double *B  = malloc((size_t)n * n * sizeof(double));
    double *C1 = malloc((size_t)n * n * sizeof(double));
    double *C2 = malloc((size_t)n * n * sizeof(double));
    double *C3 = malloc((size_t)n * n * sizeof(double));

    srand(42);
    generate_matrix(A, n);
    generate_matrix(B, n);

    double t0 = omp_get_wtime(); matmul_row(A, B, C1, n);              double t1 = omp_get_wtime();
    double t2 = omp_get_wtime(); matmul_col(A, B, C2, n);              double t3 = omp_get_wtime();
    double t4 = omp_get_wtime(); matmul_block(A, B, C3, n, block_size); double t5 = omp_get_wtime();

    printf("========================================\n");
    printf("Decomposition Strategy Comparison\n");
    printf("Matrix size: %d x %d | Threads: %d | Block size: %d\n", n, n, threads, block_size);
    printf("Row-wise    time: %f s\n", t1 - t0);
    printf("Column-wise time: %f s\n", t3 - t2);
    printf("Block (2D)  time: %f s\n", t5 - t4);
    printf("========================================\n");

    free(A); free(B); free(C1); free(C2); free(C3);
    return 0;
}