/*
 * Task 4: Compare Decomposition Strategies for Matrix Multiplication
 *
 *   - Sequential baseline
 *   - Row-wise decomposition    : parallelize outer loop over rows (i)
 *   - Column-wise decomposition : parallelize outer loop over columns (j)
 *   - Block (2D) decomposition  : divide C into tiles, assign tiles to threads
 *
 * Compile:
 * gcc -fopenmp Task4/decomp-strategies.c -o decomposition-strategies
 *
 * Run:
 * ./decomp-strategies [matrix_size] [threads] [block_size]
 *
 * Example:
 * ./decompo-strategies 500 4 64
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* ---------- Generate Matrix ---------- */
void generate_matrix(double *M, int n) {
    for (int i = 0; i < n * n; i++)
        M[i] = (double)(rand() % 10);
}

/* ---------- Sequential Matrix Multiplication ---------- */
void matmul_seq(double *A, double *B, double *C, int n) {

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {

            double sum = 0.0;

            for (int k = 0; k < n; k++) {
                sum += A[(size_t)i * n + k] *
                       B[(size_t)k * n + j];
            }

            C[(size_t)i * n + j] = sum;
        }
    }
}

/* ---------- Row-wise decomposition ---------- */
void matmul_row(double *A, double *B, double *C, int n) {

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {

        for (int j = 0; j < n; j++) {

            double sum = 0.0;

            for (int k = 0; k < n; k++) {
                sum += A[(size_t)i * n + k] *
                       B[(size_t)k * n + j];
            }

            C[(size_t)i * n + j] = sum;
        }
    }
}

/* ---------- Column-wise decomposition ---------- */
void matmul_col(double *A, double *B, double *C, int n) {

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n; j++) {

        for (int i = 0; i < n; i++) {

            double sum = 0.0;

            for (int k = 0; k < n; k++) {
                sum += A[(size_t)i * n + k] *
                       B[(size_t)k * n + j];
            }

            C[(size_t)i * n + j] = sum;
        }
    }
}

/* ---------- Block (2D) decomposition ----------
 *
 * The output matrix C is divided into square tiles.
 * Each tile is assigned to a thread.
 */
void matmul_block(double *A, double *B, double *C,
                  int n, int block_size) {

    int num_blocks = (n + block_size - 1) / block_size;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < num_blocks; bi++) {

        for (int bj = 0; bj < num_blocks; bj++) {

            int i_start = bi * block_size;
            int i_end = (i_start + block_size < n)
                        ? i_start + block_size
                        : n;

            int j_start = bj * block_size;
            int j_end = (j_start + block_size < n)
                        ? j_start + block_size
                        : n;

            for (int i = i_start; i < i_end; i++) {

                for (int j = j_start; j < j_end; j++) {

                    double sum = 0.0;

                    for (int k = 0; k < n; k++) {
                        sum += A[(size_t)i * n + k] *
                               B[(size_t)k * n + j];
                    }

                    C[(size_t)i * n + j] = sum;
                }
            }
        }
    }
}

/* ---------- Check if two matrices are equal ---------- */
int verify_result(double *C1, double *C2, int n) {

    double tolerance = 1e-9;

    for (int i = 0; i < n * n; i++) {

        if (abs(C1[i] - C2[i]) > tolerance) {
            return 0;
        }
    }

    return 1;
}

/* ---------- Main ---------- */
int main(int argc, char *argv[]) {

    /* Command-line arguments */
    int n = (argc > 1) ? atoi(argv[1]) : 500;

    int threads = (argc > 2)
                  ? atoi(argv[2])
                  : omp_get_max_threads();

    int block_size = (argc > 3)
                     ? atoi(argv[3])
                     : 64;

    omp_set_num_threads(threads);

    /* ---------- Allocate matrices ---------- */

    double *A = malloc((size_t)n * n * sizeof(double));
    double *B = malloc((size_t)n * n * sizeof(double));

    double *C_seq   = malloc((size_t)n * n * sizeof(double));
    double *C_row   = malloc((size_t)n * n * sizeof(double));
    double *C_col   = malloc((size_t)n * n * sizeof(double));
    double *C_block = malloc((size_t)n * n * sizeof(double));

    if (A == NULL || B == NULL ||
        C_seq == NULL || C_row == NULL ||
        C_col == NULL || C_block == NULL) {

        printf("Memory allocation failed.\n");

        free(A);
        free(B);
        free(C_seq);
        free(C_row);
        free(C_col);
        free(C_block);

        return 1;
    }

    /* ---------- Generate input matrices ---------- */

    srand(42);

    generate_matrix(A, n);
    generate_matrix(B, n);

    /* =========================================================
       1. Sequential Baseline
       ========================================================= */

    double t0 = omp_get_wtime();

    matmul_seq(A, B, C_seq, n);

    double t1 = omp_get_wtime();

    double seq_time = t1 - t0;

    /* =========================================================
       2. Row-wise decomposition
       ========================================================= */

    double t2 = omp_get_wtime();

    matmul_row(A, B, C_row, n);

    double t3 = omp_get_wtime();

    double row_time = t3 - t2;

    /* =========================================================
       3. Column-wise decomposition
       ========================================================= */

    double t4 = omp_get_wtime();

    matmul_col(A, B, C_col, n);

    double t5 = omp_get_wtime();

    double col_time = t5 - t4;

    /* =========================================================
       4. Block (2D) decomposition
       ========================================================= */

    double t6 = omp_get_wtime();

    matmul_block(A, B, C_block, n, block_size);

    double t7 = omp_get_wtime();

    double block_time = t7 - t6;

    /* =========================================================
       Speedup
       
       Speedup = Sequential Time / Parallel Time
       ========================================================= */

    double row_speedup =
        seq_time / row_time;

    double col_speedup =
        seq_time / col_time;

    double block_speedup =
        seq_time / block_time;

    /* =========================================================
       Efficiency
       
       Efficiency = Speedup / Number of Threads
       ========================================================= */

    double row_efficiency =
        row_speedup / threads;

    double col_efficiency =
        col_speedup / threads;

    double block_efficiency =
        block_speedup / threads;

    /* =========================================================
       Verify correctness
       ========================================================= */

    int row_correct =
        verify_result(C_seq, C_row, n);

    int col_correct =
        verify_result(C_seq, C_col, n);

    int block_correct =
        verify_result(C_seq, C_block, n);

    /* =========================================================
       Print Results
       ========================================================= */

    printf("\n");
    printf("============================================================\n");
    printf("        DECOMPOSITION STRATEGY COMPARISON\n");
    printf("============================================================\n");

    printf("Matrix size : %d x %d\n", n, n);
    printf("Threads     : %d\n", threads);
    printf("Block size  : %d\n", block_size);

    printf("------------------------------------------------------------\n");

    printf("\nExecution Time:\n");

    printf("Sequential  : %f s\n", seq_time);
    printf("Row-wise    : %f s\n", row_time);
    printf("Column-wise : %f s\n", col_time);
    printf("Block (2D)  : %f s\n", block_time);

    printf("\n------------------------------------------------------------\n");

    printf("Speedup (Sequential / Parallel):\n");

    printf("Row-wise    : %.4fx\n", row_speedup);
    printf("Column-wise : %.4fx\n", col_speedup);
    printf("Block (2D)  : %.4fx\n", block_speedup);

    printf("\n------------------------------------------------------------\n");

    printf("Efficiency (Speedup / Threads):\n");

    printf("Row-wise    : %.4f (%.2f%%)\n",
           row_efficiency,
           row_efficiency * 100.0);

    printf("Column-wise : %.4f (%.2f%%)\n",
           col_efficiency,
           col_efficiency * 100.0);

    printf("Block (2D)  : %.4f (%.2f%%)\n",
           block_efficiency,
           block_efficiency * 100.0);

    printf("\n------------------------------------------------------------\n");

    printf("Correctness Check:\n");

    printf("Row-wise    : %s\n",
           row_correct ? "PASS" : "FAIL");

    printf("Column-wise : %s\n",
           col_correct ? "PASS" : "FAIL");

    printf("Block (2D)  : %s\n",
           block_correct ? "PASS" : "FAIL");

    printf("============================================================\n");

    /* ---------- Free memory ---------- */

    free(A);
    free(B);
    free(C_seq);
    free(C_row);
    free(C_col);
    free(C_block);

    return 0;
}
