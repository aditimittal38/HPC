/*
 * Task 1: Sum / Max / Min / Average via OpenMP sections
 * Compile: gcc -fopenmp Task1/array-op-sections.c -o array-op-sections
 * Run:     ./array-op-sections <array_size> [threads]
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    int n = (argc > 1) ? atoi(argv[1]) : 1000000;
    int threads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    omp_set_num_threads(threads);

    double *arr = malloc((size_t)n * sizeof(double));
    srand(42);
    for (int i = 0; i < n; i++) arr[i] = rand() % 1000;

    /* ---------- Sequential ---------- */
    double sum_seq = 0, max_seq = arr[0], min_seq = arr[0], avg_seq;
    double t0 = omp_get_wtime();
    for (int i = 0; i < n; i++) {
        sum_seq += arr[i];
        if (arr[i] > max_seq) max_seq = arr[i];
        if (arr[i] < min_seq) min_seq = arr[i];
    }
    avg_seq = sum_seq / n;
    double t1 = omp_get_wtime();
    double seq_time = t1 - t0;

    /* ---------- OpenMP sections ---------- */
    double sum_par = 0, max_par = arr[0], min_par = arr[0], avg_par;
    double t2 = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        { double s = 0; for (int i = 0; i < n; i++) s += arr[i]; sum_par = s; }

        #pragma omp section
        { double m = arr[0]; for (int i = 0; i < n; i++) if (arr[i] > m) m = arr[i]; max_par = m; }

        #pragma omp section
        { double m = arr[0]; for (int i = 0; i < n; i++) if (arr[i] < m) m = arr[i]; min_par = m; }

        #pragma omp section
        { double s = 0; for (int i = 0; i < n; i++) s += arr[i]; avg_par = s / n; }
    }
    double t3 = omp_get_wtime();
    double par_time = t3 - t2;

    int correct = (sum_seq == sum_par) && (max_seq == max_par) &&
                  (min_seq == min_par) && (avg_seq == avg_par);

    printf("Sequential time: %f s\n", seq_time);
    printf("Sections time  : %f s\n", par_time);
    printf("Speedup        : %f\n", seq_time / par_time);
    printf("Correctness    : %s\n", correct ? "PASSED" : "FAILED");

    free(arr);
    return 0;
}