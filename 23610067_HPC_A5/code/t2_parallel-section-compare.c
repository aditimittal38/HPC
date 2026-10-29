/*
 * Task 2: parallel for vs sections for Sum/Max/Min/Even-count
 * Compile: gcc -fopenmp Task2/parallel-section-compare.c -o parallel-section-compare
 * Run:     ./parallel-section-compare <array_size> [threads]
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    int n = (argc > 1) ? atoi(argv[1]) : 1000000;
    int threads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    omp_set_num_threads(threads);

    int *arr = malloc((size_t)n * sizeof(int));
    srand(42);
    for (int i = 0; i < n; i++) arr[i] = rand() % 1000;

    /* ---------- parallel for (reduction-based) ---------- */
    long sum_for = 0; int max_for = arr[0], min_for = arr[0], even_for = 0;
    double t0 = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum_for, even_for) reduction(max:max_for) reduction(min:min_for)
    for (int i = 0; i < n; i++) {
        sum_for += arr[i];
        if (arr[i] > max_for) max_for = arr[i];
        if (arr[i] < min_for) min_for = arr[i];
        if (arr[i] % 2 == 0) even_for++;
    }
    double t1 = omp_get_wtime();
    double for_time = t1 - t0;

    /* ---------- sections (one operation per section) ---------- */
    long sum_sec = 0; int max_sec = arr[0], min_sec = arr[0], even_sec = 0;
    double t2 = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        { long s = 0; for (int i = 0; i < n; i++) s += arr[i]; sum_sec = s; }
        #pragma omp section
        { int m = arr[0]; for (int i = 0; i < n; i++) if (arr[i] > m) m = arr[i]; max_sec = m; }
        #pragma omp section
        { int m = arr[0]; for (int i = 0; i < n; i++) if (arr[i] < m) m = arr[i]; min_sec = m; }
        #pragma omp section
        { int c = 0; for (int i = 0; i < n; i++) if (arr[i] % 2 == 0) c++; even_sec = c; }
    }
    double t3 = omp_get_wtime();
    double sec_time = t3 - t2;

    printf("parallel for : %f s | sum=%ld max=%d min=%d even=%d\n", for_time, sum_for, max_for, min_for, even_for);
    printf("sections     : %f s | sum=%ld max=%d min=%d even=%d\n", sec_time, sum_sec, max_sec, min_sec, even_sec);

    free(arr);
    return 0;
}