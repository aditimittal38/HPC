/*
 * Task 3: Data-processing pipeline (normalize -> transform -> stats)
 * Parallelized by splitting the array into independent chunks, one
 * OpenMP task per chunk, each running the full pipeline on its chunk.
 * Compile: gcc -fopenmp Task3/data-processing-pipeline.c -lm -o data-processing-pipeline
 * Run:     ./data-processing-pipeline <array_size> <num_chunks> [threads]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

double find_max(double *arr, int n) {
    double m = arr[0];
    for (int i = 0; i < n; i++) if (arr[i] > m) m = arr[i];
    return m;
}

void process_chunk(double *arr, int start, int end, double global_max,
                    double *out_sum, double *out_min, double *out_max) {
    for (int i = start; i < end; i++) arr[i] /= global_max;                                   /* normalize */
    for (int i = start; i < end; i++) arr[i] = sin(arr[i]) * cos(arr[i]) + sqrt(arr[i] + 1.0); /* transform */

    double s = 0, mn = arr[start], mx = arr[start];
    for (int i = start; i < end; i++) {
        s += arr[i];
        if (arr[i] < mn) mn = arr[i];
        if (arr[i] > mx) mx = arr[i];
    }
    *out_sum = s; *out_min = mn; *out_max = mx;
}

int main(int argc, char *argv[]) {
    int n          = (argc > 1) ? atoi(argv[1]) : 4000000;
    int num_chunks = (argc > 2) ? atoi(argv[2]) : 8;
    int threads    = (argc > 3) ? atoi(argv[3]) : omp_get_max_threads();
    omp_set_num_threads(threads);

    double *seq = malloc((size_t)n * sizeof(double));
    double *par = malloc((size_t)n * sizeof(double));
    srand(42);
    for (int i = 0; i < n; i++) { double v = (rand() % 1000) / 10.0 + 1; seq[i] = v; par[i] = v; }

    double gmax = find_max(seq, n);

    /* ---------- Sequential: single "chunk" covering the whole array ---------- */
    double sum_s, min_s, max_s;
    double t0 = omp_get_wtime();
    process_chunk(seq, 0, n, gmax, &sum_s, &min_s, &max_s);
    double t1 = omp_get_wtime();
    double seq_time = t1 - t0;

    /* ---------- Task-based: one task per chunk ---------- */
    double *csum = malloc(num_chunks * sizeof(double));
    double *cmin = malloc(num_chunks * sizeof(double));
    double *cmax = malloc(num_chunks * sizeof(double));
    int chunk_size = n / num_chunks;

    double t2 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int c = 0; c < num_chunks; c++) {
                int start = c * chunk_size;
                int end = (c == num_chunks - 1) ? n : start + chunk_size;
                #pragma omp task firstprivate(start, end, c)
                process_chunk(par, start, end, gmax, &csum[c], &cmin[c], &cmax[c]);
            }
            #pragma omp taskwait
        }
    }
    double sum_p = 0, min_p = cmin[0], max_p = cmax[0];
    for (int c = 0; c < num_chunks; c++) {
        sum_p += csum[c];
        if (cmin[c] < min_p) min_p = cmin[c];
        if (cmax[c] > max_p) max_p = cmax[c];
    }
    double t3 = omp_get_wtime();
    double par_time = t3 - t2;

    int correct = (fabs(sum_s - sum_p) < 1e-3) && (fabs(min_s - min_p) < 1e-9) && (fabs(max_s - max_p) < 1e-9);

    printf("Sequential time : %f s\n", seq_time);
    printf("Task-based time : %f s\n", par_time);
    printf("Speedup         : %f\n", seq_time / par_time);
    printf("Correctness     : %s\n", correct ? "PASSED" : "FAILED");

    free(seq); free(par); free(csum); free(cmin); free(cmax);
    return 0;
}