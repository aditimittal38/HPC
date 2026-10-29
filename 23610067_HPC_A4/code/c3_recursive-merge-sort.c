/*
 * Section C - Challenge 3: OpenMP Tasks
 * Recursive Merge Sort reimplemented with OpenMP task constructs,
 * compared against a plain sequential version.
 *
 * Compile: gcc -fopenmp Challenge3/recursive-merge-sort.c -o recursive-merge-sort
 * Run:     ./recursive-merge-sort <array_size> [num_threads]
 * Example: ./recursive-merge-sort 2000000 4
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* Below this sub-array size, stop spawning tasks and sort sequentially.
   This controls task granularity (see discussion below). */
#define THRESHOLD 10000

void merge(int *arr, int *temp, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;
    while (i <= mid && j <= right)
        temp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
    while (i <= mid)   temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    for (i = left; i <= right; i++) arr[i] = temp[i];
}

void merge_sort_seq(int *arr, int *temp, int left, int right) {
    if (left >= right) return;
    int mid = left + (right - left) / 2;
    merge_sort_seq(arr, temp, left, mid);
    merge_sort_seq(arr, temp, mid + 1, right);
    merge(arr, temp, left, mid, right);
}

/* Recursive decomposition: each recursive half becomes an OpenMP task.
   Tasks are created until the sub-problem is smaller than THRESHOLD,
   at which point recursion falls back to sequential (this bounds the
   number of tasks and controls overhead). */
void merge_sort_par(int *arr, int *temp, int left, int right) {
    if (left >= right) return;
    if (right - left < THRESHOLD) {
        merge_sort_seq(arr, temp, left, right);
        return;
    }
    int mid = left + (right - left) / 2;

    #pragma omp task shared(arr, temp)
    merge_sort_par(arr, temp, left, mid);

    #pragma omp task shared(arr, temp)
    merge_sort_par(arr, temp, mid + 1, right);

    #pragma omp taskwait   /* wait for both halves before merging */
    merge(arr, temp, left, mid, right);
}

int main(int argc, char *argv[]) {
    int n       = (argc > 1) ? atoi(argv[1]) : 1000000;
    int threads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    omp_set_num_threads(threads);

    int *arr_seq = malloc((size_t)n * sizeof(int));
    int *arr_par = malloc((size_t)n * sizeof(int));
    int *temp    = malloc((size_t)n * sizeof(int));

    srand(42);
    for (int i = 0; i < n; i++) {
        int v = rand();
        arr_seq[i] = v;
        arr_par[i] = v;
    }

    double t0 = omp_get_wtime();
    merge_sort_seq(arr_seq, temp, 0, n - 1);
    double t1 = omp_get_wtime();
    double seq_time = t1 - t0;

    double t2 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        merge_sort_par(arr_par, temp, 0, n - 1);
    }
    double t3 = omp_get_wtime();
    double par_time = t3 - t2;

    int correct = 1;
    for (int i = 0; i < n; i++)
        if (arr_seq[i] != arr_par[i]) { correct = 0; break; }

    printf("========================================\n");
    printf("Recursive Decomposition: Merge Sort via OpenMP Tasks\n");
    printf("Array size: %d | Threads: %d | Task threshold: %d\n", n, threads, THRESHOLD);
    printf("Sequential time : %f s\n", seq_time);
    printf("Parallel time   : %f s\n", par_time);
    printf("Speedup         : %f\n", seq_time / par_time);
    printf("Correctness     : %s\n", correct ? "PASSED" : "FAILED");
    printf("========================================\n");

    free(arr_seq); free(arr_par); free(temp);
    return 0;
}