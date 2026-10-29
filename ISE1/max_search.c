/*
 * Parallel Maximum Search using Recursive Decomposition
 * HPC ISE-I Assignment
 *
 * Compile:
 *   gcc -O2 -fopenmp -o max_search max_search.c
 *
 * Run:
 *   ./max_search
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <limits.h>

#define CUTOFF 1000   /* fall back to sequential below this size */

/* ── Utility ─────────────────────────────────────────────── */

/* Fill array with random integers */
void fill_random(int *arr, int n) {
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000000;
}

/* Wall-clock timer using omp_get_wtime() for portability */
static inline double now(void) { return omp_get_wtime(); }

/* ── Task 1 : Sequential linear scan ─────────────────────── */

int max_sequential(const int *arr, int n) {
    int m = arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > m) m = arr[i];
    return m;
}

/* ── Task 2 : Recursive decomposition (single-threaded) ──── */

int max_recursive(const int *arr, int lo, int hi) {
    if (hi - lo <= 1)
        return arr[lo];

    /* Base-case cutoff: fall back to sequential scan */
    if (hi - lo <= CUTOFF)
        return max_sequential(arr + lo, hi - lo);

    int mid  = lo + (hi - lo) / 2;
    int left = max_recursive(arr, lo,  mid);
    int right= max_recursive(arr, mid, hi);
    return left > right ? left : right;
}

/* ── Task 3a : OpenMP tasks (recursive + omp task) ──────── */

int max_omp_task(const int *arr, int lo, int hi) {
    if (hi - lo <= CUTOFF)
        return max_sequential(arr + lo, hi - lo);

    int mid = lo + (hi - lo) / 2;
    int left, right;

    #pragma omp task shared(left)
    left  = max_omp_task(arr, lo,  mid);

    #pragma omp task shared(right)
    right = max_omp_task(arr, mid, hi);

    #pragma omp taskwait
    return left > right ? left : right;
}

/* ── Task 3b : OpenMP parallel-for with reduction ───────── */

int max_omp_reduction(const int *arr, int n) {
    int m = arr[0];
    #pragma omp parallel for reduction(max : m)
    for (int i = 1; i < n; i++)
        if (arr[i] > m) m = arr[i];
    return m;
}

/* ── Benchmark helper ────────────────────────────────────── */

typedef struct { double t; int val; } Result;

Result bench_seq(const int *arr, int n) {
    double t0 = now();
    int v = max_sequential(arr, n);
    return (Result){ now() - t0, v };
}

Result bench_rec(const int *arr, int n) {
    double t0 = now();
    int v = max_recursive(arr, 0, n);
    return (Result){ now() - t0, v };
}

Result bench_task(const int *arr, int n) {
    double t0 = now();
    int v;
    #pragma omp parallel
    #pragma omp single
    v = max_omp_task(arr, 0, n);
    return (Result){ now() - t0, v };
}

Result bench_red(const int *arr, int n) {
    double t0 = now();
    int v = max_omp_reduction(arr, n);
    return (Result){ now() - t0, v };
}

/* ── Main ────────────────────────────────────────────────── */

int main(void) {
    srand(42);

    /* --- Experiment 1 : vary array size, fix threads to max --- */
    int sizes[] = { 10000, 50000, 100000, 500000, 1000000, 5000000 };
    int ns = sizeof(sizes) / sizeof(sizes[0]);
    int max_threads = omp_get_max_threads();

    printf("=== Experiment 1 : Execution time vs Array Size (threads = %d) ===\n", max_threads);
    printf("%-12s %-12s %-12s %-12s %-12s\n",
           "Size", "Seq(s)", "Rec(s)", "OMP_Task(s)", "OMP_Red(s)");

    omp_set_num_threads(max_threads);
    for (int i = 0; i < ns; i++) {
        int n = sizes[i];
        int *arr = malloc(n * sizeof(int));
        fill_random(arr, n);

        Result rs  = bench_seq (arr, n);
        Result rr  = bench_rec (arr, n);
        Result rt  = bench_task(arr, n);
        Result rrd = bench_red (arr, n);

        printf("%-12d %-12.6f %-12.6f %-12.6f %-12.6f\n",
               n, rs.t, rr.t, rt.t, rrd.t);
        free(arr);
    }

    /* --- Experiment 2 : vary threads, fix size to 1M --- */
    int N = 1000000;
    int *arr = malloc(N * sizeof(int));
    fill_random(arr, N);
    Result rs_base = bench_seq(arr, N);

    int thread_counts[] = { 1, 2, 4, 8, 16 };
    int nt = sizeof(thread_counts) / sizeof(thread_counts[0]);

    printf("\n=== Experiment 2 : Speedup vs Threads (n = %d) ===\n", N);
    printf("%-8s %-12s %-8s %-8s | %-12s %-8s %-8s\n",
           "Threads", "Task(s)", "S_task", "E_task",
           "Red(s)", "S_red", "E_red");

    for (int i = 0; i < nt; i++) {
        int p = thread_counts[i];
        omp_set_num_threads(p);
        Result rt  = bench_task(arr, N);
        Result rrd = bench_red (arr, N);
        double st  = rs_base.t / rt.t;
        double sr  = rs_base.t / rrd.t;
        printf("%-8d %-12.6f %-8.3f %-8.3f | %-12.6f %-8.3f %-8.3f\n",
               p, rt.t, st, st/p, rrd.t, sr, sr/p);
    }

    free(arr);
    return 0;
}
