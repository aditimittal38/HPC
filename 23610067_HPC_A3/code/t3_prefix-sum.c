// Section B - Task 1: Prefix Sum (Scan)
// Sequential vs Parallel (two-pass block scan) implementation

// gcc -fopenmp Task3/prefix-sum.c -o prefix-sum

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Sequential: inherently dependent, out[i] = out[i-1] + A[i]
void prefix_sequential(double *A, double *out, long N) {
    out[0] = A[0];
    for (long i = 1; i < N; i++)
        out[i] = out[i-1] + A[i];
}

// Parallel prefix sum: classic two-pass block-scan algorithm
// Pass 1: each thread computes a local prefix sum over its own chunk
// Pass 2: compute exclusive prefix sum of per-thread chunk totals (small, done by 1 thread)
// Pass 3: each thread adds its offset to every element in its chunk
void prefix_parallel(double *A, double *out, long N, int num_threads) {
    double *chunk_totals = malloc(num_threads * sizeof(double));

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        long chunk = (N + nthreads - 1) / nthreads;
        long start = tid * chunk;
        long end = start + chunk; if (end > N) end = N;

        if (start < end) {
            out[start] = A[start];
            for (long i = start + 1; i < end; i++)
                out[i] = out[i-1] + A[i];
            chunk_totals[tid] = out[end - 1];
        } else {
            chunk_totals[tid] = 0.0;
        }

        #pragma omp barrier

        #pragma omp single
        {
            double running = 0.0;
            for (int t = 0; t < nthreads; t++) {
                double val = chunk_totals[t];
                chunk_totals[t] = running;
                running += val;
            }
        }

        #pragma omp barrier

        double offset = chunk_totals[tid];
        for (long i = start; i < end; i++)
            out[i] += offset;
    }

    free(chunk_totals);
}

int main(int argc, char *argv[]) {
    long N = 10000000;
    int num_threads = 4;
    if (argc > 1) N = atol(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);

    double *A = malloc(N * sizeof(double));
    double *out_seq = malloc(N * sizeof(double));
    double *out_par = malloc(N * sizeof(double));
    for (long i = 0; i < N; i++) A[i] = 1.0;

    double t1 = omp_get_wtime();
    prefix_sequential(A, out_seq, N);
    double t2 = omp_get_wtime();
    double seq_time = t2 - t1;

    double t3 = omp_get_wtime();
    prefix_parallel(A, out_par, N, num_threads);
    double t4 = omp_get_wtime();
    double par_time = t4 - t3;

    int correct = 1;
    for (long i = 0; i < N; i++)
        if (out_seq[i] != out_par[i]) { correct = 0; break; }

    printf("N = %ld, Threads = %d\n", N, num_threads);
    printf("Sequential time = %.6f s\n", seq_time);
    printf("Parallel   time = %.6f s\n", par_time);
    printf("Speedup = %.3f\n", seq_time / par_time);
    printf("Correctness check: %s\n", correct ? "PASS (all N elements match)" : "FAIL");
    printf("Sample: out[0]=%.0f  out[N-1]=%.0f (expected %ld)\n", out_par[0], out_par[N-1], N);

    free(A); free(out_seq); free(out_par);
    return 0;
}