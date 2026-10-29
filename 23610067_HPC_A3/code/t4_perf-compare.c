// Section B - Task 2: Performance Analysis
// Runs Dot Product and Prefix Sum at a given thread count, reports both

// gcc -fopenmp Task4/compare.c -o compare

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double dot_sequential(double *A, double *B, long N) {
    double sum = 0.0;
    for (long i = 0; i < N; i++) sum += A[i]*B[i];
    return sum;
}
double dot_parallel(double *A, double *B, long N) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < N; i++) sum += A[i]*B[i];
    return sum;
}

void prefix_sequential(double *A, double *out, long N) {
    out[0] = A[0];
    for (long i = 1; i < N; i++) out[i] = out[i-1] + A[i];
}
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
            for (long i = start+1; i < end; i++) out[i] = out[i-1] + A[i];
            chunk_totals[tid] = out[end-1];
        } else chunk_totals[tid] = 0.0;
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
        for (long i = start; i < end; i++) out[i] += offset;
    }
    free(chunk_totals);
}

int main(int argc, char *argv[]) {
    long N = 10000000;
    int num_threads = 4;
    if (argc > 1) N = atol(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);
    omp_set_num_threads(num_threads);

    double *A = malloc(N*sizeof(double));
    double *B = malloc(N*sizeof(double));
    double *out = malloc(N*sizeof(double));
    for (long i = 0; i < N; i++) { A[i] = 1.0; B[i] = 2.0; }

    // volatile sink prevents the compiler from deleting the "unused" dot product loop
    volatile double sink;
    double t1 = omp_get_wtime();
    sink = dot_sequential(A, B, N);
    double t2 = omp_get_wtime();
    double dot_seq_time = t2 - t1;

    double t3 = omp_get_wtime();
    sink = dot_parallel(A, B, N);
    double t4 = omp_get_wtime();
    double dot_par_time = t4 - t3;
    (void)sink;

    double t5 = omp_get_wtime();
    prefix_sequential(A, out, N);
    double t6 = omp_get_wtime();
    double pre_seq_time = t6 - t5;

    double t7 = omp_get_wtime();
    prefix_parallel(A, out, N, num_threads);
    double t8 = omp_get_wtime();
    double pre_par_time = t8 - t7;

    double dot_speedup = dot_seq_time / dot_par_time;
    double pre_speedup = pre_seq_time / pre_par_time;

    printf("Threads=%d\n", num_threads);
    printf("Dot Product   : Time=%.6f s  Speedup=%.3f  Efficiency=%.3f\n",
           dot_par_time, dot_speedup, dot_speedup/num_threads);
    printf("Prefix Sum    : Time=%.6f s  Speedup=%.3f  Efficiency=%.3f\n",
           pre_par_time, pre_speedup, pre_speedup/num_threads);

    free(A); free(B); free(out);
    return 0;
}