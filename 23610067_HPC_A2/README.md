# HPC Assignment 2 — OpenMP Parallel Loops & Scheduling

**PRN:** 23610067 | **Course:** High Performance Computing

---

## Theory

### `#pragma omp parallel for`

Distributes loop iterations across the thread team. The default schedule divides iterations into equal-sized chunks assigned to threads at compile time (static scheduling).

```c
#pragma omp parallel for
for (int i = 0; i < N; i++) { ... }  // each thread handles N/T iterations
```

### Reduction

When multiple threads accumulate into a shared variable, you must use `reduction` to avoid data races. Each thread gets a private copy, computes locally, then the runtime combines them at the end.

```c
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < N; i++) sum += A[i];
```

### Scheduling Strategies

| Strategy | Behaviour | Best for |
|----------|-----------|---------|
| `static` | Pre-divides iterations into equal chunks at compile time | Uniform work per iteration |
| `dynamic` | Threads pick the next chunk from a pool when they finish | Irregular / imbalanced iterations |
| `guided` | Starts with large chunks, shrinks as work runs out | Mix of light and heavy iterations |

### Loop Collapse (`collapse(n)`)

Merges n nested loops into a single iteration space before distributing. Turns `M` chunks (outer loop) into `M×P` chunks — much more parallelism available.

```c
#pragma omp parallel for collapse(2)
for (int i = 0; i < M; i++)
    for (int j = 0; j < P; j++) { ... }  // M*P independent iterations
```

### Speedup & Efficiency

- **Speedup** = T_sequential / T_parallel
- **Efficiency** = Speedup / num_threads (ideal = 1.0)

---

## Tasks

### t1 — Parallel Matrix Addition (`t1_matrix-addition.c`)
Adds two 4×4 matrices in parallel. Demonstrates thread creation, identification, and parallel `for` distribution across rows.

**Functions:**
- `main(argc, argv)` — reads thread count from CLI, fills matrices A and B, uses `#pragma omp parallel` + `#pragma omp for` to compute `C = A + B`. Each thread announces which row it computed.

### t2 — Parallel Sum of N Numbers (`t2_sum-of-numbers.c`)
Sums a large array sequentially vs in parallel with `reduction`. Measures time with `omp_get_wtime()` and computes speedup.

**Functions:**
- `sum_sequential(A, N)` — single-threaded loop summation.
- `sum_parallel(A, N)` — `#pragma omp parallel for reduction(+:sum)` version.
- `main(argc, argv)` — benchmarks both, prints N, times, and speedup.

### t3 — Parallel Matrix Multiplication (`t3_matrix-multiplication.c`)
Computes C = A × B for square matrices of configurable size. Parallelizes the outer row loop.

**Functions:**
- `alloc_matrix(rows, cols)` — malloc wrapper.
- `fill_matrix(M, rows, cols)` — fills with pattern `(i % 100) / 10`.
- `matmul_sequential(A, B, C, M, N, P)` — triple-nested loop.
- `matmul_parallel(A, B, C, M, N, P)` — `#pragma omp parallel for` on outer loop.
- `main(argc, argv)` — runs both, prints time, speedup, efficiency.

### t4 — Scheduling Strategies (`t4_scheduling-strategies.c`)
Applies matrix multiplication with three different OMP schedules and compares their runtimes.

**Functions:**
- `matmul_scheduled(A, B, C, M, N, P, schedule_type)` — dispatches to static (0), dynamic (1), or guided (2) schedule variant.
- `main(argc, argv)` — benchmarks sequential baseline, then each schedule. Prints speedup per strategy.

### t5 — Loop Collapse Optimization (`t5_loop-collapse-optim.c`)
Shows how `collapse(2)` merges the i and j loops of matrix multiplication into a larger parallel iteration space for better load distribution.

**Functions:**
- `matmul_sequential(A, B, C, M, N, P)` — sequential baseline.
- `matmul_collapsed(A, B, C, M, N, P)` — uses `#pragma omp parallel for collapse(2) private(k)`.
- `main(argc, argv)` — compares both, prints speedup and efficiency.

---

## Prerequisites

```bash
gcc with OpenMP (gcc -fopenmp)
```

---

## Files

| File | Description |
|------|-------------|
| `code/t1_matrix-addition.c` | Parallel 4×4 matrix add |
| `code/t2_sum-of-numbers.c` | Reduction sum benchmark |
| `code/t3_matrix-multiplication.c` | Parallel matmul with speedup |
| `code/t4_scheduling-strategies.c` | Static vs dynamic vs guided |
| `code/t5_loop-collapse-optim.c` | Loop collapse optimization |
| `23610067_HPC_A2.pdf` | Submitted report |
