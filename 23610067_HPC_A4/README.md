# HPC Assignment 4 — Data Decomposition Strategies

**PRN:** 23610067 | **Course:** High Performance Computing

---

## Theory

### Data Decomposition

Data decomposition is how we split a problem's data across threads or processes. The choice of decomposition affects load balance, memory access patterns (cache locality), and communication overhead.

#### Input Data Decomposition
Each processor owns a subset of the **input** data. For matrix-vector multiplication `y = Ax`:
- Split matrix A row-wise → each thread owns contiguous rows of A and the full vector x.
- All threads share the same input vector (read-only, no conflict).

#### Output Data Decomposition
Each processor owns a subset of the **output** data. For matrix-matrix multiplication `C = AB`:
- Split the output matrix C into blocks → each thread computes a different part of C.
- Reads from A and B can overlap safely (read-only from multiple threads).

### Decomposition Strategies for Matrix Multiplication

| Strategy | What's split | Cache behavior | Notes |
|----------|--------------|---------------|-------|
| **Row-wise** | Outer loop over rows (i) | Good — A rows are contiguous | Default in most parallel matmul |
| **Column-wise** | Outer loop over columns (j) | Poor — B columns are non-contiguous | Worse cache performance |
| **Block (2D tiling)** | C divided into tiles (block_size × block_size) | Best — both A and B reuse in cache | Best for large matrices |

### OpenMP Tasks

`#pragma omp task` creates an asynchronous unit of work that the runtime schedules on an available thread. Combined with `#pragma omp taskwait`, you can express recursive parallelism (e.g., merge sort) that `parallel for` cannot handle.

```c
#pragma omp parallel
#pragma omp single
{
    #pragma omp task  // spawn task for left half
    sort_left(...);
    #pragma omp task  // spawn task for right half
    sort_right(...);
    #pragma omp taskwait  // wait for both tasks
    merge(...);
}
```

**Granularity threshold:** spawning a task has overhead. Below a threshold (e.g., 10000 elements), fall back to sequential sort to avoid task overhead dominating.

---

## Tasks

### t1 — Matrix-Vector Multiplication: Input Decomposition (`t1_matrix-vector-mul.c`)
Demonstrates **input data decomposition**: the matrix A is split row-wise across threads. Each thread owns contiguous rows and reads the full vector x.

**Functions:**
- `generate_matrix(A, n)` — random fill seeded with 42.
- `generate_vector(v, n)` — random fill.
- `main(argc, argv)` — sequential `y = Ax`, then parallel with `#pragma omp parallel for schedule(static)`. Correctness check (tolerance 1e-6). Prints speedup, efficiency.

### t2 — Matrix-Matrix Multiplication: Output Decomposition (`t2_matrix-matrix-mul.c`)
Demonstrates **output data decomposition**: the output matrix C is split across threads. Uses `collapse(2)` to merge the i and j loops, giving T more parallel chunks than row-only decomposition.

**Functions:**
- `matmul_seq(A, B, C, n)` — sequential triple-loop.
- `matmul_par(A, B, C, n, sched_type)` — parallel with `collapse(2)`, dispatches to `static` (0) or `dynamic` (1) schedule.
- `main(argc, argv)` — compares sequential vs static vs dynamic. Prints speedup and efficiency for each.

### t3 — Row-wise vs Column-wise Decomposition (`t3_decomp-row-col.c`)
Compares only row-wise vs column-wise parallelization strategies to illustrate cache effects. Column-wise typically underperforms due to non-contiguous memory access.

**Functions:**
- `generate_matrix(M, n)` — sequential fill.
- `matmul_row(A, B, C, n)` — `#pragma omp parallel for schedule(static)` on outer i loop.
- *(column-wise defined further in file)* — `#pragma omp parallel for` on outer j loop.

### t4 — All Decomposition Strategies Compared (`t4_decomp-all-strategies.c`)
Full benchmark: sequential baseline, row-wise, column-wise, and 2D block (tile) decomposition. The block decomposition divides C into `block_size × block_size` tiles assigned to threads.

**Functions:**
- `matmul_seq(A, B, C, n)` — sequential baseline.
- `matmul_row(A, B, C, n)` — row-wise parallel.
- `matmul_col(A, B, C, n)` — column-wise parallel.
- `matmul_block(A, B, C, n, bs)` — 2D block decomposition with tile size `bs`.
- `main(argc, argv)` — benchmarks all four, prints speedup table.

### c3 — Recursive Merge Sort with OpenMP Tasks (`c3_recursive-merge-sort.c`)
Challenge: implements merge sort using `#pragma omp task` for recursive parallelism. Uses a threshold (default 10000) below which tasks fall back to sequential sort.

**Functions:**
- `merge(arr, temp, left, mid, right)` — standard merge of two sorted halves.
- `merge_sort_seq(arr, temp, left, right)` — recursive sequential merge sort.
- `merge_sort_tasks(arr, temp, left, right)` — if sub-array > THRESHOLD, spawn two tasks for halves, then `taskwait` and merge; else call sequential.
- `main(argc, argv)` — generates random array, times sequential vs task-based parallel sort. Verifies sortedness.

---

## Prerequisites

```bash
gcc with OpenMP
```

---

## Files

| File | Description |
|------|-------------|
| `code/t1_matrix-vector-mul.c` | Input decomposition: row-wise matrix-vector |
| `code/t2_matrix-matrix-mul.c` | Output decomposition: matmul static vs dynamic |
| `code/t3_decomp-row-col.c` | Row vs column decomposition |
| `code/t4_decomp-all-strategies.c` | Full comparison: row, col, 2D block |
| `code/c3_recursive-merge-sort.c` | Challenge: task-based merge sort |
| `23610067_HPC_A4.pdf` | Submitted report |
