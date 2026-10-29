# HPC Assignment 5 — OpenMP Sections & Tasks

**PRN:** 23610067 | **Course:** High Performance Computing

---

## Theory

### `#pragma omp sections`

Sections express **functional parallelism** — different blocks of code (sections) run concurrently on different threads. Each `section` is a chunk of independent work.

```c
#pragma omp parallel sections
{
    #pragma omp section
    task_A();          // runs on thread 0 (or whichever picks it up)
    #pragma omp section
    task_B();          // runs concurrently on another thread
}
```

Use sections when you have a small number of independent tasks that are each too coarse for a loop.  
Use `parallel for` when you have many similar iterations to distribute.

### `#pragma omp task` + `taskwait`

Tasks decouple work creation from execution. The runtime queues tasks and schedules them on idle threads. `taskwait` blocks the current task until all child tasks finish.

```c
#pragma omp task         // creates a new schedulable unit
    compute_something();
#pragma omp taskwait     // wait for all tasks created in this scope
```

Key properties:
- Recursive-friendly (merge sort, tree traversals).
- Granularity matters: spawning tiny tasks has overhead → use a threshold to fall back to sequential.

### `depend` Clauses (Task Dependency Graph)

`#pragma omp task depend(in: x) depend(out: y)` expresses data flow between tasks. The runtime builds a dependency DAG and only executes a task when all its `in` dependencies are satisfied.

```
A → {B, C} → D
```
B and C can run in parallel after A finishes. D only runs after both B and C finish.

### Pipeline Parallelism

A data pipeline applies a sequence of transformations to data. With tasks, each chunk of data can move through the pipeline stages independently. While one chunk is in stage 2, the next is in stage 1.

---

## Tasks

### t1 — Array Operations via Sections (`t1_array-ops-sections.c`)
Computes Sum, Max, Min, and Average of an array using `#pragma omp sections` — each stat is a separate section.

**Functions:**
- `main(argc, argv)` — generates array, opens a `parallel sections` region with 4 `section` blocks computing sum, max, min, count-of-evens concurrently.

### t2 — parallel for vs sections (`t2_parallel-section-compare.c`)
Benchmarks the same aggregate statistics (Sum, Max, Min, even count) computed two ways: with `parallel for` (many iterations each thread handles) vs `sections` (one stat per thread).

**Functions:**
- `main(argc, argv)` — times both approaches; shows that sections give lower latency for small N (function call overhead dominates) while `parallel for` wins for very large N.

### t3 — Data Processing Pipeline (`t3_data-pipeline.c`)
Splits an array into chunks and sends each chunk through a 3-stage pipeline (normalize → transform → stats). Each chunk is a task; the full pipeline runs on it independently.

**Functions:**
- `normalize_chunk(data, n)` — scales values to [0,1].
- `transform_chunk(data, n)` — applies sqrt/log transform.
- `stats_chunk(data, n, sum, max)` — computes stats after transform.
- `process_chunk(data, n)` — calls all three stages on a chunk.
- `main(argc, argv)` — divides array into chunks, spawns one `#pragma omp task` per chunk inside a `parallel single` region.

### t4 — Matrix Operations via Sections (`t4_matrix-ops-sections.c`)
Uses sections to perform four matrix operations (add, subtract, transpose-A, transpose-B) concurrently on separate output matrices.

**Functions:**
- `main(argc, argv)` — allocates matrices, opens `parallel sections`, each section performs a distinct matrix operation.

### t5 — Merge Sort with Tasks (`t5_merge-sort-tasks.c`)
Task-based recursive merge sort with configurable threshold and thread count.

**Functions:**
- `merge(arr, temp, l, m, r)` — merges two sorted sub-arrays.
- `merge_sort_seq(arr, temp, l, r)` — sequential recursive sort.
- `merge_sort_tasks(arr, temp, l, r, threshold)` — if size > threshold, spawn two tasks for left/right halves, taskwait, merge. Else call sequential.
- `main(argc, argv)` — benchmarks sequential vs task sort, checks correctness.

### t6 — Task Dependency Graph (`t6_task-dependency.c`)
Demonstrates `depend(in:)` / `depend(out:)` clauses to encode `A → {B, C} → D` execution order. Tasks B and C run in parallel after A; D waits for both.

**Functions:**
- `task_A/B/C/D()` — placeholder functions that simulate work (sleep/compute).
- `main()` — single parallel region with `#pragma omp single`; four tasks with depend clauses forming a diamond dependency graph.

---

## Prerequisites

```bash
gcc with OpenMP, math library (-lm needed for t3)
```

---

## Files

| File | Description |
|------|-------------|
| `code/t1_array-ops-sections.c` | Sections: sum/max/min/count concurrently |
| `code/t2_parallel-section-compare.c` | parallel for vs sections benchmark |
| `code/t3_data-pipeline.c` | Task-based 3-stage data pipeline |
| `code/t4_matrix-ops-sections.c` | 4 matrix ops in parallel sections |
| `code/t5_merge-sort-tasks.c` | Task merge sort with threshold |
| `code/t6_task-dependency.c` | Task DAG with depend clauses |
| `23610067_HPC_A5.pdf` | Submitted report |
