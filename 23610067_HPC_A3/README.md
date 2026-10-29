# HPC Assignment 3 — OpenMP Reduction & Parallel Scan

**PRN:** 23610067 | **Course:** High Performance Computing

---

## Theory

### Race Conditions & Synchronization

When multiple threads write to the same variable, the outcome depends on execution order — a **race condition**. Three ways to avoid this for accumulation:

| Method | Mechanism | Performance |
|--------|-----------|-------------|
| `critical` | Only one thread at a time enters the critical section | Slowest — serializes all updates |
| `atomic` | Hardware-level single instruction update | Faster than critical, but still serialized per update |
| `reduction` | Each thread keeps a private copy, combined at the end | Fastest — fully parallel, O(log T) combine step |

### Dot Product

Dot product of vectors A and B: `sum = Σ A[i] * B[i]`. Trivially parallelizable — all multiplications are independent, accumulation requires reduction.

### Prefix Sum (Scan)

Output `out[i] = A[0] + A[1] + ... + A[i]`. Sequential by nature (each element depends on the previous), but can be parallelized with a **two-pass block scan**:

1. **Local scan:** each thread computes prefix sum over its own chunk.
2. **Offset computation:** sequentially compute the cumulative sum of each chunk's total.
3. **Apply offsets:** each thread adds its chunk's offset to every element in its chunk.

This yields O(N/T) parallel work instead of O(N) sequential.

---

## Tasks

### t1 — Array Sum: critical vs atomic vs reduction (`t1_array-sum-reduction.c`)
Compares three synchronization strategies for summing an array. Shows that `reduction` is fastest; `critical` can be slower than sequential due to lock contention.

**Functions:**
- `sum_sequential(A, N)` — simple loop.
- `sum_critical(A, N)` — parallel for + `#pragma omp critical` on the update.
- `sum_reduction(A, N)` — `#pragma omp parallel for reduction(+:sum)`.
- `main(argc, argv)` — benchmarks all three, checks correctness, prints speedups.

### t2 — Dot Product (`t2_dot-product.c`)
Sequential vs parallel dot product using reduction. Demonstrates that embarrassingly parallel operations scale well.

**Functions:**
- `dot_sequential(A, B, N)` — `Σ A[i]*B[i]` single-threaded.
- `dot_parallel(A, B, N)` — same with `reduction(+:sum)`.
- `main(argc, argv)` — prints result, correctness, speedup, efficiency.

### t3 — Prefix Sum (`t3_prefix-sum.c`)
Two-pass block-scan parallel prefix sum vs sequential scan.

**Functions:**
- `prefix_sequential(A, out, N)` — `out[i] = out[i-1] + A[i]`.
- `prefix_parallel(A, out, N, num_threads)` — three-phase: local scan → offset scan (single) → apply offsets. Uses `#pragma omp barrier` to synchronize phases.
- `main(argc, argv)` — correctness check (all N elements must match), speedup.

### t4 — Performance Comparison (`t4_perf-compare.c`)
Runs both dot product and prefix sum at a given thread count in one binary. Good for filling a comparison table across different thread counts.

**Functions:**  
Contains `dot_sequential`, `dot_parallel`, `prefix_sequential`, `prefix_parallel` (same as above) and a `main` that runs all four and prints side-by-side speedup/efficiency.

### c1 — Dot Product: critical vs atomic vs reduction (`c1_dot-product-sync-methods.c`)
Challenge task — directly compares all three synchronization primitives for dot product, showing the relative cost of each.

**Functions:**
- `dot_critical(A, B, N)` — per-element critical section.
- `dot_atomic(A, B, N)` — `#pragma omp atomic` update.
- `dot_reduction(A, B, N)` — `reduction(+:sum)`.
- `main(argc, argv)` — prints all three times and their ratio relative to reduction.

---

## Prerequisites

```bash
gcc with OpenMP
```

---

## Files

| File | Description |
|------|-------------|
| `code/t1_array-sum-reduction.c` | critical vs atomic vs reduction |
| `code/t2_dot-product.c` | Parallel dot product |
| `code/t3_prefix-sum.c` | Two-pass parallel prefix sum |
| `code/t4_perf-compare.c` | Combined benchmark runner |
| `code/c1_dot-product-sync-methods.c` | Challenge: sync method comparison |
| `23610067_HPC_A3.pdf` | Submitted report |
