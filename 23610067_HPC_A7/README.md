# HPC Assignment 7 — MPI Collective Communication & Matrix Parallelism

**PRN:** 23610067 | **Course:** High Performance Computing

---

## Theory

### Collective Communication

Unlike point-to-point (one sender, one receiver), **collective operations** involve all processes in a communicator simultaneously. Every process must call the collective — it acts as an implicit barrier.

| Function | Pattern | Description |
|----------|---------|-------------|
| `MPI_Bcast(buf, n, type, root, comm)` | 1 → all | Root sends to every other process |
| `MPI_Scatter(send, sc, t, recv, rc, t, root, comm)` | 1 → N chunks | Root distributes equal chunks to each process |
| `MPI_Gather(send, sc, t, recv, rc, t, root, comm)` | N chunks → 1 | Root collects one chunk from every process |
| `MPI_Scatterv(...)` | 1 → N unequal | Like Scatter but each process gets a different count |
| `MPI_Gatherv(...)` | N unequal → 1 | Like Gather but each process sends a different count |
| `MPI_Reduce(send, recv, n, t, op, root, comm)` | N → 1 (reduced) | Combines values with an operation (SUM, MAX, MIN, …) |
| `MPI_Allreduce(...)` | N → all (reduced) | All processes receive the reduced result |

### Why `MPI_Scatterv` Instead of `MPI_Scatter`?

`MPI_Scatter` requires equal-sized chunks. When `N % size != 0` (the matrix rows don't divide evenly), `MPI_Scatterv` is needed: it accepts separate `sendcounts[]` and `displs[]` arrays so each rank can receive a different number of rows.

### MPI Matrix-Vector Multiplication

1. **Root** allocates full matrix A (N×N) and vector X.
2. `MPI_Bcast(X, ...)` — all processes receive X.
3. `MPI_Scatterv(A, ...)` — each process receives its row-block of A.
4. Each process computes its partial result `Y_local = A_local × X`.
5. `MPI_Gatherv(Y_local, ...)` — root collects all partial Y blocks.

### Block vs Column Matrix-Matrix Multiply

| Strategy | Description | Advantages |
|----------|-------------|-----------|
| **Row-wise** | Each process owns rows of A | Simple; good cache locality for A |
| **Column-wise** | Each process owns columns of B | Columns distribute C computation |
| **Block (2D)** | 2D grid of processes; each owns a sub-block | Better load balance for square matrices; typical in ScaLAPACK |

### Speedup & Efficiency (MPI)

```
Speedup   = T_seq / T_mpi
Efficiency = Speedup / num_processes
```

Communication overhead (Bcast, Scatter, Gather) grows with number of processes, so efficiency decreases at large process counts — the plots in Task4 illustrate this.

---

## Tasks

### t1 — Broadcast (`t1_broadcast.c`)
Process 0 initialises an integer value and broadcasts it to all processes with `MPI_Bcast`. Every process prints the received value.

**Functions:**
- `main(argc, argv)` — Init → rank 0 sets `value=100` → `MPI_Bcast` → all print → Finalize.

### t2 — Scatter, Gather & Sum (`t2_scatter-gather-sum.c`)
Root scatters a 16-element array evenly across 4 processes (4 elements each). Each process computes its local sum. Sums are gathered back and the global sum is printed.

**Functions:**
- `main(argc, argv)` — `MPI_Scatter` → local sum loop → `MPI_Gather` → root sums gathered partial sums.

### t3 — MPI Reduce (`t3_mpi-reduce.c`)
After scatter, each process computes local sum, max, and min. `MPI_Reduce` with `MPI_SUM`, `MPI_MAX`, `MPI_MIN` combines results at root — cleaner than manual gather+loop.

**Functions:**
- `main(argc, argv)` — scatter → local stats → 3 × `MPI_Reduce` (SUM/MAX/MIN) → root prints globals.

### t4 — Matrix-Vector MPI vs Sequential (`t4_matrix-vector-mpi.c`, `t4_matrix-vector-seq.c`)
Parallel MPI matrix-vector multiply (y = Ax) using `MPI_Bcast` + `MPI_Scatterv` + `MPI_Gatherv`. Compared against a sequential baseline. Breakdown of broadcast/scatter/compute/gather times is printed.

**Functions (MPI):**
- `main(argc, argv)` — builds count/offset arrays for Scatterv/Gatherv → Bcast(X) → Scatterv(A rows) → local dot products → Gatherv(Y) → time breakdown via `MPI_Reduce(MPI_MAX)`.

**Files:**
- `t4_performance-analysis.py` — reads `performance_results.csv` and generates speedup/efficiency/time plots.

### t5 — Matrix-Matrix MPI vs Sequential (`t5_matrix-matrix-mpi.c`, `t5_matrix-matrix-seq.c`)
Parallel matrix-matrix multiply (C = A × B). Root scatters rows of A; all processes broadcast B; each process computes its block of C; results are gathered.

**Functions (MPI):**
- `main(argc, argv)` — Bcast(B) → Scatter(A rows) → local C_block = A_local × B → Gather(C blocks) → timing and checksum.

### t6 — Block vs Column Matrix Multiply (`t6_block-matrix-mul.c`, `t6_column-matrix-mul.c`)
Compares two decomposition strategies for matrix multiplication on a 4-process 2×2 grid.

- **Block:** `t6_block-matrix-mul.c` — each process owns a sub-block of both A and B; uses send/recv to exchange B-blocks.
- **Column:** `t6_column-matrix-mul.c` — each process owns columns of B; computes its columns of C independently.

### t7 — MPI_Scatterv Uneven Distribution (`t7_scatter-v.c`)
Demonstrates `MPI_Scatterv` when rows don't divide evenly among 3 processes (10 rows → P0: 4, P1: 3, P2: 3). Each process computes a local sum of its rows and gathers results.

**Functions:**
- `main(argc, argv)` — builds `rows_per_process[]` and `offsets[]` arrays → `MPI_Scatterv` → local row sums → `MPI_Gatherv` → root prints.

---

## Prerequisites

```bash
mpicc --version   # OpenMPI or MPICH
mpirun --version
python3 matplotlib pandas  # for t4_performance-analysis.py
```

---

## Files

| File | Description |
|------|-------------|
| `code/t1_broadcast.c` | MPI_Bcast root → all |
| `code/t2_scatter-gather-sum.c` | Scatter / Gather with local partial sums |
| `code/t3_mpi-reduce.c` | MPI_Reduce SUM/MAX/MIN |
| `code/t4_matrix-vector-mpi.c` | Parallel MV multiply with Scatterv/Gatherv |
| `code/t4_matrix-vector-seq.c` | Sequential MV multiply baseline |
| `code/t4_performance-analysis.py` | Plots speedup/efficiency from CSV |
| `code/t5_matrix-matrix-mpi.c` | Parallel MM multiply |
| `code/t5_matrix-matrix-seq.c` | Sequential MM multiply baseline |
| `code/t6_block-matrix-mul.c` | Block (2D grid) decomposition |
| `code/t6_column-matrix-mul.c` | Column decomposition |
| `code/t7_scatter-v.c` | Scatterv with uneven row counts |
| `23610067_HPC_A7.pdf` | Submitted report |
