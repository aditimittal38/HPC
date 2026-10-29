# HPC Assignment 8 — Hybrid Parallelism: OpenMP + MPI

**PRN:** 23610067 | **Course:** High Performance Computing

---

## Theory

### Why Hybrid OpenMP + MPI?

| Model | Parallelism | Scope | Memory |
|-------|-------------|-------|--------|
| OpenMP | Thread-level | Single node | Shared |
| MPI | Process-level | Multi-node cluster | Distributed |
| **Hybrid** | Both | Multi-node | Shared within, distributed across |

On a modern cluster: each **node** has multiple cores → use OpenMP threads within a node. Multiple **nodes** → use MPI across nodes. The hybrid model avoids creating an MPI process per core (which saturates the network) while still using all available cores.

### Two Applications

#### 1. Matrix-Matrix Multiplication (C = A × B)

| Version | Strategy |
|---------|----------|
| Sequential | Triple nested loop, single core |
| OpenMP | `#pragma omp parallel for schedule(static)` on outer loop (row decomposition) |
| MPI | Bcast(B) → Scatter(rows of A) → local matmul → Gather(rows of C); pads N to divisible-by-size |

**OpenMP inner loop:** Each thread gets a chunk of rows (static schedule); inner j,k loops are sequential per thread — good cache locality.

**MPI strategy:** Every process needs all of B, so B is broadcast. A is row-scattered. Each process computes its block of C rows. N is padded to be divisible by `size` to allow equal `MPI_Scatter`.

#### 2. 2D Convolution

Given an N×N input image and a K×K kernel, compute the (N−K+1)×(N−K+1) output where each output pixel is the dot product of the kernel with the corresponding input patch.

| Version | Strategy |
|---------|----------|
| Sequential | 4 nested loops (i, j, u, v) |
| OpenMP | `#pragma omp parallel for collapse(2)` over (i, j) outer loops; schedule variants: static/dynamic/guided |
| MPI | `MPI_Bcast` the input → each process computes its row-slice of output (with halo rows for kernel overlap) → `MPI_Gather` output rows |

**Halo rows:** Because each output row `i` needs input rows `i` to `i+K-1`, the last process needs K−1 extra "halo" rows below its partition — handled by computing the actual needed `input_rows` per process.

**collapse(2):** Flattens i and j loops into one large iteration space for better thread distribution when M is small.

### Performance Metrics

```
Speedup   = T_seq / T_parallel
Efficiency = Speedup / num_threads_or_processes
```

`results.csv` stores timing data across thread/process counts and matrix sizes. `generate_graphs.py` plots speedup, efficiency, and time curves.

---

## Tasks

### t1 — Matrix-Matrix Multiply: Sequential, OpenMP, MPI

**`t1_matrix-matrix-seq.c`** — Baseline triple-loop matmul. Accepts `N` as CLI arg. Uses `clock_gettime` for accurate timing.

**`t1_matrix-matrix-omp.c`** — OpenMP version. CLI args: `N threads`. `#pragma omp parallel for schedule(static)` distributes rows. Reports thread count and wall-clock time via `omp_get_wtime()`.

**`t1_matrix-matrix-mpi.c`** — MPI version. CLI arg: `N`. Pads N to divisible by `size`. Bcast(B) → Scatter(A rows) → local matmul → Gather(C rows). Reports wall-clock time from all processes, takes max with `MPI_Reduce(MPI_MAX)`.

### t2 — 2D Convolution: Sequential, OpenMP, MPI

**`t2_conv-seq.c`** — Sequential 2D convolution with a 3×3 averaging kernel. CLI arg: `N`. Uses `clock_gettime` for timing. Output size = (N−2) × (N−2).

**`t2_conv-omp.c`** — OpenMP version. CLI args: `N threads sched` (0=static, 1=dynamic, 2=guided). Uses `collapse(2)` on the (i,j) output loops.

**`t2_conv-mpi.c`** — MPI version. CLI arg: `N`. `MPI_Bcast` the full input to all processes; each process computes its output row-slice (accounts for halo rows needed by kernel overlap). `MPI_Gather` collects output.

### Tools

**`generate_graphs.py`** — Reads `results.csv` and produces 8 performance plots in `Performance/` (threads/processes vs time, speedup, efficiency; size vs time for both matmul and convolution).

**`run_experiments.sh`** (in `code/`) — Runs all three versions of both tasks with varying thread/process counts and matrix sizes, writing timing data to `results.csv`.

---

## Prerequisites

```bash
gcc with OpenMP   # for t1/t2 omp versions
mpicc, mpirun     # for t1/t2 mpi versions
python3 matplotlib pandas  # for generate_graphs.py
```

---

## Files

| File | Description |
|------|-------------|
| `code/t1_matrix-matrix-seq.c` | Sequential matrix multiply |
| `code/t1_matrix-matrix-omp.c` | OpenMP parallel matrix multiply |
| `code/t1_matrix-matrix-mpi.c` | MPI distributed matrix multiply |
| `code/t2_conv-seq.c` | Sequential 2D convolution |
| `code/t2_conv-omp.c` | OpenMP parallel 2D convolution (schedule variants) |
| `code/t2_conv-mpi.c` | MPI distributed 2D convolution with halo |
| `code/generate_graphs.py` | Plot speedup/efficiency from results.csv |
| `code/run_experiments.sh` | Automated benchmark runner |
| `Performance/` | Pre-generated performance graphs |
| `results.csv` | Benchmark timing data |
| `COMMANDS_REFERENCE.md` | Quick compile/run command reference |
| `23610067_HPC_A8.pdf` | Submitted report |
