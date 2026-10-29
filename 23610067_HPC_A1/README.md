# HPC Assignment 1 — Environment Setup & OpenMP Basics

**PRN:** 23610067 | **Course:** High Performance Computing

---

## Theory

### What is OpenMP?

OpenMP (Open Multi-Processing) is a shared-memory parallel programming API for C, C++, and Fortran. It works via **compiler directives** (`#pragma omp ...`) that tell the compiler to spawn and coordinate threads. The underlying model is **fork-join**: the master thread forks a team of worker threads at a parallel region, and they join (synchronize) at the end.

Key concepts:
- **Thread:** the smallest unit of execution; all threads in a team share the same memory space.
- `omp_get_thread_num()` — returns the calling thread's ID (0 = master).
- `omp_get_num_threads()` — returns total threads active in the current parallel region.
- `#pragma omp parallel` — spawns a team of threads; every thread runs the enclosed block.

### FLOPS (Floating-Point Operations Per Second)

Theoretical peak FLOPS measures how fast a CPU *can* compute, assuming perfect utilization:

```
Peak FLOPS = Cores × Frequency (Hz) × SIMD_width (FP32 values) × FMA_factor
```

- **SIMD (Single Instruction Multiple Data):** one instruction operates on multiple values simultaneously (SSE=4, AVX/AVX2=8, AVX-512=16 FP32 values).
- **FMA (Fused Multiply-Add):** one instruction does `a*b + c`, counting as **2 FLOPs**.

---

## Tasks

### t1 — GCC Hello World (`t1_gcc-hello.c`)
Verifies GCC is installed and OpenMP flag (`-fopenmp`) is accepted. Prints `Hello World` — the simplest possible C program.

**Functions:**
- `main()` — prints a greeting and exits.

### t2 — OpenMP Hello World (`t2_openmp-hello.c`)
Verifies OpenMP is functional. Each thread in the parallel team prints its own ID.

**Functions:**
- `main()` — opens a `#pragma omp parallel` region; every thread executes `printf`.

### t3 — Parallel Square Computation (`t3_parallel-square.c`)
Distributes 1000 squaring operations across threads using `#pragma omp parallel for`. Demonstrates loop-level parallelism — iterations are divided among threads automatically.

**Functions:**
- `main()` — parallel for loop; each iteration computes `i*i` and prints the thread ID that handled it.

### t4 — Theoretical Peak FLOPS Calculator (`t4_flops-calculator.py`)
Reads CPU specs via `lscpu` and `/proc/cpuinfo`, detects SIMD capabilities, and calculates theoretical peak FP32 performance.

**Functions:**
- `get_lscpu()` — runs `lscpu` and returns the output string.
- `get_value(output, key)` — extracts a field value by regex.
- `main` script body — detects cores, frequency, SIMD type, computes peak FLOPS in GFLOPs and TFLOPs.

---

## Prerequisites

```bash
gcc --version          # needs gcc with OpenMP support
python3 --version      # for the FLOPS calculator (Linux only)
```

---

## Files

| File | Description |
|------|-------------|
| `code/t1_gcc-hello.c` | GCC installation check |
| `code/t2_openmp-hello.c` | OpenMP thread spawning |
| `code/t3_parallel-square.c` | Parallel for loop — squaring |
| `code/t4_flops-calculator.py` | Theoretical peak FLOPS (Linux) |
| `23610067_HPC_A1.pdf` | Submitted report |
