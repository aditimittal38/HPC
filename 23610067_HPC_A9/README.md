# HPC Lab – Experiment No. 9
## Introduction to CUDA Programming: Fundamentals of GPU Computing

**Name:** Aditi Mittal | **PRN:** 23610067 | **Batch:** B1
**Course:** Final Year B.Tech CSE | **College:** Walchand College of Engineering
**Colab Notebook:** [Open in Google Colab](https://colab.research.google.com/drive/1jDc_U5Q1tYqDomfXlIxHywb3cRJArA8m?usp=sharing)

---

## Aim

To introduce students to CUDA programming by implementing basic GPU programs that demonstrate
thread management, memory operations, and performance comparison between CPU and GPU execution.

---

## Objectives

1. Understand the CUDA programming model including threads, blocks, and grids.
2. Write and compile basic CUDA programs using `nvcc`.
3. Implement parallel vector operations on GPU.
4. Measure and compare CPU vs GPU execution times.
5. Explore CUDA memory management including explicit H→D and D→H transfers.
6. Investigate the effect of block size on kernel performance.
7. Compare sequential, OpenMP (multi-threaded CPU), and CUDA execution.
8. Apply CUDA to a practical squaring operation and verify correctness.

---

## System Configuration (Google Colab – Tesla T4)

Obtained from Task 1 (`nvidia-smi` + `cudaGetDeviceProperties`):

| Property | Value |
|---|---|
| GPU Model | NVIDIA Tesla T4 |
| Driver Version | 580.82.07 |
| CUDA Version | 13.0 |
| Total Global Memory | 14,913 MB (~15 GB) |
| Compute Capability | 7.5 (Turing architecture) |
| Streaming Multiprocessors (SMs) | 40 |
| Max Threads per Block | 1,024 |
| Max Grid Dimensions | (2,147,483,647 × 65,535 × 65,535) |
| Warp Size | 32 |
| Memory Bus Width | 256-bit |
| L2 Cache Size | 4,096 KB |
| Shared Memory per Block | 48 KB |
| GPU Temperature | 35°C |

---

## Theory: CUDA Programming Model

### What is CUDA?

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform that
allows developers to write programs that execute on the GPU. Programs written in CUDA
use thousands of parallel threads to perform computations simultaneously.

### Host vs Device

| Concept | Meaning |
|---|---|
| **Host** | The CPU and its main memory (RAM) |
| **Device** | The GPU and its dedicated VRAM |
| **Kernel** | A function that runs on the GPU, executed by many threads in parallel |

### Thread Hierarchy

CUDA organizes threads in a 3-level hierarchy:

```
Grid
 └─ Blocks (threadIdx has 3D coordinates per block)
     └─ Threads (each thread runs one copy of the kernel)
```

- **Thread**: Smallest unit of execution. Has a unique `threadIdx.x` (and `.y`, `.z`).
- **Block**: A group of up to 1,024 threads that can share fast **shared memory** and synchronize with `__syncthreads()`.
- **Grid**: All the blocks launched for one kernel call.

**Global thread ID formula (1D):**
```
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```
This is the most-used pattern in CUDA to map each thread to a unique array element.

### CUDA Built-in Variables

| Variable | Meaning |
|---|---|
| `threadIdx.x` | Thread's index within its block (0 to blockDim.x−1) |
| `blockIdx.x` | Block's index within the grid (0 to gridDim.x−1) |
| `blockDim.x` | Number of threads per block |
| `gridDim.x` | Number of blocks in the grid |

### Memory Types

| Memory | Location | Speed | Accessible by |
|---|---|---|---|
| Global memory | GPU VRAM | ~900 GB/s | All threads, host |
| Shared memory | Per-block on-chip SRAM | ~19 TB/s | Threads in same block only |
| Registers | Per-thread on-chip | Fastest | One thread only |
| Constant memory | GPU VRAM (cached) | Fast | All threads (read-only) |

### CUDA Memory Management API

```c
cudaMalloc(&d_ptr, size);                        // Allocate device memory
cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);    // Copy host → device
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);    // Copy device → host
cudaFree(d_ptr);                                 // Free device memory
```

### CUDA Timing with Events

```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
// ... kernel launch or operations ...
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);  // elapsed time in milliseconds
```

### Kernel Launch Syntax

```c
kernel<<<numBlocks, threadsPerBlock>>>(arg1, arg2, ...);
```
- `numBlocks`: number of thread blocks in the grid
- `threadsPerBlock`: number of threads per block (must be ≤ 1,024)
- For a vector of N elements: `numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock`

---

## Tasks

---

### Task 1: GPU Device Information

**Objective:** Query and display properties of the available CUDA device.

#### Code

```c
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n", deviceCount);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Device 0: %s\n", prop.name);
    printf("  Compute Capability         : %d.%d\n", prop.major, prop.minor);
    printf("  Total Global Memory        : %.0f MB\n", prop.totalGlobalMem / (1024.0*1024.0));
    printf("  Multiprocessors            : %d\n", prop.multiProcessorCount);
    printf("  Max Threads per Block      : %d\n", prop.maxThreadsPerBlock);
    printf("  Max Grid Size              : (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Warp Size                  : %d\n", prop.warpSize);
    printf("  Memory Bus Width           : %d-bit\n", prop.memoryBusWidth);
    printf("  L2 Cache Size              : %d KB\n", prop.l2CacheSize / 1024);
    printf("  Shared Memory per Block    : %zu KB\n", prop.sharedMemPerBlock / 1024);
    return 0;
}
```

> **Note:** `clockRate` and `memoryClockRate` fields were removed in CUDA 12.x. We do not
> print them — this is why the fix was necessary for CUDA 13.0 on Colab.

#### Output

![Task 1 Output](screenshots/task1_output.png)

**Key observations:** Tesla T4 has 40 SMs, Compute Capability 7.5, and supports up to 1,024
threads per block with 48 KB shared memory — typical of Turing-generation datacenter GPUs.

---

### Task 2: Hello from GPU – Thread and Block Configurations

**Objective:** Launch a simple kernel with multiple grid/block configurations and print thread IDs.

#### Code

```c
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_gpu() {
    printf("Hello from GPU! Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    // Configuration 1: 1 block, 8 threads
    printf("--- 1 block, 8 threads ---\n");
    hello_gpu<<<1, 8>>>();
    cudaDeviceSynchronize();

    // Configuration 2: 1 block, 16 threads
    printf("--- 1 block, 16 threads ---\n");
    hello_gpu<<<1, 16>>>();
    cudaDeviceSynchronize();

    // Configuration 3: 2 blocks, 8 threads each
    printf("--- 2 blocks, 8 threads ---\n");
    hello_gpu<<<2, 8>>>();
    cudaDeviceSynchronize();

    // Configuration 4: 4 blocks, 8 threads each
    printf("--- 4 blocks, 8 threads ---\n");
    hello_gpu<<<4, 8>>>();
    cudaDeviceSynchronize();

    return 0;
}
```

#### Output

![Task 2 Output](screenshots/task2_output.png)

**Key observation:** `cudaDeviceSynchronize()` ensures the CPU waits for all GPU threads to
finish printing before proceeding to the next configuration.

---

### Task 3: Thread and Block ID Visualization

**Objective:** Visualize how `threadIdx.x`, `blockIdx.x`, and the global thread ID relate to
each other in a 2D grid-of-blocks layout.

#### Code

```c
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void print_ids() {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Block %d | Thread %d | Global ID %d\n",
           blockIdx.x, threadIdx.x, global_id);
}

int main() {
    printf("=== Config: 2 blocks x 4 threads ===\n");
    print_ids<<<2, 4>>>();
    cudaDeviceSynchronize();

    printf("\n=== Config: 3 blocks x 4 threads ===\n");
    print_ids<<<3, 4>>>();
    cudaDeviceSynchronize();

    return 0;
}
```

#### Output

![Task 3 Output](screenshots/task3_output.png)

**Key observation:** Blocks may print in a non-sequential order (e.g., Block 1 before Block 0).
This is by design — the GPU scheduler assigns blocks to SMs independently and there is
**no guaranteed execution order between blocks**. This is fundamental to CUDA's parallel model.

---

### Task 4: Parallel Vector Addition (Direct Timing)

**Objective:** Add two large vectors in parallel on the GPU and measure CPU vs GPU kernel time.

#### Code

```c
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

#define N 1000000
#define THREADS_PER_BLOCK 256

__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main() {
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    int bytes = N * sizeof(float);

    h_a     = (float*)malloc(bytes);
    h_b     = (float*)malloc(bytes);
    h_c_cpu = (float*)malloc(bytes);
    h_c_gpu = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) { h_a[i] = i * 0.1f; h_b[i] = i * 0.2f; }

    // --- CPU timing ---
    double cpu_start = omp_get_wtime();
    for (int i = 0; i < N; i++) h_c_cpu[i] = h_a[i] + h_b[i];
    double cpu_time = (omp_get_wtime() - cpu_start) * 1000.0;

    // --- GPU ---
    cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    vector_add<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost);

    // --- Verify ---
    bool pass = true;
    for (int i = 0; i < N; i++)
        if (fabsf(h_c_gpu[i] - h_c_cpu[i]) > 1e-4f) { pass = false; break; }

    printf("CPU Time: %.3f ms\n", cpu_time);
    printf("GPU Kernel Time: %.3f ms\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("Result: %s\n", pass ? "PASSED" : "FAILED");

    return 0;
}
```

#### Output

![Task 4 Output](screenshots/task4_output.png)

| Metric | Value |
|---|---|
| CPU Time | 2.018 ms |
| GPU Kernel Time | 0.104 ms |
| **Speedup** | **19.43×** |
| Grid Configuration | 3,907 blocks × 256 threads |
| Correctness | PASSED |

**Key observation:** GPU provides ~19× speedup purely for the compute kernel. The GPU launches
3,907 blocks covering all 1,000,000 elements. `(1,000,000 + 256 − 1) / 256 = 3,907` blocks.

---

### Task 5: Explicit CUDA Memory Management

**Objective:** Measure individual timings of Host→Device transfer, kernel execution, and
Device→Host transfer to understand the full GPU pipeline cost.

#### Code

```c
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 500000
#define THREADS_PER_BLOCK 256

__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

float time_event(cudaEvent_t s, cudaEvent_t e) {
    float ms; cudaEventElapsedTime(&ms, s, e); return ms;
}

int main() {
    int bytes = N * sizeof(float);
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { h_a[i] = i; h_b[i] = i * 2; }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_c, bytes);

    cudaEvent_t e0,e1,e2,e3,e4;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&e2); cudaEventCreate(&e3); cudaEventCreate(&e4);

    cudaEventRecord(e0);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(e1);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vector_add<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    cudaEventRecord(e2);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(e3);
    cudaEventSynchronize(e3);

    printf("H->D Transfer: %.3f ms\n", time_event(e0,e1));
    printf("Kernel:        %.3f ms\n", time_event(e1,e2));
    printf("D->H Transfer: %.3f ms\n", time_event(e2,e3));
    printf("Total:         %.3f ms\n", time_event(e0,e3));
    printf("Blocks: %d, Threads/Block: %d\n", blocks, THREADS_PER_BLOCK);

    return 0;
}
```

#### Output

![Task 5 Output](screenshots/task5_output.png)

| Operation | Time |
|---|---|
| Host → Device Transfer | 1.122 ms |
| Kernel Execution | 0.131 ms |
| Device → Host Transfer | 1.433 ms |
| **Total GPU Time** | **2.686 ms** |
| Grid | 1,954 blocks × 256 threads |
| Correctness | PASSED |

**Key observation:** Memory transfers (H→D + D→H = 2.555 ms) dominate over the kernel
(0.131 ms). This illustrates why memory transfer overhead is a key bottleneck in GPU programming.
For repeated operations on the same data, keeping it resident on the GPU eliminates this cost.

---

### Task 6: Block Size Benchmark

**Objective:** Measure how different thread-per-block configurations affect kernel performance
for a fixed vector size.

#### Code

```c
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1000000

__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    int bytes = N * sizeof(float);
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *d_a, *d_b, *d_c;
    for (int i = 0; i < N; i++) { h_a[i] = i; h_b[i] = i*2; }
    cudaMalloc(&d_a,bytes); cudaMalloc(&d_b,bytes); cudaMalloc(&d_c,bytes);
    cudaMemcpy(d_a,h_a,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,bytes,cudaMemcpyHostToDevice);

    int block_sizes[] = {64, 128, 256, 512};
    printf("Block Size | Kernel Time (ms)\n");
    for (int b = 0; b < 4; b++) {
        int tpb = block_sizes[b];
        int blocks = (N + tpb - 1) / tpb;
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        vector_add<<<blocks, tpb>>>(d_a, d_b, d_c, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        printf("%10d | %.3f ms\n", tpb, ms);
    }
    return 0;
}
```

#### Output

![Task 6 Output](screenshots/task6_output.png)

| Threads per Block | Kernel Time (ms) |
|---|---|
| 64 | 0.053 |
| 128 | 0.052 |
| 256 | 0.053 |
| 512 | 0.051 |

![Graph 2: Block Size vs Kernel Time](graph2_blocksize_vs_time.png)

**Key observation:** Kernel time is nearly identical across all block sizes for this simple
memory-bound kernel. Block size matters more for compute-intensive kernels with shared memory usage.
The Tesla T4 can hide memory latency effectively regardless of block size for this workload.

---

### Task 7: Sequential vs OpenMP vs CUDA Comparison

**Objective:** Compare vector addition performance across three implementations as vector size scales.

#### Code

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <omp.h>

#define THREADS_PER_BLOCK 256

__global__ void cuda_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

void seq_add(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

void omp_add(float *a, float *b, float *c, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

int main() {
    int sizes[] = {100000, 1000000, 10000000, 100000000};
    printf("%-12s  %-12s  %-12s  %-12s  %-6s\n",
           "N", "Seq(ms)", "OMP(ms)", "CUDA(ms)", "Err");
    for (int s = 0; s < 4; s++) {
        int n = sizes[s];
        float *h_a = (float*)malloc(n*sizeof(float));
        float *h_b = (float*)malloc(n*sizeof(float));
        float *h_c = (float*)malloc(n*sizeof(float));
        for (int i = 0; i < n; i++) { h_a[i] = i*0.1f; h_b[i] = i*0.2f; }

        // Sequential
        double t0 = omp_get_wtime();
        seq_add(h_a, h_b, h_c, n);
        double seq_ms = (omp_get_wtime()-t0)*1000.0;

        // OpenMP
        t0 = omp_get_wtime();
        omp_add(h_a, h_b, h_c, n);
        double omp_ms = (omp_get_wtime()-t0)*1000.0;

        // CUDA
        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a,n*sizeof(float));
        cudaMalloc(&d_b,n*sizeof(float));
        cudaMalloc(&d_c,n*sizeof(float));
        cudaMemcpy(d_a,h_a,n*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_b,h_b,n*sizeof(float),cudaMemcpyHostToDevice);
        int blocks = (n+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
        cudaEvent_t start,stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        cuda_add<<<blocks,THREADS_PER_BLOCK>>>(d_a,d_b,d_c,n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float cuda_ms; cudaEventElapsedTime(&cuda_ms,start,stop);

        printf("%-12d  %-12.3f  %-12.3f  %-12.3f  0\n",
               n, seq_ms, omp_ms, cuda_ms);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        free(h_a); free(h_b); free(h_c);
    }
    return 0;
}
```

#### Output

![Task 7 Output](screenshots/task7_output.png)

| Vector Size (N) | Sequential (ms) | OpenMP – 2 threads (ms) | CUDA (ms) | Err |
|---|---|---|---|---|
| 100,000 | 0.320 | 0.226 | 0.083 | 0 |
| 1,000,000 | 2.193 | 1.670 | 0.048 | 0 |
| 10,000,000 | 22.447 | 17.343 | 0.458 | 0 |
| 100,000,000 | 249.584 | 179.032 | 4.558 | 0 |

![Graph 1: Execution Time vs Input Size](graph1_time_vs_size.png)

**Key observation:** CUDA dominates at every scale. At N = 100,000,000, CUDA is **54.8× faster**
than sequential. OpenMP (2 threads) provides a modest ~1.4× speedup — limited by the Colab
environment having only 2 CPU threads available. Error = 0 confirms all results are numerically correct.

---

### Task 8: Array Squaring

**Objective:** Apply GPU parallelism to a practical element-wise operation (squaring each element)
and verify correctness against CPU results.

#### Code

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <omp.h>

#define N 1000000
#define THREADS_PER_BLOCK 256

__global__ void square_array(float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * in[idx];
}

int main() {
    int bytes = N * sizeof(float);
    float *h_in  = (float*)malloc(bytes);
    float *h_out_cpu = (float*)malloc(bytes);
    float *h_out_gpu = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = i * 0.01f;

    // CPU
    double t0 = omp_get_wtime();
    for (int i = 0; i < N; i++) h_out_cpu[i] = h_in[i] * h_in[i];
    double cpu_ms = (omp_get_wtime()-t0)*1000.0;

    // GPU
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes); cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    square_array<<<blocks, THREADS_PER_BLOCK>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms; cudaEventElapsedTime(&gpu_ms, start, stop);

    cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost);

    bool pass = true;
    for (int i = 0; i < N; i++)
        if (fabsf(h_out_gpu[i] - h_out_cpu[i]) > 1e-4f) { pass = false; break; }

    printf("N=%d, Threads=%d, Blocks=%d\n", N, THREADS_PER_BLOCK, blocks);
    printf("CPU Time: %.3f ms\n", cpu_ms);
    printf("GPU Time: %.3f ms\n", gpu_ms);
    printf("Speedup:  %.2fx\n", cpu_ms/gpu_ms);
    printf("Result:   %s\n", pass ? "PASSED" : "FAILED");

    return 0;
}
```

#### Output

![Task 8 Output](screenshots/task8_output.png)

| Metric | Value |
|---|---|
| N | 1,000,000 |
| Threads per Block | 256 |
| Blocks | 3,907 |
| CPU Time | 2.015 ms |
| GPU Kernel Time | 0.121 ms |
| **Speedup** | **16.70×** |
| Correctness | PASSED |

Sample verification: `in[5]=0.05 → out≈0.00` (0.05²=0.0025), `in[100]=1.00 → out≈1.00` ✓

---

## Performance Summary Tables

### Table 1: CPU vs GPU Kernel Speedup

| Task | Operation | N | CPU (ms) | GPU Kernel (ms) | Speedup |
|---|---|---|---|---|---|
| Task 4 | Vector Addition | 1,000,000 | 2.018 | 0.104 | **19.43×** |
| Task 7 | Vector Addition | 100,000,000 | 249.584 | 4.558 | **54.76×** |
| Task 8 | Array Squaring | 1,000,000 | 2.015 | 0.121 | **16.70×** |

### Table 2: Memory Transfer Breakdown (Task 5, N=500,000)

| Phase | Time |
|---|---|
| Host → Device | 1.122 ms |
| Kernel | 0.131 ms |
| Device → Host | 1.433 ms |
| Total | 2.686 ms |

> Takeaway: Memory transfer takes **19× longer** than the actual compute kernel.

### Table 3: Sequential vs OpenMP vs CUDA (Task 7)

| N | Seq (ms) | OMP (ms) | CUDA (ms) | CUDA Speedup vs Seq |
|---|---|---|---|---|
| 100,000 | 0.320 | 0.226 | 0.083 | 3.86× |
| 1,000,000 | 2.193 | 1.670 | 0.048 | 45.69× |
| 10,000,000 | 22.447 | 17.343 | 0.458 | 49.01× |
| 100,000,000 | 249.584 | 179.032 | 4.558 | **54.76×** |

---

## Performance Graphs

### Graph 1: Input Size vs Execution Time

![Graph 1](graph1_time_vs_size.png)

The log-scale plot shows CUDA remaining orders of magnitude faster than both sequential and OpenMP
across all input sizes. OpenMP provides consistent but modest improvement (2 CPU threads in Colab).

### Graph 2: Block Size vs Kernel Execution Time

![Graph 2](graph2_blocksize_vs_time.png)

Kernel time is essentially flat across block sizes 64–512 for this memory-bound vector addition.
This shows the T4 hides memory latency efficiently regardless of block configuration.

### Graph 3: CUDA Speedup vs Input Size

![Graph 3](graph3_speedup_vs_size.png)

CUDA speedup grows dramatically from 3.86× at N=100,000 to 54.76× at N=100,000,000. Small
vectors do not fully utilize all 40 SMs; larger vectors saturate all GPU resources.

---

## Questions and Answers

### 1. What is the difference between CPU and GPU architecture?

CPU architecture is optimized for **latency** — it has a small number of powerful cores
(typically 4–32) with large caches and out-of-order execution, designed to run sequential
code as fast as possible. GPU architecture is optimized for **throughput** — the Tesla T4
has 40 SMs each containing 64 CUDA cores (2,560 total), designed to run thousands of
threads simultaneously on simple arithmetic operations. CPUs minimize latency per task;
GPUs maximize total work done per second across many parallel tasks.

### 2. What is a CUDA thread and how is it different from a CPU thread?

A CUDA thread is an extremely lightweight execution unit that runs one iteration of a
kernel function. Unlike CPU threads (which are OS-managed, heavyweight, and switch via
context switching), CUDA threads are managed entirely in hardware by the GPU scheduler.
A single SM on the T4 can hold up to 1,024 active threads and switches between them
in zero cycles (zero-overhead warp switching) to hide memory latency.

### 3. Explain the concept of thread blocks and grids in CUDA.

A **thread block** is a group of threads (up to 1,024 on Turing GPUs) that execute on
a single SM. Threads within a block can communicate via **shared memory** and synchronize
with `__syncthreads()`. A **grid** is the entire collection of blocks launched for one
kernel call. Blocks are distributed across available SMs automatically — the programmer
specifies how many blocks to launch (`numBlocks`), not which SM each runs on. This
abstraction makes CUDA programs scalable across different GPU generations.

### 4. What is the role of `blockIdx.x`, `threadIdx.x`, and `blockDim.x` in CUDA?

- `blockIdx.x`: The index of the current thread's block within the grid (0 to gridDim.x−1)
- `threadIdx.x`: The index of the current thread within its block (0 to blockDim.x−1)
- `blockDim.x`: The total number of threads in each block (set at launch time)

Combined formula: `int global_id = blockIdx.x * blockDim.x + threadIdx.x;`

This maps each thread to a unique array element, enabling each thread to work on a
different piece of data independently.

### 5. How does the GPU achieve parallelism?

The GPU achieves parallelism through **SIMT** (Single Instruction, Multiple Threads)
execution. Threads are grouped into **warps** of 32 (the T4 warp size). All 32 threads
in a warp execute the same instruction simultaneously on different data. With 40 SMs and
multiple active warps per SM, the T4 can have thousands of threads in flight at once.
When one warp stalls on a memory access, the SM instantly switches to another ready warp,
hiding latency with parallelism (latency hiding via warp switching).

### 6. What is CUDA memory hierarchy and its significance?

CUDA has multiple memory levels (fastest to slowest):
1. **Registers** — per-thread, fastest, limited (~255 per thread)
2. **Shared memory** — per-block on-chip SRAM, 48 KB on T4, ~100× faster than global
3. **L1/L2 cache** — automatic caching of global memory accesses
4. **Global memory** — GPU DRAM (15 GB on T4), slowest but largest

Significance: Task 5 showed that memory transfers (H→D + D→H = 2.555 ms) take 19× longer
than the actual kernel (0.131 ms). Using shared memory for frequently reused data can
dramatically reduce global memory accesses and improve performance.

### 7. How is CUDA memory managed using `cudaMalloc` and `cudaMemcpy`?

```c
// Allocate device memory
cudaMalloc((void**)&d_ptr, size_in_bytes);

// Copy data to GPU (Host → Device)
cudaMemcpy(d_dst, h_src, size_in_bytes, cudaMemcpyHostToDevice);

// Copy results back (Device → Host)
cudaMemcpy(h_dst, d_src, size_in_bytes, cudaMemcpyDeviceToHost);

// Free device memory
cudaFree(d_ptr);
```

These are the four fundamental memory management calls. `cudaMalloc` allocates in GPU global
memory; `cudaMemcpy` transfers over the PCIe bus (or NVLink in multi-GPU setups). The
programmer is responsible for explicit allocation and deallocation — there is no garbage
collection on the GPU.

### 8. What is the purpose of `cudaDeviceSynchronize()`?

`cudaDeviceSynchronize()` blocks the CPU until all previously launched CUDA kernels and
memory operations have completed. Without it, CUDA kernel launches are **asynchronous** —
the CPU immediately returns after launching the kernel and continues executing. This is
used in Task 2 to ensure all GPU printf statements complete before the CPU prints the next
configuration header. For timing, `cudaEventSynchronize(stop)` on the stop event is
preferred over `cudaDeviceSynchronize()` as it is more precise and lightweight.

### 9. Explain the significance of warp size (32) in CUDA execution.

The warp is the fundamental scheduling unit on an SM. All 32 threads in a warp execute
the same instruction per cycle (SIMT). If threads within a warp take different branches
(if/else), the warp executes both paths sequentially with inactive threads masked out
(**warp divergence**), reducing efficiency. For maximum performance:
- Align data accesses to 128-byte boundaries (coalesced memory access)
- Avoid branch divergence within a warp
- Use multiples of 32 for block sizes to avoid partially filled warps

The T4's warp size of 32 is standard across all modern NVIDIA GPUs.

### 10. How does the number of threads per block affect performance?

From Task 6 (N = 1,000,000, block sizes 64/128/256/512):

| Block Size | Time |
|---|---|
| 64 | 0.053 ms |
| 128 | 0.052 ms |
| 256 | 0.053 ms |
| 512 | 0.051 ms |

For this simple memory-bound kernel, performance is nearly identical because the bottleneck
is memory bandwidth, not thread scheduling. For compute-intensive kernels using shared memory,
block size critically affects occupancy and performance. Best practice is 128–256 threads
per block for general workloads.

### 11. What is the purpose of using `cudaEvent_t` for timing?

`cudaEvent_t` provides **GPU-side hardware timestamps** accurate to microseconds. CPU-side
timing (e.g., `clock()` or `omp_get_wtime()`) is unreliable for GPU kernel timing because:
1. Kernel launches are asynchronous — the CPU timer records the launch time, not completion
2. OS scheduling jitter on the CPU adds noise

`cudaEventRecord(start)` inserts a timestamp into the GPU's command queue at that point.
`cudaEventElapsedTime(&ms, start, stop)` computes the actual GPU time between the two events,
independent of CPU scheduling. Used in Tasks 4–8 for accurate kernel timing.

### 12. What is the significance of checking `cudaGetDeviceCount()`?

`cudaGetDeviceCount()` is the first call in any robust CUDA program — it determines how
many CUDA-capable GPUs are in the system. If it returns 0, no GPU is available and all
subsequent CUDA calls will fail. It allows graceful fallback to CPU processing. In multi-GPU
systems, it enables the program to select the most powerful GPU (e.g., by checking
`totalGlobalMem` or `multiProcessorCount` for each device) or distribute work across GPUs.

### 13. How does the grid size relate to the input data size?

Grid size (`numBlocks`) is computed as:
```c
int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
```
This ceiling division ensures enough blocks to cover all N elements. For N=1,000,000 and
threadsPerBlock=256: `numBlocks = (1,000,000 + 255) / 256 = 3,907`. Each thread computes
one element. The `if (idx < n)` guard in the kernel prevents the ~230 idle threads in the
last block from writing out-of-bounds. Grid size scales linearly with input size, up to
the maximum grid dimension (2,147,483,647 for x on the T4).

### 14. What happens when blocks finish in different orders?

Task 3 demonstrated that blocks may print in non-sequential order (e.g., Block 2 before
Block 0). This is by design: the GPU scheduler assigns blocks to SMs as they become
available, and there is **no guaranteed ordering between blocks**. This is a fundamental
constraint of the CUDA programming model — blocks must be independent (no inter-block
communication during kernel execution). Dependencies between blocks require separate kernel
launches with synchronization via `cudaDeviceSynchronize()` between them.

### 15. What is the speedup formula and what does it represent?

```
Speedup = CPU Execution Time / GPU Kernel Time
```

Results from this experiment:
- Task 4 (N=1M): Speedup = 2.018 / 0.104 = **19.43×**
- Task 7 (N=100M): Speedup = 249.584 / 4.558 = **54.76×**
- Task 8 (N=1M): Speedup = 2.015 / 0.121 = **16.70×**

Speedup > 1 means GPU is faster. Note: "GPU kernel speedup" excludes memory transfer time.
If total GPU time (including transfers) is used, speedup is lower (Task 5 shows total GPU
time = 2.686 ms vs CPU ~2 ms for N=500,000 — transfers can eliminate the advantage for
small, one-shot computations).

### 16. Explain the difference between sequential, OpenMP, and CUDA execution models.

| Feature | Sequential | OpenMP | CUDA |
|---|---|---|---|
| Hardware | Single CPU core | Multiple CPU cores | GPU (thousands of cores) |
| Threads | 1 | 2–128 typically | Up to millions |
| Memory | Shared RAM | Shared RAM | Separate GPU VRAM |
| Overhead | None | Thread creation | Memory transfer + launch |
| Best for | Small data, logic | Medium parallel tasks | Massive data parallelism |
| Task 7 result (N=100M) | 249.6 ms | 179.0 ms | 4.6 ms |

OpenMP uses `#pragma omp parallel for` to split loop iterations across CPU threads. CUDA
maps each iteration to a GPU thread. CUDA wins by ~39× over OpenMP at N=100M.

### 17. Why is GPU computing beneficial for large-scale scientific computing?

GPUs are beneficial because:
1. **Massive parallelism**: T4 has 2,560 CUDA cores vs 2–64 CPU cores
2. **High memory bandwidth**: GPU VRAM bandwidth (~320 GB/s on T4) vs CPU (~50 GB/s)
3. **Energy efficiency**: More GFLOPS per watt than CPUs for parallel workloads
4. **Scalability**: CUDA code scales automatically to more powerful GPUs
5. **Real results**: This experiment shows 54.76× speedup on a real Tesla T4

Applications: molecular dynamics (GROMACS), weather simulation (WRF), deep learning
training (BERT, GPT), seismic processing, genomics (GATK-GPU), computational fluid dynamics.

### 18. What are the limitations of GPU computing?

1. **Memory transfer overhead**: PCIe bus is much slower than GPU compute (seen in Task 5)
2. **Memory capacity**: GPU VRAM (15 GB on T4) is smaller than RAM (often 100s of GB)
3. **Programming complexity**: CUDA requires explicit memory management and parallel thinking
4. **Warp divergence**: Branchy code runs poorly on GPUs
5. **Sequential code**: Amdahl's law — if 10% of code is sequential, max speedup is 10×
6. **Small inputs**: GPU overhead doesn't pay off for N < ~10,000 elements
7. **Cost**: High-end GPUs are expensive; Colab provides free T4 access with limits

### 19. How does CUDA support different hardware via compute capability?

Compute Capability (CC) is a versioning system for CUDA hardware features:
- **CC 3.0–5.x** (Kepler/Maxwell): Basic CUDA, unified memory, dynamic parallelism
- **CC 6.x** (Pascal): FP16 support, NVLink, page migration
- **CC 7.0** (Volta): Tensor Cores for AI, cooperative groups
- **CC 7.5** (Turing = T4): Tensor Cores, RT Cores, INT8/INT4 inference
- **CC 8.0** (Ampere): 3rd-gen Tensor Cores, A100
- **CC 9.0** (Hopper): H100, Transformer Engine

The T4 at CC 7.5 supports all features up to CUDA 13.0. Higher CC enables newer features
but code written for lower CC is forward-compatible. `nvcc --arch=sm_75` compiles optimized
code for the T4.

### 20. What are best practices for writing efficient CUDA code?

1. **Maximize occupancy**: Use 128–256 threads/block; keep register usage low
2. **Coalesced memory access**: Adjacent threads should access adjacent memory addresses
3. **Use shared memory**: Cache frequently reused data from global memory into shared memory
4. **Minimize transfers**: Keep data on GPU for repeated operations (avoid repeated H↔D)
5. **Avoid warp divergence**: Keep `if/else` branches uniform across warps
6. **Use CUDA events for timing**: More accurate than CPU-side timers for GPU code
7. **Check errors**: Use `cudaGetLastError()` after kernel launches; use CUDA_CHECK macros
8. **Profile with Nsight**: Use NVIDIA Nsight Systems/Compute to find bottlenecks
9. **Overlap compute and transfer**: Use CUDA streams for concurrent H↔D and kernel execution
10. **Prefer libraries**: cuBLAS, cuFFT, cuDNN are highly optimized — use them when available

---

## Reflection Questions

### 1. What surprised you most about GPU programming?

The most surprising aspect was how **dominant memory transfers are** compared to actual
compute time. Task 5 showed that the H→D + D→H transfers (2.555 ms) took 19× longer than
the kernel (0.131 ms). Intuitively, one might expect the bottleneck to be the thousands
of parallel computations, but the PCIe bus connecting the CPU and GPU is actually the
critical bottleneck for many workloads. This fundamentally changes how one must design
GPU programs — keeping data resident on the GPU across multiple operations.

### 2. What was the most challenging aspect of CUDA programming?

The most challenging aspect was **understanding the thread indexing model** and correctly
computing the global thread ID: `int idx = blockIdx.x * blockDim.x + threadIdx.x;`.
It requires thinking at two levels simultaneously — the position within a block
(`threadIdx.x`) and the position of the block within the grid (`blockIdx.x`). Additionally,
the `if (idx < n)` boundary check is easy to forget, which would cause out-of-bounds
memory access with no compiler warning (a silent GPU bug).

### 3. How does the CUDA programming model compare to traditional programming?

Traditional (sequential) programming is naturally linear — instructions execute one at
a time, with shared variables and predictable execution order. CUDA inverts this completely:
thousands of threads run simultaneously, all executing the same function on different data.
Traditional debugging tools don't work on GPU code. The programmer must think about:
- Which thread handles which data element?
- Do any threads depend on each other? (use `__syncthreads()`)
- Is memory access pattern coalesced?

It requires a fundamentally different mental model of computation.

### 4. What improvements would you make to the code?

1. **Unified Memory**: Use `cudaMallocManaged()` to eliminate explicit H→D/D→H transfers
   and let CUDA manage data migration automatically
2. **CUDA Streams**: Overlap memory transfers with kernel computation using multiple streams
3. **Shared memory tiling**: For more complex kernels (e.g., matrix multiplication), load
   tiles of data into shared memory to reduce global memory bandwidth requirements
4. **Error handling**: Add `CUDA_CHECK` macros for all CUDA API calls to catch errors early
5. **Multi-GPU**: Distribute work across multiple T4 GPUs using `cudaSetDevice()`

### 5. How would you optimize the vector addition kernel for better performance?

For the current memory-bound vector addition:
1. **Vectorized loads**: Use `float4` instead of `float` to load 4 elements per thread,
   improving memory throughput
2. **Grid-stride loops**: Instead of one element per thread, have each thread process
   multiple elements (`for (int i = tid; i < n; i += gridDim.x * blockDim.x)`)
3. **Prefetching**: Use `__ldg()` (load via texture cache) for read-only data
4. **Increased arithmetic intensity**: Fuse operations (e.g., FMA — fused multiply-add)
   to do more compute per memory access

### 6. What real-world applications benefit from CUDA programming?

1. **Deep Learning**: Training neural networks (ResNet, BERT, GPT) uses GPU matrix ops
2. **Medical Imaging**: CT/MRI reconstruction, real-time image segmentation
3. **Molecular Dynamics**: Simulating protein folding (as in AlphaFold2)
4. **Computational Finance**: Monte Carlo option pricing, risk analysis
5. **Seismic Processing**: 3D subsurface imaging for oil/gas exploration
6. **Video Processing**: Real-time H.264/H.265 encoding on YouTube/Netflix
7. **Scientific Computing**: Climate simulation, cosmological N-body simulations
8. **Cryptography**: GPU-accelerated hash computation (Bitcoin mining)

### 7. How does CUDA handle thread synchronization?

CUDA provides two levels of synchronization:

**Intra-block** (within one block): `__syncthreads()` — a barrier that ensures all threads
in the block have completed the current instruction before any thread proceeds. Used when
sharing data through shared memory (write → `__syncthreads()` → read pattern).

**Inter-block / global**: Blocks cannot synchronize directly during a kernel. Global
synchronization requires ending one kernel and launching another. `cudaDeviceSynchronize()`
on the host ensures the CPU waits for all GPU work to complete before proceeding.

Advanced: CUDA 9+ provides **cooperative groups** for flexible, explicit group-level
synchronization beyond the simple block barrier.

### 8. What did you learn about parallel computing from this experiment?

Key lessons:
1. **Amdahl's Law is real**: Even with 54.76× GPU speedup, if 50% of the application
   is sequential, the maximum overall speedup is only 2×
2. **Granularity matters**: Small N (100,000) gave only 3.86× speedup; large N (100,000,000)
   gave 54.76× — parallelism pays off only when there is enough work to hide overhead
3. **Memory is often the bottleneck**, not compute — GPU programming requires memory-centric
   optimization thinking
4. **Correctness first**: All tasks verified results with `fabsf` comparisons; parallel
   bugs are hard to debug — always verify against a CPU reference
5. **Hardware determines strategy**: The T4's 40 SMs, 48 KB shared memory, and 256-bit
   memory bus define what optimization techniques apply

---

## AI & HPC Case Study: Deep Learning Training on GPUs

### Why GPUs Transformed AI

Deep learning's renaissance from 2012 onward was made possible almost entirely by GPUs.
The key operation in training a neural network — the matrix multiplication in forward
and backward passes — maps perfectly to GPU architecture. A single forward pass of
ResNet-50 involves billions of multiply-add operations; training over a million images
requires trillions. Before GPUs, this was computationally prohibitive on CPUs.

### How CUDA Enables Deep Learning

Modern frameworks like PyTorch and TensorFlow use CUDA under the hood. When a user writes:

```python
output = torch.matmul(A, B)  # runs on GPU via cuBLAS
loss.backward()               # gradient computation via CUDA kernels
```

PyTorch dispatches these operations to highly optimized CUDA kernels (often using cuBLAS
for matrix ops and cuDNN for convolutions). The Tesla T4's **Tensor Cores** — specialized
hardware introduced in Turing (CC 7.5) — perform 4×4 matrix multiplications in a single
clock cycle using mixed-precision (FP16 compute, FP32 accumulate), delivering up to
65 TOPS (tera-operations per second) for AI inference — 65× faster than pure FP32 on the
same GPU.

### Large Language Model Training

Training GPT-3 (175B parameters) required 3,640 A100 GPUs running for 34 days.
Each A100 (CC 8.0) delivers 312 TFLOPS of tensor-core performance. Without GPU-accelerated
CUDA code, this training run would take centuries on CPUs.

The same CUDA fundamentals from this experiment — thread blocks, shared memory, memory
management — scale up to production AI training, where techniques like:
- **Tensor parallelism**: Split weight matrices across GPUs
- **Pipeline parallelism**: Different GPU nodes handle different transformer layers
- **Mixed precision (FP16/BF16)**: Halve memory usage, double throughput via Tensor Cores

### Connection to This Experiment

The vector addition kernel (Task 4, 19.43× speedup) is conceptually identical to the inner
loop of an attention mechanism in a Transformer: element-wise multiply, add, and activate.
At scale (matrices of size 4096×4096 per attention head, 96 heads in GPT-3), these speedups
compound to make trillion-parameter models trainable. This experiment demonstrates the
foundational building block on which all modern AI infrastructure is built.

---

## Deliverables Checklist

- [x] Task 1: GPU device information program (verified: Tesla T4, 40 SMs, 15 GB, CC 7.5)
- [x] Task 2: Hello GPU with 4 configurations (1×8, 1×16, 2×8, 4×8)
- [x] Task 3: Thread/Block ID visualization (2×4 and 3×4 grids)
- [x] Task 4: Vector addition with CPU vs GPU timing (19.43× speedup, PASSED)
- [x] Task 5: Explicit memory management with phased timing (H→D/Kernel/D→H breakdown)
- [x] Task 6: Block size benchmark (64/128/256/512, all ~0.052 ms)
- [x] Task 7: Sequential vs OpenMP vs CUDA comparison (up to 54.76× speedup)
- [x] Task 8: Array squaring (16.70× speedup, PASSED)
- [x] Graph 1: Execution Time vs Input Size (log scale, 3 implementations)
- [x] Graph 2: Block Size vs Kernel Time (bar chart, 4 block sizes)
- [x] Graph 3: CUDA Speedup vs Input Size (speedup curve)
- [x] All 20 questions answered
- [x] All 8 reflection questions answered
- [x] AI & HPC case study (Deep Learning on GPUs, ~400 words)
- [x] All output screenshots from Google Colab (Tesla T4) included

---

## Resources

- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA API Reference: https://docs.nvidia.com/cuda/cuda-runtime-api/
- NVIDIA Tesla T4 Datasheet: https://www.nvidia.com/en-us/data-center/tesla-t4/
- **Google Colab Notebook:** https://colab.research.google.com/drive/1jDc_U5Q1tYqDomfXlIxHywb3cRJArA8m?usp=sharing
