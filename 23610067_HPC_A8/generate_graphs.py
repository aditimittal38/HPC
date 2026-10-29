import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── load data ──────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path   = os.path.join(script_dir, "results.csv")

rows = []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({
            "app":   r["app"],
            "model": r["model"],
            "param": int(r["param"]),
            "size":  int(r["size"]),
            "time":  float(r["time_sec"]),
        })

def get_time(app, model, param, size):
    for r in rows:
        if r["app"] == app and r["model"] == model and r["param"] == param and r["size"] == size:
            return r["time"]
    return None

# ── colour palette ─────────────────────────────────────────────────────────
SEQ_C   = "#4e79a7"
OMP_C   = "#f28e2b"
MPI_C   = "#59a14f"
SCHED   = {"static": "#f28e2b", "dynamic": "#e15759", "guided": "#76b7b2"}

STYLE = dict(linewidth=2, marker="o", markersize=6)
plt.rcParams.update({"font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.4, "figure.dpi": 150})

sizes_matmul = [500, 1000, 2000]
sizes_conv   = [1000, 2000, 4000]
params       = [2, 4, 8]


# ══════════════════════════════════════════════════════════════════════════
#  TASK 1 – Matrix–Matrix Multiplication
# ══════════════════════════════════════════════════════════════════════════

# Graph 1A – Threads/Processes vs Execution Time (size=1000)
fig, ax = plt.subplots()
N = 1000
seq_t = get_time("matmul", "sequential", 1, N)
ax.axhline(seq_t, color=SEQ_C, linestyle="--", linewidth=1.8, label=f"Sequential ({seq_t:.3f}s)")
omp_t = [get_time("matmul", "openmp", p, N) for p in params]
mpi_t = [get_time("matmul", "mpi",    p, N) for p in params]
ax.plot(params, omp_t, color=OMP_C, label="OpenMP", **STYLE)
ax.plot(params, mpi_t, color=MPI_C, label="MPI",    **STYLE)
ax.set_xlabel("Threads / Processes")
ax.set_ylabel("Execution Time (s)")
ax.set_title("MatMul – Threads/Processes vs Execution Time (N=1000)")
ax.set_xticks(params)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(script_dir, "matmul_threads_vs_time.png"))
plt.close()

# Graph 1B – Threads/Processes vs Speedup (size=1000)
fig, ax = plt.subplots()
omp_sp = [seq_t / t for t in omp_t]
mpi_sp = [seq_t / t for t in mpi_t]
ax.plot(params, omp_sp, color=OMP_C, label="OpenMP", **STYLE)
ax.plot(params, mpi_sp, color=MPI_C, label="MPI",    **STYLE)
ax.plot(params, params, color="gray", linestyle="--", linewidth=1.4, label="Ideal")
ax.set_xlabel("Threads / Processes")
ax.set_ylabel("Speedup")
ax.set_title("MatMul – Threads/Processes vs Speedup (N=1000)")
ax.set_xticks(params)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(script_dir, "matmul_threads_vs_speedup.png"))
plt.close()

# Graph 1C – Threads/Processes vs Efficiency (size=1000)
fig, ax = plt.subplots()
omp_eff = [s / p for s, p in zip(omp_sp, params)]
mpi_eff = [s / p for s, p in zip(mpi_sp, params)]
ax.plot(params, omp_eff, color=OMP_C, label="OpenMP", **STYLE)
ax.plot(params, mpi_eff, color=MPI_C, label="MPI",    **STYLE)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.4, label="Ideal")
ax.set_xlabel("Threads / Processes")
ax.set_ylabel("Efficiency")
ax.set_title("MatMul – Threads/Processes vs Efficiency (N=1000)")
ax.set_xticks(params)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(script_dir, "matmul_threads_vs_efficiency.png"))
plt.close()

# Graph 1D – Problem Size vs Execution Time (p=4)
fig, ax = plt.subplots()
seq_t4 = [get_time("matmul", "sequential", 1, N) for N in sizes_matmul]
omp_t4 = [get_time("matmul", "openmp",     4, N) for N in sizes_matmul]
mpi_t4 = [get_time("matmul", "mpi",        4, N) for N in sizes_matmul]
ax.plot(sizes_matmul, seq_t4, color=SEQ_C, label="Sequential", **STYLE)
ax.plot(sizes_matmul, omp_t4, color=OMP_C, label="OpenMP 4T",  **STYLE)
ax.plot(sizes_matmul, mpi_t4, color=MPI_C, label="MPI 4P",     **STYLE)
ax.set_xlabel("Matrix Size N (N×N)")
ax.set_ylabel("Execution Time (s)")
ax.set_title("MatMul – Problem Size vs Execution Time (4 threads/processes)")
ax.set_xticks(sizes_matmul)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(script_dir, "matmul_size_vs_time.png"))
plt.close()


# ══════════════════════════════════════════════════════════════════════════
#  TASK 2 – 2D Convolution
# ══════════════════════════════════════════════════════════════════════════

# Graph 2A – Threads/Processes vs Execution Time (size=2000)
fig, ax = plt.subplots()
N = 2000
seq_t = get_time("conv", "sequential", 1, N)
ax.axhline(seq_t, color=SEQ_C, linestyle="--", linewidth=1.8, label=f"Sequential ({seq_t:.4f}s)")
omp_st = [get_time("conv", "openmp_static",  p, N) for p in params]
omp_dy = [get_time("conv", "openmp_dynamic", p, N) for p in params]
omp_gu = [get_time("conv", "openmp_guided",  p, N) for p in params]
mpi_t  = [get_time("conv", "mpi",            p, N) for p in params]
ax.plot(params, omp_st, color=SCHED["static"],  label="OpenMP static",  **STYLE)
ax.plot(params, omp_dy, color=SCHED["dynamic"], label="OpenMP dynamic", **STYLE)
ax.plot(params, omp_gu, color=SCHED["guided"],  label="OpenMP guided",  **STYLE)
ax.plot(params, mpi_t,  color=MPI_C,            label="MPI",            **STYLE)
ax.set_xlabel("Threads / Processes")
ax.set_ylabel("Execution Time (s)")
ax.set_title("Conv – Threads/Processes vs Execution Time (N=2000)")
ax.set_xticks(params)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(script_dir, "conv_threads_vs_time.png"))
plt.close()

# Graph 2B – Threads/Processes vs Speedup (size=2000, static schedule)
fig, ax = plt.subplots()
omp_sp = [seq_t / t for t in omp_st]
mpi_sp = [seq_t / t for t in mpi_t]
ax.plot(params, omp_sp, color=OMP_C, label="OpenMP (static)", **STYLE)
ax.plot(params, mpi_sp, color=MPI_C, label="MPI",             **STYLE)
ax.plot(params, params, color="gray", linestyle="--", linewidth=1.4, label="Ideal")
ax.set_xlabel("Threads / Processes")
ax.set_ylabel("Speedup")
ax.set_title("Conv – Threads/Processes vs Speedup (N=2000)")
ax.set_xticks(params)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(script_dir, "conv_threads_vs_speedup.png"))
plt.close()

# Graph 2C – Threads/Processes vs Efficiency (size=2000, static schedule)
fig, ax = plt.subplots()
omp_eff = [s / p for s, p in zip(omp_sp, params)]
mpi_eff = [s / p for s, p in zip(mpi_sp, params)]
ax.plot(params, omp_eff, color=OMP_C, label="OpenMP (static)", **STYLE)
ax.plot(params, mpi_eff, color=MPI_C, label="MPI",             **STYLE)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.4, label="Ideal")
ax.set_xlabel("Threads / Processes")
ax.set_ylabel("Efficiency")
ax.set_title("Conv – Threads/Processes vs Efficiency (N=2000)")
ax.set_xticks(params)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(script_dir, "conv_threads_vs_efficiency.png"))
plt.close()

# Graph 2D – Problem Size vs Execution Time (p=4 static for OpenMP)
fig, ax = plt.subplots()
seq_t4 = [get_time("conv", "sequential",    1, N) for N in sizes_conv]
omp_t4 = [get_time("conv", "openmp_static", 4, N) for N in sizes_conv]
mpi_t4 = [get_time("conv", "mpi",           4, N) for N in sizes_conv]
ax.plot(sizes_conv, seq_t4, color=SEQ_C, label="Sequential", **STYLE)
ax.plot(sizes_conv, omp_t4, color=OMP_C, label="OpenMP 4T",  **STYLE)
ax.plot(sizes_conv, mpi_t4, color=MPI_C, label="MPI 4P",     **STYLE)
ax.set_xlabel("Input Size N (N×N)")
ax.set_ylabel("Execution Time (s)")
ax.set_title("Conv – Problem Size vs Execution Time (4 threads/processes)")
ax.set_xticks(sizes_conv)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(script_dir, "conv_size_vs_time.png"))
plt.close()

# ── summary table ──────────────────────────────────────────────────────────
print("\n=== TASK 1: Matrix-Matrix Multiplication Performance Table ===")
header = f"{'Size':>6}  {'Model':>15}  {'Threads':>7}  {'Time(s)':>10}  {'Speedup':>8}  {'Efficiency':>10}"
print(header)
print("-" * len(header))
for N in sizes_matmul:
    seq_t = get_time("matmul", "sequential", 1, N)
    print(f"{N:>6}  {'Sequential':>15}  {'1':>7}  {seq_t:>10.4f}  {'1.00':>8}  {'1.000':>10}")
    for model, label in [("openmp", "OpenMP"), ("mpi", "MPI")]:
        for p in params:
            t  = get_time("matmul", model, p, N)
            sp = seq_t / t
            ef = sp / p
            print(f"{N:>6}  {label:>15}  {p:>7}  {t:>10.4f}  {sp:>8.3f}  {ef:>10.3f}")

print("\n=== TASK 2: 2D Convolution Performance Table ===")
header2 = f"{'Size':>6}  {'Model':>20}  {'Threads':>7}  {'Time(s)':>10}  {'Speedup':>8}  {'Efficiency':>10}"
print(header2)
print("-" * len(header2))
for N in sizes_conv:
    seq_t = get_time("conv", "sequential", 1, N)
    print(f"{N:>6}  {'Sequential':>20}  {'1':>7}  {seq_t:>10.4f}  {'1.00':>8}  {'1.000':>10}")
    for model, label in [("openmp_static", "OpenMP-static"),
                         ("openmp_dynamic", "OpenMP-dynamic"),
                         ("openmp_guided", "OpenMP-guided"),
                         ("mpi", "MPI")]:
        for p in params:
            t  = get_time("conv", model, p, N)
            sp = seq_t / t
            ef = sp / p
            print(f"{N:>6}  {label:>20}  {p:>7}  {t:>10.4f}  {sp:>8.3f}  {ef:>10.3f}")

print("\nAll graphs saved.")
