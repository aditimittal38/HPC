# HPC Assignments — 23610067

High Performance Computing lab assignments covering OpenMP, MPI, Hybrid parallelism, and CUDA.

---

## Assignments

| # | Topic |
|---|-------|
| A1 | Environment Setup & OpenMP Basics |
| A2 | OpenMP Parallel Loops & Scheduling |
| A3 | OpenMP Reduction & Parallel Scan |
| A4 | Data Decomposition Strategies |
| A5 | OpenMP Sections & Tasks |
| A6 | MPI Basics: Point-to-Point Communication |
| A7 | MPI Collective Communication & Matrix Parallelism |
| A8 | Hybrid Parallelism: OpenMP + MPI |
| A9 | CUDA Fundamentals |
| A10 | Advanced CUDA: Tiled Matmul, Reduction, Histogram |
| A11 | CUDA Optimization & cuBLAS |

Each folder contains a `README.md` with theory, task descriptions, and sample output, plus a `run.sh` to compile and run all files.

---

## Setup on Ubuntu Linux

### 1. Install packages

```bash
sudo apt update
sudo apt install -y gcc libgomp1 openmpi-bin libopenmpi-dev \
    python3 python3-matplotlib python3-pandas python3-numpy
```

### 2. Transfer files

```bash
# Option A: SCP from Windows (run on Linux)
scp -r yourwindowsuser@<windows-ip>:"C:/Users/z0054ufe/Desktop/clg/HPC" ~/HPC

# Option B: zip on Windows, transfer, then unzip
unzip HPC.zip -d ~/
```

### 3. Make scripts executable

```bash
cd ~/HPC
chmod +x run.sh
chmod +x 23610067_HPC_A*/run.sh
```

### 4. Run

```bash
./run.sh               # runs A1–A8 (OpenMP + MPI)
./run.sh --cuda        # runs A1–A11 (includes CUDA — requires NVIDIA GPU)
./run.sh --only A3     # runs a single assignment
```

---

## CUDA Setup (A9–A11 only)

Required only if you have an NVIDIA GPU.

```bash
# Verify GPU is detected
nvidia-smi

# Install CUDA toolkit
sudo apt install -y nvidia-cuda-toolkit

# Verify compiler
nvcc --version
```

Then run with `--cuda`:

```bash
./run.sh --cuda
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `not enough slots` from mpirun | Already handled — all MPI scripts use `--oversubscribe` |
| `Aborting — no permission to run as root` | Add `--allow-run-as-root` to `mpirun` lines, or don't run as root |
| `python3: command not found` | `sudo apt install python3` |
| `nvcc: command not found` | `sudo apt install nvidia-cuda-toolkit` |
| Compilation fails for one assignment | Run that assignment individually: `./run.sh --only A6` |

---

## Repo Structure

```
HPC/
├── run.sh                        ← master runner (this repo)
├── 23610067_HPC_A1/
│   ├── README.md                 ← theory + task descriptions
│   ├── run.sh                    ← compile & run all files
│   ├── code/                     ← all source files (t1_*.c, t2_*.c ...)
│   └── 23610067_HPC_A1.pdf      ← submitted report
├── 23610067_HPC_A2/ ...
└── ...
```
