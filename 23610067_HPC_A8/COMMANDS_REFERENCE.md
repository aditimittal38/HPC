# A8 Commands Reference
## 23610067 – HPC Assignment 8

All commands run inside **WSL Ubuntu** unless noted.
Open WSL: `wsl -d Ubuntu` in Windows Terminal / PowerShell.

---

## 0. Navigate to the A8 directory

```bash
cd /mnt/c/Users/z0054ufe/Desktop/clg/HPC/23610067_HPC_A8
```

---

## 1. Task 1 – Matrix–Matrix Multiplication

### Compile

```bash
cd Task1

# Sequential
gcc -O2 -o matrix-matrix-seq matrix-matrix-seq.c -lm

# OpenMP
gcc -O2 -fopenmp -o matrix-matrix-omp matrix-matrix-omp.c -lm

# MPI
mpicc -O2 -o matrix-matrix-mpi matrix-matrix-mpi.c -lm
```

### Run – Sequential

```bash
# Usage: ./matrix-matrix-seq <matrix_size>
./matrix-matrix-seq 500
./matrix-matrix-seq 1000
./matrix-matrix-seq 2000
```

### Run – OpenMP

```bash
# Usage: ./matrix-matrix-omp <matrix_size> <num_threads>
./matrix-matrix-omp 500  2
./matrix-matrix-omp 500  4
./matrix-matrix-omp 500  8

./matrix-matrix-omp 1000 2
./matrix-matrix-omp 1000 4
./matrix-matrix-omp 1000 8

./matrix-matrix-omp 2000 2
./matrix-matrix-omp 2000 4
./matrix-matrix-omp 2000 8
```

### Run – MPI

```bash
# Usage: mpirun -n <num_processes> ./matrix-matrix-mpi <matrix_size>
mpirun -n 2 ./matrix-matrix-mpi 500
mpirun -n 4 ./matrix-matrix-mpi 500
mpirun -n 8 ./matrix-matrix-mpi 500

mpirun -n 2 ./matrix-matrix-mpi 1000
mpirun -n 4 ./matrix-matrix-mpi 1000
mpirun -n 8 ./matrix-matrix-mpi 1000

mpirun -n 2 ./matrix-matrix-mpi 2000
mpirun -n 4 ./matrix-matrix-mpi 2000
mpirun -n 8 ./matrix-matrix-mpi 2000
```

---

## 2. Task 2 – 2D Convolution

### Compile

```bash
cd ../Task2

# Sequential
gcc -O2 -o conv-seq conv-seq.c -lm

# OpenMP
gcc -O2 -fopenmp -o conv-omp conv-omp.c -lm

# MPI
mpicc -O2 -o conv-mpi conv-mpi.c -lm
```

### Run – Sequential

```bash
# Usage: ./conv-seq <input_size>
./conv-seq 1000
./conv-seq 2000
./conv-seq 4000
```

### Run – OpenMP

```bash
# Usage: ./conv-omp <input_size> <num_threads> <schedule>
# schedule: 0=static  1=dynamic  2=guided

# Static schedule
./conv-omp 1000 2 0
./conv-omp 1000 4 0
./conv-omp 1000 8 0

./conv-omp 2000 2 0
./conv-omp 2000 4 0
./conv-omp 2000 8 0

./conv-omp 4000 2 0
./conv-omp 4000 4 0
./conv-omp 4000 8 0

# Dynamic schedule
./conv-omp 2000 4 1

# Guided schedule
./conv-omp 2000 4 2
```

### Run – MPI

```bash
# Usage: mpirun -n <num_processes> ./conv-mpi <input_size>
mpirun -n 2 ./conv-mpi 1000
mpirun -n 4 ./conv-mpi 1000
mpirun -n 8 ./conv-mpi 1000

mpirun -n 2 ./conv-mpi 2000
mpirun -n 4 ./conv-mpi 2000
mpirun -n 8 ./conv-mpi 2000

mpirun -n 2 ./conv-mpi 4000
mpirun -n 4 ./conv-mpi 4000
mpirun -n 8 ./conv-mpi 4000
```

---

## 3. Run All Experiments at Once (automated)

```bash
cd /mnt/c/Users/z0054ufe/Desktop/clg/HPC/23610067_HPC_A8/Performance
bash run_experiments.sh
# Results saved to: Performance/results.csv
```

---

## 4. Generate Graphs and Performance Tables

Run from Windows (Python installed):

```cmd
cd C:\Users\z0054ufe\Desktop\clg\HPC
python 23610067_HPC_A8\Performance\generate_graphs.py
```

Generated PNG files in `Performance/`:
- `matmul_threads_vs_time.png`
- `matmul_threads_vs_speedup.png`
- `matmul_threads_vs_efficiency.png`
- `matmul_size_vs_time.png`
- `conv_threads_vs_time.png`
- `conv_threads_vs_speedup.png`
- `conv_threads_vs_efficiency.png`
- `conv_size_vs_time.png`

---

## 5. Build Completed Report

```cmd
cd C:\Users\z0054ufe\Desktop\clg\HPC
python 23610067_HPC_A8\Performance\build_report.py
# Output: 23610067_HPC_A8\23610067_HPC_A8_completed.docx
```

---

## 6. Check CPU Info (WSL)

```bash
nproc                          # number of logical cores
lscpu | grep -E "Model|Socket|Core|Thread"
cat /proc/cpuinfo | grep "model name" | head -1
```

---

## 7. Full Sequence to Reproduce Everything from Scratch

```bash
# Step 1 – compile (run inside WSL)
wsl -d Ubuntu -- sh -c '
  cd /mnt/c/Users/z0054ufe/Desktop/clg/HPC/23610067_HPC_A8/Task1 &&
  gcc -O2 -o matrix-matrix-seq matrix-matrix-seq.c -lm &&
  gcc -O2 -fopenmp -o matrix-matrix-omp matrix-matrix-omp.c -lm &&
  mpicc -O2 -o matrix-matrix-mpi matrix-matrix-mpi.c -lm &&
  echo "Task1 compiled OK"
'

wsl -d Ubuntu -- sh -c '
  cd /mnt/c/Users/z0054ufe/Desktop/clg/HPC/23610067_HPC_A8/Task2 &&
  gcc -O2 -o conv-seq conv-seq.c -lm &&
  gcc -O2 -fopenmp -o conv-omp conv-omp.c -lm &&
  mpicc -O2 -o conv-mpi conv-mpi.c -lm &&
  echo "Task2 compiled OK"
'

# Step 2 – run experiments (run inside WSL)
wsl -d Ubuntu -- bash /mnt/c/Users/z0054ufe/Desktop/clg/HPC/23610067_HPC_A8/Performance/run_experiments.sh

# Step 3 – generate graphs & report (run in Windows cmd/PowerShell)
python C:\Users\z0054ufe\Desktop\clg\HPC\23610067_HPC_A8\Performance\generate_graphs.py
python C:\Users\z0054ufe\Desktop\clg\HPC\23610067_HPC_A8\Performance\build_report.py
```

---

## File Structure

```
23610067_HPC_A8/
├── Task1/
│   ├── matrix-matrix-seq.c      # Sequential matmul
│   ├── matrix-matrix-omp.c      # OpenMP matmul
│   ├── matrix-matrix-mpi.c      # MPI matmul
│   ├── matrix-matrix-seq        # Compiled binary (Linux/WSL)
│   ├── matrix-matrix-omp        # Compiled binary
│   └── matrix-matrix-mpi        # Compiled binary
├── Task2/
│   ├── conv-seq.c               # Sequential 2D convolution
│   ├── conv-omp.c               # OpenMP 2D convolution
│   ├── conv-mpi.c               # MPI 2D convolution (with halo rows)
│   ├── conv-seq                 # Compiled binary
│   ├── conv-omp                 # Compiled binary
│   └── conv-mpi                 # Compiled binary
├── Performance/
│   ├── run_experiments.sh       # Bash script: compile + run all
│   ├── results.csv              # Raw timing data
│   ├── generate_graphs.py       # Python: generate 8 PNG graphs
│   ├── build_report.py          # Python: build completed docx
│   ├── matmul_threads_vs_time.png
│   ├── matmul_threads_vs_speedup.png
│   ├── matmul_threads_vs_efficiency.png
│   ├── matmul_size_vs_time.png
│   ├── conv_threads_vs_time.png
│   ├── conv_threads_vs_speedup.png
│   ├── conv_threads_vs_efficiency.png
│   └── conv_size_vs_time.png
├── 23610067_HPC_A8.docx         # Original template
├── 23610067_HPC_A8_completed.docx  # Completed report (SUBMIT THIS)
└── COMMANDS_REFERENCE.md        # This file
```
