#!/bin/bash
# HPC Assignments — Master Runner
# Run from the root HPC directory (the folder containing all A1–A11 subdirs).
#
# ── Prerequisites (Ubuntu) ───────────────────────────────────────────────────
#   sudo apt update
#   sudo apt install gcc libgomp1 openmpi-bin libopenmpi-dev python3
#
# A9–A11 are CUDA assignments and require an NVIDIA GPU + CUDA toolkit:
#   sudo apt install nvidia-cuda-toolkit
#   (or install via https://developer.nvidia.com/cuda-downloads)
#   They are skipped by default; pass --cuda to include them.
#
# ── Usage ────────────────────────────────────────────────────────────────────
#   chmod +x run.sh        # make executable (first time only)
#   ./run.sh               # run A1–A8 (OpenMP + MPI)
#   ./run.sh --cuda        # run A1–A11 (includes CUDA assignments)
#   ./run.sh --only A3     # run a single assignment, e.g. A3
# ─────────────────────────────────────────────────────────────────────────────

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

RUN_CUDA=false
ONLY=""

for arg in "$@"; do
    case "$arg" in
        --cuda)   RUN_CUDA=true ;;
        --only)   ;;
        A[0-9]|A[0-9][0-9]) ONLY="$arg" ;;
    esac
done

# Parse --only <assignment>
for i in "$@"; do
    if [ "$PREV" = "--only" ]; then ONLY="$i"; fi
    PREV="$i"
done

run_assignment() {
    local name="$1"       # e.g. "A1"
    local dir="$2"        # full path to assignment folder
    local label="$3"      # human-readable title

    if [ -n "$ONLY" ] && [ "$ONLY" != "$name" ]; then
        return
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  ${name}: ${label}"
    echo "════════════════════════════════════════════════════════════════"

    if [ ! -f "$dir/run.sh" ]; then
        echo "  [SKIP] run.sh not found in $dir"
        return
    fi

    bash "$dir/run.sh" || echo "  [WARN] ${name} exited with errors — continuing."
}

run_assignment "A1" "$ROOT/23610067_HPC_A1"  "Environment Setup & OpenMP Basics"
run_assignment "A2" "$ROOT/23610067_HPC_A2"  "OpenMP Parallel Loops & Scheduling"
run_assignment "A3" "$ROOT/23610067_HPC_A3"  "OpenMP Reduction & Parallel Scan"
run_assignment "A4" "$ROOT/23610067_HPC_A4"  "Data Decomposition Strategies"
run_assignment "A5" "$ROOT/23610067_HPC_A5"  "OpenMP Sections & Tasks"
run_assignment "A6" "$ROOT/23610067_HPC_A6"  "MPI Basics: Point-to-Point"
run_assignment "A7" "$ROOT/23610067_HPC_A7"  "MPI Collective Communication & Matrix Parallelism"
run_assignment "A8" "$ROOT/23610067_HPC_A8"  "Hybrid Parallelism: OpenMP + MPI"

if [ "$RUN_CUDA" = true ]; then
    run_assignment "A9"  "$ROOT/23610067_HPC_A9"  "CUDA Fundamentals"
    run_assignment "A10" "$ROOT/23610067_HPC_A10" "Advanced CUDA: Tiled Matmul, Reduction, Histogram"
    run_assignment "A11" "$ROOT/23610067_HPC_A11" "CUDA Optimization & cuBLAS"
else
    if [ -z "$ONLY" ]; then
        echo ""
        echo "────────────────────────────────────────────────────────────────"
        echo "  A9–A11 (CUDA) skipped. Run with --cuda to include them."
        echo "  Requires: NVIDIA GPU + nvidia-cuda-toolkit"
        echo "────────────────────────────────────────────────────────────────"
    fi
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  All done."
echo "════════════════════════════════════════════════════════════════"
