"""Collect Task 4 timings in Ubuntu WSL and build the performance report."""

import csv
import re
import statistics
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
SIZES = (500, 1000, 2000)
PROCESSES = (2, 4, 8)
REPEATS = 3
WSL = ("wsl", "-d", "Ubuntu", "--")
METRICS = {
    "mpi": "MPI Time",
    "broadcast": "Broadcast Time",
    "scatter": "Scatter Time",
    "compute": "Computation Time",
    "min_compute": "Minimum Computation Time",
    "gather": "Gather Time",
}


def run(*command):
    return subprocess.run(
        [*WSL, *map(str, command)], check=True, capture_output=True, text=True
    ).stdout


def measure(output, label):
    match = re.search(rf"^{re.escape(label)} = ([0-9.]+) seconds$", output, re.M)
    if not match:
        raise ValueError(f"Missing {label} in output:\n{output}")
    return float(match.group(1))


def save_csv(filename, rows, columns):
    with (HERE / filename).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot(filename, title, xlabel, ylabel, xvalues, series, percent=False):
    figure, axis = plt.subplots(figsize=(8, 5))
    for label, values in series.items():
        axis.plot(xvalues, values, marker="o", linewidth=2, label=label)
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel, xticks=xvalues)
    if percent:
        axis.set_ylim(bottom=0)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(HERE / filename, dpi=160)
    plt.close(figure)


def sequential_time(binary, size):
    output = run(binary, size)
    checksum = re.search(r"^Result checksum = ([0-9.]+)$", output, re.M)
    if not checksum or float(checksum.group(1)) != size * size:
        raise ValueError(f"Unexpected sequential result for N={size}:\n{output}")
    return measure(output, "Sequential Time")


def mpi_measurements(binary, size, processes):
    output = run("mpirun", "-np", processes, binary, size)
    checksum = re.search(r"^Result checksum = ([0-9.]+)$", output, re.M)
    if not checksum or float(checksum.group(1)) != size * size:
        raise ValueError(f"Unexpected MPI result for N={size}, p={processes}:\n{output}")
    return {key: measure(output, label) for key, label in METRICS.items()}


def collect_results(seq_binary, mpi_binary):
    samples = []
    results = []
    for size in SIZES:
        seq_times = []
        for repetition in range(1, REPEATS + 1):
            seconds = sequential_time(seq_binary, size)
            seq_times.append(seconds)
            samples.append({"size": size, "processes": 1, "repeat": repetition,
                            "metric": "sequential", "seconds": seconds})

        sequential = statistics.mean(seq_times)
        for processes in PROCESSES:
            readings = {key: [] for key in METRICS}
            for repetition in range(1, REPEATS + 1):
                for key, seconds in mpi_measurements(mpi_binary, size, processes).items():
                    readings[key].append(seconds)
                    samples.append({"size": size, "processes": processes,
                                    "repeat": repetition, "metric": key, "seconds": seconds})

            averages = {key: statistics.mean(values) for key, values in readings.items()}
            speedup = sequential / averages["mpi"]
            results.append({"size": size, "processes": processes,
                            "sequential_seconds": sequential,
                            "mpi_seconds": averages["mpi"],
                            "speedup": speedup,
                            "efficiency_percent": 100 * speedup / processes,
                            **{f"{key}_seconds": averages[key] for key in METRICS
                               if key != "mpi"}})
            print(f"N={size}, p={processes}: {averages['mpi']:.6f} s, "
                  f"speedup={speedup:.2f}", flush=True)

    return samples, results


def main():
    source_dir = f"/mnt/{HERE.drive[0].lower()}/{'/'.join(HERE.parts[1:])}"
    build_dir = run("mktemp", "-d").strip()
    seq_binary = f"{build_dir}/matrix-vector-seq"
    mpi_binary = f"{build_dir}/matrix-vector-mpi"
    run("gcc", "-O2", f"{source_dir}/matrix-vector-seq.c", "-o", seq_binary)
    run("mpicc", "-O2", f"{source_dir}/matrix-vector-mpi.c", "-o", mpi_binary)
    samples, results = collect_results(seq_binary, mpi_binary)

    save_csv("performance_samples.csv", samples,
             ("size", "processes", "repeat", "metric", "seconds"))
    save_csv("performance_results.csv", results, tuple(results[0]))

    by_size = {size: [row for row in results if row["size"] == size]
               for size in SIZES}
    plot("processes_vs_time.png", "Processes vs execution time", "Processes",
         "MPI time (ms)", PROCESSES,
         {f"{size} x {size}": [1000 * row["mpi_seconds"] for row in rows]
          for size, rows in by_size.items()})
    plot("processes_vs_speedup.png", "Processes vs speedup", "Processes",
         "Speedup", PROCESSES,
         {f"{size} x {size}": [row["speedup"] for row in rows]
          for size, rows in by_size.items()})
    plot("processes_vs_efficiency.png", "Processes vs efficiency", "Processes",
         "Efficiency (%)", PROCESSES,
         {f"{size} x {size}": [row["efficiency_percent"] for row in rows]
          for size, rows in by_size.items()}, percent=True)
    plot("size_vs_time.png", "Matrix size vs execution time", "Matrix dimension N",
         "Time (ms)", SIZES,
         {"Sequential": [1000 * by_size[size][0]["sequential_seconds"]
                         for size in SIZES],
          **{f"MPI p={processes}": [1000 * next(
              row["mpi_seconds"] for row in by_size[size]
              if row["processes"] == processes) for size in SIZES]
             for processes in PROCESSES}})

    lines = [
        "# A7 Task 4: MPI matrix-vector performance analysis", "",
        "## Method",
        f"Compiled with `-O2` in Ubuntu WSL; {REPEATS} independent runs per "
        "configuration. Values below are arithmetic means of the program's "
        "internal wall-clock timings, excluding allocation, initialization, "
        "process startup, and printing. MPI time includes broadcast, scatter, "
        "computation, and gather; it is the maximum across ranks. Sequential "
        "time covers computation only. MPI collective phase maxima can come "
        "from different ranks and must not be summed as elapsed time.", "",
        "Speedup = sequential time / MPI time; efficiency = 100 x speedup / p. "
        "Both sequential and MPI result checksums are checked for every run.", "",
        "## Results", "",
        "| N x N | Processes | Sequential (ms) | MPI (ms) | Speedup | Efficiency (%) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(f"| {row['size']} x {row['size']} | {row['processes']} | "
                     f"{1000 * row['sequential_seconds']:.3f} | "
                     f"{1000 * row['mpi_seconds']:.3f} | "
                     f"{row['speedup']:.2f} | {row['efficiency_percent']:.1f} |")

    lines += ["", "## Graphs", "",
              "![Processes vs execution time](processes_vs_time.png)", "",
              "![Processes vs speedup](processes_vs_speedup.png)", "",
              "![Processes vs efficiency](processes_vs_efficiency.png)", "",
              "![Matrix size vs execution time](size_vs_time.png)", "",
              "## Communication, computation, and load imbalance", "",
              "The following times (ms) are mean per-run maximum rank durations "
              "for each phase. The minimum computation column is the mean of "
              "per-run rank minima; it helps identify variation between ranks.", "",
              "| N | p | Broadcast | Scatter | Compute min | Compute max | Gather |",
              "|---:|---:|---:|---:|---:|---:|---:|",]
    for row in results:
        lines.append(f"| {row['size']} | {row['processes']} | "
                     + " | ".join(f"{1000 * row[key + '_seconds']:.3f}"
                                  for key in ("broadcast", "scatter", "min_compute",
                                              "compute", "gather")) + " |")
    lines += ["", "Broadcast, scatter, and gather include waiting inside MPI "
              "collectives, so these timings are communication-phase costs, "
              "not pure network transfer. Differences between maximum and "
              "minimum local compute times indicate possible imbalance, but "
              "also include scheduler and cache effects. Rows are distributed "
              "as evenly as possible (at most one row difference).", "",
              "## Interpretation", "",
              f"All nine configurations ran slower than sequential here: "
              f"the best measured speedup was {max(row['speedup'] for row in results):.2f}. "
              f"At N=2000, increasing from 2 to 8 ranks reduced MPI time "
              f"from {1000 * by_size[2000][0]['mpi_seconds']:.2f} to "
              f"{1000 * by_size[2000][2]['mpi_seconds']:.2f} ms, but "
              f"efficiency at 8 ranks was only "
              f"{by_size[2000][2]['efficiency_percent']:.1f}%. "
              f"Scatter took {1000 * by_size[2000][2]['scatter_seconds']:.2f} ms "
              f"at N=2000, p=8, compared with "
              f"{1000 * by_size[2000][2]['compute_seconds']:.2f} ms of "
              "maximum local computation. Communication-phase costs and "
              "waiting dominate this implementation. For strong scaling, "
              "compare process counts at fixed N; increasing N grows both "
              "the N-squared matrix data and computation. Differences in "
              "minimum and maximum computation time may indicate imbalance "
              "or scheduling effects. These short WSL measurements are noisy "
              "and cannot establish statistical significance.", "",
              "Reproduce from PowerShell: `py -3.14 A7/Task4/performance_analysis.py` "
              "(requires Matplotlib and Ubuntu WSL with gcc, mpicc, and mpirun).", ""]
    (HERE / "performance_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()