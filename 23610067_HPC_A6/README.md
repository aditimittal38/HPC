# HPC Assignment 6 — MPI Basics: Point-to-Point Communication

**PRN:** 23610067 | **Course:** High Performance Computing

---

## Theory

### What is MPI?

MPI (Message Passing Interface) is the standard for **distributed-memory** parallel programming. Unlike OpenMP (shared memory, one node), MPI processes have their own private address space and communicate explicitly by sending and receiving messages over a network (or shared memory on one machine).

Key concepts:
- **Rank:** a process's unique ID within a communicator (0, 1, 2, ...).
- **Communicator:** a group of processes that can talk to each other. `MPI_COMM_WORLD` = all processes.
- **Size:** total number of processes in the communicator.

### MPI Lifecycle

```c
MPI_Init(&argc, &argv);          // initialise MPI runtime
MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // get my rank
MPI_Comm_size(MPI_COMM_WORLD, &size);   // get total processes
// ... parallel work ...
MPI_Finalize();                  // shut down MPI
```

### Point-to-Point Communication

| Function | Description |
|----------|-------------|
| `MPI_Send(buf, count, type, dest, tag, comm)` | Blocking send — returns when message is sent |
| `MPI_Recv(buf, count, type, src, tag, comm, status)` | Blocking receive — blocks until message arrives |
| `MPI_Sendrecv(...)` | Send and receive simultaneously — avoids deadlock |

**Blocking** means the call does not return until the operation is complete. If all processes block on `Send` waiting for a `Recv`, you get a **deadlock** — avoided by careful ordering or using `MPI_Sendrecv`.

### MPI Groups & Communicators

`MPI_Comm_group` extracts the process group from a communicator. `MPI_Group_incl` creates a new group containing a subset of ranks. `MPI_Comm_create` builds a new communicator from a group — lets you restrict communication to a subset of processes.

### Communication Time & Bandwidth

```
Latency (α)    = time for an empty message (overhead)
Bandwidth (β)  = bytes per second of sustained transfer
Transfer time  = α + message_size / β
```

Measured by sending messages of increasing sizes and timing them with `MPI_Wtime()`.

---

## Tasks

### t2 — Hello MPI (`t2_hello-mpi.c`)
Every process prints its rank and the total number of processes. The entry point for any MPI program.

**Functions:**
- `main(argc, argv)` — Init → Comm_rank → Comm_size → printf → Finalize.

### t3 — Rank & Size (`t3_rank-size.c`)
Each process reports its rank and total size. Demonstrates that every process runs the same code but with a different rank.

**Functions:**
- `main(argc, argv)` — Init, Comm_rank, Comm_size, conditional print.

### t4 — Group Communication (`t4_group-comm.c`)
Creates a sub-communicator containing only even-ranked processes, then has only those processes communicate within it.

**Functions:**
- `main(argc, argv)` — extracts world group, creates an even-rank group via `MPI_Group_incl`, creates a new communicator with `MPI_Comm_create`, prints rank within both.

### t5 — Send & Receive (string) (`t5_send-receive.c`)
Process 0 sends a string message to process 1; process 1 receives and prints it.

**Functions:**
- `main(argc, argv)` — rank 0: `MPI_Send` of a char array; rank 1: `MPI_Recv` into buffer and print.

### t6 — Integer Send (`t6_integer-send.c`)
Process 0 sends a single integer to all other processes. Demonstrates basic non-root receives.

**Functions:**
- `main(argc, argv)` — rank 0 loops over ranks 1..size-1, sending an integer; others receive.

### t7 — Array Send (`t7_array-send.c`)
Process 0 sends an array to process 1. Demonstrates `count > 1` in `MPI_Send`.

**Functions:**
- `main(argc, argv)` — rank 0: `MPI_Send` of int array with count > 1; rank 1: `MPI_Recv` and print.

### c1 — Ping-Pong Benchmark (`c1_ping-pong.c`)
Measures round-trip communication time between two processes. Process 0 sends, process 1 bounces it back; time / 2 = one-way latency.

**Functions:**
- `main(argc, argv)` — loops over message sizes (1 B to 1 MB), measures RTT with `MPI_Wtime`, reports latency and bandwidth.

### ca — Non-blocking SendRecv (`ca_send-recv-nonblock.c`)
Advanced challenge: uses `MPI_Sendrecv` to simultaneously send and receive between processes — avoids the deadlock risk of paired `Send`/`Recv`.

**Functions:**
- `main(argc, argv)` — each process sends its rank to its "partner" and receives simultaneously.

### pa — Communication Time Profiler (`pa_communication-time.c`)
Performance activity: measures communication time as a function of message size, prints a table suitable for plotting latency and bandwidth.

**Functions:**
- `main(argc, argv)` — sweeps message sizes, times `MPI_Send`/`MPI_Recv` pairs, prints size vs time table.

---

## Prerequisites

```bash
mpicc --version    # OpenMPI or MPICH
mpirun --version
```

---

## Files

| File | Description |
|------|-------------|
| `code/t2_hello-mpi.c` | Hello World in MPI |
| `code/t3_rank-size.c` | Rank and size identification |
| `code/t4_group-comm.c` | Sub-communicator from even ranks |
| `code/t5_send-receive.c` | String message pass |
| `code/t6_integer-send.c` | Integer broadcast via send/recv |
| `code/t7_array-send.c` | Array point-to-point transfer |
| `code/c1_ping-pong.c` | Ping-pong latency/bandwidth |
| `code/ca_send-recv-nonblock.c` | MPI_Sendrecv simultaneous |
| `code/pa_communication-time.c` | Communication time profiler |
| `23610067_HPC_A6.pdf` | Submitted report |
