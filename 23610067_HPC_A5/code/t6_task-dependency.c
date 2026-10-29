/*
 * Task 6: Task dependency graph A -> {B, C} -> D using depend clauses
 * Compile: gcc -fopenmp Task6/task-dependency.c -o task-dependency
 * Run:     ./task-dependency
 */
#include <stdio.h>
#include <unistd.h>
#include <omp.h>

void taskA(int *x) { printf("Task A running\n"); *x = 1; sleep(1); }
void taskB(int *x, int *y) { printf("Task B running\n"); *y = *x + 1; sleep(1); }
void taskC(int *x, int *z) { printf("Task C running\n"); *z = *x + 2; sleep(1); }
void taskD(int *y, int *z) { printf("Task D running, result = %d\n", *y + *z); }

int main() {
    int x, y, z;
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task depend(out: x)
            taskA(&x);

            #pragma omp task depend(in: x) depend(out: y)
            taskB(&x, &y);

            #pragma omp task depend(in: x) depend(out: z)
            taskC(&x, &z);

            #pragma omp task depend(in: y, z)
            taskD(&y, &z);
        }
    }
    return 0;
}