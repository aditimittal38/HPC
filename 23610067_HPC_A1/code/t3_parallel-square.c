#include<stdio.h> 
#include<omp.h> 
int main() { 
    int i; 
    #pragma omp parallel for 
    for(i=0; i<1000; i++) {
        int square = i * i; 
        printf("Thread ID: %d | Total Threads: %d | Number: %d | Square: %d\n", omp_get_thread_num(), omp_get_num_threads(), i, square);
    }
        return 0; 
}
// gcc -fopenmp Task2/square.c -o square