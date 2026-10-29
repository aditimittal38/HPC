// gcc -fopenmp Task1/openmp-verification.c -o openmp-verification

#include<stdio.h>
#include<omp.h>

int main()
{
    #pragma omp parallel
    {
        printf("Hello from Thread %d\n", omp_get_thread_num());
    }

    return 0;
}
