#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include<math.h>
#define M 1024
#define N 100000
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
__global__ void add(float *a, float *b, float *c) {
       int index = threadIdx.x + blockIdx.x * blockDim.x;
       if (index < N) c[index] = a[index] + b[index];
}
inline void check_cuda_errors(const char *filename, const int line_number) {
#ifdef DEBUG
       cudaDeviceSynchronize();
       cudaError_t error = cudaGetLastError();
       if(error != cudaSuccess) {
              printf("CUDA error at %s:%i: %s\n", filename, line_number,     cudaGetErrorString(error));
       exit(-1);
       }
#endif
}
__global__ void foo(int *ptr) {
       *ptr =7;
}

void random_floats(float *a){
       for(int i = 0;i<N;i++){
       
              float x=(float)rand()/((float)RAND_MAX/10);
              a[i] = x;
       }
}

void check_results(float *a,float *b,float *c){
       float d[N];
       for(int i = 0;i<N;i++){
              d[i] = a[i]+b[i];
              if(abs(d[i]-c[i])>1e-12){
                     printf("%d\t%f\t%f\n",i,d[i],c[i]);
                     break;
                     }
       }
}

int main(void) {
       srand(time(NULL));
       float *a, *b, *c;
       int size = N * sizeof(float);
// Allocate space
       cudaMallocManaged((void**)&a, size);
       cudaMallocManaged((void**)&b, size);
       cudaMallocManaged((void**)&c, size);
       random_floats(a); //setup input values
       random_floats(b);
       cudaEvent_t start, stop;
       float time = 0;
       cudaEventCreate(&start);
       cudaEventCreate(&stop);
       cudaEventRecord(start , 0);
       add <<<(N+M-1)/M, M>>> (a, b, c); // Launch add() kernel on GPU
       cudaEventRecord( stop,0);
       cudaEventSynchronize(stop);
       cudaEventElapsedTime(&time, start, stop);
       printf("Elapsed time CUDA: %.2f ms\n", time);
       cudaEventDestroy(start);
       cudaEventDestroy (stop);
       cudaDeviceSynchronize();
       foo<<<N/M, M>>>(0);
       check_cuda_errors(__FILE__, __LINE__);
       check_results(a, b, c);
// Cleanup
       cudaFree(a);
       cudaFree(b);
       cudaFree(c);
       return 0;
}
